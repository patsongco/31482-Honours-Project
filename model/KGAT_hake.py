import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
from utility.helper import edge_softmax_fix

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gcn':
            self.W = nn.Linear(self.in_dim, self.out_dim)       # W in Equation (6)
        elif aggregator_type == 'graphsage':
            self.W = nn.Linear(self.in_dim * 2, self.out_dim)   # W in Equation (7)
        elif aggregator_type == 'bi-interaction':
            self.W1 = nn.Linear(self.in_dim, self.out_dim)      # W1 in Equation (8)
            self.W2 = nn.Linear(self.in_dim, self.out_dim)      # W2 in Equation (8)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()


    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        # Equation (3) & (10)
        # DGL: dgl-cu90(0.4.1)
        # Get different results when using `dgl.function.sum`, and the randomness is due to `atomicAdd`
        # Use `dgl.function.sum` when training model to speed up
        # Use custom function to ensure deterministic behavior when predicting
        if mode == 'predict':
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), lambda nodes: {'N_h': torch.sum(nodes.mailbox['side'], 1)})
        else:
            g.update_all(dgl.function.u_mul_e('node', 'att', 'side'), dgl.function.sum('side', 'N_h'))

        if self.aggregator_type == 'gcn':
            # Equation (6) & (9)
            out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))                         # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'graphsage':
            # Equation (7) & (9)
            out = self.activation(self.W(torch.cat([g.ndata['node'], g.ndata['N_h']], dim=1)))      # (n_users + n_entities, out_dim)

        elif self.aggregator_type == 'bi-interaction':
            # Equation (8) & (9)
            out1 = self.activation(self.W1(g.ndata['node'] + g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out2 = self.activation(self.W2(g.ndata['node'] * g.ndata['N_h']))                       # (n_users + n_entities, out_dim)
            out = out1 + out2
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out


class KGAT_hake(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT_hake, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.aggregation_type = args.aggregation_type
        self.conv_dim_list = [args.entity_dim*2] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        #added hake hyperparameters
        self.adversarial_temperature = args.adversarial_temperature
        self.modulus_weight = args.modulus_weight
        self.phase_weight = args.phase_weight
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.entity_dim]),
            requires_grad=False
        )

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim*2)
        self.entity_user_embed.weight.data.uniform_(-self.embedding_range.item(), self.embedding_range.item())

        self.relation_embedding = nn.Parameter(torch.zeros(self.n_relations, self.relation_dim*3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, self.relation_dim:2 * self.relation_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * self.relation_dim:3 * self.relation_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[self.phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[self.modulus_weight]]))

        self.pi = 3.14159262358979323846

        if (self.use_pretrain == 1) and (user_pre_embed is not None) and (item_pre_embed is not None):
            other_entity_embed = nn.Parameter(torch.Tensor(self.n_entities - item_pre_embed.shape[0], self.entity_dim))
            nn.init.xavier_uniform_(other_entity_embed, gain=nn.init.calculate_gain('relu'))
            entity_user_embed = torch.cat([item_pre_embed, other_entity_embed, user_pre_embed], dim=0)
            self.entity_user_embed.weight = nn.Parameter(entity_user_embed)

        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type))


    def att_score(self, edges):
        # Equation (4) modified to refelect HAKE distance score function
        h_embed = self.entity_user_embed(edges.dst['id']).unsqueeze(1)                              #(n_edge, 1, entity_dim * 2)
        t_embed = self.entity_user_embed(edges.src['id']).unsqueeze(1)                              #(n_edge, 1, entity_dim * 2)
        r_embed = torch.index_select(                                                               #(n_edge, 1, entity_dim * 3)
            self.relation_embedding,
            dim=0,
            index=edges.data['type']
        ).unsqueeze(1)

        #split embeddings into phase/modulus parts
        phase_head, mod_head = torch.chunk(h_embed, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(r_embed, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(t_embed, 2, dim=2)

        #constrain phase part between 0-2pi
        phase_head = phase_head / (self.embedding_range.item() / self.pi)                            #(n_edge, 1, entity_dim)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)                    #(n_edge, 1, entity_dim)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)                            #(n_edge, 1, entity_dim)

        #restrict mod_relation to positive
        mod_relation = torch.abs(mod_relation)                                                       #(n_edge, 1, entity_dim)
        bias_relation = torch.clamp(bias_relation, max=1)                                            #(n_edge, 1, entity_dim)
        indicator = (bias_relation < -mod_relation)                                                  #(n_edge, 1, entity_dim)
        bias_relation[indicator] = -mod_relation[indicator]

        #calculate modulus score: HAKE:eq(1)
        mod_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)
        mod_score = torch.norm(mod_score, dim=2) * self.modulus_weight                                # (n_edge, 1)

        #calculate phase score: HAKE:eq(2)
        phase_score = phase_head + (phase_relation - phase_tail)
        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight     # (n_edge, 1)

        #calcualte distance function score: HAKE:eq(3)
        att = (phase_score + mod_score)                                                               # (n_edge, 1)
        return {'att': att}


    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
            #apply_edges: Update the features of the specified edges by the provided function.
            g.apply_edges(self.att_score, edge_idxs)

        # Equation (5)
        g.edata['att'] = edge_softmax_fix(g, g.edata.pop('att'))
        return g.edata.pop('att')


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """

        # Equation (4) modified to refelect HAKE distance score function
        h_embed = self.entity_user_embed(h).unsqueeze(1)                                                    #(kg_batch_size, 1, entity_dim * 2)
        pos_t_embed = self.entity_user_embed(pos_t).unsqueeze(1)                                            #(kg_batch_size, 1, entity_dim * 2)
        neg_t_embed = self.entity_user_embed(neg_t).unsqueeze(1)                                            #(kg_batch_size, 1, entity_dim * 2)
        r_embed = torch.index_select(                                                                       #(kg_batch_size, 1, entity_dim * 3)
            self.relation_embedding,
            dim=0,
            index=r
        ).unsqueeze(1)

        #split embeddings into phase/modulus parts
        phase_head, mod_head = torch.chunk(h_embed, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(r_embed, 3, dim=2)
        pos_phase_tail, pos_mod_tail = torch.chunk(pos_t_embed, 2, dim=2)
        neg_phase_tail, neg_mod_tail = torch.chunk(neg_t_embed, 2, dim=2)

        #constrain phase part between 0-2pi
        phase_head = phase_head / (self.embedding_range.item() / self.pi)                                   #(kg_batch_size, 1, entity_dim)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)                           #(kg_batch_size, 1, entity_dim)
        pos_phase_tail = pos_phase_tail / (self.embedding_range.item() / self.pi)                           #(kg_batch_size, 1, entity_dim)
        neg_phase_tail = neg_phase_tail / (self.embedding_range.item() / self.pi)                           #(kg_batch_size, 1, entity_dim)

        #restrict mod_relation to positive  
        mod_relation = torch.abs(mod_relation)                                                              #(kg_batch_size, 1, relation_dim)
        bias_relation = torch.clamp(bias_relation, max=1)                                                   #(kg_batch_size, 1, relation_dim)
        indicator = (bias_relation < -mod_relation)                                                         #(kg_batch_size, 1, relation_dim)
        bias_relation[indicator] = -mod_relation[indicator]

        #calculate positive modulus score: HAKE:eq(1)
        pos_mod_score = mod_head * (mod_relation + bias_relation) - pos_mod_tail * (1 - bias_relation)
        pos_mod_score = torch.norm(pos_mod_score, dim=2) * self.modulus_weight                              # (kg_batch_size, 1)

        #calculate negative modulus score: HAKE:eq(1)
        neg_mod_score = mod_head * (mod_relation + bias_relation) - neg_mod_tail * (1 - bias_relation)
        neg_mod_score = torch.norm(neg_mod_score, dim=2) * self.modulus_weight                              # (kg_batch_size, 1)

        #calculate positive phase score: HAKE:eq(2)
        pos_phase_score = phase_head + (phase_relation - pos_phase_tail)
        pos_phase_score = torch.sum(torch.abs(torch.sin(pos_phase_score / 2)), dim=2) * self.phase_weight   # (kg_batch_size, 1)

        #calculate negative phase score: HAKE:eq(2)
        neg_phase_score = phase_head + (phase_relation - neg_phase_tail)
        neg_phase_score = torch.sum(torch.abs(torch.sin(neg_phase_score / 2)), dim=2) * self.phase_weight   # (kg_batch_size, 1)

        #calcualte positive distance function score: HAKE:eq(3)
        pos_score = self.gamma.item() - (pos_phase_score + pos_mod_score)                                   # (kg_batch_size, 1)
        
        #calcualte negative distance function score: HAKE:eq(3)
        neg_score = self.gamma.item() - (neg_phase_score + neg_mod_score)                                   # (kg_batch_size, 1)
        
        #calculate loss function: HAKE:eq(4)
        negative_score = (F.softmax(neg_score * self.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-neg_score)).sum(dim=1)

        positive_score = F.logsigmoid(pos_score).squeeze(dim=1)

        # positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        # negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        kg_loss = (-positive_score - negative_score) / 2
        kg_loss = torch.mean(kg_loss)

        #l2 loss only applied to modulus part
        l2_loss = _L2_loss_mean(mod_head) + _L2_loss_mean(mod_relation) + _L2_loss_mean(bias_relation) + _L2_loss_mean(pos_mod_tail) + _L2_loss_mean(neg_mod_tail)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def cf_embedding(self, mode, g):
        g = g.local_var()
        ego_embed = self.entity_user_embed(g.ndata['id'])
        all_embed = [ego_embed]

        for i, layer in enumerate(self.aggregator_layers):
            ego_embed = layer(mode, g, ego_embed)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        # Equation (11)
        all_embed = torch.cat(all_embed, dim=1)         # (n_users + n_entities, cf_concat_dim)
        return all_embed


    def cf_score(self, mode, g, user_ids, item_ids):
        """
        user_ids:   number of users to evaluate   (n_eval_users)
        item_ids:   number of items to evaluate   (n_eval_items)
        """
        all_embed = self.cf_embedding(mode, g)          # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                # (n_eval_users, cf_concat_dim)
        item_embed = all_embed[item_ids]                # (n_eval_items, cf_concat_dim)

        # Equation (12)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))    # (n_eval_users, n_eval_items)
        return cf_score


    def calc_cf_loss(self, mode, g, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        all_embed = self.cf_embedding(mode, g)                      # (n_users + n_entities, cf_concat_dim)
        user_embed = all_embed[user_ids]                            # (cf_batch_size, cf_concat_dim (128+128+64+32 = 352))
        item_pos_embed = all_embed[item_pos_ids]                    # (cf_batch_size, cf_concat_dim)
        item_neg_embed = all_embed[item_neg_ids]                    # (cf_batch_size, cf_concat_dim)

        # Equation (12)
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)   # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)   # (cf_batch_size)

        # Equation (13)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_cf_loss':
            return self.calc_cf_loss(mode, *input)
        if mode == 'calc_kg_loss':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.cf_score(mode, *input)


