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
            #DGLGraph.update_all(message_func, reduce_func)
            #dgl.function.u_mul_e(lhs_field, rhs_field, out)
            #dgl.function.sum(msg, out)
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


class KGAT_rotatE(nn.Module):

    def __init__(self, args,
                 n_users, n_entities, n_relations,
                 user_pre_embed=None, item_pre_embed=None):

        super(KGAT_rotatE, self).__init__()
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

        #added rotatE hyperparameters
        self.adversarial_temperature = args.adversarial_temperature
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

        #set ent embedding range
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.entity_dim*2]),
            requires_grad=False
        )

        #create and init ent_user embed
        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.entity_dim*2)
        nn.init.uniform_(
			tensor = self.entity_user_embed.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)

        #set rel embedding range
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.relation_dim]),
            requires_grad=False
        )

        #create and init rel embed
        self.relation_embedding = nn.Embedding(self.n_relations, self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )

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
        # Equation (4) modified to refelect RotatE distance score function
        h_embed = self.entity_user_embed(edges.src['id']).unsqueeze(1)                        #(n_edge, 1, entity_dim * 2)
        t_embed = self.entity_user_embed(edges.dst['id']).unsqueeze(1)                        #(n_edge, 1, entity_dim * 2)                     
        r_embed = self.relation_embedding(edges.data['type']).unsqueeze(1)                    #(n_edge, 1, relation_dim)

        #split ent embeddings into real/imaginary parts
        re_head, im_head = torch.chunk(h_embed, 2, dim=-1)                                    #(n_edge, 1, entity_dim)
        re_tail, im_tail = torch.chunk(t_embed, 2, dim=-1)                                    #(n_edge, 1, entity_dim)
        

        phase_relation = r_embed / (self.rel_embedding_range.item() / self.pi)                #(n_edge, 1, relation_dim)
        
        #split rel embeddings into real/imaginary parts
        re_relation = torch.cos(phase_relation)                                               #(n_edge, 1, relation_dim)
        im_relation = torch.sin(phase_relation)                                               #(n_edge, 1, relation_dim)
    
        #calculate distance function score: RotatE:eq(3)
        re_score = re_head * re_relation - im_head * im_relation                              #(n_edge, 1, entity_dim)
        im_score = re_head * im_relation + im_head * re_relation                              #(n_edge, 1, entity_dim)
        re_score = re_score - re_tail                                                         #(n_edge, 1, entity_dim)
        im_score = im_score - im_tail                                                         #(n_edge, 1, entity_dim)
        score = torch.stack([re_score, im_score], dim = 0)                                    #(2, n_edge, 1, entity_dim)
        att = score.norm(dim = 0).sum(dim = -1)                                               #(n_edge, 1)
        return {'att': att}


    def compute_attention(self, g):
        g = g.local_var()
        for i in range(self.n_relations):
            edge_idxs = g.filter_edges(lambda edge: edge.data['type'] == i)
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

        # Equation (4) modified to refelect RotatE distance score function
        h_embed = self.entity_user_embed(h)                      #(kg_batch_size, 1, entity_dim * 2)
        pos_t_embed = self.entity_user_embed(pos_t)                        #(kg_batch_size, 1, entity_dim * 2)                    
        neg_t_embed = self.entity_user_embed(neg_t)             #(kg_batch_size, 1, entity_dim * 2)
        r_embed = self.relation_embedding(r)                    #(kg_batch_size, 1, relation_dim)

        #split ent embeddings into real/imaginary parts
        re_head, im_head = torch.chunk(h_embed, 2, dim=-1)                                    #(kg_batch_size, 1, entity_dim)
        re_pos_tail, im_pos_tail = torch.chunk(pos_t_embed, 2, dim=-1)                                    #(kg_batch_size, 1, entity_dim)
        re_neg_tail, im_neg_tail = torch.chunk(neg_t_embed, 2, dim=-1)                                    #(kg_batch_size, 1, entity_dim)

        phase_relation = r_embed / (self.rel_embedding_range.item() / self.pi)                #(kg_batch_size, 1, relation_dim)
        
        #split rel embeddings into real/imaginary parts
        re_relation = torch.cos(phase_relation)                                               #(kg_batch_size, 1, relation_dim)
        im_relation = torch.sin(phase_relation)                                               #(kg_batch_size, 1, relation_dim)
    
        #calculate distance function score: RotatE:eq(3)
        re_score = re_head * re_relation - im_head * im_relation                              #(kg_batch_size, 1, entity_dim)
        im_score = re_head * im_relation + im_head * re_relation                               #(kg_batch_size, 1, entity_dim)
        
        #calculate pos score: RotatE:eq(3)
        pos_re_score = re_score - re_pos_tail                                                          #(kg_batch_size, 1, entity_dim)
        pos_im_score = im_score - im_pos_tail                                                          #(kg_batch_size, 1, entity_dim)
        pos_score = torch.stack([pos_re_score, pos_im_score], dim = 0)                                    #(2, kg_batch_size, 1, entity_dim)
        pos_score = pos_score.norm(dim = 0).sum(dim = -1)                                            #(kg_batch_size, 1)
        final_pos_score = self.gamma.item() - pos_score 
        
        #calculate neg score: RotatE:eq(3)
        neg_re_score = re_score - re_neg_tail                                                        #(kg_batch_size, 1, entity_dim)
        neg_im_score = im_score - im_neg_tail                                                         #(kg_batch_size, 1, entity_dim)
        neg_score = torch.stack([neg_re_score, neg_im_score], dim = 0)                                    #(2, kg_batch_size, 1, entity_dim)
        neg_score = neg_score.norm(dim = 0).sum(dim = -1)                                               #(kg_batch_size, 1)
        final_neg_score = self.gamma.item() - neg_score 

        #calculate loss function: HAKE:eq(4)
        negative_score = (F.softmax(final_neg_score * self.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-final_neg_score)).sum(dim=1)

        positive_score = F.logsigmoid(final_pos_score).squeeze(dim=1)
        
        kg_loss = (-positive_score - negative_score) / 2
        loss = torch.mean(kg_loss)

        # l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        # loss = kg_loss + self.kg_l2loss_lambda * l2_loss
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
        user_embed = all_embed[user_ids]                            # (cf_batch_size, cf_concat_dim)
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


