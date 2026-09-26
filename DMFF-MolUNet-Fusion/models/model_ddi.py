# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, to_dense_batch, degree
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, SetTransformerAggregation
from torch_geometric.nn.models import MLP
from torch.nn.utils.weight_norm import weight_norm
from collections import defaultdict
from .MolUnet import MolUnetEncoder
from .layers import BasicBlock, MPBlock, AtomEmbedding, BondEmbedding, FPNN
    
class ScoreJK(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(in_dim, in_dim),
                                nn.ReLU(),
                                nn.Linear(in_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.nn[0].weight)
        self.nn[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.nn[-1].weight)
        self.nn[-1].bias.data.zero_()

    def forward(self, src):
        src = torch.stack(src, dim=1)
        a = self.nn(src)
        a = torch.softmax(a, dim=1)
        out = (a * src).sum(dim=1)
        return out
    
class UnetDDI(nn.Module):
    def __init__(self, config, num_class=1):
        super().__init__()
        
        self.unet = DDIUnetEncoder(config)
        
        out_dim = config['model']['hidden_dim']
        self.out_dim = out_dim
        num_pool_layer = config['model']['num_pool_layer']
        self.num_pool_layer = num_pool_layer
        graph_pool = config['predict']['graph_pool']
        self.graph_pool = graph_pool
        self.jk = config['predict']['jk']
        self.dropout_rate = config['predict']['dropout_rate']

        self.query_fp = config['FP']['query_fp']
        if self.query_fp:
            self.fpnn = FPNN(config)
        
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" else out_dim
        if self.jk == "score":
            self.score_jk = ScoreJK(out_dim)
        
        #Different kind of graph pooling
        if graph_pool == "sum":
            self.pool = global_add_pool
        elif graph_pool == "mean":
            self.pool = global_mean_pool
        elif graph_pool == "max":
            self.pool = global_max_pool
        elif graph_pool == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1))
                
        elif graph_pool[:-1] == "set2set":
            set2set_iter = int(graph_pool[-1])
            self.pool = Set2Set(feature_dim, set2set_iter)
            feature_dim = feature_dim * 2
        elif graph_pool == "ST":
            num_seed_points = config['ST']['num_seed_points']
            num_encoder_blocks = config['ST']['num_encoder_blocks']
            num_decoder_blocks = config['ST']['num_decoder_blocks']
            heads = config['ST']['heads']
            self.pool = SetTransformerAggregation(feature_dim,
                                                  num_seed_points=num_seed_points,
                                                  num_encoder_blocks=num_encoder_blocks,
                                                  num_decoder_blocks=num_decoder_blocks,
                                                  heads=heads,
                                                  )
            feature_dim *= num_seed_points
        else:
            raise ValueError("Invalid graph pooling type.")

        self.feature_dim = feature_dim
        self.attr_decoder = MLP([feature_dim, feature_dim*2, out_dim])

        if self.query_fp:
            out_dim *= 2
        self.predict = nn.Linear(out_dim, num_class) 
        self.reset_parameters()
        self.extra_init()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.predict.bias.data.zero_()
        self.attr_decoder.reset_parameters()
        
    def extra_init(self):
        raise NotImplementedError

    def forward(self, h, t, rels):
        raise NotImplementedError

    
    def do_jk(self, src):
        if self.jk == "cat":
            x = torch.cat(src, dim = 1)
        elif self.jk == "last":
            x = src[-1]
        elif self.jk == "max":
            x = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            x = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            x = torch.sum(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "score":
            self.jk_score, x = self.score_jk(src)
        else:
            raise ValueError("Invalid JK type.")
        return x
        
    def cal_attention(self, g):
        batch = g.batch
        xs, es = self.unet(g)

        x = self.do_jk(xs)
        attn = self.pool.gate_nn(x)
        attn = softmax(attn, batch)
        return attn.view(-1)
    
    @property
    def pool_info(self):
        return self.unet.pool_info

class UnetDDI_SA(UnetDDI):
    """
    Adapted from [SA-DDI] model.py in https://github.com/guaguabujianle/SA-DDI
    """
    def extra_init(self):
        out_dim = self.out_dim
        hidden_dim = self.feature_dim
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim * 2),
            nn.PReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )
    
        self.rmodule = nn.Embedding(86, out_dim)
    
        self.w_j = nn.Linear(hidden_dim, out_dim)
        self.w_i = nn.Linear(hidden_dim, out_dim)
    
        self.prj_j = nn.Linear(hidden_dim, out_dim)
        self.prj_i = nn.Linear(hidden_dim, out_dim)
        
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, data):
        h, t, rel, label = data
        xs_h, xs_t = self.unet(h, t)

        x_h = self.do_jk(xs_h)
        x_t = self.do_jk(xs_t)
        
        g_h = self.pool(x_h, h.batch)
        g_t = self.pool(x_t, t.batch)
        
        g_h_align = g_h.repeat_interleave(degree(t.batch, dtype=t.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h.batch, dtype=h.batch.dtype), dim=0)

        
        h_scores = (self.w_i(x_h) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h.batch, dim=0)
        # Equation (10)
        t_scores = (self.w_j(x_t) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t.batch, dim=0)
        # Equation (11)
        h_final = global_add_pool(x_h * g_t_align * h_scores.unsqueeze(-1), h.batch)
        t_final = global_add_pool(x_t * g_h_align * t_scores.unsqueeze(-1), t.batch)
        # End of SSIM

        pair = torch.cat([h_final, t_final], dim=-1)
        rfeat = self.rmodule(rel)
        logit = (self.lin(pair) * rfeat).sum(-1)
        
        loss = self.criterion(logit, label)
        return logit, loss

class UnetDDI_SSI(UnetDDI):
    """
    Adapted from [SSI-DDI] models.py in https://github.com/kanz76/SSI-DDI
    """
    def extra_init(self):
        self.lins = nn.ModuleList()
        for i in range(self.num_pool_layer):
            lin = nn.Linear(self.out_dim, 64)
            self.lins.append(lin)
            
        self.co_attention = CoAttentionLayer(64)
        self.KGE = RESCAL(86, 64)
        
        self.criterion = SigmoidLoss()
        
    def forward(self, data):
        h, t, rel, label = data
        xs_h, xs_t = self.unet(h, t)
        
        # repr_h = [self.lins[i](global_add_pool(x_h, h.batch)) for i, x_h in enumerate(xs_h)]
        # repr_t = [self.lins[i](global_add_pool(x_t, t.batch)) for i, x_t in enumerate(xs_t)]
        
        repr_h = [global_add_pool(x_h, h.batch) for i, x_h in enumerate(xs_h)]
        repr_t = [global_add_pool(x_t, t.batch) for i, x_t in enumerate(xs_t)]
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        self.a = attentions
        scores = self.KGE(kge_heads, kge_tails, rel, attentions)
        
        p_score = scores[label==1]
        n_score = scores[label==0]
        
        loss, loss_p, loss_n = self.criterion(p_score, n_score)
        return scores, loss
    
    def mask_pred(self, xs_h, xs_t, rel):
        repr_h = [x_h.sum(0) for i, x_h in enumerate(xs_h)]
        repr_t = [x_t.sum(0) for i, x_t in enumerate(xs_t)]
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        
        kge_heads = repr_h
        kge_tails = repr_t
        # attentions = self.a
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rel, attentions)
        return scores
    
    def visual_read(self, h, t):
        xs_h, xs_t = self.unet(h, t)
        
        # repr_h = [self.lins[i](global_add_pool(x_h, h.batch)) for i, x_h in enumerate(xs_h)]
        # repr_t = [self.lins[i](global_add_pool(x_t, t.batch)) for i, x_t in enumerate(xs_t)]
        
        repr_h = [global_add_pool(x_h, h.batch) for i, x_h in enumerate(xs_h)]
        repr_t = [global_add_pool(x_t, t.batch) for i, x_t in enumerate(xs_t)]
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        kge_heads = repr_h
        kge_tails = repr_t

        attentions = self.co_attention(kge_heads, kge_tails)
        # attentions = None
        
        return attentions

class DDIBasicBlock(BasicBlock):
    def down(self, h, t):
        self.buffer_h = h.clone()
        self.buffer_t = t.clone()
        q_h = global_mean_pool(h.x, h.batch)
        q_t = global_mean_pool(t.x, t.batch)
        
        h, info_h, self.s_h = self.pool(h, query=q_t)
        t, info_t, self.s_t = self.pool(t, query=q_h)
        
        self.__pool_info = [info_h, info_t]
        return h, t
    
    def up(self, h, t):
        h = self.unpool(h, self.buffer_h, self.__pool_info[0])
        t = self.unpool(t, self.buffer_t, self.__pool_info[1])
        return h, t
    
    @property
    def pool_info(self):
        return self.__pool_info
    

class DDIUnetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        use_one_hot = config['model']['use_one_hot']
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        embedding_jk = config['model']['embedding_jk']
        self.jump = config['model']['jump']
        self.jump_weight = config['model']['jump_weight']

        self.atom_embedding = AtomEmbedding(out_dim, use_one_hot, embedding_jk)
        self.bond_embedding = BondEmbedding(out_dim, use_one_hot, embedding_jk)
        
        self.blocks = nn.Sequential()
        
        for i in range(num_pool_layer):
            self.blocks.append(DDIBasicBlock(out_dim, config))
        
        self.num_pool_layer = num_pool_layer
        
        self.mp_up = nn.ModuleList()
        self.mp_down = nn.ModuleList()
        self.mp_init = MPBlock(out_dim, config)
        
        for i in range(num_pool_layer):
            mp = nn.ModuleList()
            self.mp_down.append(MPBlock(out_dim, config))
            for j in range(i+1):
                mp.insert(0, MPBlock(out_dim, config))
            self.mp_up.append(mp)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        pass
    
    def forward(self, h, t):    
        h = h.clone()
        t = t.clone()
        h.x = self.atom_embedding(h.x)
        h.edge_attr = self.bond_embedding(h.edge_attr)
        t.x = self.atom_embedding(t.x)
        t.edge_attr = self.bond_embedding(t.edge_attr)
        
        if h.batch is None:
            h.batch = h.edge_index.new_zeros(h.x.size(0))
        if t.batch is None:
            t.batch = t.edge_index.new_zeros(t.x.size(0))

        x_jump_h, x_jump_t = [], []
        emx_h, emx_t = [], []
        # eme = []
        
        h = self.mp_init(h)
        t = self.mp_init(t)
        
        for i in range(self.num_pool_layer):
            xh = h.x; xt = t.x
            x_jump_h.append(xh)
            x_jump_t.append(xt)
            
            # pooling         
            h, t = self.blocks[i].down(h, t)
            
            # message passing
            h = self.mp_down[i](h)
            t = self.mp_down[i](t)
            hp = h.clone()
            tp = t.clone()
            
            w = 1
            for j in range(i, -1, -1):
                # unpooling
                hp, tp = self.blocks[j].up(hp, tp)       
                if self.jump is not None:
                    hp.x = hp.x + x_jump_h[j] / w
                    tp.x = tp.x + x_jump_t[j] / w
                # message passing
                hp = self.mp_up[i][j](hp)
                tp = self.mp_up[i][j](tp)
                # jump connection
                if self.jump is not None:
                    x_jump_h[j] = self.__jump(x_jump_h[j], hp)
                    x_jump_t[j] = self.__jump(x_jump_t[j], tp)
                    w += 1
                    
            xh_out = hp.x; xt_out = tp.x        
            emx_h.append(xh_out)
            emx_t.append(xt_out)
        self.x = [emx_h, emx_t]
        return emx_h, emx_t
    
    def __jump(self, xj, g):
        if self.jump == 'straight':
            return g.x.clone()
        elif self.jump == 'all':
            return xj + g.x

    @property
    def pool_info(self):
        out_h = defaultdict(list)
        out_t = defaultdict(list)
        for block in self.blocks:
            for k, v in block.pool_info[0].items():
                out_h[k].append(v)
            for k, v in block.pool_info[1].items():
                out_t[k].append(v)
        return [out_h, out_t]


"""
============================= From SSI-DDI ================================
"""
class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores

        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    
    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)
        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)
        rels = rels.view(-1, self.n_features, self.n_features)

        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
          scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))
        return scores 
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature
    
    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights= F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()
        
        return (p_loss + n_loss) / 2, p_loss, n_loss 

















