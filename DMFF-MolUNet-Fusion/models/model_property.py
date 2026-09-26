# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, SetTransformerAggregation
from torch_geometric.nn.models import MLP
from .MolUnet import MolUnetEncoder
from .layers import FPNN

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
        return a, out

    
class UnetProperty(nn.Module):
    def __init__(self, config, num_class):
        super().__init__()
        self.unet = MolUnetEncoder(config)
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
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

        
        self.attr_decoder = MLP([feature_dim, feature_dim*2, out_dim])

        if self.query_fp:
            out_dim *= 2
        self.predict = nn.Linear(out_dim, num_class) 
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.predict.bias.data.zero_()
        self.attr_decoder.reset_parameters()

    def forward(self, g):
        batch = g.batch
        fp_hidden=None
        if self.query_fp:
            fp_hidden, fp_attr = self.fpnn(g.fp)
        xs, _, _ = self.unet(g, query=fp_hidden)

        x = self.__do_jk(xs)
        g_embd = self.pool(x, batch)
        g_embd = self.attr_decoder(g_embd)

        g_embd = F.dropout(g_embd, p=self.dropout_rate, training=self.training)
        if self.query_fp:
            fp_attr = F.dropout(fp_attr, p=self.dropout_rate, training=self.training)
            self.fp_attr = fp_attr
            g_embd = torch.cat([g_embd, fp_attr], dim=-1)
        logits = self.predict(g_embd)

        return x, logits
    
    def __do_jk(self, src):
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

        x = self.__do_jk(xs)
        attn = self.pool.gate_nn(x)
        attn = softmax(attn, batch)
        return attn.view(-1)
    
    @property
    def pool_info(self):
        return self.unet.pool_info


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


