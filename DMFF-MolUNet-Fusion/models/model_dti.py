# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.models import MLP
from torch.nn.utils.weight_norm import weight_norm
from .MolUnet import MolUnetEncoder
from .layers import FPNN

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True,
                 vocab_size=27):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                           padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
    
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
     
class UnetDTI(nn.Module):
    def __init__(self, config, protein_vocab_size=27):
        super().__init__()
        
        # =========================================
        #           DrugBAN settings
        # =========================================
        protein_emb_dim = 128
        num_filters = [128, 128, 128]
        kernel_size = [3, 6, 9]
        protein_padding = True
        ban_heads = 2
        mlp_in_dim = 256
        mlp_hidden_dim = 512
        mlp_out_dim = 128
        out_binary = 1
        self.graph_feature_dim = mlp_in_dim
        
        out_dim = config['model']['hidden_dim'] = 128
        self.query_fp = config['FP']['query_fp'] = True
        config['FP']['hidden_dims'] = num_filters
        
        
        self.protein_extractor = ProteinCNN(
            protein_emb_dim, num_filters, kernel_size, protein_padding,
            vocab_size=protein_vocab_size)
        self.bcn = weight_norm(
            BANLayer(v_dim=out_dim, q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
        # ========================================= 
        config['MP']['norm'] = "BatchNorm"
        config['MP']['act'] = "ReLU"
        config['MP']['heads'] = 2
        config['FP']['norm'] = "BatchNorm"
        config['FP']['act'] = "ReLU"
        config['predict']['jk'] = "cat"
        config['predict']['graph_pool'] = "mean"
        
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        graph_pool = config['predict']['graph_pool']
        self.graph_pool = graph_pool
        self.dropout_rate = config['predict']['dropout_rate']
        self.pool_first = config['predict']['pool_first']
        self.jk = config['predict']['jk']
        
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" and not self.pool_first else out_dim
        self.unet = MolUnetEncoder(config)
        
        
            
        feature_dim = num_pool_layer * out_dim if self.jk == "cat" and not self.pool_first else out_dim
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
            if self.pool_first:
                self.pool = nn.Sequential(
                    *[GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1)) for i in range(num_pool_layer)])
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(feature_dim, 1))
                
        elif graph_pool[:-1] == "set2set":
            set2set_iter = int(graph_pool[-1])
            if self.pool_first:
                self.pool = nn.Sequential(
                    *[Set2Set(feature_dim, set2set_iter) for i in range(num_pool_layer)])
            else:
                self.pool = Set2Set(feature_dim, set2set_iter)
            feature_dim = feature_dim * 2
        else:
            raise ValueError("Invalid graph pooling type.")
        
        if self.pool_first:
            feature_dim *= num_pool_layer
        
        # self.attr_decoder_mol = MLP([feature_dim, out_dim])
        self.attr_decoder_mol = MLP([feature_dim, feature_dim*2, out_dim])
        # self.attr_decoder_pro = MLP([128, 256, out_dim])
        
        # final_dim = out_dim
        # self.predict = nn.Sequential(nn.Linear(final_dim, final_dim),
        #                              nn.BatchNorm1d(final_dim),
        #                              nn.ReLU(),
        #                              nn.Linear(final_dim, 1))
        
        self.gnn = DrugGCN(out_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        pass

    
    def forward(self, g, t, mask=None, return_features=False):
        t = self.protein_extractor(t)
        batch = g.batch
        
        t_embd = t.mean(dim=1)
        query = t_embd
        # if self.query_fp:
        #     query = self.t_trans(t_embd)
        xs, _, _ = self.unet(g, query=query)
        
        x = self.__do_jk(xs)
        x = self.attr_decoder_mol(x)
        # dx, dm = to_dense_batch(x, batch)
        # # print(dx.shape)
        # f, att = self.bcn(dx, t) 
        
        # f = F.dropout(f, p=self.dropout_rate, training=self.training)
        
        # logits = self.mlp_classifier(f)
        
        att, logits, graph_feature = self.BAN_pred(x, batch, t, mask)
        if return_features:
            return (x, att), logits, graph_feature
        return (x, att), logits
    
    def BAN_pred(self, x, batch, t, mask=None):
        if mask is not None:
            x = x[mask]
            batch = batch[mask]
        dx, dm = to_dense_batch(x, batch)
        f, att = self.bcn(dx, t) 
        f = F.dropout(f, p=self.dropout_rate, training=self.training)
        logits = self.mlp_classifier(f)
        return att, logits, f
    
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
            x = self.score_jk(src)
        else:
            raise ValueError("Invalid JK type.")
        return x
        
    def cal_attention(self, g):
        batch = g.batch
        xs, es = self.unet(g)
        if self.pool_first:
            attn = [pool.gate_nn(x) for pool, x in zip(self.pool, xs)]
            attn = [softmax(a, batch) for a in attn]
        else:
            x = self.__do_jk(xs)
            attn = self.pool.gate_nn(x)
            attn = softmax(attn, batch)
        return attn.view(-1)
    
    @property
    def pool_info(self):
        return self.unet.pool_info



# ============================== DrugBAN =====================================
from torch_geometric.nn import GCN, GCNConv

class UNetBAN(MolUnetEncoder):
    def __init__(self, config):
        super().__init__(config)
        out_dim = config['model']['hidden_dim']
        self.atom_embedding = nn.Linear(74, out_dim)
        self.bond_embedding = nn.Linear(12, out_dim)
    
    
    
    
class DrugGCN(nn.Module):
    def __init__(self, dim_embedding=128, activation=None):
        super(DrugGCN, self).__init__()
        self.init_transform = nn.Linear(74, dim_embedding, bias=False)
        self.gnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.reslins = nn.ModuleList()
        for i in range(3):
            self.gnns.append(GCNConv(dim_embedding, dim_embedding))
            self.norms.append(nn.BatchNorm1d(dim_embedding))
            self.reslins.append(nn.Linear(dim_embedding, dim_embedding))
        # self.gnns = GCN(dim_embedding, dim_embedding, num_layers=3,
        #                 act="relu", norm="batchnorm")
    def forward(self, g):
        x = g.x
        x = self.init_transform(x)
        edge_index = g.edge_index
        for i in range(3):
            x_new = self.norms[i](self.gnns[i](x, edge_index)).relu()
            x = x_new + self.reslins[i](x)
        
        # x = self.gnns(x, edge_index)
        dx, _ = to_dense_batch(x, g.batch)
        return dx


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        # self.dropout = nn.Dropout(dropout[1])
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)  # b x 1 x d
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k  # sum-pooling
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """

    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/bc.py
    """

    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2, .5], k=3):
        super(BCNet, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim;
        self.q_dim = q_dim
        self.h_dim = h_dim;
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else:
            v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
            return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v)  # b x v x d
        q_ = self.q_net(q)  # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits



