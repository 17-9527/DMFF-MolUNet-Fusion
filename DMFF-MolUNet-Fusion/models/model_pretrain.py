# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, mask_feature
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, GlobalAttention, Set2Set, SetTransformerAggregation
from torch_geometric.nn.models import MLP
from torch_geometric.utils.smiles import x_map, e_map
from .MolUnet import MolUnetEncoder, ConvUnetEncoder

class MaskedUnet(MolUnetEncoder):
    def mask_attr(self, x, edge_attr,
                  node_mask=None, edge_mask=None):
        if node_mask is not None:
            x = x.masked_fill(node_mask.view(-1,1), 0)
        if edge_mask is not None:
            edge_attr = edge_attr.masked_fill(edge_mask.view(-1,1), 0)
        return x, edge_attr
    
    def forward(self, g, query=None,
                node_mask=None, edge_mask=None):    
        g = g.clone()
        x = self.atom_embedding(g.x)
        edge_attr = self.bond_embedding(g.edge_attr)
        
        x = F.dropout(x, p=0.2, training=self.training)
        edge_attr = F.dropout(edge_attr, p=0.2, training=self.training)
        
        g.x, g.edge_attr = self.mask_attr(x, edge_attr, node_mask, edge_mask)

        if g.batch is None:
            g.batch = g.edge_index.new_zeros(g.x.size(0))
        # g.edge_attr = None
        xs = []
        emx = []
        eme = []
        pool_features = []
        g = self.mp_init(g)
        
        for i in range(self.num_pool_layer):
            x = g.x
            xs.append(x)
            # pooling
            g = self.blocks[i].down(g, query=query)
            # message passing
            g = self.mp_down[i](g)
            gp = g.clone()
            x_pool = g.x
            pool_features.append(x_pool)
            w = 1
            for j in range(i, -1, -1):
                # unpooling
                gp = self.blocks[j].up(gp)       
                if self.jump is not None:
                    gp.x = gp.x + xs[j] / w
                # message passing
                gp = self.mp_up[i][j](gp)
                # jump connection
                if self.jump is not None:
                    xs[j] = self.__jump(xs[j], gp)
                    w += 1
            em_x = gp.x
            em_e = gp.edge_attr
            eme.append(em_e)
            emx.append(em_x)
        self.x = emx
        return emx, eme, pool_features
    
    def __jump(self, xj, g):
        if self.jump == 'straight':
            return g.x.clone()
        elif self.jump == 'all':
            return xj + g.x

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
    
class CompIdentify(nn.Module):
    def __init__(self, in_dim, num_pool_layer):
        super().__init__()
        self.proj_fg = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 39),
        )
        self.fg_bce = nn.BCEWithLogitsLoss()

        self.proj_brics = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
        self.brics_bce = nn.BCEWithLogitsLoss()
        # self.brics_bce = nn.BCELoss()

        self.proj_scaffold = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
        self.scaffold_bce = nn.BCEWithLogitsLoss()
        
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.proj_fg[0].weight)
        self.proj_fg[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj_fg[2].weight)
        self.proj_fg[2].bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.proj_brics[0].weight)
        self.proj_brics[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj_brics[2].weight)
        self.proj_brics[2].bias.data.zero_()
        
        torch.nn.init.xavier_uniform_(self.proj_scaffold[0].weight)
        self.proj_scaffold[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj_scaffold[2].weight)
        self.proj_scaffold[2].bias.data.zero_()
        
    def forward(self, xs, es, label_fg, label_brics, label_scaffold):

        pred_fg = self.proj_fg(xs)
        pred_brics = self.proj_brics(es)
        pred_scaffold = self.proj_scaffold(xs)
        loss_fg = self.fg_bce(pred_fg, label_fg)
        loss_brics = self.brics_bce(pred_brics, label_brics)
        loss_scaffold = self.scaffold_bce(pred_scaffold, label_scaffold)
        loss_comp = loss_fg + loss_brics + loss_scaffold

        return loss_comp, [loss_fg.item(), loss_brics.item(), loss_scaffold.item()]
        
class DescriptorReg(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 200),
        )
        self.loss_func = nn.MSELoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.proj[0].weight)
        self.proj[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj[-1].weight)
        self.proj[-1].bias.data.zero_()
        
    def forward(self, h_g, label_desc):
        pred_desc = self.proj(h_g)
        nan_mask = torch.isnan(label_desc)
        loss_desc = self.loss_func(pred_desc[~nan_mask], label_desc[~nan_mask])
        # print(loss_desc)
        return loss_desc
 
        
class ChemblCls(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 1310),
        )
        self.loss_func = nn.BCEWithLogitsLoss(reduction = "none")
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.proj[0].weight)
        self.proj[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj[-1].weight)
        self.proj[-1].bias.data.zero_()
        
    def forward(self, h_g, y):
        pred = self.proj(h_g)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        loss_mat = self.loss_func(pred, (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        
        return loss
    
class MultiTaskCls(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_task):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, num_task),
        )
        self.loss_func = nn.BCEWithLogitsLoss()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.proj[0].weight)
        self.proj[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.proj[-1].weight)
        self.proj[-1].bias.data.zero_()
        
    def forward(self, h_g, y):
        pred = self.proj(h_g)
        mask = y >= 0
        pred = pred[mask]
        y = y[mask]
        loss = self.loss_func(pred, y)
        return loss
    
class GraphSim(nn.Module):
    def __init__(self, config, eps=1e-6):
        super().__init__()
        self.eps=eps
        self.cos = nn.CosineSimilarity(dim=2, eps=self.eps)
        self.loss_func = nn.MSELoss()
        
    def forward(self, h_g, fp):
        # Tanimoto sim by fingerprint
        c = torch.matmul(fp, fp.T)
        cnt = fp.sum(dim=1)
        if self.eps is not None:
            cnt[cnt < self.eps] = self.eps
        a = cnt.unsqueeze(0)
        b = cnt.unsqueeze(1)
        sim_fp = c / (a + b - c)
        # cosine sim by graph embedding
        sim_cos = self.cos(h_g.unsqueeze(0), h_g.unsqueeze(1))
        loss = self.loss_func(sim_cos, sim_fp)
        return loss


class UnetPretrainS2(nn.Module):
    def __init__(self, config, mol_dict=None):
        super().__init__()
        self.unet = MolUnetEncoder(config)
        in_dim = config['model']['hidden_dim']
        hidden_dim = config['pretrain']['pred_hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        graph_pool = config['pretrain']['graph_pool']
        self.jk = config['pretrain']['jk']
        self.dropout_rate = config['pretrain']['dropout_rate']
        
        feature_dim = num_pool_layer * in_dim if self.jk == "cat" else in_dim
        if self.jk == "score":
            self.score_jk = ScoreJK(in_dim)
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
            self.pool = SetTransformerAggregation(feature_dim, 2, 1, 1, heads=4)
            feature_dim *= 2
        else:
            raise ValueError("Invalid graph pooling type.")
        
        # self.attr_decoder = MLP([feature_dim, feature_dim*2, in_dim])
        # self.chembl_cls = ChemblCls(in_dim, hidden_dim)
        # self.desc_reg = DescriptorReg(in_dim, hidden_dim)
        
        self.chembl_cls = ChemblCls(feature_dim, hidden_dim)
        self.desc_reg = DescriptorReg(feature_dim, hidden_dim)
        self.graph_sim = GraphSim(config)
        
    def forward(self, g):
        xs, es, xp = self.unet(g)
        x = self.__do_jk(xs)
        h_g = self.pool(x, g.batch)
        
        # h_g = self.attr_decoder(h_g)
        # h_g = F.dropout(h_g, p=self.dropout_rate, training=self.training)
        
        loss_chembl = self.chembl_cls(h_g, g.y)
        loss_desc = self.desc_reg(h_g, g.desc)
        loss_gsim = self.graph_sim(h_g, g.fpsim)
    
        
        separate_loss = [loss_chembl.item(), loss_desc.item(), loss_gsim.item()]
        loss = loss_chembl + loss_desc + loss_gsim
        
        # separate_loss = [loss_chembl.item(), loss_gsim.item()]
        # loss = loss_chembl + loss_gsim
        
        # separate_loss = [loss_chembl.item()]
        # loss = loss_chembl
        return separate_loss, loss
    
    def __do_jk(self, src):
        if self.jk == "cat":
            src = torch.cat(src, dim = 1)
        elif self.jk == "last":
            src = src[-1]
        elif self.jk == "max":
            src = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            src = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            src = torch.sum(torch.stack(src, dim = 0), dim = 0)
        return src
    
    @property
    def pool_info(self):
        return self.unet.pool_info

            

class UnetPretrainS1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.unet = MaskedUnet(config)
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        self.comp_idt = CompIdentify(out_dim, num_pool_layer)
        # self.desc_reg = DescriptorReg(out_dim, num_pool_layer, config)
        
        self.jk = config['pretrain']['jk']
        pred_dim = num_pool_layer * out_dim if self.jk == "cat" else out_dim
        self.lin_node_pred_w = nn.Linear(pred_dim, len(x_map['atomic_num']))
        self.lin_edge_pred_w = nn.Linear(pred_dim, len(e_map['bond_type']))
        self.lin_node_pred_m = nn.Linear(pred_dim, len(x_map['atomic_num']))
        self.lin_edge_pred_m = nn.Linear(pred_dim, len(e_map['bond_type']))
        self.loss_func = nn.CrossEntropyLoss()

        self.comp_idt = CompIdentify(pred_dim, num_pool_layer)
        
        # self.graph_sim = GraphSim(config)
        # self.cgss = CGSSLoss(0.05)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_node_pred_w.weight)
        self.lin_node_pred_w.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.lin_edge_pred_w.weight)
        self.lin_edge_pred_w.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.lin_node_pred_m.weight)
        self.lin_node_pred_m.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.lin_edge_pred_m.weight)
        self.lin_edge_pred_m.bias.data.zero_()
        
    def weighted_mask_pred(self, g):
        node_mask = g.node_mask
        edge_mask = g.edge_mask
        node_label = g.x[node_mask][:, 0]
        edge_label = g.edge_attr[edge_mask][:, 0]
        xs, es = self.unet.conv_forward(g, node_mask=node_mask, edge_mask=edge_mask)
        x = self.__do_jk(xs)
        e = self.__do_jk(es)
        e = self.edge_pred_attr(x, e, g.edge_index)
        node_pred = self.lin_node_pred_w(x[node_mask])
        edge_pred = self.lin_edge_pred_w(e[edge_mask])
        loss_node = self.loss_func(node_pred, node_label)
        loss_edge = self.loss_func(edge_pred, edge_label)
        return loss_node, loss_edge, x, e
    
    def motif_mask_pred(self, g):
        node_mask = g.node_mask_motif
        edge_mask = g.edge_mask_motif
        node_label = g.x[node_mask][:, 0]
        edge_label = g.edge_attr[edge_mask][:, 0]
        xs, es, xp = self.unet(g, node_mask=node_mask, edge_mask=edge_mask)
        x = self.__do_jk(xs)
        e = self.__do_jk(es)
        e = self.edge_pred_attr(x, e, g.edge_index)
        node_pred = self.lin_node_pred_w(x[node_mask])
        edge_pred = self.lin_edge_pred_w(e[edge_mask])
        loss_node = self.loss_func(node_pred, node_label)
        loss_edge = self.loss_func(edge_pred, edge_label)
        return loss_node, loss_edge, x, e, xp
        
        
    def forward(self, g):
        loss_node_w, loss_edge_w, _, _ = self.weighted_mask_pred(g)
        separate_loss_w = [loss_node_w.item(), loss_edge_w.item()]
        loss_w = loss_node_w + loss_edge_w
        
        loss_node_m, loss_edge_m, x, e, xp = self.motif_mask_pred(g)
        separate_loss_m = [loss_node_m.item(), loss_edge_m.item()]
        loss_m = loss_node_m + loss_edge_m
        
        node_mask_c = g.node_mask_motif.logical_not()
        edge_mask_c = g.edge_mask_motif.logical_not()
        loss_comp, single_loss_comp = self.comp_idt(x[node_mask_c], e[edge_mask_c],
                                                    g.label_fg[node_mask_c],
                                                    g.label_brics[edge_mask_c],
                                                    g.label_scaffold[node_mask_c])


        # loss_comp, single_loss_comp = self.comp_idt(x, e,
        #                                             g.label_fg,
        #                                             g.label_brics,
        #                                             g.label_scaffold)
        
        separate_loss = separate_loss_w + separate_loss_m + single_loss_comp
        loss = loss_w + loss_m + loss_comp
        
        # separate_loss = separate_loss_w + separate_loss_m
        # loss = loss_w + loss_m
        
        # separate_loss = separate_loss_m + single_loss_comp
        # loss = loss_m + loss_comp

        return separate_loss, loss

    def __do_jk(self, src):
        if self.jk == "cat":
            src = torch.cat(src, dim = 1)
        elif self.jk == "last":
            src = src[-1]
        elif self.jk == "max":
            src = torch.max(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "mean":
            src = torch.mean(torch.stack(src, dim = 0), dim = 0)
        elif self.jk == "sum" or self.jk == "add":
            src = torch.sum(torch.stack(src, dim = 0), dim = 0)
        return src
    
    def edge_pred_attr(self, x, e, edge_index):
        _, e = to_undirected(edge_index, e, reduce='mean')
        e = e + x[edge_index[0]] + x[edge_index[1]]
        return e
    
    @property
    def pool_info(self):
        return self.unet.pool_info










































