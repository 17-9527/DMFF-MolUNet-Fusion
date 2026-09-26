# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:12:06 2023

@author: Fanding Xu
"""

import torch.nn as nn
from torch_geometric.utils import mask_feature
from collections import defaultdict
import torch.nn.functional as F
from .layers import BasicBlock, MPBlock, AtomEmbedding, BondEmbedding

    
class MolUnetEncoder(nn.Module):
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
            self.blocks.append(BasicBlock(out_dim, config))
        
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

    def forward(self, g, query=None):    
        g = g.clone()
        x = self.atom_embedding(g.x)
        edge_attr = self.bond_embedding(g.edge_attr)
        # x = F.dropout(x, p=0.2, training=self.training)
        # edge_attr = F.dropout(edge_attr, p=0.2, training=self.training)
        
        g.x = x
        g.edge_attr = edge_attr
        
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
    
    def conv_forward(self, g, attn_query=None,
                     node_mask=None, edge_mask=None):
        g = g.clone()
        x = self.atom_embedding(g.x)
        edge_attr = self.bond_embedding(g.edge_attr)
        g.x, g.edge_attr = self.mask_attr(x, edge_attr, node_mask, edge_mask)

        if g.batch is None:
            g.batch = g.edge_index.new_zeros(g.x.size(0))
        xs = []
        emx = []
        eme = []

        g = self.mp_init(g)

        for i in range(self.num_pool_layer):
            x = g.x
            xs.append(x)
            # message passing
            g = self.mp_down[i](g)
            gp = g.clone()
            w = 1
            for j in range(i, -1, -1):
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
        return emx, eme
        
        
    @property
    def pool_info(self):
        out = defaultdict(list)
        for block in self.blocks:
            for k, v in block.pool_info.items():
                out[k].append(v)
        return out


class ConvUnetEncoder(nn.Module):
    def __init__(self, config,
                 node_mask_ratio=None,
                 edge_mask_ratio=None):
        super().__init__()
        
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        
        use_one_hot = config['model']['use_one_hot']
        out_dim = config['model']['hidden_dim']
        num_pool_layer = config['model']['num_pool_layer']
        embedding_jk = config['model']['embedding_jk']
        self.jump = config['model']['jump']
        self.jump_weight = config['model']['jump_weight']
        
        self.atom_embedding = AtomEmbedding(out_dim, use_one_hot, embedding_jk)
        self.bond_embedding = BondEmbedding(out_dim, use_one_hot, embedding_jk)
        
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
            
           
    def forward(self, g, attn_query=None):    
        g.x = self.atom_embedding(g.x)
        g.edge_attr = self.bond_embedding(g.edge_attr)
        
        if self.node_mask_ratio is not None:
            g.x, node_mask = mask_feature(g.x, p=self.node_mask_ratio, mode='row')

        if self.node_mask_ratio is not None:
            g.edge_attr, edge_mask = mask_feature(g.edge_attr, p=self.edge_mask_ratio, mode='row')
        
        if g.batch is None:
            g.batch = g.edge_index.new_zeros(g.x.size(0))
        # g.edge_attr = None
        xs = []
        emx = []
        eme = []

        g = self.mp_init(g)

        for i in range(self.num_pool_layer):
            x = g.x
            xs.append(x)
            # message passing
            g = self.mp_down[i](g)
            gp = g.clone()

            w = 1
            for j in range(i, -1, -1):

                if self.jump is not None:
                    gp.x = gp.x + xs[j] / w
                # message passing
                gp = self.mp_up[i][j](gp)
                # jump connection
                if self.jump is not None:
                    xs[j] = self.__jump(xs[j], gp)
                    w += 1
            
            em_x = gp.x
            # em_graph = global_add_pool(em_x, gp.batch)
            em_e = gp.edge_attr
            eme.append(em_e)
            emx.append(em_x)
        self.x = emx
        return emx, eme, node_mask.view(-1), edge_mask.view(-1)
    
    def __jump(self, xj, g):
        if self.jump == 'straight':
            return g.x.clone()
        elif self.jump == 'all':
            return xj + g.x
















