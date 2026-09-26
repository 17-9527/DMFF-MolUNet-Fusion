# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:44:19 2023

@author: Fanding Xu
"""
import torch
import math
import torch.nn as nn

from torch_geometric.utils import scatter, softmax, to_undirected, normalized_cut, one_hot
# from torch_scatter import scatter
from torch_geometric.nn import GINEConv, GCNConv
from torch_geometric.nn.models import MLP
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import GIN, GAT, GCN, PNA, GraphSAGE, MessagePassing, global_mean_pool
from torch_geometric.nn.conv import GraphConv, SuperGATConv, TransformerConv
from torch_geometric.nn.models.basic_gnn import BasicGNN
# from torch_geometric.utils.smiles import x_map, e_map
from typing import Final
# from .dropmessage.backbone import bpGAT
from .utils import clustering, edge_reduce, coalesce_with_mask, generate_edge_batch, x_map, e_map

class AtomEmbedding(nn.Module):
    def __init__(self, out_dim, use_one_hot=True, jk='cat'):
        super().__init__()
        self.dims = len(x_map)
        self.use_one_hot = use_one_hot
        self.jk = jk
        if self.use_one_hot:
            self.num_classes = [len(v) for v in x_map.values()]
            len_one_hot = sum(self.num_classes)
            self.pre_lin = MLP([len_one_hot, 1024, out_dim])
            self.pre_lin.reset_parameters()
        else:
            if jk == 'cat':
                type_dim = out_dim // 2
                other_dim = out_dim // 2 // (len(x_map)-1)
            else:
                type_dim, other_dim = (out_dim, out_dim)
            self.embeddings = nn.ModuleList()
            for k, v in x_map.items():
                if k == 'atomic_num':
                    self.embeddings.append(nn.Embedding(len(v), type_dim))
                else:
                    self.embeddings.append(nn.Embedding(len(v), other_dim))
            self.reset_parameters()        
            
    def reset_parameters(self):
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
    
    def forward(self, x):
        if self.use_one_hot:
            embds = []
            for i in range(self.dims):
                embds.append(one_hot(x[:, i], self.num_classes[i],
                             dtype=torch.float32))
            out = torch.cat(embds, dim=-1)
            out = self.pre_lin(out)
        else:
            embds = []
            for i in range(self.dims):
                embds.append(self.embeddings[i](x[:, i]))
            if self.jk == 'cat':
                out = torch.cat(embds, dim=-1)
            elif self.jk == 'sum':
                out = torch.sum(torch.stack(embds, dim = 0), dim = 0)
            elif self.jk == 'mean':
                out = torch.mean(torch.stack(embds, dim = 0), dim = 0)
        return out
    
    
    
class BondEmbedding(nn.Module):
    def __init__(self, out_dim, use_one_hot=True, jk='cat'):
        super().__init__()
        self.dims = len(e_map)
        self.use_one_hot = use_one_hot
        self.jk = jk
        if self.use_one_hot:
            self.num_classes = [len(v) for v in e_map.values()]
            len_one_hot = sum(self.num_classes)
            self.pre_lin = MLP([len_one_hot, 1024, out_dim])
            self.pre_lin.reset_parameters()
        else:
            if jk == 'cat':
                type_dim = out_dim // 2
                other_dim = out_dim // 2 // (len(e_map)-1)
            else:
                type_dim, other_dim = (out_dim, out_dim)
            self.embeddings = nn.ModuleList()
            for k, v in e_map.items():
                if k == 'bond_type':
                    self.embeddings.append(nn.Embedding(len(v), type_dim))
                else:
                    self.embeddings.append(nn.Embedding(len(v), other_dim))
            self.reset_parameters()        
            
    def reset_parameters(self):
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
    
    def forward(self, edge_attr):
        if self.use_one_hot:
            embds = []
            for i in range(self.dims):
                embds.append(one_hot(edge_attr[:, i], self.num_classes[i],
                             dtype=torch.float32))
            out = torch.cat(embds, dim=-1)
            out = self.pre_lin(out)
        else:
            embds = []
            for i in range(self.dims):
                embds.append(self.embeddings[i](edge_attr[:, i]))
            if self.jk == 'cat':
                out = torch.cat(embds, dim=-1)
            elif self.jk == 'sum':
                out = torch.sum(torch.stack(embds, dim = 0), dim = 0)
            elif self.jk == 'mean':
                out = torch.mean(torch.stack(embds, dim = 0), dim = 0)
        return out 

class FPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config['FP']['in_dim']
        hidden_dims = config['FP']['hidden_dims']
        act = activation_resolver(config['FP']['act'])
        out_dim = config['model']['hidden_dim']
        norm = config['FP']['norm']
        
        channel_list = [in_dim] + hidden_dims
        self.mlp_encoder = MLP(channel_list, act=act, norm=norm)
        self.mlp_encoder.reset_parameters()
        hidden_dim = hidden_dims[-1]
        self.attr_decoder = MLP([hidden_dim, hidden_dim*2, out_dim], act=act, norm=norm)
        self.attr_decoder.reset_parameters()
        
    def forward(self, fp):
        fp_hidden = self.mlp_encoder(fp)
        fp_attr = self.attr_decoder(fp_hidden)
        return fp_hidden, fp_attr

def edge_prop(edge_index, edge_attr):
    edge_sums = scatter(edge_attr, edge_index[1], dim=0, reduce='add')[edge_index[0]]
    edge_out = edge_sums + edge_attr
    return edge_out

    
class MESPool(torch.nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.in_dim = in_dim
        self.dk = math.sqrt(in_dim)
        
        self.threshold = config['pool']['threshold']
        # self.threshold = nn.Threshold(config['pool']['threshold'], 0)
        
        self.cut_norm = config['pool']['cut_norm']
        self.score_act = activation_resolver(config['pool']['act'])
        self.mp_edge = config['MP']['use_edge']  
        
        self.q_lin = nn.Linear(in_dim, in_dim)
        self.k_lin = nn.Linear(in_dim, in_dim)
        self.v_lin = nn.Linear(in_dim, in_dim)
        

        self.act = nn.PReLU()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = MPBlock(in_dim, config, num_layer=1)
        # self.conv = PoolGCN(in_dim)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, 1, bias=False),
            nn.LeakyReLU()
        )
        self.reset_parameters()
         
    def reset_parameters(self):
        # uniform(self.in_dim, self.k_weight)
        # uniform(self.in_dim, self.q_weight)
        
        torch.nn.init.xavier_uniform_(self.q_lin.weight)
        self.q_lin.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.k_lin.weight)
        self.k_lin.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.v_lin.weight)
        self.v_lin.bias.data.zero_()
        self.norm.reset_parameters()
        torch.nn.init.xavier_uniform_(self.attn[0].weight)
        
    def forward(self, graph, batch_size=None,
                num_nodes=None, query=None):
        
        x = graph.x
        # print(graph.x)
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch = graph.batch

        edge_batch = generate_edge_batch(edge_index, batch)
        if batch_size is None:
            batch_size = batch[-1] + 1
        if num_nodes is None:
            num_nodes = x.size(0) 
        
        ################ edge_update ################
        if edge_attr is not None:
            edge_message = scatter(edge_attr, edge_index[1], dim=0, reduce='mean')[edge_index[0]]
            # edge_attr = torch.cat([edge_attr + edge_message, x[edge_index[0]]], dim=-1)
            edge_attr = (edge_attr + edge_message + x[edge_index[0]]) / 3
            _, unit_attr = to_undirected(edge_index, edge_attr, num_nodes, reduce='mean')
            V = self.v_lin(edge_attr)
            
        else:
            unit_attr = x[edge_index[0]] + x[edge_index[1]]
            
        ################ scoring ################
        K = self.k_lin(unit_attr)
        Q = self.q_lin(query)[edge_batch]
        
        score = (K * Q).sum(-1) / (Q.norm(p=2, dim=-1) * K.norm(p=2, dim=-1))

        if self.cut_norm:
            score = normalized_cut(edge_index, score)
            
        if self.threshold is None:
            score = softmax(score, edge_batch, dim=0)
        else:
            score = self.score_act(score)
   
        scores_mean = scatter(score, edge_batch, dim=0, reduce='mean')[edge_batch]
        perm = score < scores_mean
        if self.threshold is not None:
            perm = (score <= self.threshold) & perm        
        perm = perm.nonzero(as_tuple=False).view(-1)
        #########################################
            
        if edge_attr is not None:
            edge_attr = V * score.unsqueeze(1)
            edge_attr = self.norm(self.act(edge_attr))
                
                  
        ################ shrinkage ################
        comp, x_mask, edge_mask, edge_target = clustering(x, edge_index, perm)
        batch_out = scatter(batch, comp, dim=-1, reduce='any')
        num_nodes = comp.max().item() + 1
        
        x_sparse, _ = self.conv.seperate_forward(x, edge_target, edge_attr[edge_mask])
        
        # Attention
        attn = self.attn(x_sparse)
        attn = softmax(attn, comp, dim=0)
        
        # Reduce
        x_out = scatter(attn * x_sparse, comp, dim=0, dim_size=num_nodes, reduce='sum')

        ################ extraction ################
        edge_index_out, edge_attr_out, mask = edge_reduce(edge_index, edge_attr, comp, edge_mask) 
        edge_index_out, edge_attr_out, idx, c_perm = coalesce_with_mask(edge_index_out, edge_attr_out, num_nodes=x_out.size(0))

        graph.x = x_out
        graph.edge_index = edge_index_out
        graph.edge_attr = edge_attr_out
        graph.batch = batch_out
        pool_info = dict(comp=comp, edge_target=edge_target,
                         perm=perm, idx=idx, mask=mask, c_perm=c_perm,
                         batch=batch_out, edge_index_out=edge_index)
        # pool_info = [comp, edge_target, idx, mask, batch_out, edge_index_out]
        return graph, pool_info, score  
    
    
class Unpool(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.edge_update = nn.Sequential(nn.Linear(in_dim, in_dim),
                                         nn.LayerNorm(in_dim),
                                         nn.ReLU())
        
        
    def forward(self, graph, buffer, pool_info):
        comp = pool_info['comp']
        idx = pool_info['idx']
        mask = pool_info['mask']
        c_perm = pool_info['c_perm']
        
        x = graph.x.index_select(0, comp)
        graph.x = x
        edge_index = buffer.edge_index
        
        if graph.edge_attr is not None:
            if idx is not None:
                graph.edge_attr = graph.edge_attr.index_select(0, idx)
            if c_perm is not None:
                graph.edge_attr = scatter(graph.edge_attr, c_perm, dim=0, reduce='any')  
            
            edge_attr = torch.zeros_like(buffer.edge_attr)
            edge_attr[mask] = graph.edge_attr
            
            edge_message = scatter(edge_attr, edge_index[1], dim=0, reduce='mean')[edge_index[0]]
            edge_attr = (edge_attr + edge_message + x[edge_index[0]]) / 3
            edge_attr = self.edge_update(edge_attr)
            
            graph.edge_attr = edge_attr
        
        graph.edge_index = edge_index
        graph.batch = buffer.batch
        return graph


class BasicBlock(nn.Module):
    def __init__(self, in_dim, config):
        super().__init__()
        self.pool = MESPool(in_dim, config)
        self.unpool = Unpool(in_dim)
        self.query_fp = config['FP']['query_fp']
        if self.query_fp:
            fp_dim = config['FP']['hidden_dims'][-1]
            self.combine = nn.Sequential(nn.Linear(in_dim + fp_dim, in_dim),
                                         nn.LayerNorm(in_dim),
                                         nn.ReLU()
                                         )
            torch.nn.init.xavier_uniform_(self.combine[0].weight)
            self.combine[0].bias.data.zero_()
        self.__pool_info = None
        
    def down(self, g, query=None, graph_attr=None):
        self.buffer = g.clone()

        if graph_attr is None:
            graph_attr = global_mean_pool(g.x, g.batch)
            
        if self.query_fp:
            query = torch.cat([query, graph_attr], dim=-1)
            query = self.combine(query)
            # query = query + graph_attr
            # query = query + global_mean_pool(g.x, g.batch)
        else:
            query = graph_attr
            
        g, info, self.s = self.pool(g, query=query)
        self.__pool_info = info
        return g
        
    def up(self, g, bypass=False):
        g = self.unpool(g, self.buffer, self.pool_info)
        return g
    
    @property
    def pool_info(self):
        return self.__pool_info
    


class MPBlock(nn.Module):
    def __init__(self, in_dim, config, num_layer=None):
        super().__init__()
        if num_layer is None: num_layer = config['MP']['num_mp_layer']
        mp_method = config['MP']['method']
        mp_norm = config['MP']['norm']
        mp_act = config['MP']['act']
        mp_act_first = config['MP']['act_first']
        mp_jk = config['MP']['jk']
        self.mp_edge = config['MP']['use_edge']

        params = {'in_channels': in_dim,
                  'hidden_channels': in_dim,
                  'num_layers': num_layer,
                  'out_channels': in_dim,
                  'norm': mp_norm,
                  'act': mp_act,
                  'act_first': mp_act_first,
                  'jk': mp_jk}
        if mp_method == 'PNA':
            params['aggregators'] = ['mean', 'min', 'max', 'std']
            params['scalers'] = ['identity', 'amplification', 'attenuation']
            params['deg'] = config['deg']
            params['towers'] = config['MP']['heads']
            
        if mp_method in ['GAT', 'SuperGAT', 'GraphTransformer']:
            params['heads'] = config['MP']['heads']
            
        if mp_method == 'GAT':
            params['v2'] = True
            
        if self.mp_edge:
            if mp_method in ['GAT', 'GIN', 'PNA', 'GraphTransformer']:
                if mp_method == 'GIN': mp_method = 'GINE'
                params['edge_dim'] = in_dim
            else:
                self.node_edge_update = nn.Linear(in_dim, in_dim)
                torch.nn.init.xavier_uniform_(self.node_edge_update.weight)
                self.node_edge_update.bias.data.zero_()
        
        self.MP = eval(mp_method)(**params)
        self.MP.reset_parameters()
        
        
    def forward(self, g):
        g.x, g.edge_attr = self.seperate_forward(g.x, g.edge_index, g.edge_attr)
        return g
    
    def seperate_forward(self, x, edge_index, edge_attr):
        if hasattr(self, 'node_edge_update'):
            e_message = scatter(edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce='mean')
            x = self.node_edge_update(x+e_message)
        if self.mp_edge:
            x = self.MP(x, edge_index, edge_attr = edge_attr)
        else:
            x = self.MP(x, edge_index)
        return x, edge_attr  


class PoolGCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.MP = GCNConv(in_dim, in_dim)
        
    def seperate_forward(self, x, edge_index, edge_weight):
        x = self.MP(x, edge_index, edge_weight)
        return x



class GINE(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINEConv(mlp, **kwargs)

class SuperGAT(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False
        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'SuperGATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        return SuperGATConv(in_channels, out_channels, heads=heads, concat=concat,
                            dropout=self.dropout.p, **kwargs)


class GraphTransformer(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels, out_channels, **kwargs) -> MessagePassing:
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        if getattr(self, '_is_conv_to_out', False):
            concat = False
        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'TransformerConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads
        return TransformerConv(in_channels, out_channels, heads=heads, concat=concat,
                               dropout=self.dropout.p, **kwargs)


class BPGAT(BasicGNN):
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)
        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False
        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'SuperGATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        return bpGAT(in_channels, heads=heads, add_self_loops=False,
                            dropout=0.25, **kwargs)





















































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    