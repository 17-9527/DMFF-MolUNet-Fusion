# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:36:56 2023

@author: Fanding Xu
"""

import torch
import os
import numpy as np
import random
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, degree, scatter, subgraph, to_dense_adj, index_sort
# from torch_scatter import scatter
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Any, Dict, List

# copyed from torch_geometric.utils.smiles
# torch_geometric == 2.5.0
x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(filename, " successfully loaded")
    return obj

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

def concat_group_by(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    adapted from
    https://github.com/rusty1s/pytorch_scatter/issues/398
    """
    index_count = torch.bincount(index)
    fill_count = index_count.max() - index_count
    fill_zeros = (torch.ones_like(x[0])*-1).repeat(fill_count.sum(), *([1]*(len(x.shape)-1)))
    fill_index = torch.arange(0, fill_count.shape[0], device=x.device).repeat_interleave(fill_count)
    index_ = torch.cat([index, fill_index], dim=0)
    x_ = torch.cat([x, fill_zeros], dim=0)
    x_ = x_[torch.argsort(index_, stable=True)].view(index_count.shape[0], index_count.max()*x.size(1), x.size(-1))
    return x_

def get_dataset_deg(dataset:str):
    load_path = f'data/processed/{dataset}_scaffold_split.pkl'
    data_tr, data_va, data_te = read_pickle(load_path)
    max_degree = -1
    for data in data_tr:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    
    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in data_tr:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def get_deg_from_list(data_list):
    max_degree = -1
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    
    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in data_list:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def generate_link(batch_h, batch_t):
    batch_size = batch_h.max() + 1
    
    amt_h = scatter(batch_h.new_ones(batch_h.size()), batch_h, dim=0, reduce='add')
    amt_t = scatter(batch_h.new_ones(batch_t.size()), batch_t, dim=0, reduce='add')
    sizes = torch.stack([amt_h, amt_t], dim=1)
    ones = list(map(lambda x: amt_h.new_ones([x[0], x[1]]), sizes))
    A_inter = torch.block_diag(*ones)
    inter_index = A_inter.to_sparse_coo().indices()
    
    inter_batch = torch.arange(batch_size, dtype=batch_h.dtype, device=batch_h.device)
    repeat = sizes[:, 0] * sizes[:, 1]
    inter_batch = inter_batch.repeat_interleave(repeat)
    
    return inter_index, inter_batch


def unpack_graph(graph):
    x = graph.x
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    batch = graph.batch
    line_graph_edge_index = graph.line_graph_edge_index
    return [x, edge_index, edge_attr, batch, line_graph_edge_index]
    # edge_index_batch = graph.edge_index_batch
    # return [x, edge_index, edge_attr, batch, line_graph_edge_index, edge_index_batch]

def generate_edge_batch(edge_index, batch):
    return batch.index_select(0, edge_index[0])


def connection(edge_target, num_nodes):
    """
    This function is to find which nodes will be pooled together as a cluster and return an index
    """
    csr = to_scipy_sparse_matrix(edge_target, num_nodes=num_nodes)
    _, comp = connected_components(csr)
    return torch.tensor(comp, dtype=torch.int64, device=edge_target.device)


def clustering(x, edge_index, perm):   
    edge_target = edge_index[..., perm] # edge_index that will be pooled
    comp = connection(edge_target, x.size(0)) # an index for which nodes will be pooled togethor 
    
    x_mask = index_to_mask(edge_target[0], size=x.size(0))
    edge_mask = index_to_mask(perm, size=edge_index.size(1))

    return comp, x_mask, edge_mask, edge_target



def edge_reduce(edge_index, edge_attr, comp, edge_mask):
    row, col = edge_index
    row = comp[row]
    col = comp[col]
    mask = edge_mask.logical_not()
    row = row[mask] # dropout the pooled edges
    col = col[mask] # dropout the pooled edges
    edge_index = torch.stack([row, col], dim=0)
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    return edge_index, edge_attr, mask


def coalesce_with_mask(
    edge_index,
    edge_attr,
    num_nodes,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
    ):
    nnz = edge_index.size(1)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])
    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None:
            edge_attr = edge_attr[perm]
    
    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index, edge_attr, None, perm


    edge_index = edge_index[:, mask]

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not().cumsum(dim=0))
    if edge_attr is not None:
        edge_attr = scatter(edge_attr, idx, 0, dim_size, reduce)
    return edge_index, edge_attr, idx, perm

def separate_idx(comp):
    uniq, count = comp.unique(return_counts=True)
    super_id = uniq[count>1]
    nodes = [(comp==n).nonzero(as_tuple=False).view(-1) for n in super_id]
    return nodes, super_id


def subgraph_adj(subset, edge_index):
    edge_index, _ = subgraph(subset, edge_index)
    _, edge_index = edge_index.unique(return_inverse=True)
    adj = to_dense_adj(edge_index).squeeze()
    return adj


def sort_edge_index_perm(  # noqa: F811
    edge_index,
    num_nodes = None,
    sort_by_row: bool = True):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    idx = edge_index[1 - int(sort_by_row)] * num_nodes
    idx += edge_index[int(sort_by_row)]
    _, perm = index_sort(idx, max_value=num_nodes * num_nodes)
    # edge_index = edge_index[:, perm]

    return perm


def unique_edge(edge_index, edge_attr=None):
    mask = edge_index[0] < edge_index[1]
    if edge_attr is not None:
        return edge_index[:, mask], edge_attr[mask]
    return edge_index[:, mask]

def assign_edge_index(edge_index, mask, comp, sorted_index=None, idx=None):
    if idx is not None:
        mask = mask[idx]
    direct_index = edge_index[:, mask]
    row, col = sorted_index if sorted_index is not None else edge_index
    row = comp[row]; col = comp[col]
    row = row[mask]; col = col[mask] # dropout the pooled edges
    comp_index = torch.stack([row, col], dim=0)
    p = sort_edge_index_perm(comp_index)
    sorted_index = comp_index[:, p]
    assigned_index = direct_index[:, p]
    
    return assigned_index, sorted_index




























































































