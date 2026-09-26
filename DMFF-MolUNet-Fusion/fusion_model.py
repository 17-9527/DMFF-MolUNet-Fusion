"""Standalone DMFF-DTA and NestedMolUNet fusion model.

This module is intentionally isolated from both source repositories. It accepts
three tensors/objects produced by the two original data pipelines:

* ``molecule_graph``: a PyG graph batch with ``x``, ``edge_index`` and
  ``edge_attr``;
* ``protein_tokens``: integer protein tokens with shape ``[batch, length]``;
* ``smiles_tokens``: integer SMILES tokens with shape ``[batch, length]``.

The NestedMolUNet branch models molecular graph/protein interactions. The
sequence branch follows the DMFF sequence encoder pattern. Their pooled
representations are fused by a learned gate before regression.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from models.model_dti import UnetDTI


class SequenceBranch(nn.Module):
    """DMFF-style bidirectional sequence encoder for SMILES and proteins."""

    def __init__(self, smiles_vocab_size: int, protein_vocab_size: int,
                 token_dim: int = 256, hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, token_dim,
                                             padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, token_dim,
                                              padding_idx=0)
        self.smiles_input_projection = nn.Linear(token_dim, hidden_dim)
        self.protein_input_projection = nn.Linear(token_dim, hidden_dim)
        self.smiles_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout)
        self.protein_encoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True,
            bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 4

    @staticmethod
    def masked_mean(values: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(self, smiles_tokens: torch.Tensor,
                protein_tokens: torch.Tensor) -> torch.Tensor:
        smiles = self.smiles_embedding(smiles_tokens.long())
        protein = self.protein_embedding(protein_tokens.long())
        smiles = self.smiles_input_projection(smiles)
        protein = self.protein_input_projection(protein)
        smiles, _ = self.smiles_encoder(smiles)
        protein, _ = self.protein_encoder(protein)
        smiles = self.masked_mean(smiles, smiles_tokens)
        protein = self.masked_mean(protein, protein_tokens)
        return self.dropout(torch.cat([smiles, protein], dim=-1))


class DMFFMolUNetFusion(nn.Module):
    """Fuse NestedMolUNet graph interactions with DMFF sequence features."""

    def __init__(self, config: dict, deg: torch.Tensor,
                 smiles_vocab_size: int = 46, protein_vocab_size: int = 27,
                 sequence_hidden_dim: int = 128):
        super().__init__()
        model_config = copy.deepcopy(config)
        model_config['deg'] = deg
        self.graph_branch = UnetDTI(
            model_config, protein_vocab_size=26)
        self.sequence_branch = SequenceBranch(
            smiles_vocab_size=smiles_vocab_size,
            protein_vocab_size=protein_vocab_size,
            hidden_dim=sequence_hidden_dim,
            dropout=config['predict']['dropout_rate'],
        )
        graph_dim = self.graph_branch.graph_feature_dim
        sequence_dim = self.sequence_branch.output_dim
        fusion_dim = 128
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.sequence_projection = nn.Sequential(
            nn.Linear(sequence_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(config['predict']['dropout_rate']),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, molecule_graph: object, protein_tokens: torch.Tensor,
                smiles_tokens: torch.Tensor) -> torch.Tensor:
        _, _, graph_feature = self.graph_branch(
            molecule_graph, protein_tokens, return_features=True)
        graph_feature = self.graph_projection(graph_feature)
        sequence_feature = self.sequence_projection(
            self.sequence_branch(smiles_tokens, protein_tokens))
        gate = self.gate(torch.cat([graph_feature, sequence_feature], dim=-1))
        fused = gate * graph_feature + (1.0 - gate) * sequence_feature
        return self.predictor(fused).squeeze(-1)


__all__ = ['DMFFMolUNetFusion']
