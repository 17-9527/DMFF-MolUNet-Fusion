"""Run a minimal end-to-end forward and backward check for the fusion model."""

import argparse

import torch
import yaml
from torch_geometric.data import Batch

from dataset.databuild import from_smiles
from fusion_model import DMFFMolUNetFusion
from models.utils import get_deg_from_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.config, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    graph = from_smiles('CCO', get_fp=False)
    deg = get_deg_from_list([graph])
    model = DMFFMolUNetFusion(config, deg).to(device)
    model.train()

    molecule_graph = Batch.from_data_list([graph, graph]).to(device)
    protein_tokens = torch.zeros(
        (2, 1200), dtype=torch.long, device=device)
    protein_tokens[:, :6] = torch.tensor(
        [3, 5, 6, 7, 8, 9], dtype=torch.long, device=device)
    smiles_tokens = torch.tensor(
        [[3, 10, 11, 2, 0, 0], [3, 10, 11, 2, 0, 0]],
        dtype=torch.long,
        device=device,
    )
    target = torch.tensor([7.0, 7.0], dtype=torch.float32, device=device)

    prediction = model(molecule_graph, protein_tokens, smiles_tokens)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()

    print('prediction_shape:', tuple(prediction.shape))
    print('loss:', float(loss.detach()))
    print('status: forward and backward succeeded')


if __name__ == '__main__':
    main()