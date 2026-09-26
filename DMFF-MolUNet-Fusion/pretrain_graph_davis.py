"""Pretrain the NestedMolUNet/UnetDTI graph branch on Davis affinity data."""

import argparse
import random

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

from build_vocab import WordVocab
from dataset.databuild import from_smiles
from models.model_dti import UnetDTI
from models.utils import get_deg_from_list
from utils import get_cindex, get_rm2


CHARPROTSET = {
    'A': 1, 'C': 2, 'B': 3, 'E': 4, 'D': 5, 'G': 6, 'F': 7,
    'I': 8, 'H': 9, 'K': 10, 'M': 11, 'L': 12, 'O': 13, 'N': 14,
    'Q': 15, 'P': 16, 'S': 17, 'R': 18, 'U': 19, 'T': 20,
    'W': 21, 'V': 22, 'Y': 23, 'X': 24, 'Z': 25,
}


class DavisGraphDataset(Dataset):
    def __init__(self, frame, graphs, protein_length=1200):
        self.frame = frame.reset_index(drop=True)
        self.graphs = graphs
        self.protein_length = protein_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        tokens = [CHARPROTSET.get(char.upper(), 0)
                  for char in row['target_sequence'][:self.protein_length]]
        tokens += [0] * (self.protein_length - len(tokens))
        return (self.graphs[row['compound_iso_smiles']],
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(float(row['affinity']), dtype=torch.float32))


def collate(batch):
    graphs, proteins, labels = zip(*batch)
    return (Batch.from_data_list(list(graphs)), torch.stack(proteins),
            torch.stack(labels))


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    labels_all = []
    predictions_all = []
    for graph, proteins, labels in loader:
        graph = graph.to(device)
        proteins = proteins.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        _, predictions = model(graph, proteins)
        predictions = predictions.squeeze(-1)
        loss = F.mse_loss(predictions, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        labels_all.append(labels.detach().cpu())
        predictions_all.append(predictions.detach().cpu())
    labels_all = torch.cat(labels_all).numpy()
    predictions_all = torch.cat(predictions_all).numpy()
    return total_loss / len(labels_all), labels_all, predictions_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='davis', choices=['davis'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='checkpoint/DTI/davis.pt')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    frame = pd.read_csv('davis_processed_300_add_range.csv').dropna(
        subset=['compound_iso_smiles', 'target_sequence', 'affinity'])
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    split = int(len(frame) * 0.75)
    train_frame = frame.iloc[:split]
    valid_frame = frame.iloc[split:]
    graphs = {smiles: from_smiles(smiles, get_fp=False)
              for smiles in frame['compound_iso_smiles'].unique()}
    deg = get_deg_from_list(list(graphs.values()))
    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    config['deg'] = deg
    model = UnetDTI(config).to(device)
    train_loader = DataLoader(
        DavisGraphDataset(train_frame, graphs), batch_size=args.batch_size,
        shuffle=False, drop_last=True, collate_fn=collate)
    valid_loader = DataLoader(
        DavisGraphDataset(valid_frame, graphs), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mse = float('inf')

    print(f'dataset=davis rows={len(frame)} train={len(train_frame)} '
          f'valid={len(valid_frame)} device={device}')
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, device, True)
        with torch.no_grad():
            valid_loss, labels, predictions = run_epoch(
                model, valid_loader, optimizer, device, False)
        cindex = get_cindex(labels, predictions)
        rm2 = get_rm2(labels, predictions)
        print(f'epoch={epoch} train_mse={train_loss:.6f} '
              f'valid_mse={valid_loss:.6f} valid_cindex={cindex:.6f} '
              f'valid_rm2={rm2:.6f}')
        if valid_loss < best_mse:
            best_mse = valid_loss
            torch.save(model.state_dict(), args.output)
            print(f'saved={args.output}')


if __name__ == '__main__':
    main()