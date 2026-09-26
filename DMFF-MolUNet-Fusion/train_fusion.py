"""Train the fusion model on the bundled Davis or KIBA CSV data."""

import argparse
import random
from collections import OrderedDict

import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

from build_vocab import WordVocab
from dataset.databuild import from_smiles
from fusion_model import DMFFMolUNetFusion
from models.utils import get_deg_from_list
from utils import get_cindex, get_rm2


class FusionDataset(Dataset):
    def __init__(self, rows, graphs, smiles_vocab, protein_vocab,
                 smiles_length=540, protein_length=1200):
        self.rows = rows.reset_index(drop=True)
        self.graphs = graphs
        self.smiles_vocab = smiles_vocab
        self.protein_vocab = protein_vocab
        self.smiles_length = smiles_length
        self.protein_length = protein_length

    def __len__(self):
        return len(self.rows)

    def _smiles_tokens(self, smiles):
        tokens = []
        index = 0
        while index < len(smiles):
            if (index + 1 < len(smiles)
                    and smiles[index:index + 2] in self.smiles_vocab.stoi):
                tokens.append(self.smiles_vocab.stoi[smiles[index:index + 2]])
                index += 2
            else:
                tokens.append(self.smiles_vocab.stoi.get(
                    smiles[index], self.smiles_vocab.unk_index))
                index += 1
        tokens = [self.smiles_vocab.sos_index] + tokens
        tokens = tokens[:self.smiles_length]
        return tokens + [self.smiles_vocab.pad_index] * (
            self.smiles_length - len(tokens))

    def _protein_tokens(self, sequence):
        tokens = [self.protein_vocab.sos_index]
        tokens.extend(self.protein_vocab.stoi.get(
            character, self.protein_vocab.unk_index) for character in sequence)
        tokens = tokens[:self.protein_length]
        return tokens + [self.protein_vocab.pad_index] * (
            self.protein_length - len(tokens))

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return (
            self.graphs[row['compound_iso_smiles']],
            torch.tensor(self._protein_tokens(row['target_sequence']),
                         dtype=torch.long),
            torch.tensor(self._smiles_tokens(row['compound_iso_smiles']),
                         dtype=torch.long),
            torch.tensor(float(row['affinity']), dtype=torch.float32),
        )


def collate(batch):
    graphs, proteins, smiles, labels = zip(*batch)
    return (Batch.from_data_list(list(graphs)), torch.stack(proteins),
            torch.stack(smiles), torch.stack(labels))


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    count = 0
    predictions = []
    targets = []
    for graph, protein, smiles, labels in loader:
        graph = graph.to(device)
        protein = protein.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(graph, protein, smiles)
        loss = F.mse_loss(prediction, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        count += labels.numel()
        predictions.append(prediction.detach().cpu())
        targets.append(labels.detach().cpu())
    return (total_loss / max(count, 1), torch.cat(targets).numpy(),
            torch.cat(predictions).numpy())


def load_compatible_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise ValueError(f'unsupported checkpoint format: {checkpoint_path}')
    model_state = model.graph_branch.state_dict()
    compatible = {
        key: value for key, value in checkpoint.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.graph_branch.load_state_dict(compatible, strict=False)
    print(f'pretrained={checkpoint_path} '
          f'loaded={len(compatible)}/{len(model_state)} graph parameters')


def load_dmff_sequence_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    prefix_map = {
        'smiles_embed.': 'smiles_embedding.',
        'protein_embed.': 'protein_embedding.',
        'smiles_input_fc.': 'smiles_input_projection.',
        'protein_input_fc.': 'protein_input_projection.',
        'smiles_lstm.': 'smiles_encoder.',
        'protein_lstm.': 'protein_encoder.',
    }
    sequence_state = model.sequence_branch.state_dict()
    compatible = {}
    for source_key, value in checkpoint.items():
        for source_prefix, target_prefix in prefix_map.items():
            if source_key.startswith(source_prefix):
                target_key = target_prefix + source_key[len(source_prefix):]
                if (target_key in sequence_state
                        and sequence_state[target_key].shape == value.shape):
                    compatible[target_key] = value
                break
    model.sequence_branch.load_state_dict(compatible, strict=False)
    print(f'pretrained={checkpoint_path} '
          f'loaded={len(compatible)}/{len(sequence_state)} sequence parameters')


def make_optimizer(model, fusion_lr, pretrained_lr, include_pretrained):
    fusion_parameters = list(model.graph_projection.parameters())
    fusion_parameters += list(model.sequence_projection.parameters())
    fusion_parameters += list(model.gate.parameters())
    fusion_parameters += list(model.predictor.parameters())
    parameter_groups = [{'params': fusion_parameters, 'lr': fusion_lr}]
    if include_pretrained:
        pretrained_parameters = list(model.graph_branch.parameters())
        pretrained_parameters += list(model.sequence_branch.parameters())
        parameter_groups.insert(0, {
            'params': pretrained_parameters,
            'lr': pretrained_lr,
        })
    return torch.optim.Adam(parameter_groups)


def set_pretrained_trainable(model, trainable):
    for parameter in model.graph_branch.parameters():
        parameter.requires_grad = trainable
    for parameter in model.sequence_branch.parameters():
        parameter.requires_grad = trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['davis', 'kiba'], default='davis')
    parser.add_argument('--limit', type=int, default=0,
                        help='maximum rows; 0 uses the complete CSV')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained',
                        default='checkpoint/DTI/bindingdb.pt')
    parser.add_argument('--dmff-pretrained',
                        default='/DMFF-DTA/Model/davis_processed_default.pt')
    parser.add_argument('--fusion-lr', type=float, default=1e-3)
    parser.add_argument('--pretrained-lr', type=float, default=1e-5)
    parser.add_argument('--freeze-epochs', type=int, default=3,
                        help='freeze pretrained branches for this many epochs')
    parser.add_argument('--output', default='checkpoint/DTI/fusion_davis.pt')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    csv_path = f'{args.dataset}_processed_300_add_range.csv'
    frame = pd.read_csv(csv_path).dropna(
        subset=['compound_iso_smiles', 'target_sequence', 'affinity'])
    if args.limit > 0:
        frame = frame.head(args.limit)
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError('at least two rows are required for BatchNorm training')

    smiles_vocab = WordVocab.load_vocab('Vocab/smiles_vocab.pkl')
    protein_vocab = WordVocab.load_vocab('Vocab/protein_vocab.pkl')
    unique_smiles = frame['compound_iso_smiles'].unique()
    graphs = {smiles: from_smiles(smiles, get_fp=False)
              for smiles in unique_smiles}
    deg = get_deg_from_list(list(graphs.values()))

    split = max(2, int(len(frame) * 0.75))
    split = min(split, len(frame) - 1)
    train_frame = frame.iloc[:split]
    valid_frame = frame.iloc[split:]
    train_set = FusionDataset(train_frame, graphs, smiles_vocab, protein_vocab)
    valid_set = FusionDataset(valid_frame, graphs, smiles_vocab, protein_vocab)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, drop_last=True,
                              collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    with open('config.yaml', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    model = DMFFMolUNetFusion(
        config, deg, smiles_vocab_size=max(len(smiles_vocab), 46),
        protein_vocab_size=max(len(protein_vocab), 27)).to(device)
    load_compatible_weights(model, args.pretrained)
    load_dmff_sequence_weights(model, args.dmff_pretrained)
    set_pretrained_trainable(model, args.freeze_epochs == 0)
    optimizer = make_optimizer(
        model, args.fusion_lr, args.pretrained_lr,
        include_pretrained=args.freeze_epochs == 0)
    best_valid_loss = float('inf')
    best_epoch = 0

    print(f'dataset={args.dataset} rows={len(frame)} '
          f'train={len(train_set)} valid={len(valid_set)} device={device}')
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            set_pretrained_trainable(model, True)
            optimizer = make_optimizer(
                model, args.fusion_lr, args.pretrained_lr,
                include_pretrained=True)
            print('unfroze pretrained branches')
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, device, True)
        with torch.no_grad():
            valid_loss, valid_targets, valid_predictions = run_epoch(
                model, valid_loader, optimizer, device, False)
        valid_cindex = get_cindex(valid_targets, valid_predictions)
        valid_rm2 = get_rm2(valid_targets, valid_predictions)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.output)
            best_marker = f' saved_best={args.output}'
        else:
            best_marker = ''
        print(f'epoch={epoch} train_mse={train_loss:.6f} '
              f'valid_mse={valid_loss:.6f} '
              f'valid_cindex={valid_cindex:.6f} '
              f'valid_rm2={valid_rm2:.6f} best_epoch={best_epoch}'
              f'{best_marker}')


if __name__ == '__main__':
    main()