# DMFF-MolUNet-Fusion

This directory is an isolated fusion workspace. The original `DMFF-DTA` and `NestedMolUNet` directories remain unchanged.

## Contents

- `fusion_model.py`: sequence and graph fusion model.
- `main.py`, `main_warm_up.py`, `dataset.py`, `model.py`: original DMFF-DTA training code copied into this workspace.
- `models/`, `dataset/`, `config.yaml`: NestedMolUNet implementation and configuration.
- `checkpoint/`: copied pretrained checkpoints.
- `davis_processed_300_add_range.csv`, `kiba_processed_300_add_range.csv`, and contact maps: copied runtime data.

## Model inputs

`DMFFMolUNetFusion` expects:

- a PyTorch Geometric molecule batch with `x`, `edge_index`, and `edge_attr`;
- integer protein tokens shaped `[batch, protein_length]`;
- integer SMILES tokens shaped `[batch, smiles_length]`.

Construct the model with the PNA degree histogram from the training molecular graphs:

```python
from fusion_model import DMFFMolUNetFusion
from models.utils import get_deg_from_list

config['deg'] = get_deg_from_list(training_graphs)
model = DMFFMolUNetFusion(config, config['deg'])
```

The original training entry points are retained for reference. Fusion training should be wired to a dedicated loader before running a full experiment.
