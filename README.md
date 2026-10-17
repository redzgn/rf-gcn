# RF-GCN: Receptive-Field Regularized Graph Convolutional Network

> **Hop-Distance-Weighted Label Propagation for Semi-Supervised Node Classification**

RF-GCN extends a standard 2-layer GCN with an auxiliary label-propagation loss that
pushes unlabeled nodes toward the labels of nearby training nodes, weighted by hop
distance and local structural homophily. The paper evaluates it in the label-scarce
regime (5 labels per class).

## Key Results (5 labels / class, 10 seeds, 400 epochs)

GCN → RF-GCN, mean over seeds 1–10 (Table 4 of the paper):

| Dataset     | Accuracy        | Balanced accuracy | Macro-F1        | ROC-AUC         |
|-------------|-----------------|-------------------|-----------------|-----------------|
| Actor       | 20.83 → 22.09   | 20.13 → 20.22     | 17.99 → 16.92   | 50.80 → 51.03   |
| Minesweeper | 70.82 → 72.07   | 57.14 → 55.80     | 55.40 → 54.49   | 64.06 → 62.09   |
| PubMed      | 71.56 → 71.51   | 72.16 → 72.84     | 71.38 → 71.19   | 86.35 → 86.97   |
| Tolokers    | 61.52 → 72.91   | 59.83 → 56.46     | 54.23 → 54.85   | 62.94 → 57.61   |

RF-GCN uses the configuration selected on validation accuracy (see Hyperparameters).
The only significant difference (paired Wilcoxon test, Holm-corrected) is the accuracy
gain on Tolokers, which comes from predicting the majority class more often. On PubMed,
supervision beyond the receptive field significantly improves a one-layer GCN (Table 6).

## Dataset Information

All datasets are loaded via [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
without modification to features, edges, or labels.

| Dataset     | Nodes  | Edges   | Features | Classes | Source |
|-------------|--------|---------|----------|---------|--------|
| Actor       | 7,600  | 26,659  | 932      | 5       | Pei et al. (2020) — Geom-GCN |
| Minesweeper | 10,000 | 39,402  | 7        | 2       | Platonov et al. (2023) |
| PubMed      | 19,717 | 44,324  | 500      | 3       | Yang et al. (2016) — Planetoid |
| Tolokers    | 11,758 | 519,000 | 10       | 2       | Platonov et al. (2023) |

Edges are undirected. Actor is distributed as a directed graph (29,926 stored edges).

**Dataset sources:**
- **Actor**: https://github.com/graphdml-uiuc-jlu/geom-gcn
- **Minesweeper / Tolokers**: https://github.com/yandex-research/heterophilous-graphs (DOI: 10.48550/arxiv.2302.11640)
- **PubMed**: available via `torch_geometric.datasets.Planetoid`

The `data.tar.gz` archive in this repo packages the PyG `.pt` files for convenience.
Alternatively, they can be re-downloaded by running `python run_experiment.py --download`.

## Requirements

```bash
pip install torch torch_geometric pandas pyyaml scikit-learn scipy
```

Tested with Python 3.11, PyTorch 2.5.1, PyTorch Geometric 2.7.0, CUDA 12.1.
CPU-only machines are supported (slower but correct).

## Usage Instructions

### Main comparison (GCN vs RF-GCN, accuracy) in 4 commands

```bash
git clone https://github.com/redzgn/rf-gcn.git
cd rf-gcn
tar -zxvf data.tar.gz
python run_experiment.py
```

Results are printed to the terminal and saved under `outputs/`.

### Reproduce all result tables of the paper

```bash
tar -zxvf data.tar.gz
cd experiments
python run_experiments.py           # 1,560 training runs + label propagation (~30 min on an RTX 4090)
python diagnostics.py preliminary   # dataset and RF-pair statistics (Tables 1 and S5)
python diagnostics.py selected
python make_tables.py               # Tables 1, 4-7 and S1-S5 -> results/tables.md, results/latex/
```

`run_experiments.py` appends to `results/runs.csv` and skips finished runs, so it can be
stopped and restarted; `python run_experiments.py valsel` runs a single group (groups
are listed in the script). On a GPU, PyTorch Geometric's scatter operations are not
deterministic, so re-runs can differ from the paper by a few tenths of a percentage point
on some datasets.

### Smoke test (under 30 seconds)

```bash
python run_experiment.py --quick
```

Runs Tolokers, 1 seed, 100 epochs. Verifies the installation is working.

### Full option reference

```bash
# Paper setting (default): 4 datasets, 10 seeds, 400 epochs, 5 labels/class
python run_experiment.py

# Custom dataset and seed subset
python run_experiment.py --datasets PubMed Tolokers --seeds 1 2 3

# Label-rate sensitivity sweep (3 / 5 / 10 / 20 labels per class)
python run_experiment.py --sweep_labels

# Override epochs or device
python run_experiment.py --epochs 200 --device cuda
```

## Code Information

```
run_experiment.py     entry point: training loop, checkpointing, results table
src/
  config.yaml         all hyperparameters
  model.py            2-layer GCN backbone
  data_utils.py       dataset loading + label subsampling
  rf_pairs.py         BFS-based RF pair construction with homophily weighting
  losses.py           hop-weighted label CE loss + symmetric KL loss
  requirements.txt    pinned dependencies

experiments/          scripts that produce the result tables of the paper
  run_experiments.py  training runs: GCN, RF-GCN, APPNP (± RF loss), label propagation
  diagnostics.py      dataset and RF-pair statistics (no training)
  make_tables.py      result tables from the CSV files (Markdown and LaTeX)
  results/            created at runtime

data/                 extracted from data.tar.gz
  Actor/data.pt
  Minesweeper/data.pt
  PubMed/data.pt
  Tolokers/data.pt

outputs/              created at runtime
  raw_results.csv     one row per dataset x method x seed
  summary.csv         mean ± std per dataset x method
```

## Methodology

VanillaGCN trains with standard cross-entropy on labeled nodes:

```
loss = CE(logits[train_mask], y[train_mask])
```

RF-GCN adds a receptive-field regularization term:

```
loss = CE(logits[train_mask], y[train_mask])
     + lambda_rf * RF_loss(logits, rf_pairs)
```

`rf_pairs` are (source, destination) node pairs sampled at hop distances
`[min_hop, max_hop]` from each labeled training node. Each pair weight is
`alpha^hop × source_homophily`, where `source_homophily` is the fraction of
the source node's labeled 1-hop neighbors that share its class, suppressing
label propagation from structurally unreliable anchor nodes.

RF pairs are constructed once via BFS before training and held fixed across
all epochs.

## Hyperparameters

| Parameter          | Value | Description                            |
|--------------------|-------|----------------------------------------|
| labels_per_class   | 5     | labeled nodes per class                |
| epochs             | 400   | training budget (paper setting)        |
| lambda_rf          | 2.0   | RF loss weight                         |
| min_hop            | 1     | closest hop for RF pairs               |
| max_hop            | 4     | furthest hop for RF pairs              |
| alpha              | 0.5   | distance decay per hop                 |
| pairs_per_source   | 8     | pairs sampled per source node per hop  |
| hidden_dim         | 64    | GCN hidden dimension                   |
| dropout            | 0.5   | dropout rate                           |
| lr                 | 0.01  | Adam learning rate                     |
| weight_decay       | 5e-4  | L2 regularization                      |

`lambda_rf`, `max_hop`, `alpha` and `pairs_per_source` were selected on mean validation
accuracy over the four datasets (Tables 3 and S1 of the paper). The same values are used
for all datasets.

## Citations

If you use this code or the RF-GCN method, please cite:

```bibtex
@article{zheng2025rfgcn,
  title   = {{RF-GCN}: Receptive-Field Regularized Graph Convolutional Network},
  author  = {Zheng, Guineng and Peng, Ting and Wang, Chuanchuan and Wen, Haifeng},
  journal = {PeerJ Computer Science},
  year    = {2025}
}
```

**Dataset citations:**

- Pei et al. (2020). *Geom-GCN: Geometric Graph Convolutional Networks.* ICLR 2020. https://github.com/graphdml-uiuc-jlu/geom-gcn
- Yang et al. (2016). *Revisiting Semi-Supervised Learning with Graph Embeddings.* ICML 2016.
- Platonov et al. (2023). *A Critical Look at the Evaluation of GNNs under Heterophily.* DOI: 10.48550/arxiv.2302.11640

## License

This project is released under the [MIT License](LICENSE).
