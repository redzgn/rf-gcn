# RF-GCN: Receptive-Field Regularized Graph Convolutional Network

> **Hop-Distance-Weighted Label Propagation for Semi-Supervised Node Classification**

RF-GCN extends a standard 2-layer GCN with an auxiliary label-propagation loss that
pushes unlabeled nodes toward the labels of nearby training nodes, weighted by hop
distance and local structural homophily. In the label-scarce regime (5 labels per
class), RF-GCN consistently outperforms a vanilla GCN baseline.

## Key Results (5 labels / class, 5 seeds, 400 epochs)

| Dataset     | VanillaGCN          | RF-GCN              | Delta   |
|-------------|---------------------|---------------------|---------|
| Tolokers    | 61.70 ± 10.11       | 71.93 ± 5.97        | +10.2%  |
| Actor       | 20.58 ± 1.13        | 22.24 ± 1.91        | +1.7%   |
| PubMed      | 70.73 ± 1.52        | 72.48 ± 3.24        | +1.8%   |
| Minesweeper | 68.58 ± 13.35       | 70.04 ± 11.67       | +1.5%   |

Results match Table 1 of the paper exactly.

## Dataset Information

All datasets are loaded via [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
without modification to features, edges, or labels.

| Dataset     | Nodes  | Edges   | Features | Classes | Source |
|-------------|--------|---------|----------|---------|--------|
| Actor       | 7,600  | 29,926  | 932      | 5       | Pei et al. (2020) — Geom-GCN |
| Minesweeper | 10,000 | 39,402  | 7        | 2       | Platonov et al. (2023) |
| PubMed      | 19,717 | 44,324  | 500      | 3       | Yang et al. (2016) — Planetoid |
| Tolokers    | 11,758 | 519,000 | 10       | 2       | Platonov et al. (2023) |

**Dataset sources:**
- **Actor**: https://github.com/graphdml-uiuc-jlu/geom-gcn
- **Minesweeper / Tolokers**: https://github.com/yandex-research/heterophilous-graphs (DOI: 10.48550/arxiv.2302.11640)
- **PubMed**: available via `torch_geometric.datasets.Planetoid`

The `data.tar.gz` archive in this repo packages the PyG `.pt` files for convenience.
Alternatively, they can be re-downloaded by running `python run_experiment.py --download`.

## Requirements

```bash
pip install torch torch_geometric pandas pyyaml
```

Tested with Python 3.10, PyTorch 2.5.1, PyTorch Geometric 2.7.0, CUDA 12.1.
CPU-only machines are supported (slower but correct).

## Usage Instructions

### Reproduce paper results in 3 commands

```bash
git clone https://github.com/redzgn/rf-gcn.git
cd rf-gcn
tar -zxvf data.tar.gz
python run_experiment.py
```

Results are printed to the terminal and saved under `outputs/`.

### Smoke test (under 30 seconds)

```bash
python run_experiment.py --quick
```

Runs Tolokers, 1 seed, 100 epochs. Verifies the installation is working.

### Full option reference

```bash
# Paper setting (default): 4 datasets, 5 seeds, 400 epochs, 5 labels/class
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
| max_hop            | 3     | furthest hop for RF pairs              |
| alpha              | 0.7   | distance decay per hop                 |
| pairs_per_source   | 16    | pairs sampled per source node per hop  |
| hidden_dim         | 64    | GCN hidden dimension                   |
| dropout            | 0.5   | dropout rate                           |
| lr                 | 0.01  | Adam learning rate                     |
| weight_decay       | 5e-4  | L2 regularization                      |

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
