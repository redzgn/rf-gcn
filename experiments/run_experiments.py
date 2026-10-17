#!/usr/bin/env python3
"""Experiments behind the result tables of the RF-GCN paper (Tables 4-7, S1-S4).

Reuses the training loop of ../run_experiment.py (epoch picked by
validation accuracy, 400 epochs, paper hyperparameters) and only varies
settings. Every run logs test accuracy, balanced accuracy, macro-F1, ROC-AUC
and the predicted-minority rate. Results are appended to results/runs.csv and
finished runs are skipped on restart.

Usage:  python run_experiments.py [group ...]      (default: all groups)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import APPNP, LabelPropagation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from run_experiment import set_seed  # noqa: E402
from model import GCN  # noqa: E402
from data_utils import load_dataset, subsample_train_mask  # noqa: E402
from rf_pairs import build_rf_pairs  # noqa: E402
from losses import rf_loss  # noqa: E402

DATA = REPO / "data"
OUT = HERE / "results" / "runs.csv"
OUT.parent.mkdir(exist_ok=True)
DATASETS = ["Actor", "Minesweeper", "PubMed", "Tolokers"]
SEEDS = list(range(1, 11))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preliminary setting; the ablations vary one factor from it. The
# validation-selected setting (Table 3 of the paper) is the "valsel" group.
BASE = dict(backbone="gcn", layers=2, lpc=5, lam=0.0, min_hop=1, max_hop=3,
            alpha=0.7, P=16, excl_test=False)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


GROUPS = {
    # E1/E2/E4: lambda grid; lambda is then picked on validation only
    "main": [cfg(lam=l) for l in (0.0, 0.5, 1.0, 2.0)],
    # R1.3: hop sets
    "hops": [cfg(lam=2.0, min_hop=a, max_hop=b)
             for a, b in ((1, 1), (2, 2), (3, 3), (1, 2))],
    # R1.3c / R2.4: depth L vs reach H
    "depth": [cfg(layers=L, lam=l, max_hop=H)
              for L in (1, 2, 3)
              for l, H in [(0.0, 3)] + [(2.0, h) for h in (1, 2, 3, 4)]],
    # R1.5 / R2.4: alpha (1.0 = uniform hop weighting) and P
    "alpha": [cfg(lam=2.0, alpha=a) for a in (0.5, 0.9, 1.0)],
    "P": [cfg(lam=2.0, P=p) for p in (4, 8, 32)],
    # R2.5: destinations restricted to non-test nodes
    "excl_test": [cfg(lam=2.0, excl_test=True)],
    # R1.6: label budget
    "budget": [cfg(lpc=k, lam=l) for k in (3, 10, 20) for l in (0.0, 2.0)],
    # E6: configuration chosen on validation only (coordinate-wise over the
    # H, alpha and P grids above: H=4, alpha=0.5, P=8; lambda=2.0)
    "valsel": [cfg(lam=2.0, max_hop=4, alpha=0.5, P=8)],
    # Label budget and APPNP with the validation-selected configuration
    # (GCN / APPNP baselines with lam=0 do not depend on RF settings)
    "budget_vs": [cfg(lpc=k, lam=2.0, max_hop=4, alpha=0.5, P=8) for k in (3, 10, 20)],
    "appnp_vs": [cfg(backbone="appnp", lam=2.0, max_hop=4, alpha=0.5, P=8)],
    # E5 / R2.1: one extra backbone (APPNP, also a named comparison method)
    "appnp": [cfg(backbone="appnp", lam=l) for l in (0.0, 2.0)],
}

KEY = ["dataset", "seed", "backbone", "layers", "lpc", "lam", "min_hop",
       "max_hop", "alpha", "P", "excl_test"]


class APPNPNet(nn.Module):
    """MLP + personalized-PageRank propagation (K=10, alpha=0.1, Gasteiger et al.)."""

    def __init__(self, in_channels, hidden, out_channels, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden)
        self.lin2 = nn.Linear(hidden, out_channels)
        self.prop = APPNP(K=10, alpha=0.1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.dropout(F.relu(self.lin1(x)), self.dropout, self.training)
        return self.prop(self.lin2(x), edge_index)


def metrics(scores, y, prefix):
    """scores: [n, C] probabilities (or non-negative label scores)."""
    pred = scores.argmax(1)
    C = scores.shape[1]
    out = {
        f"{prefix}acc": float((pred == y).mean()),
        f"{prefix}bal_acc": float(balanced_accuracy_score(y, pred)),
        f"{prefix}macro_f1": float(f1_score(y, pred, average="macro")),
        f"{prefix}pred_minority": float((pred == np.bincount(y, minlength=C).argmin()).mean()),
    }
    if C == 2:
        out[f"{prefix}auc"] = float(roc_auc_score(y, scores[:, 1]))
    else:
        s = scores / np.clip(scores.sum(1, keepdims=True), 1e-12, None)
        s = np.where(s.sum(1, keepdims=True) > 0, s, 1.0 / C)
        out[f"{prefix}auc"] = float(roc_auc_score(y, s, multi_class="ovr", average="macro"))
    return out


def build_pairs_excluding_test(data, seed, alpha, P, max_hop=3):
    """Same sampling as rf_pairs.build_rf_pairs, but test nodes are never chosen
    as destinations (paths may still pass through them; labels stay hidden)."""
    import random
    from rf_pairs import _local_homophily
    rng = random.Random(int(seed))
    n = int(data.num_nodes)
    adj = [[] for _ in range(n)]
    for u, v in data.edge_index.cpu().t().tolist():
        if u != v:
            adj[u].append(v)
            adj[v].append(u)
    adj = [list(set(a)) for a in adj]
    y, tm = data.y.cpu().tolist(), data.train_mask.cpu().tolist()
    test = data.test_mask.cpu().tolist()
    sources = torch.where(data.train_mask.cpu())[0].tolist()
    phi = {s: _local_homophily(s, adj, y, tm) for s in sources}
    rng.shuffle(sources)
    src_all, dst_all, w_all = [], [], []
    for s in sources:
        visited, frontier = {s}, [s]
        for hop in range(1, max_hop + 1):
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
            frontier = nxt
            cand = [v for v in nxt if not test[v]]
            rng.shuffle(cand)
            for t in cand[:P]:
                src_all.append(s)
                dst_all.append(t)
                w_all.append(float(alpha) ** hop * phi[s])
    dev = data.x.device
    src_t = torch.tensor(src_all, dtype=torch.long, device=dev)
    return {"src": src_t, "dst": torch.tensor(dst_all, dtype=torch.long, device=dev),
            "weight": torch.tensor(w_all, dtype=torch.float, device=dev),
            "source_label": data.y[src_t], "num_pairs": len(src_all)}


_cache = {}


def get_data(name):
    if name not in _cache:
        _cache[name] = load_dataset(name, str(DATA), device=DEV)
    return _cache[name].clone()


def run_one(name, seed, c):
    set_seed(seed)
    data = subsample_train_mask(get_data(name), c["lpc"], seed=seed)
    C = int(data.y.max()) + 1
    if c["backbone"] == "gcn":
        model = GCN(data.num_features, 64, C, num_layers=c["layers"], dropout=0.5).to(DEV)
    else:
        model = APPNPNet(data.num_features, 64, C).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    pairs = None
    if c["lam"] > 0:
        if c["excl_test"]:
            pairs = build_pairs_excluding_test(data, seed, c["alpha"], c["P"], c["max_hop"])
        else:
            pairs = build_rf_pairs(data, seed=seed, min_hop=c["min_hop"], max_hop=c["max_hop"],
                                   alpha=c["alpha"], pairs_per_source=c["P"],
                                   pair_graph="undirected")
    best_val, best_ep, best_logits = -1.0, -1, None
    for ep in range(1, 401):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if pairs is not None:
            loss = loss + c["lam"] * rf_loss(logits, pairs, loss_type="label_ce")
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            ev = model(data.x, data.edge_index)
        va = (ev.argmax(1)[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        if va > best_val:
            best_val, best_ep, best_logits = va, ep, ev.detach()
    prob = torch.softmax(best_logits, 1).cpu().numpy()
    y = data.y.cpu().numpy()
    vm, tm = data.val_mask.cpu().numpy(), data.test_mask.cpu().numpy()
    row = {**c, "dataset": name, "seed": seed, "method": "rf" if c["lam"] > 0 else "base",
           "best_epoch": best_ep, "num_pairs": pairs["num_pairs"] if pairs else 0}
    row.update(metrics(prob[vm], y[vm], "val_"))
    row.update(metrics(prob[tm], y[tm], "test_"))
    return row


def run_label_propagation(rows_done):
    """Classical label propagation (Zhou et al.); alpha picked on validation accuracy."""
    out = HERE / "results" / "label_prop.csv"
    if out.exists():
        return
    rows = []
    for name in DATASETS:
        for seed in SEEDS:
            data = subsample_train_mask(get_data(name), 5, seed=seed)
            y = data.y.cpu().numpy()
            vm, tm = data.val_mask.cpu().numpy(), data.test_mask.cpu().numpy()
            best = None
            for a in (0.5, 0.7, 0.9, 0.99):
                lp = LabelPropagation(num_layers=50, alpha=a)
                s = lp(data.y, data.edge_index, mask=data.train_mask).cpu().numpy()
                v = metrics(s[vm], y[vm], "val_")
                if best is None or v["val_acc"] > best[0]["val_acc"]:
                    best = (v, metrics(s[tm], y[tm], "test_"), a)
            rows.append({"dataset": name, "seed": seed, "lp_alpha": best[2], **best[0], **best[1]})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"label propagation -> {out}")


def main():
    groups = sys.argv[1:] or list(GROUPS) + ["lp"]
    done = set()
    rows = []
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = {tuple(r[k] for k in KEY) for r in rows}
    if "lp" in groups:
        run_label_propagation(done)
    todo = []
    for g in groups:
        for c in GROUPS.get(g, []):
            for name in DATASETS:
                for seed in SEEDS:
                    key = tuple({**c, "dataset": name, "seed": seed}[k] for k in KEY)
                    if key not in done:
                        done.add(key)
                        todo.append((g, name, seed, c))
    print(f"{len(todo)} runs to do on {DEV}", flush=True)
    t0 = time.time()
    for i, (g, name, seed, c) in enumerate(todo, 1):
        r = run_one(name, seed, c)
        r["group"] = g
        rows.append(r)
        if i % 20 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[{i}/{len(todo)}] {time.time() - t0:.0f}s  last: {g} {name} s{seed} "
                  f"test_acc={r['test_acc']:.3f} auc={r['test_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
