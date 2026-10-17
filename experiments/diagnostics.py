#!/usr/bin/env python3
"""Data and RF-pair diagnostics (Tables 1, 2 and S5 of the paper; no training).

E2 / R1.4   class counts per split, majority-class baseline
R1.8        edge convention and average degree
R2.3        phi(s) statistics
R1.5        same-class agreement at exact hop h (graph level and for RF pairs)
R1.5b       destination multiplicity and conflicting targets, by hop
R2.5        share of RF pairs / weight whose destination is a test node

Writes results/diagnostics.csv (one row per dataset x hop where relevant)
and prints a readable summary.

Usage:  python diagnostics.py [preliminary|selected]
RF-pair statistics depend on the pair configuration: "preliminary" is the
submitted setting (H=3, alpha=0.7, P=16, -> diagnostics.csv), "selected" the
validation-selected one (H=4, alpha=0.5, P=8, -> diagnostics_selected.csv).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.utils import homophily

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from data_utils import load_dataset, subsample_train_mask  # noqa: E402
from rf_pairs import build_rf_pairs, _local_homophily  # noqa: E402

DATA = HERE.parent / "data"
DATASETS = ["Actor", "Minesweeper", "PubMed", "Tolokers"]
SEEDS = list(range(1, 11))
CONFIGS = {"preliminary": (0.7, 3, 16, "diagnostics.csv"),
           "selected": (0.5, 4, 8, "diagnostics_selected.csv")}
ALPHA, H, P, OUT_NAME = CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "preliminary"]


def undirected_csr(edge_index, n):
    ei = edge_index[:, edge_index[0] != edge_index[1]].numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.int8).tocsr()
    return A


def exact_hop_sets(A, s, max_hop):
    """Nodes at exact distance 1..max_hop from s (vectorised BFS on CSR)."""
    n = A.shape[0]
    seen = np.zeros(n, bool)
    seen[s] = True
    frontier = np.array([s])
    out = {}
    for h in range(1, max_hop + 1):
        nb = np.unique(A.indices[np.concatenate(
            [np.arange(A.indptr[u], A.indptr[u + 1]) for u in frontier])]) if len(frontier) else np.array([], int)
        nb = nb[~seen[nb]]
        seen[nb] = True
        out[h] = nb
        frontier = nb
    return out


rows, summary = [], []
for name in DATASETS:
    raw = torch.load(DATA / name / "data.pt", map_location="cpu", weights_only=False)
    y = raw.y.long().numpy()
    n = raw.num_nodes
    C = int(y.max()) + 1
    A = undirected_csr(raw.edge_index, n)
    ei = raw.edge_index[:, raw.edge_index[0] != raw.edge_index[1]]
    stored = {tuple(e) for e in ei.t().tolist()}
    recip = np.mean([(v, u) in stored for u, v in stored])
    E_und = A.nnz // 2
    masks = {"all": np.ones(n, bool), "train": raw.train_mask.numpy(),
             "val": raw.val_mask.numpy(), "test": raw.test_mask.numpy()}
    info = {"dataset": name, "nodes": n, "stored_edges": len(stored),
            "reciprocity": round(recip, 3), "undirected_edges": E_und,
            "avg_degree": round(2 * E_und / n, 2), "features": int(raw.x.shape[1])}
    # Table 2: homophily on the stored edges; "edge_insensitive" is the
    # class-insensitive edge homophily of Lim et al. (2021)
    for m in ("edge", "node", "edge_insensitive"):
        info[f"homophily_{m}"] = round(float(homophily(ei, raw.y, method=m)), 3)
    for k, m in masks.items():
        cnt = np.bincount(y[m], minlength=C)
        info[f"{k}_n"] = int(m.sum())
        info[f"{k}_counts"] = "/".join(map(str, cnt))
        info[f"{k}_props"] = "/".join(f"{c / cnt.sum():.3f}" for c in cnt)
    pt = np.bincount(y[masks["test"]], minlength=C) / masks["test"].sum()
    info["majority_test_acc"] = round(100 * pt.max(), 2)
    prior = np.bincount(y, minlength=C) / n
    info["chance_agreement"] = round(float((prior ** 2).sum()), 3)

    # Graph-level same-class agreement at exact hop h (1,000 random sources)
    rng = np.random.default_rng(0)
    agree = defaultdict(list)
    for s in rng.choice(n, size=min(1000, n), replace=False):
        for h, nodes in exact_hop_sets(A, s, H).items():
            if len(nodes):
                agree[h].append((y[nodes] == y[s]).mean())
    for h in range(1, H + 1):
        info[f"graph_agree_h{h}"] = round(float(np.mean(agree[h])), 3)

    # RF pairs under the paper setting, 10 seeds
    per_seed = defaultdict(list)
    for seed in SEEDS:
        data = load_dataset(name, str(DATA), device="cpu")
        data = subsample_train_mask(data, 5, seed=seed)
        pr = build_rf_pairs(data, seed=seed, min_hop=1, max_hop=H, alpha=ALPHA,
                            pairs_per_source=P, pair_graph="undirected")
        src, dst, w = pr["src"].numpy(), pr["dst"].numpy(), pr["weight"].numpy()
        trm = data.train_mask.numpy()
        adj = [A.indices[A.indptr[u]:A.indptr[u + 1]].tolist() for u in range(n)]
        sources = np.where(trm)[0]
        phi = {s: _local_homophily(s, adj, y.tolist(), trm.tolist()) for s in sources}
        has_nbr = [trm[adj[s]].any() if adj[s] else False for s in sources]
        per_seed["src_with_labeled_nbr"].append(np.mean(has_nbr))
        per_seed["phi_eq_1"].append(np.mean([phi[s] == 1.0 for s in sources]))
        per_seed["mean_phi"].append(np.mean([phi[s] for s in sources]))
        ph = np.array([phi[s] for s in src])
        hop = np.full(len(src), -1)
        ok = ph > 0
        hop[ok] = np.rint(np.log(w[ok] / ph[ok]) / np.log(ALPHA)).astype(int)
        same = y[dst] == y[src]
        per_seed["pairs"].append(len(src))
        per_seed["pair_agree"].append(same.mean())
        per_seed["dst_test_share"].append(raw.test_mask.numpy()[dst].mean())
        for h in range(1, H + 1):
            sel = hop == h
            per_seed[f"pair_agree_h{h}"].append(same[sel].mean() if sel.any() else np.nan)
        for h in ["all"] + list(range(1, H + 1)):
            sel = np.ones(len(src), bool) if h == "all" else hop == h
            cnt, labs = defaultdict(int), defaultdict(set)
            for s, d in zip(src[sel], dst[sel]):
                cnt[d] += 1
                labs[d].add(int(y[s]))
            mult = np.array(list(cnt.values())) if cnt else np.array([0])
            per_seed[f"unique_dst_{h}"].append(len(cnt))
            per_seed[f"mean_mult_{h}"].append(mult.mean())
            per_seed[f"max_mult_{h}"].append(mult.max())
            per_seed[f"single_src_{h}"].append((mult == 1).mean() if cnt else 0)
            per_seed[f"conflict_share_{h}"].append(np.mean([len(v) > 1 for v in labs.values()]) if labs else 0)
    for k, v in per_seed.items():
        info[k] = round(float(np.nanmean(v)), 3)
        if k.startswith("max_mult"):
            info[k] = int(np.max(v))
    rows.append(info)

df = pd.DataFrame(rows)
out = HERE / "results" / OUT_NAME
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
with pd.option_context("display.max_columns", None, "display.width", 250):
    print(df.set_index("dataset").T.to_string())
print(f"\n-> {out}")
