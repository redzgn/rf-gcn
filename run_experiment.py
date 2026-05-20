#!/usr/bin/env python3
"""
RF-GCN: Hop-Distance-Weighted Label Propagation

Usage:
  python run_experiment.py                    # paper setting: 5 datasets, 5 seeds
  python run_experiment.py --quick            # smoke test: 1 dataset, 1 seed, 100 epochs
  python run_experiment.py --sweep_labels     # label-rate sensitivity: 3/5/10/20 lpc
  python run_experiment.py --datasets PubMed Tolokers
  python run_experiment.py --seeds 1 2 3 4 5 6 7 8 9 10
"""

import sys
import os
from pathlib import Path

# Repo root = directory containing this script
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import argparse
import random
import time
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
from copy import deepcopy

from model import GCN
from data_utils import load_dataset, subsample_train_mask, dataset_stats
from rf_pairs import build_rf_pairs
from losses import rf_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits, y, mask):
    if mask is None or mask.sum() == 0:
        return None
    return float((logits.argmax(1)[mask] == y[mask]).float().mean().item())


def load_config():
    cfg_path = Path(__file__).parent / "src" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the repo root (parent of experiments_3/)
    for key in ("data_root", "output_dir"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = str(REPO_ROOT / cfg[key])
    return cfg


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_one(dataset_name, variant, seed, cfg, device, labels_per_class):
    set_seed(seed)

    data = load_dataset(
        dataset_name,
        data_root=cfg["data_root"],
        device=device,
    )
    data = subsample_train_mask(data, labels_per_class, seed=seed)

    stats = dataset_stats(dataset_name, data)
    num_classes = stats["num_classes"]

    tcfg = cfg["training"]
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=int(tcfg["hidden_dim"]),
        out_channels=num_classes,
        num_layers=int(tcfg["num_layers"]),
        dropout=float(tcfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    method = variant["method"]
    rcfg = cfg["rf"]

    pairs = None
    if method == "rf_gcn":
        pairs = build_rf_pairs(
            data=data,
            seed=seed,
            min_hop=int(rcfg["min_hop"]),
            max_hop=int(rcfg["max_hop"]),
            alpha=float(rcfg["alpha"]),
            pairs_per_source=int(rcfg["pairs_per_source"]),
            pair_graph=str(rcfg["pair_graph"]),
        )

    lambda_rf = float(variant.get("lambda_rf", 0.0))
    loss_type = str(variant.get("loss_type", "label_ce"))
    rampup = int(rcfg.get("rampup_epochs", 0))
    epochs = int(tcfg["epochs"])

    best_val_acc = -1.0
    best_epoch = -1
    best_test_acc = None
    best_state = None

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])

        rf_loss_val = 0.0
        if method == "rf_gcn" and pairs is not None and epoch > rampup:
            rl = rf_loss(logits, pairs, loss_type=loss_type)
            loss = loss + lambda_rf * rl
            rf_loss_val = float(rl.item())

        loss.backward()
        optimizer.step()

        # Validation checkpoint
        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, data.edge_index)

        val_acc = accuracy(eval_logits, data.y, data.val_mask)
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_test_acc = accuracy(eval_logits, data.y, data.test_mask)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    runtime = time.time() - t0

    # Restore best checkpoint for final eval
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    row = {
        **stats,
        "labels_per_class": labels_per_class,
        "method": variant["name"],
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_acc": round(best_val_acc, 6) if best_val_acc >= 0 else None,
        "best_test_acc": round(best_test_acc, 6) if best_test_acc is not None else None,
        "lambda_rf": lambda_rf,
        "loss_type": loss_type if method == "rf_gcn" else "",
        "rampup_epochs": rampup,
        "num_rf_pairs": pairs["num_pairs"] if pairs else 0,
        "runtime_sec": round(runtime, 2),
    }
    return row


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def build_summary(df):
    """Per-dataset × method mean ± std over seeds."""
    agg = (
        df.groupby(["dataset", "method"])
        .agg(
            mean_test=("best_test_acc", "mean"),
            std_test=("best_test_acc", "std"),
            mean_val=("best_val_acc", "mean"),
            mean_epoch=("best_epoch", "mean"),
            num_rf_pairs=("num_rf_pairs", "mean"),
            runtime_sec=("runtime_sec", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    agg["std_test"] = agg["std_test"].fillna(0.0)
    return agg


def print_results_table(df, best_rf_variant):
    """Print the main comparison table: VanillaGCN vs best RF-GCN variant."""
    summary = build_summary(df)

    vanilla = summary[summary["method"] == "VanillaGCN"][
        ["dataset", "mean_test", "std_test"]
    ].rename(columns={"mean_test": "vanilla_mean", "std_test": "vanilla_std"})

    rf = summary[summary["method"] == best_rf_variant][
        ["dataset", "mean_test", "std_test"]
    ].rename(columns={"mean_test": "rf_mean", "std_test": "rf_std"})

    comp = vanilla.merge(rf, on="dataset", how="outer")
    comp["delta"] = comp["rf_mean"] - comp["vanilla_mean"]

    col_w = 18
    hdr = (
        f"{'Dataset':<14} | "
        f"{'VanillaGCN':<{col_w}} | "
        f"{best_rf_variant:<{col_w}} | "
        f"{'Delta':>7}"
    )
    sep = "-" * len(hdr)

    print(f"\n{'=' * len(hdr)}")
    lpc = int(df["labels_per_class"].iloc[0]) if "labels_per_class" in df.columns else "?"
    print(f"RF-GCN vs VanillaGCN  |  {lpc} labels/class  |  seeds = {df['seed'].nunique()}")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    deltas = []
    for _, row in comp.sort_values("dataset").iterrows():
        v_str = f"{row['vanilla_mean']*100:.2f} ± {row['vanilla_std']*100:.2f}"
        r_str = f"{row['rf_mean']*100:.2f} ± {row['rf_std']*100:.2f}"
        d = row["delta"]
        d_str = f"{d*100:+.2f}"
        print(f"{row['dataset']:<14} | {v_str:<{col_w}} | {r_str:<{col_w}} | {d_str:>7}")
        deltas.append(d)

    print(sep)
    n = len(deltas)
    wins = sum(1 for d in deltas if d > 0.001)
    ties = sum(1 for d in deltas if abs(d) <= 0.001)
    losses = sum(1 for d in deltas if d < -0.001)
    mean_delta = sum(deltas) / n if n else 0.0
    print(f"{'Mean delta':<14}   {'':>{col_w}}   {'':>{col_w}}   {mean_delta*100:+.2f}")
    print(f"Win/Tie/Loss: {wins}/{ties}/{losses} out of {n} datasets")
    print("=" * len(hdr))


def print_all_variants_table(df):
    """Print a compact table showing all RF-GCN variants vs Vanilla."""
    summary = build_summary(df)
    methods = [m for m in df["method"].unique() if m != "VanillaGCN"]
    datasets = sorted(df["dataset"].unique())

    vanilla_acc = {
        row["dataset"]: row["mean_test"]
        for _, row in summary[summary["method"] == "VanillaGCN"].iterrows()
    }

    print("\n--- Delta vs VanillaGCN (percentage points) ---")
    header = f"{'Dataset':<14}" + "".join(f"  {m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for ds in datasets:
        row_str = f"{ds:<14}"
        v = vanilla_acc.get(ds)
        for m in methods:
            sub = summary[(summary["dataset"] == ds) & (summary["method"] == m)]
            if sub.empty or v is None:
                row_str += f"  {'N/A':>14}"
            else:
                delta = (sub["mean_test"].iloc[0] - v) * 100
                row_str += f"  {delta:>+13.2f}"
        print(row_str)
    print("-" * len(header))


def print_label_rate_table(df):
    """For label rate sweep runs."""
    summary = build_summary(df)
    rates = sorted(df["labels_per_class"].unique())
    datasets = sorted(df["dataset"].unique())
    methods = sorted(df["method"].unique())

    print("\n--- Label Rate Sensitivity (test acc %) ---")
    print(f"{'Dataset':<14} {'Method':<18}" + "".join(f"  {r:>4}lpc" for r in rates))
    print("-" * (32 + 9 * len(rates)))
    for ds in datasets:
        for m in methods:
            row_str = f"{ds:<14} {m:<18}"
            for r in rates:
                sub = df[(df["dataset"] == ds) & (df["method"] == m) & (df["labels_per_class"] == r)]
                if sub.empty:
                    row_str += f"  {'N/A':>6}"
                else:
                    acc = sub["best_test_acc"].mean() * 100
                    row_str += f"  {acc:>6.2f}"
            print(row_str)
    print("-" * (32 + 9 * len(rates)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RF-GCN experiments 3")
    p.add_argument("--config", default=None, help="Path to config.yaml (auto-detected by default)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--variants", nargs="*", default=None, help="Variant names to run")
    p.add_argument("--labels_per_class", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output_dir", default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Tolokers only, seed=1, 100 epochs — for smoke testing")
    mode.add_argument("--sweep_labels", action="store_true",
                      help="Sweep labels_per_class in [3, 5, 10, 20]")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    output_dir = args.output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Mode selection
    if args.quick:
        datasets = ["Tolokers"]
        seeds = [1]
        cfg["training"]["epochs"] = 100
        label_rates = [int(args.labels_per_class or cfg["labels_per_class"])]
    elif args.sweep_labels:
        datasets = args.datasets or cfg["datasets"]
        seeds = args.seeds or cfg["seeds"]
        label_rates = [3, 5, 10, 20]
    else:
        datasets = args.datasets or cfg["datasets"]
        seeds = args.seeds or cfg["seeds"]
        label_rates = [int(args.labels_per_class or cfg["labels_per_class"])]

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    # Variant selection
    all_variants = cfg["variants"]
    if args.variants:
        keep = set(args.variants)
        all_variants = [v for v in all_variants if v["name"] in keep]
        missing = keep - {v["name"] for v in all_variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    seeds = [int(s) for s in seeds]

    print(f"Device     : {device}")
    print(f"Datasets   : {datasets}")
    print(f"Seeds      : {seeds}")
    print(f"Label rates: {label_rates}")
    print(f"Variants   : {[v['name'] for v in all_variants]}")
    print(f"Epochs     : {cfg['training']['epochs']}")
    print(f"Output     : {output_dir}")
    print()

    raw_path = os.path.join(output_dir, "raw_results.csv")
    rows = []

    total = len(label_rates) * len(datasets) * len(all_variants) * len(seeds)
    done = 0

    for lpc in label_rates:
        for dataset_name in datasets:
            for variant in all_variants:
                for seed in seeds:
                    done += 1
                    tag = f"[{done}/{total}] {dataset_name} | {variant['name']} | seed={seed} | lpc={lpc}"
                    print(tag, end=" ... ", flush=True)
                    t0 = time.time()

                    try:
                        row = train_one(dataset_name, variant, seed, cfg, device, lpc)
                        rows.append(row)
                        elapsed = time.time() - t0
                        print(
                            f"test={row['best_test_acc']:.4f}  "
                            f"val={row['best_val_acc']:.4f}  "
                            f"epoch={row['best_epoch']}  "
                            f"pairs={row['num_rf_pairs']}  "
                            f"{elapsed:.1f}s"
                        )
                    except Exception as e:
                        print(f"FAILED: {e}")
                        rows.append({
                            "dataset": dataset_name,
                            "method": variant["name"],
                            "seed": seed,
                            "labels_per_class": lpc,
                            "failed": True,
                            "error": str(e),
                        })

                    pd.DataFrame(rows).to_csv(raw_path, index=False)

    df = pd.DataFrame(rows)
    if "failed" in df.columns:
        df = df[df["failed"].isna() | (df["failed"] == False)].copy()

    df.to_csv(raw_path, index=False)
    print(f"\nRaw results: {raw_path}")

    if df.empty:
        print("No successful runs.")
        return

    # Summaries
    summary_path = os.path.join(output_dir, "summary.csv")
    build_summary(df).to_csv(summary_path, index=False)

    if args.sweep_labels:
        print_label_rate_table(df)
    else:
        # Find the best RF-GCN variant by mean test acc across datasets
        rf_variants = [v["name"] for v in all_variants if v["method"] == "rf_gcn"]
        if rf_variants and "VanillaGCN" in df["method"].values:
            summary = build_summary(df)
            rf_means = {
                m: summary[summary["method"] == m]["mean_test"].mean()
                for m in rf_variants
                if m in summary["method"].values
            }
            best_rf = max(rf_means, key=rf_means.get) if rf_means else None

            print_all_variants_table(df)
            if best_rf:
                print_results_table(df, best_rf)

    print(f"\nSummary CSV : {summary_path}")


if __name__ == "__main__":
    main()
