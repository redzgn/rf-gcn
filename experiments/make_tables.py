#!/usr/bin/env python3
"""Build the result tables of the paper from results/runs.csv, label_prop.csv,
diagnostics.csv and diagnostics_selected.csv.
Output: results/tables.md and the LaTeX tables in results/latex/
(table*.tex for the manuscript, S*.tex for the supplement)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
runs = pd.read_csv(RES / "runs.csv")
lp = pd.read_csv(RES / "label_prop.csv")
diag = pd.read_csv(RES / "diagnostics.csv").set_index("dataset")
diag_sel = pd.read_csv(RES / "diagnostics_selected.csv").set_index("dataset")
DATASETS = ["Actor", "Minesweeper", "PubMed", "Tolokers"]
METRICS = [("test_acc", "Acc"), ("test_bal_acc", "Bal. acc"),
           ("test_macro_f1", "Macro-F1"), ("test_auc", "ROC-AUC")]
DEFAULT = dict(backbone="gcn", layers=2, lpc=5, min_hop=1, max_hop=3,
               alpha=0.7, P=16, excl_test=False)
out = []


def pick(lam, **kw):
    c = {**DEFAULT, **kw, "lam": lam}
    m = np.ones(len(runs), bool)
    for k, v in c.items():
        m &= np.isclose(runs[k], v) if isinstance(v, float) else (runs[k] == v)
    d = runs[m]
    assert d.groupby("dataset").seed.nunique().eq(10).all(), (c, len(d))
    return d.set_index(["dataset", "seed"]).sort_index()


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a, b, metric, ds):
    x, y = a.loc[ds, metric].values * 100, b.loc[ds, metric].values * 100
    d = y - x
    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.wilcoxon(y, x).pvalue if np.any(d != 0) else 1.0
    return d.mean(), ci, p, int((d > 0).sum())


def majority_f1(ds):
    """Macro-F1 of the majority-class predictor: F1 = 2p/(1+p) on that class, 0 on the others."""
    p, C = diag.loc[ds].majority_test_acc / 100, diag.loc[ds].all_counts.count("/") + 1
    return 100 * 2 * p / (1 + p) / C


def ms(d, metric, ds):
    v = d.loc[ds, metric].values * 100
    return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"


def delta_row(label, base, var, metrics=("test_acc", "test_bal_acc", "test_auc")):
    cells = []
    for ds in DATASETS:
        for m in metrics:
            dm, ci, p, _ = paired(base, var, m, ds)
            star = "*" if p < 0.05 else ""
            cells.append(f"{dm:+.1f}{star}")
    return f"| {label} | " + " | ".join(cells) + " |"


def delta_header(first):
    h = f"| {first} | " + " | ".join(f"{ds[:5]} {m}" for ds in DATASETS for m in ("Acc", "Bal", "AUC")) + " |"
    return h + "\n|" + "---|" * (1 + 3 * len(DATASETS))


# ---------------------------------------------------------------- Table A
out.append("## A. Class distribution per split (E2, R1.4)\n")
out.append("| Dataset | Classes | All | Train split | Val | Test | Majority-class test acc. |")
out.append("|---|---|---|---|---|---|---|")
for ds in DATASETS:
    r = diag.loc[ds]
    out.append(f"| {ds} | {r.all_counts.count('/') + 1} | {r.all_props} | {r.train_props} | "
               f"{r.val_props} | {r.test_props} | {r.majority_test_acc:.2f} |")
out.append("\nSplit sizes are 80/10/10 of the nodes (one fixed random split shipped in `data.pt`).\n")

# ---------------------------------------------------------------- Table B
out.append("## B. Validation-only selection of λ (E1, R1.1)\n")
lams = [0.5, 1.0, 2.0]
val = {l: pick(l).groupby("dataset").val_acc.mean() * 100 for l in [0.0] + lams}
out.append("| λ | " + " | ".join(DATASETS) + " | Mean over datasets |")
out.append("|---|" + "---|" * (len(DATASETS) + 1))
for l in [0.0] + lams:
    v = val[l]
    out.append(f"| {l} | " + " | ".join(f"{v[ds]:.2f}" for ds in DATASETS) + f" | {v.mean():.2f} |")
lam_star = max(lams, key=lambda l: val[l].mean())
out.append(f"\nMean **validation** accuracy (10 seeds). A single λ for all datasets is chosen by the "
           f"highest mean validation accuracy: **λ = {lam_star}**. Test data play no part.\n")

out.append("### B2. Validation-only selection of H, α and P (E6), one at a time, λ = 2.0\n")
out.append("| Parameter | Value | " + " | ".join(DATASETS) + " | Mean |")
out.append("|---|---|" + "---|" * (len(DATASETS) + 1))
chosen = {}
for par, vals in (("max_hop", (1, 2, 3, 4)), ("alpha", (0.5, 0.7, 0.9, 1.0)), ("P", (4, 8, 16, 32))):
    means = {}
    for v in vals:
        vv = pick(lam_star, **{par: v}).groupby("dataset").val_acc.mean() * 100
        means[v] = vv.mean()
        out.append(f"| {'H' if par == 'max_hop' else par} | {v} | " + " | ".join(f"{vv[ds]:.2f}" for ds in DATASETS) + f" | {vv.mean():.2f} |")
    chosen[par] = max(means, key=means.get)
out.append(f"\nSelected on validation: H = {chosen['max_hop']}, α = {chosen['alpha']}, P = {chosen['P']} "
           "(preliminary setting: H = 3, α = 0.7, P = 16). The main comparison (Tables C, H, I) uses the selected setting.\n")
MAIN = dict(max_hop=chosen["max_hop"], alpha=chosen["alpha"], P=chosen["P"])

# ---------------------------------------------------------------- Table C
gcn, rf = pick(0.0), pick(lam_star, **MAIN)
out.append(f"## C. Main results, 5 labels/class, 10 seeds (E2, E4, R1.4, R2.2, R2.6)\n")
out.append("| Dataset | Method | " + " | ".join(n for _, n in METRICS) + " |")
out.append("|---|---|" + "---|" * len(METRICS))
for ds in DATASETS:
    out.append(f"| {ds} | Majority class | {diag.loc[ds].majority_test_acc:.2f} | "
               f"{100 / (diag.loc[ds].all_counts.count('/') + 1):.2f} | {majority_f1(ds):.2f} | 50.00 |")
    l = lp[lp.dataset == ds]
    out.append(f"| | Label propagation | " + " | ".join(
        f"{l[m].mean() * 100:.2f} ± {l[m].std() * 100:.2f}" for m, _ in METRICS) + " |")
    out.append(f"| | GCN | " + " | ".join(ms(gcn, m, ds) for m, _ in METRICS) + " |")
    out.append(f"| | RF-GCN | " + " | ".join(ms(rf, m, ds) for m, _ in METRICS) + " |")
out.append("\nMean ± SD over seeds 1–10. Majority-class balanced accuracy is 1/C. "
           "Multi-class ROC-AUC is macro one-vs-rest.\n")

out.append("### C2. Paired difference RF-GCN − GCN (percentage points)\n")
out.append("| Dataset | " + " | ".join(n for _, n in METRICS) + " | RF-GCN wins (acc) |")
out.append("|---|" + "---|" * (len(METRICS) + 1))
pvals = {m: holm([paired(gcn, rf, m, ds)[2] for ds in DATASETS]) for m, _ in METRICS}
for i, ds in enumerate(DATASETS):
    cells = []
    for m, _ in METRICS:
        dm, ci, p, w = paired(gcn, rf, m, ds)
        cells.append(f"{dm:+.2f} [{dm - ci:+.2f}, {dm + ci:+.2f}], p={pvals[m][i]:.3f}")
    out.append(f"| {ds} | " + " | ".join(cells) + f" | {paired(gcn, rf, 'test_acc', ds)[3]}/10 |")
out.append("\nMean paired difference with 95% t-interval; p = Wilcoxon signed-rank, Holm-adjusted "
           "across the four datasets for each metric.\n")
out.append("Share of test nodes predicted as the minority class (binary datasets): " + "; ".join(
    f"{ds} GCN {gcn.loc[ds, 'test_pred_minority'].mean() * 100:.1f}% vs RF-GCN "
    f"{rf.loc[ds, 'test_pred_minority'].mean() * 100:.1f}% (true {100 * float(diag.loc[ds].test_props.split('/')[1]):.1f}%)"
    for ds in ("Minesweeper", "Tolokers")) + ".\n")

# ---------------------------------------------------------------- Table D
out.append(f"## D. Hop sets (R1.3), λ={lam_star}; change vs GCN in pp (* p<0.05, unadjusted)\n")
out.append("Tables D–G vary one factor at a time from the preliminary setting (H = 3, α = 0.7, P = 16).\n")
rf_pre = pick(lam_star)
out.append(delta_header("Hops"))
for a, b in ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3)):
    lab = "{" + ",".join(map(str, range(a, b + 1))) + "}"
    out.append(delta_row(lab, gcn, pick(lam_star, min_hop=a, max_hop=b)))
out.append("")

# ---------------------------------------------------------------- Table E
out.append("## E. Depth L vs reach H (R1.3c, R2.4); change vs GCN of the same depth\n")
out.append(delta_header("L, H"))
for L in (1, 2, 3):
    base = pick(0.0, layers=L)
    for Hh in (1, 2, 3, 4):
        tag = "H≤L" if Hh <= L else "H>L"
        out.append(delta_row(f"L={L}, H={Hh} ({tag})", base, pick(lam_star, layers=L, max_hop=Hh)))
out.append("\nGCN accuracy by depth: " + "; ".join(
    f"L={L}: " + ", ".join(f"{ds[:5]} {pick(0.0, layers=L).loc[ds, 'test_acc'].mean() * 100:.1f}"
                           for ds in DATASETS) for L in (1, 2, 3)) + "\n")

# ---------------------------------------------------------------- Table F
out.append("## F. Hop decay α and pairs per hop P (R1.5, R2.4); change vs GCN\n")
out.append(delta_header("Setting"))
for a in (0.5, 0.7, 0.9, 1.0):
    out.append(delta_row(f"α={a}" + (" (uniform)" if a == 1.0 else ""), gcn, pick(lam_star, alpha=a)))
for p in (4, 8, 16, 32):
    out.append(delta_row(f"P={p}", gcn, pick(lam_star, P=p)))
out.append("")

# ---------------------------------------------------------------- Table G
out.append("## G. Destinations restricted to non-test nodes (R2.5)\n")
out.append(delta_header("Variant"))
ex = pick(lam_star, excl_test=True)
out.append(delta_row("RF-GCN, all nodes eligible", gcn, rf_pre))
out.append(delta_row("RF-GCN, test nodes excluded", gcn, ex))
out.append(delta_row("excluded − all eligible", rf_pre, ex))
out.append("")

# ---------------------------------------------------------------- Table H
out.append("## H. Label budget (R1.6): test accuracy / balanced accuracy / ROC-AUC\n")
out.append("| Dataset | Labels/class | GCN acc | RF-GCN acc | Δ acc | Δ bal. acc | Δ AUC |")
out.append("|---|---|---|---|---|---|---|")
for ds in DATASETS:
    for k in (3, 5, 10, 20):
        g, r = pick(0.0, lpc=k), pick(lam_star, lpc=k, **MAIN)
        d = [paired(g, r, m, ds) for m in ("test_acc", "test_bal_acc", "test_auc")]
        out.append(f"| {ds if k == 3 else ''} | {k} | {ms(g, 'test_acc', ds)} | {ms(r, 'test_acc', ds)} | " +
                   " | ".join(f"{x[0]:+.2f}{'*' if x[2] < 0.05 else ''}" for x in d) + " |")
out.append("")

# ---------------------------------------------------------------- Table I
out.append("## I. Second backbone: APPNP (E5, R2.1)\n")
ag, ar = pick(0.0, backbone="appnp"), pick(lam_star, backbone="appnp", **MAIN)
out.append("| Dataset | " + " | ".join(f"APPNP {n}" for _, n in METRICS[:1] + METRICS[3:]) +
           " | " + " | ".join(f"+RF {n}" for _, n in METRICS[:1] + METRICS[3:]) + " | Δ acc | Δ bal. acc | Δ AUC |")
out.append("|---|" + "---|" * 7)
for ds in DATASETS:
    d = [paired(ag, ar, m, ds) for m in ("test_acc", "test_bal_acc", "test_auc")]
    out.append(f"| {ds} | {ms(ag, 'test_acc', ds)} | {ms(ag, 'test_auc', ds)} | {ms(ar, 'test_acc', ds)} | "
               f"{ms(ar, 'test_auc', ds)} | " + " | ".join(f"{x[0]:+.2f}{'*' if x[2] < 0.05 else ''}" for x in d) + " |")
out.append("")

# ---------------------------------------------------------------- Table J
out.append("## J. RF pair diagnostics, 5 labels/class, 10 seeds (R2.3, R1.5, R1.5b, R2.5)\n")
keys = [("chance_agreement", "Chance same-class agreement Σp²"),
        ("graph_agree_h1", "Graph agreement, h=1"), ("graph_agree_h2", "Graph agreement, h=2"),
        ("graph_agree_h3", "Graph agreement, h=3"),
        ("pair_agree_h1", "RF-pair agreement, h=1"), ("pair_agree_h2", "RF-pair agreement, h=2"),
        ("pair_agree_h3", "RF-pair agreement, h=3"),
        ("src_with_labeled_nbr", "Sources with ≥1 labeled 1-hop neighbour"),
        ("phi_eq_1", "Sources with φ(s)=1"),
        ("pairs", "RF pairs"), ("unique_dst_all", "Unique destinations"),
        ("mean_mult_all", "Mean sources per destination"), ("max_mult_all", "Max sources per destination"),
        ("single_src_all", "Destinations with a single source"),
        ("conflict_share_all", "Destinations with conflicting targets"),
        ("conflict_share_1", "  … at h=1"), ("conflict_share_2", "  … at h=2"), ("conflict_share_3", "  … at h=3"),
        ("dst_test_share", "Pairs whose destination is a test node")]
out.append("| Quantity | " + " | ".join(DATASETS) + " |")
out.append("|---|" + "---|" * len(DATASETS))
for k, lab in keys:
    out.append(f"| {lab} | " + " | ".join(f"{diag.loc[ds, k]:g}" for ds in DATASETS) + " |")
out.append("\nRF pairs under the validation-selected configuration (H = 4, α = 0.5, P = 8):\n")
out.append("| Quantity | " + " | ".join(DATASETS) + " |")
out.append("|---|" + "---|" * len(DATASETS))
sel_keys = [(k, lab) for k, lab in keys if k.startswith(("pair", "pairs", "unique", "mean_mult", "max_mult",
                                                         "single", "conflict", "dst_test"))]
sel_keys[3:3] = [("pair_agree_h4", "RF-pair agreement, h=4")]
sel_keys.insert(-1, ("conflict_share_4", "  … at h=4"))
for k, lab in sel_keys:
    out.append(f"| {lab} | " + " | ".join(f"{diag_sel.loc[ds, k]:g}" for ds in DATASETS) + " |")
out.append("\nGraph agreement: fraction of nodes at exact distance h from 1,000 random nodes that share "
           "their class. RF-pair agreement: fraction of sampled RF pairs whose destination truly has "
           "the source's label (sources are class-balanced, so chance is 1/C).\n")
out.append(f"Edge convention (R1.8): Actor is stored directed (reciprocity {diag.loc['Actor', 'reciprocity']}); "
           f"{diag.loc['Actor', 'undirected_edges']} undirected edges, average degree {diag.loc['Actor', 'avg_degree']}.\n")

(RES / "tables.md").write_text("\n".join(out))
print("\n".join(out))


# ====================================================================== LaTeX
# The same numbers typeset for the paper (table*.tex) and the supplement (S*.tex).
# The tables use \cmid{a-b}; define it as \cmidrule(lr){a-b} in the preamble.
TEX = RES / "latex"
TEX.mkdir(exist_ok=True)
LAM = r"\lambda_{\mathrm{RF}}"
SEL_TXT = rf"${LAM}={lam_star}$, $H={MAIN['max_hop']}$, $\alpha={MAIN['alpha']}$, $P={MAIN['P']}$"
PRE_TXT = rf"${LAM}={lam_star}$, $H=3$, $\alpha=0.7$, $P=16$"
DMETRICS = ("test_acc", "test_bal_acc", "test_auc")
TEXNAME = {"test_acc": "Acc", "test_bal_acc": r"Bal.\ acc", "test_macro_f1": "Macro-F1", "test_auc": "ROC-AUC"}
STAR = r"$^{*}\,p<0.05$, two-sided Wilcoxon signed-rank test, unadjusted."
NCLS = {ds: diag.loc[ds].all_counts.count("/") + 1 for ds in DATASETS}


def sgn(x, star=False, nd=1):
    return f"${x:+.{nd}f}" + ("^{*}" if star else "") + "$"


def tpm(d, metric, ds):
    v = d.loc[ds, metric].values * 100
    return rf"${v.mean():.2f} \pm {v.std(ddof=1):.2f}$"


def num(n):
    return f"{int(n):,}".replace(",", "{,}")


def write_table(name, caption, label, body, size=r"\small", colsep=None):
    lines = ["% Generated by experiments/make_tables.py; do not edit by hand.",
             r"\begin{table}[htbp]", r"\centering", size]
    if colsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{colsep}}}")
    lines += [rf"\caption{{{caption}}}", rf"\label{{{label}}}"] + body + [r"\end{table}"]
    (TEX / f"{name}.tex").write_text("\n".join(lines) + "\n")


def delta_tabular(rows, first):
    """rows: ("delta", label, base, var) | ("abs", label, frame) | ("group", text) | ("rule",)"""
    n = len(DATASETS)
    body = [rf"\begin{{tabular}}{{l{'rrr' * n}}}", r"\toprule",
            " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{ds}}}" for ds in DATASETS) + r" \\",
            "".join(rf"\cmid{{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
            f"{first} & " + " & ".join(["Acc", "Bal", "AUC"] * n) + r" \\", r"\midrule"]
    for r in rows:
        if r[0] == "rule":
            body.append(r"\midrule")
        elif r[0] == "group":
            body.append(rf"\multicolumn{{{1 + 3 * n}}}{{l}}{{{r[1]}}} \\")
        elif r[0] == "abs":
            body.append(r[1] + " & " + " & ".join(
                f"{r[2].loc[ds, m].mean() * 100:.1f}" for ds in DATASETS for m in DMETRICS) + r" \\")
        else:
            cells = []
            for ds in DATASETS:
                for m in DMETRICS:
                    dm, _, p, _ = paired(r[2], r[3], m, ds)
                    cells.append(sgn(dm, p < 0.05))
            body.append(r[1] + " & " + " & ".join(cells) + r" \\")
    return body + [r"\bottomrule", r"\end{tabular}"]


# ---------------------------------------------------------------- Table 1 (A)
body = [r"\textbf{(a)} Graph statistics\par\smallskip",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Dataset & Nodes & Edges & Features & Classes & Avg.\ degree \\", r"\midrule"]
for ds in DATASETS:
    r = diag.loc[ds]
    body.append(f"{ds} & {num(r.nodes)} & {num(r.undirected_edges)} & {num(r.features)} & "
                f"{NCLS[ds]} & {r.avg_degree:.2f} \\\\")
SPLITS = ("all", "train", "val", "test")
body += [r"\bottomrule", r"\end{tabular}", r"\par\bigskip",
         r"\textbf{(b)} Nodes per class and split (\%)\par\smallskip",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Dataset & Class & All & Train & Validation & Test \\", r"\midrule"]
for i, ds in enumerate(DATASETS):
    if i:
        body.append(r"\addlinespace")
    counts = {sp: [int(x) for x in diag.loc[ds, f"{sp}_counts"].split("/")] for sp in SPLITS}
    top = int(np.argmax(counts["test"]))
    for c in range(NCLS[ds]):
        cells = [f"{num(counts[sp][c])} ({100 * counts[sp][c] / sum(counts[sp]):.1f})" for sp in SPLITS]
        if c == top:
            cells[-1] = rf"\textbf{{{cells[-1]}}}"
        body.append(f"{ds if c == 0 else ''} & {c} & " + " & ".join(cells) + r" \\")
    body.append(r" & Total & " + " & ".join(num(sum(counts[sp])) for sp in SPLITS) + r" \\")
body += [r"\bottomrule", r"\end{tabular}"]
write_table("table1_datasets",
            r"Datasets. (a) Graph statistics. Edges are undirected with self-loops removed, and the average "
            r"degree is $2|E|/|V|$. Actor is distributed as a directed graph "
            rf"({num(diag.loc['Actor', 'stored_edges'])} stored edges, reciprocity "
            rf"{diag.loc['Actor', 'reciprocity']:.2f}); GCN propagates over the stored edges, while RF pair "
            r"construction uses the undirected graph. (b) Number of nodes per class, with the percentage in "
            r"parentheses, in the whole graph and in each part of the fixed random 80/10/10 split used in all "
            r"experiments. The labeled set (5 nodes per class) is drawn from the training part and is "
            r"class-balanced. Bold: the most frequent test class, whose share equals the test accuracy of a "
            r"classifier that always predicts it.",
            "tab:datasets-structure", body)

# ---------------------------------------------------------------- Table 4 (C + I)
appnp, appnp_rf = pick(0.0, backbone="appnp"), pick(lam_star, backbone="appnp", **MAIN)
body = [r"\begin{tabular}{llcccc}", r"\toprule",
        "Dataset & Method & " + " & ".join(TEXNAME[m] for m, _ in METRICS) + r" \\", r"\midrule"]
for i, ds in enumerate(DATASETS):
    if i:
        body.append(r"\midrule")
    body.append(f"{ds} & Majority class & ${diag.loc[ds].majority_test_acc:.2f}$ & ${100 / NCLS[ds]:.2f}$ & "
                f"${majority_f1(ds):.2f}$ & $50.00$ \\\\")
    l = lp[lp.dataset == ds]
    body.append(" & Label propagation & " + " & ".join(
        rf"${l[m].mean() * 100:.2f} \pm {l[m].std() * 100:.2f}$" for m, _ in METRICS) + r" \\")
    for lab, d in (("GCN", gcn), ("RF-GCN", rf), ("APPNP", appnp), ("APPNP+RF", appnp_rf)):
        body.append(f" & {lab} & " + " & ".join(tpm(d, m, ds) for m, _ in METRICS) + r" \\")
body += [r"\bottomrule", r"\end{tabular}"]
minority = "; ".join(
    f"{ds}: GCN {gcn.loc[ds, 'test_pred_minority'].mean() * 100:.1f}\\%, RF-GCN "
    f"{rf.loc[ds, 'test_pred_minority'].mean() * 100:.1f}\\% "
    f"(true share {100 * float(diag.loc[ds].test_props.split('/')[1]):.1f}\\%)" for ds in ("Minesweeper", "Tolokers"))
write_table("table4_main",
            r"Test performance (\%) at 5 labels per class, mean $\pm$ standard deviation over 10 seeds. "
            rf"RF-GCN and APPNP+RF use the validation-selected configuration ({SEL_TXT}). The majority-class "
            r"predictor always outputs the most frequent test class, so its balanced accuracy is $100/C$ and "
            r"its ROC-AUC is 50. Label propagation uses the propagation weight with the highest validation "
            r"accuracy in $\{0.5, 0.7, 0.9, 0.99\}$. APPNP uses $K=10$ propagation steps and teleport "
            r"probability 0.1. For multi-class datasets, ROC-AUC is the macro average of one-vs-rest AUCs. "
            rf"Share of test nodes predicted as the minority class: {minority}.",
            "tab:main", body)

# ---------------------------------------------------------------- Table 5 (C2)
def paired_panel(title, base, var):
    rows = [rf"\multicolumn{{6}}{{l}}{{\textit{{{title}}}}} \\"]
    p_holm = {m: holm([paired(base, var, m, ds)[2] for ds in DATASETS]) for m, _ in METRICS}
    for i, ds in enumerate(DATASETS):
        for j, (m, _) in enumerate(METRICS):
            dm, ci, _, wins = paired(base, var, m, ds)
            p = p_holm[m][i]
            ptxt = f"{p:.3f}" if p >= 0.05 else rf"\textbf{{{p:.3f}}}"
            rows.append(f"{ds if j == 0 else ''} & {TEXNAME[m]} & {sgn(dm, nd=2)} & "
                        f"$[{dm - ci:+.2f}, {dm + ci:+.2f}]$ & {ptxt} & {wins}/10 \\\\")
    return rows


body = ([r"\begin{tabular}{llrcrr}", r"\toprule",
         r"Dataset & Metric & $\Delta$ (pp) & 95\% CI & $p_{\mathrm{Holm}}$ & Wins \\", r"\midrule"]
        + paired_panel(r"(a) RF-GCN $-$ GCN", gcn, rf) + [r"\midrule"]
        + paired_panel(r"(b) APPNP+RF $-$ APPNP", appnp, appnp_rf) + [r"\bottomrule", r"\end{tabular}"])
write_table("table5_paired",
            r"Paired differences between each RF variant and its backbone (percentage points) over seeds "
            r"1--10; for a given seed, both methods use the same labeled nodes and initialization. Both RF "
            rf"variants use the validation-selected configuration ({SEL_TXT}). CI: 95\% $t$-interval of the "
            r"mean difference. $p_{\mathrm{Holm}}$: two-sided Wilcoxon signed-rank test, Holm-adjusted across "
            r"the four datasets for each metric; bold if below 0.05. Wins: seeds on which the RF variant scores "
            r"higher.",
            "tab:paired", body)

# ---------------------------------------------------------------- Table 6 (E)
rows = []
for L in (1, 2, 3):
    base = pick(0.0, layers=L)
    rows += ([("rule",)] if L > 1 else []) + [("abs", rf"GCN, $L={L}$", base)]
    rows += [("delta", rf"$H={Hh}$" + (r"$^{\dagger}$" if Hh > L else ""), base,
              pick(lam_star, layers=L, max_hop=Hh)) for Hh in (1, 2, 3, 4)]
write_table("table6_depth",
            r"Depth $L$ versus supervision reach $H$ at 5 labels per class, 10 seeds. For each GCN depth $L$, "
            r"the GCN row gives absolute test performance (\%), and the $H$ rows give the mean paired change of "
            r"RF-GCN over the GCN of the same depth (pp) when RF destinations lie at hops $1,\dots,H$. "
            r"$^{\dagger}\,H>L$: destinations beyond the receptive field. Bal: balanced accuracy. "
            rf"{PRE_TXT}. {STAR}",
            "tab:depth", delta_tabular(rows, "Setting"), size=r"\footnotesize", colsep="3pt")

# ---------------------------------------------------------------- Table 7 (H)
body = [r"\begin{tabular}{lrccrrr}", r"\toprule",
        r"Dataset & Labels/class & GCN acc & RF-GCN acc & $\Delta$Acc & $\Delta$Bal & $\Delta$AUC \\",
        r"\midrule"]
for i, ds in enumerate(DATASETS):
    if i:
        body.append(r"\addlinespace")
    for k in (3, 5, 10, 20):
        g, r = pick(0.0, lpc=k), pick(lam_star, lpc=k, **MAIN)
        d = [paired(g, r, m, ds) for m in DMETRICS]
        body.append(f"{ds if k == 3 else ''} & {k} & {tpm(g, 'test_acc', ds)} & {tpm(r, 'test_acc', ds)} & " +
                    " & ".join(sgn(x[0], x[2] < 0.05, nd=2) for x in d) + r" \\")
body += [r"\bottomrule", r"\end{tabular}"]
write_table("table7_budget",
            r"Label budget. Test accuracy (\%) of GCN and RF-GCN, mean $\pm$ standard deviation over 10 seeds, "
            r"and the mean paired change of RF-GCN over GCN (pp) in accuracy, balanced accuracy and ROC-AUC. "
            rf"RF-GCN uses the validation-selected configuration ({SEL_TXT}). {STAR}",
            "tab:budget", body)

# ---------------------------------------------------------------- Table S1 (B, B2)
ncol = 2 + len(DATASETS) + 1
body = [rf"\begin{{tabular}}{{ll{'r' * (len(DATASETS) + 1)}}}", r"\toprule",
        "Parameter & Value & " + " & ".join(DATASETS) + r" & Mean \\", r"\midrule",
        rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{(a) ${LAM}$, with $H=3$, $\alpha=0.7$, $P=16$}}}} \\"]
for l in [0.0] + lams:
    v = val[l]
    mean = f"{v.mean():.2f}" if l != lam_star else rf"\textbf{{{v.mean():.2f}}}"
    body.append(f"${LAM}$ & {'0 (GCN)' if l == 0 else l} & " +
                " & ".join(f"{v[ds]:.2f}" for ds in DATASETS) + f" & {mean} \\\\")
body += [r"\midrule",
         rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{(b) One at a time from $H=3$, $\alpha=0.7$, $P=16$, with ${LAM}={lam_star}$}}}} \\"]
for par, sym, vals in (("max_hop", "$H$", (1, 2, 3, 4)), ("alpha", r"$\alpha$", (0.5, 0.7, 0.9, 1.0)),
                       ("P", "$P$", (4, 8, 16, 32))):
    for v in vals:
        vv = pick(lam_star, **{par: v}).groupby("dataset").val_acc.mean() * 100
        mean = f"{vv.mean():.2f}" if v != chosen[par] else rf"\textbf{{{vv.mean():.2f}}}"
        body.append(f"{sym} & {v} & " + " & ".join(f"{vv[ds]:.2f}" for ds in DATASETS) + f" & {mean} \\\\")
body += [r"\bottomrule", r"\end{tabular}"]
write_table("S1_selection",
            r"Validation-only selection of the RF-GCN hyperparameters: mean validation accuracy (\%) over 10 "
            r"seeds at 5 labels per class; test data are not used. (a) The RF weight, with the other settings at "
            r"their preliminary values. (b) $H$, $\alpha$ and $P$, each varied from the preliminary setting. "
            r"Each value is chosen by the highest mean over the four datasets (bold). The selected configuration "
            rf"({SEL_TXT}) is used for the main comparison.",
            "tab:S-selection", body)

# ---------------------------------------------------------------- Tables S2-S4 (D, F, G)
rows = [("delta", rf"$\{{{','.join(map(str, range(a, b + 1)))}\}}$", gcn, pick(lam_star, min_hop=a, max_hop=b))
        for a, b in ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3))]
write_table("S2_hops",
            r"Hop sets. Mean paired change of RF-GCN over GCN (pp) when RF destinations are drawn only from the "
            rf"listed hops; two-layer GCN, {PRE_TXT}, 10 seeds. Bal: balanced accuracy. {STAR}",
            "tab:S-hops", delta_tabular(rows, "Hops"), size=r"\footnotesize", colsep="4pt")

rows = ([("delta", rf"$\alpha={a}$" + (" (uniform)" if a == 1.0 else ""), gcn, pick(lam_star, alpha=a))
         for a in (0.5, 0.7, 0.9, 1.0)] + [("rule",)]
        + [("delta", f"$P={p}$", gcn, pick(lam_star, P=p)) for p in (4, 8, 16, 32)])
write_table("S3_alpha_P",
            r"Hop-decay factor $\alpha$ ($\alpha=1$: uniform weighting over hops) and pairs per hop $P$, each "
            r"varied from the preliminary setting ($H=3$, $\alpha=0.7$, $P=16$, "
            rf"${LAM}={lam_star}$). Mean paired change of RF-GCN over GCN (pp), 10 seeds. Bal: balanced "
            rf"accuracy. {STAR}",
            "tab:S-alpha-P", delta_tabular(rows, "Setting"), size=r"\footnotesize", colsep="4pt")

rows = [("delta", "All nodes eligible", gcn, rf_pre), ("delta", "Test nodes excluded", gcn, ex), ("rule",),
        ("delta", r"Excluded $-$ all eligible", rf_pre, ex)]
write_table("S4_exclude_test",
            r"Destinations restricted to non-test nodes. The first two rows give the mean paired change of "
            r"RF-GCN over GCN (pp); the last row compares the two RF-GCN variants directly. Paths to "
            r"destinations may pass through test nodes, whose labels are never used. "
            rf"{PRE_TXT}, 10 seeds. Bal: balanced accuracy. {STAR}",
            "tab:S-exclude-test", delta_tabular(rows, "Destinations"), size=r"\footnotesize", colsep="4pt")

# ---------------------------------------------------------------- Table S5 (J)
n = len(DATASETS)
span = lambda x: rf"\multicolumn{{2}}{{c}}{{{x}}}"
body = [rf"\begin{{tabular}}{{l{'rr' * n}}}", r"\toprule",
        "Quantity & " + " & ".join(span(ds) for ds in DATASETS) + r" \\",
        "".join(rf"\cmid{{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n)), r"\midrule",
        rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{\textit{{(a) Independent of the RF configuration}}}} \\",
        r"Chance same-class agreement $\sum_c p_c^2$ & " + " & ".join(
            span(f"{diag.loc[ds, 'chance_agreement']:.3f}") for ds in DATASETS) + r" \\"]
for h in (1, 2, 3, 4):
    src = diag if h <= 3 else diag_sel
    body.append(f"Same-class agreement in the graph, $h={h}$ & " + " & ".join(
        span(f"{src.loc[ds, f'graph_agree_h{h}']:.3f}") for ds in DATASETS) + r" \\")
body += [r"Sources with $\geq 1$ labeled 1-hop neighbour (\%) & " + " & ".join(
             span(f"{100 * diag.loc[ds, 'src_with_labeled_nbr']:.1f}") for ds in DATASETS) + r" \\",
         r"Sources with $\phi(s)=1$ (\%) & " + " & ".join(
             span(f"{100 * diag.loc[ds, 'phi_eq_1']:.1f}") for ds in DATASETS) + r" \\",
         r"\midrule",
         rf"\multicolumn{{{1 + 2 * n}}}{{l}}{{\textit{{(b) Sampled RF pairs}}}} \\",
         " & " + " & ".join(["Pre", "Sel"] * n) + r" \\"]


def pair_row(label, key, fmt, scale=1.0):
    cells = []
    for ds in DATASETS:
        for d in (diag, diag_sel):
            cells.append(format(d.loc[ds, key] * scale, fmt) if key in d.columns else "--")
    return f"{label} & " + " & ".join(cells) + r" \\"


body += [pair_row("RF pairs per seed", "pairs", ".1f"),
         r"Chance agreement $\approx 1/C$ & " + " & ".join(span(f"{1 / NCLS[ds]:.3f}") for ds in DATASETS) + r" \\",
         pair_row("RF-pair agreement, all hops", "pair_agree", ".3f")]
body += [pair_row(rf"\quad at $h={h}$", f"pair_agree_h{h}", ".3f") for h in (1, 2, 3, 4)]
for label, key, fmt, scale in (("Unique destinations", "unique_dst", ".1f", 1),
                               ("Mean sources per destination", "mean_mult", ".2f", 1),
                               ("Max sources per destination", "max_mult", ".0f", 1),
                               (r"Destinations with one source (\%)", "single_src", ".1f", 100),
                               (r"Destinations with conflicting targets (\%)", "conflict_share", ".1f", 100)):
    body.append(pair_row(label + ", all hops", f"{key}_all", fmt, scale))
    body += [pair_row(rf"\quad at $h={h}$", f"{key}_{h}", fmt, scale) for h in (1, 2, 3, 4)]
body += [pair_row(r"Pairs with a test-node destination (\%)", "dst_test_share", ".1f", 100),
         r"\bottomrule", r"\end{tabular}"]
write_table("S5_diagnostics",
            r"RF pair diagnostics at 5 labels per class, averaged over seeds 1--10 (maximum: over seeds). "
            r"(a) Quantities that do not depend on the RF configuration. Same-class agreement in the graph is "
            r"the fraction of nodes at exact distance $h$ from 1,000 random nodes that share their class; its "
            r"chance level is $\sum_c p_c^2$. (b) Sampled RF pairs under the preliminary configuration (Pre: "
            r"$H=3$, $\alpha=0.7$, $P=16$) and the validation-selected configuration (Sel: "
            rf"$H={MAIN['max_hop']}$, $\alpha={MAIN['alpha']}$, $P={MAIN['P']}$). RF-pair agreement is the "
            r"fraction of pairs whose destination has the source's label; labeled sources are class-balanced, "
            r"so its chance level is approximately $1/C$. Destination statistics are given over all hops and "
            r"per hop $h$. A destination has conflicting targets when it is paired with sources of different "
            r"labels.",
            "tab:S-diagnostics", body, size=r"\footnotesize", colsep="4pt")
print(f"\nLaTeX tables -> {TEX}")
