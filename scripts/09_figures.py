#!/usr/bin/env python3
"""Step 9 - regenerate Figures 1-6 from the CSVs.

Figure 1  Macro-F1 (mean +/- SD, 3 seeds) for all model variants x 4 tasks
Figure 2  (left) D4 per-class precision/recall before and after IWN
          (right) surgical feature alignment rho_k vs IWN-induced F1 gain
Figure 3  Component ablation heatmaps: absolute macro-F1 and Delta vs baseline
Figure 4  (left) speed-accuracy Pareto frontier
          (right) vocabulary sensitivity
Figure 5  Training dynamics: initial vs final loss; loss reduction + best epoch
Figure 6  Leave-one-out F1 drop per indicator group (heatmap)

``--source run`` (default) plots this repository's results; ``--source paper``
plots the published values from :mod:`surgellm.paper_reference`.  Each figure
is watermarked with its source so the two can never be confused.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surgellm import paper_reference as PR
from surgellm.data.schema import TASK_ORDER, TASKS
from surgellm.utils.io import ensure_dir

TASK_LABEL = {t: f"{t} {TASKS[t].source}" for t in TASK_ORDER}
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def _load(tables: Path, name: str, fallback: Optional[List[Dict]] = None
          ) -> Optional[pd.DataFrame]:
    p = tables / name
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame(fallback) if fallback else None


def _stamp(fig, source: str) -> None:
    fig.text(0.995, 0.005,
             f"source: {source}", ha="right", va="bottom",
             fontsize=6, color="#888888")


def fig1(df: Optional[pd.DataFrame], out: Path, source: str) -> None:
    if df is None:
        return
    df = df[df["model"].notna()].copy()
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, tid in enumerate(TASK_ORDER):
        if tid not in df.columns:
            continue
        sd_col = f"{tid}_sd"
        sd = df[sd_col].values if sd_col in df.columns else None
        ax.bar(x + (i - 1.5) * width, df[tid].values, width,
               yerr=sd, capsize=2, label=TASK_LABEL[tid], color=PALETTE[i])
    best = df["avg_f1"].idxmax() if "avg_f1" in df.columns else None
    if best is not None:
        ax.annotate("* best avg", xy=(x[df.index.get_loc(best)], 1.005),
                    ha="center", fontsize=9, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.5, 1.03)
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.set_title("Figure 1: Macro-F1 (mean $\\pm$ SD) across model variants and tasks")
    ax.grid(axis="y", alpha=0.25)
    _stamp(fig, source)
    fig.tight_layout()
    fig.savefig(out / "figure1_main_results.png", dpi=160)
    plt.close(fig)


def fig2(t5: Optional[pd.DataFrame], t15: Optional[pd.DataFrame],
         out: Path, source: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    if t5 is not None and {"hum_p", "hum_r", "llm_p", "llm_r"} <= set(t5.columns):
        rows = t5[~t5["variant"].astype(str).str.startswith("Delta")]
        rows = rows[rows.get("backbone", "RoBERTa") == "RoBERTa"] \
            if "backbone" in rows.columns else rows
        labels = ["Hum P", "Hum R", "LLM P", "LLM R"]
        cols = ["hum_p", "hum_r", "llm_p", "llm_r"]
        xs = np.arange(len(labels))
        for j, (_, r) in enumerate(rows.iterrows()):
            vals = [r[c] for c in cols]
            ax.bar(xs + (j - 0.5) * 0.35, vals, 0.35,
                   label=str(r["variant"]), color=PALETTE[j])
        if len(rows) == 2:
            a, b = rows.iloc[0], rows.iloc[1]
            for k, c in enumerate(cols):
                if pd.notna(a[c]) and pd.notna(b[c]):
                    ax.annotate(f"{b[c] - a[c]:+.2f}", xy=(xs[k], max(a[c], b[c]) + .01),
                                ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
    ax.set_title("D4 per-class breakdown before/after IWN")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    if t15 is not None:
        tasks = t15["task"].tolist()
        xs = np.arange(len(tasks))
        ax.plot(xs, t15["rho_no_iwn"], "o--", label=r"$\rho_k$ (no IWN)",
                color=PALETTE[0])
        ax.plot(xs, t15["rho_iwn"], "s-", label=r"$\rho_k$ (IWN)", color=PALETTE[1])
        ax.set_xticks(xs)
        ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=8)
        ax.set_ylabel(r"Alignment $\rho_k$ (MC estimate)")
        ax2 = ax.twinx()
        if "f1_gain" in t15.columns:
            ax2.bar(xs, t15["f1_gain"], 0.35, alpha=0.3, color=PALETTE[2],
                    label=r"$\Delta$F1 (IWN)")
            ax2.set_ylabel(r"$\Delta$Macro-F1 (IWN)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title(r"Alignment $\rho_k$ vs IWN gain")
    ax.grid(alpha=0.25)

    _stamp(fig, source)
    fig.suptitle("Figure 2: the IWN mechanism", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figure2_iwn_mechanism.png", dpi=160)
    plt.close(fig)


def fig3(t7: Optional[pd.DataFrame], out: Path, source: str) -> None:
    if t7 is None:
        return
    rob = t7[t7["backbone"] == "RoBERTa"] if "backbone" in t7.columns else t7
    if rob.empty:
        return
    labels = ["Baseline", "+ Prefix (G)", "+ Gate (FULL)", "+ IWN (full)"][:len(rob)]
    tasks = [t for t in TASK_ORDER if t in rob.columns]
    absolute = rob[tasks].values.astype(float)
    delta = absolute - absolute[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, data, title, cmap, center in (
        (axes[0], absolute, "(a) Absolute Macro-F1", "YlGn", None),
        (axes[1], delta, r"(b) $\Delta$ vs baseline", "RdBu", 0.0),
    ):
        if center is not None:
            lim = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1e-6)
            im = ax.imshow(data, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        else:
            im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:+.3f}" if center is not None
                        else f"{data[i, j]:.3f}",
                        ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.035)
    _stamp(fig, source)
    fig.suptitle("Figure 3: component ablation (RoBERTa)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figure3_component_ablation.png", dpi=160)
    plt.close(fig)


def fig4(t12: Optional[pd.DataFrame], t9: Optional[pd.DataFrame],
         out: Path, source: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    if t12 is not None:
        tcol = "train_seconds" if "train_seconds" in t12.columns else "train_s"
        pcol = "pareto_efficient" if "pareto_efficient" in t12.columns else "pareto"
        for _, r in t12.iterrows():
            eff = bool(r.get(pcol, False))
            ax.scatter(r[tcol], r["avg_f1"], s=70 if eff else 35,
                       color=PALETTE[1] if eff else PALETTE[0],
                       edgecolor="k" if eff else "none", zorder=3,
                       marker="*" if eff else "o")
            ax.annotate(str(r["model"]), (r[tcol], r["avg_f1"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        front = t12[t12[pcol].astype(bool)].sort_values(tcol) if pcol in t12 else None
        if front is not None and len(front) > 1:
            ax.plot(front[tcol], front["avg_f1"], "--", color=PALETTE[1], alpha=0.6)
        ax.set_xlabel("Training time (s)")
        ax.set_ylabel("Avg Macro-F1")
    ax.set_title("(a) Speed-accuracy trade-off")
    ax.grid(alpha=0.25)

    ax = axes[1]
    if t9 is not None:
        rows = t9[~t9["vocab"].astype(str).str.startswith(("Delta", "Auto /"))]
        xs = np.arange(len(rows))
        ax.bar(xs - 0.2, rows["avg"], 0.4, label="Avg F1", color=PALETTE[0])
        if "D4" in rows.columns:
            ax.bar(xs + 0.2, rows["D4"], 0.4, label="D4 F1", color=PALETTE[2])
        ax.set_xticks(xs)
        ax.set_xticklabels(rows["vocab"], rotation=15, ha="right", fontsize=8)
        cur = rows[rows["vocab"] == "Curated"]["avg"]
        rnd = rows[rows["vocab"] == "Random"]["avg"]
        if len(cur) and len(rnd):
            ax.annotate(f"{float(rnd.iloc[0]) - float(cur.iloc[0]):+.3f}",
                        xy=(1, float(rnd.iloc[0])), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=PALETTE[2])
        ax.set_ylabel("Macro-F1")
        ax.legend(fontsize=8)
    ax.set_title("(b) Vocabulary sensitivity")
    ax.grid(axis="y", alpha=0.25)

    _stamp(fig, source)
    fig.suptitle("Figure 4: efficiency and vocabulary sensitivity", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figure4_efficiency_vocab.png", dpi=160)
    plt.close(fig)


def fig5(t6: Optional[pd.DataFrame], out: Path, source: str) -> None:
    if t6 is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.scatter(t6["init_loss"], t6["final_loss"], c=range(len(t6)),
               cmap="viridis", s=50, zorder=3)
    for _, r in t6.iterrows():
        ax.annotate(str(r["model"]), (r["init_loss"], r["final_loss"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Initial loss (epoch 1)")
    ax.set_ylabel("Final loss (best checkpoint)")
    ax.set_title("(a) Initial vs final loss")
    ax.grid(alpha=0.25)

    ax = axes[1]
    xs = np.arange(len(t6))
    ax.bar(xs, t6["delta_loss"], color=PALETTE[0], alpha=0.8)
    ax.set_ylabel(r"Loss reduction ($\Delta$)")
    ax.set_xticks(xs)
    ax.set_xticklabels(t6["model"], rotation=40, ha="right", fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(xs, t6["best_epoch"], "kD--", ms=5, label="best epoch")
    ax2.set_ylabel("Best epoch")
    ax2.legend(fontsize=8)
    ax.set_title("(b) Loss reduction and convergence epoch")
    _stamp(fig, source)
    fig.suptitle("Figure 5: training dynamics (seed 0)", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figure5_training_dynamics.png", dpi=160)
    plt.close(fig)


def fig6(t11: Optional[pd.DataFrame], out: Path, source: str) -> None:
    if t11 is None:
        return
    gcol = "group_removed" if "group_removed" in t11.columns else "group"
    tasks = [t for t in TASK_ORDER if t in t11.columns]
    data = t11[tasks].values.astype(float).T
    fig, ax = plt.subplots(figsize=(11, 3.6))
    lim = float(np.nanmin(data))
    im = ax.imshow(data, cmap="Reds_r", vmin=lim, vmax=0.0, aspect="auto")
    ax.set_xticks(range(len(t11)))
    ax.set_xticklabels(t11[gcol], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABEL[t] for t in tasks], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, label=r"$\Delta$Macro-F1 (drop)")
    ax.set_title("Figure 6: leave-one-out F1 drop per surgical indicator group")
    _stamp(fig, source)
    fig.tight_layout()
    fig.savefig(out / "figure6_leave_one_out.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="run", choices=["run", "paper"])
    ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tables = Path(args.tables)
    out = ensure_dir(args.out or (ROOT / "results" / "figures" /
                                  ("paper" if args.source == "paper" else "run")))

    if args.source == "paper":
        t3 = pd.DataFrame(PR.as_rows(PR.TABLE3_MAIN))
        t5 = pd.DataFrame(PR.as_rows(PR.TABLE5_IWN_ABLATION))
        t6 = pd.DataFrame(PR.as_rows(PR.TABLE6_TRAINING_DYNAMICS))
        t7 = pd.DataFrame(PR.as_rows(PR.TABLE7_COMPONENTS))
        t9 = pd.DataFrame(PR.as_rows(PR.TABLE9_VOCAB_CONTROL))
        t11 = pd.DataFrame(PR.as_rows(PR.TABLE11_LEAVE_ONE_OUT))
        t12 = pd.DataFrame(PR.as_rows(PR.TABLE12_EFFICIENCY))
        t15 = pd.DataFrame(PR.as_rows(PR.TABLE15_RHO))
        t15["f1_gain"] = [0.001, 0.004, 0.018, 0.181]       # Table 5 deltas
        label = "paper (TrustNLP 2026, pp. 600-617)"
    else:
        t3 = _load(tables, "table3_main_results.csv")
        t5 = _load(tables, "table5_iwn_ablation.csv")
        t6 = _load(tables, "table6_training_dynamics.csv")
        t7 = _load(tables, "table7_component_ablation.csv")
        t9 = _load(tables, "table9_vocab_control.csv")
        t11 = _load(tables, "table11_leave_one_out.csv")
        t12 = _load(tables, "table12_efficiency.csv")
        t15 = _load(tables, "table15_rho_estimates.csv")
        label = "this repository (demo run)"

    fig1(t3, out, label)
    fig2(t5, t15, out, label)
    fig3(t7, out, label)
    fig4(t12, t9, out, label)
    fig5(t6, out, label)
    fig6(t11, out, label)

    made = sorted(p.name for p in out.glob("*.png"))
    print(f"[09] {len(made)} figures -> {out}")
    for m in made:
        print(f"     {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
