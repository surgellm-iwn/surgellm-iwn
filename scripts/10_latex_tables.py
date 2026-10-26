#!/usr/bin/env python3
"""Step 10 - LaTeX fragments for every table, from the CSVs.

Writes ``results/tables/latex/*.tex``, one ``\\begin{table}`` fragment per
table, ready to ``\\input`` into the manuscript.  ``--source paper``
regenerates the published tables instead, which is useful for checking that
the fragment layout matches the paper before swapping in new numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from surgellm import paper_reference as PR
from surgellm.data.schema import TASK_ORDER
from surgellm.utils.io import ensure_dir, to_latex_table

SPECS = [
    dict(csv="table2_corpus_statistics.csv", tex="table2_corpus.tex",
         columns=["task", "id", "n", "n_c", "pct_minority", "source"],
         headers=["Task", "ID", "$n$", "$n_c$", "\\% min.", "Source"],
         caption="Corpus statistics after stratified capping.", label="tab:corpus"),
    dict(csv="table3_main_results.csv", tex="table3_main.tex",
         columns=["model", "family", "params_m"] + TASK_ORDER + ["avg_f1", "delta_vs_baseline_roberta", "train_s"],
         headers=["Model", "Family", "Par. (M)", "D1", "D2", "D3", "D4", "Avg F1", "$\\Delta$", "T(s)"],
         caption="Main results: macro-F1 mean over three seeds.", label="tab:main",
         bold=["avg_f1"] + TASK_ORDER),
    dict(csv="table4_significance.csv", tex="table4_significance.tex",
         columns=["comparison", "task", "delta", "p_bh", "significant"],
         headers=["Comparison", "Task", "$\\Delta$", "$p$ (BH)", "Sig."],
         caption="Significance tests: BH-corrected $p$-values.", label="tab:sig"),
    dict(csv="table5_iwn_ablation.csv", tex="table5_iwn.tex",
         columns=["variant"] + TASK_ORDER + ["hum_p", "hum_r", "llm_p", "llm_r"],
         headers=["Variant", "D1", "D2", "D3", "D4", "Hum P", "Hum R", "LLM P", "LLM R"],
         caption="IWN ablation, including D$_4$ per-class breakdown.", label="tab:iwn"),
    dict(csv="table6_training_dynamics.csv", tex="table6_dynamics.tex",
         columns=["model", "init_loss", "final_loss", "best_epoch", "delta_loss"],
         headers=["Model", "Init. loss", "Final loss", "Best ep.", "$\\Delta$Loss"],
         caption="Training dynamics summary (seed 0).", label="tab:dynamics"),
    dict(csv="table7_component_ablation.csv", tex="table7_components.tex",
         columns=["model", "backbone", "P", "G", "E", "I"] + TASK_ORDER + ["avg", "delta"],
         headers=["Model", "Backbone", "P", "G", "E", "I", "D1", "D2", "D3", "D4", "Avg", "$\\Delta$"],
         caption="Component ablation across backbones.", label="tab:components"),
    dict(csv="table8_group_count.csv", tex="table8_groupcount.tex",
         columns=["n_groups"] + TASK_ORDER + ["avg"],
         headers=["$|\\mathcal{V}|$", "D1", "D2", "D3", "D4", "Avg"],
         caption="Sensitivity to the number of surgical groups.", label="tab:groupcount",
         bold=["avg"]),
    dict(csv="table9_vocab_control.csv", tex="table9_vocab.tex",
         columns=["vocab"] + TASK_ORDER + ["avg"],
         headers=["Vocab.", "D1", "D2", "D3", "D4", "Avg"],
         caption="Random-vocabulary control.", label="tab:vocab"),
    dict(csv="table10_surface_ablation.csv", tex="table10_surface.tex",
         columns=["config"] + TASK_ORDER + ["avg"],
         headers=["Config.", "D1", "D2", "D3", "D4", "Avg"],
         caption="Surface-features ablation.", label="tab:surface"),
    dict(csv="table11_leave_one_out.csv", tex="table11_loo.tex",
         columns=["group_removed"] + TASK_ORDER,
         headers=["Group removed", "D1", "D2", "D3", "D4"],
         caption="Leave-one-out per-group F1 drop.", label="tab:loo"),
    dict(csv="table12_efficiency.csv", tex="table12_efficiency.tex",
         columns=["model", "parameters", "train_seconds", "overhead_x", "avg_f1",
                  "f1_per_min", "efficiency", "pareto_efficient"],
         headers=["Model", "Par.", "T(s)", "Overhead", "Avg F1", "F1/min", "Eff", "Pareto"],
         caption="Speed-accuracy trade-off.", label="tab:efficiency"),
    dict(csv="table14_per_seed.csv", tex="table14_perseed.tex",
         columns=["model", "seed0", "seed1", "seed2", "mean", "sd"],
         headers=["Model", "Seed 0", "Seed 1", "Seed 2", "Mean", "SD"],
         caption="Per-seed Avg F1.", label="tab:perseed"),
    dict(csv="table15_rho_estimates.csv", tex="table15_rho.tex",
         columns=["task", "name", "rho_no_iwn", "rho_iwn", "rho_increase", "f1_gain"],
         headers=["Task", "Corpus", "$\\rho_k$ (no IWN)", "$\\rho_k$ (IWN)",
                  "$\\Delta\\rho_k$", "$\\Delta$F1"],
         caption="Empirical $\\rho_k$ estimates.", label="tab:rho"),
    dict(csv="table16_multilingual.csv", tex="table16_multilingual.tex",
         columns=["config", "french", "german"],
         headers=["Configuration", "French", "German"],
         caption="Preliminary multilingual results.", label="tab:multilingual"),
    dict(csv="theorem1_bound.csv", tex="theorem1_bound.tex",
         columns=["task", "n_train", "generalization_term", "approximation_term",
                  "bound_total", "excess_risk_proxy", "bound_holds"],
         headers=["Task", "$N_k$", "Gen. term", "Approx. term", "Bound",
                  "Excess risk", "Holds"],
         caption="Theorem 1: Eq. (16) terms evaluated on the trained model.",
         label="tab:thm1"),
]

PAPER_TABLES = {
    "table2_corpus_statistics.csv": None,
    "table3_main_results.csv": PR.TABLE3_MAIN,
    "table4_significance.csv": PR.TABLE4_SIGNIFICANCE,
    "table5_iwn_ablation.csv": PR.TABLE5_IWN_ABLATION,
    "table6_training_dynamics.csv": PR.TABLE6_TRAINING_DYNAMICS,
    "table7_component_ablation.csv": PR.TABLE7_COMPONENTS,
    "table8_group_count.csv": PR.TABLE8_GROUP_COUNT,
    "table9_vocab_control.csv": PR.TABLE9_VOCAB_CONTROL,
    "table10_surface_ablation.csv": PR.TABLE10_SURFACE,
    "table11_leave_one_out.csv": PR.TABLE11_LEAVE_ONE_OUT,
    "table12_efficiency.csv": PR.TABLE12_EFFICIENCY,
    "table14_per_seed.csv": PR.TABLE14_PER_SEED,
    "table15_rho_estimates.csv": PR.TABLE15_RHO,
    "table16_multilingual.csv": PR.TABLE16_MULTILINGUAL,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="run", choices=["run", "paper"])
    ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tables = Path(args.tables)
    out = ensure_dir(args.out or (tables / "latex" /
                                  ("paper" if args.source == "paper" else "run")))
    made: List[str] = []

    for spec in SPECS:
        if args.source == "paper":
            ref = PAPER_TABLES.get(spec["csv"])
            if ref is None:
                continue
            df = pd.DataFrame(PR.as_rows(ref))
        else:
            p = tables / spec["csv"]
            if not p.exists() or p.stat().st_size == 0:
                continue
            df = pd.read_csv(p)

        cols = [c for c in spec["columns"] if c in df.columns]
        if not cols:
            continue
        heads = [h for c, h in zip(spec["columns"], spec["headers"]) if c in df.columns]
        to_latex_table(
            df.to_dict("records"), cols, spec["caption"], spec["label"],
            out / spec["tex"], headers=heads,
            bold_max_columns=[c for c in spec.get("bold", []) if c in df.columns],
        )
        made.append(spec["tex"])

    print(f"[10] {len(made)} LaTeX fragments -> {out}")
    for m in made:
        print(f"     {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
