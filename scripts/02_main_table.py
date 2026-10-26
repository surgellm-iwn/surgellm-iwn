#!/usr/bin/env python3
"""Step 2 - the main multi-seed comparison.

Runs every model variant across seeds {0,1,2} and emits:

``table3_main_results.csv``        Table 3 (macro-F1 mean +/- SD, Avg F1, Delta, T(s))
``table14_per_seed.csv``           Table 14 (per-seed Avg F1)
``table6_training_dynamics.csv``   Table 6 (initial/final loss, best epoch)
``table5_iwn_ablation.csv``        Table 5 (IWN vs FULL, D4 per-class P/R)
``parameter_report.csv``           §3.1 shared-parameter share, §H overhead
``per_seed_raw.json``              everything, for the significance step

In ``--mode demo`` the offline ``tiny`` backbone stands in for every
encoder; each row is still a real training run, but the numbers are the
repository's, not the paper's, and are labelled ``source=run``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from surgellm.data.registry import load_all_tasks
from surgellm.data.schema import TASK_ORDER
from surgellm.eval.failure_analysis import efficiency_row, pareto_frontier
from surgellm.experiment import RunSpec, run_single
from surgellm.models.backbone import BACKBONE_REGISTRY
from surgellm.utils.io import ensure_dir, fmt_mean_sd, write_csv, write_json
from surgellm.utils.seeding import SEEDS

#: The eleven rows of Table 3, as (display name, variant, paper backbone).
MODEL_GRID: List[tuple] = [
    ("T5-base",                "t5",            "t5"),
    ("Baseline-DistilBERT",    "baseline",      "distilbert"),
    ("Baseline-BERT",          "baseline",      "bert"),
    ("Baseline-RoBERTa",       "baseline",      "roberta"),
    ("SURGeLLM-S-DistilBERT",  "surgellm-s",    "distilbert"),
    ("SURGeLLM-S-BERT",        "surgellm-s",    "bert"),
    ("SURGeLLM-G-RoBERTa",     "surgellm-g",    "roberta"),
    ("SURGeLLM-FULL-RoBERTa",  "surgellm-full", "roberta"),
    ("SURGeLLM-FULL-ALBERT",   "surgellm-full", "albert"),
    ("SURGeLLM-IWN-RoBERTa",   "surgellm-iwn",  "roberta"),
    ("SURGeLLM-IWN-BERT",      "surgellm-iwn",  "bert"),
]

REFERENCE_MODEL = "Baseline-RoBERTa"   # the Delta column is relative to this


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"],
                    help="demo = offline tiny backbone; paper = real HF encoders")
    ap.add_argument("--provider", default=None, choices=[None, "synthetic", "hf"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--scale", type=float, default=0.20)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of display names to run")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    demo = args.mode == "demo"
    provider = args.provider or ("synthetic" if demo else "hf")
    out = ensure_dir(Path(args.out) / "tables")
    logs = ensure_dir(Path(args.out) / "logs")
    ckpt_dir = ensure_dir(ROOT / "checkpoints")

    grid = [g for g in MODEL_GRID
            if args.models is None or g[0] in set(args.models)]

    per_seed: Dict[str, Dict[str, List[float]]] = {}
    per_seed_avg: Dict[str, List[float]] = {}
    dynamics: Dict[str, Dict] = {}
    times: Dict[str, List[float]] = {}
    param_rows: List[Dict] = []
    d4_breakdown: Dict[str, Dict] = {}

    task_cache: Dict[int, Dict] = {}
    t_start = time.time()
    total = len(grid) * len(args.seeds)
    done = 0

    for display, variant, backbone in grid:
        per_seed[display] = {t: [] for t in TASK_ORDER}
        per_seed_avg[display] = []
        times[display] = []

        for seed in args.seeds:
            if seed not in task_cache:
                task_cache[seed] = load_all_tasks(provider=provider, seed=seed,
                                                  scale=args.scale)
            # The backbone key is always the paper's.  In demo mode
            # ``offline_backbone`` swaps in that backbone's offline stand-in
            # from DEMO_TINY_CONFIG, so the eleven rows stay eleven distinct
            # models instead of collapsing onto one.
            spec = RunSpec(
                variant=variant, backbone=backbone, seed=seed,
                provider=provider, offline_backbone=demo,
                scale=args.scale, profile="demo" if demo else "paper",
                checkpoint_dir=str(ckpt_dir),
            )
            art = run_single(spec, tasks=task_cache[seed], verbose=False)
            done += 1

            for tid, m in art.result.test_metrics.items():
                per_seed[display][tid].append(float(m["macro_f1"]))
            per_seed_avg[display].append(float(art.avg_f1))
            times[display].append(float(art.result.wall_clock_s))

            if seed == args.seeds[0]:
                dynamics[display] = {
                    "init_loss": art.result.initial_loss,
                    "final_loss": art.result.final_loss,
                    "best_epoch": art.result.best_epoch,
                    "delta_loss": art.result.loss_reduction,
                    "early_stopped": art.result.early_stopped,
                    "epochs_run": art.result.epochs_run,
                }
                rep = art.result.parameter_report
                param_rows.append({
                    "model": display, "source": "run",
                    "paper_backbone": backbone,
                    "reported_backbone_params_m": round(
                        BACKBONE_REGISTRY[backbone].reported_params / 1e6, 1),
                    **{k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in rep.items()},
                })
                if "D4" in art.result.test_metrics:
                    m = art.result.test_metrics["D4"]
                    d4_breakdown[display] = {
                        "hum_p": m.get("precision_human"), "hum_r": m.get("recall_human"),
                        "llm_p": m.get("precision_llm"), "llm_r": m.get("recall_llm"),
                    }

            el = time.time() - t_start
            print(f"[02] {done}/{total} {display} seed={seed} "
                  f"avg={art.avg_f1:.4f} ({el / 60:.1f} min elapsed)", flush=True)

    # ---------------- Table 3 -------------------------------------------
    ref_avg = float(np.mean(per_seed_avg.get(REFERENCE_MODEL, [np.nan])))
    table3: List[Dict] = []
    for display, variant, backbone in grid:
        row: Dict = {
            "model": display, "source": "run",
            "family": variant, "paper_backbone": backbone,
            "params_m": round(BACKBONE_REGISTRY[backbone].reported_params / 1e6, 1),
        }
        for tid in TASK_ORDER:
            vals = per_seed[display][tid]
            row[tid] = round(float(np.mean(vals)), 4) if vals else float("nan")
            row[f"{tid}_sd"] = round(float(np.std(vals)), 4) if vals else float("nan")
            row[f"{tid}_fmt"] = (fmt_mean_sd(np.mean(vals), np.std(vals))
                                 if vals else "")
        avg = float(np.mean(per_seed_avg[display]))
        row["avg_f1"] = round(avg, 4)
        row["avg_f1_sd"] = round(float(np.std(per_seed_avg[display])), 4)
        row["delta_vs_baseline_roberta"] = round(avg - ref_avg, 4)
        row["train_s"] = round(float(np.mean(times[display])), 1)
        row["n_seeds"] = len(per_seed_avg[display])
        table3.append(row)

    write_csv(table3, out / "table3_main_results.csv")

    # ---------------- Table 14 ------------------------------------------
    table14 = []
    for display in per_seed_avg:
        r = {"model": display, "source": "run"}
        for i, s in enumerate(args.seeds):
            r[f"seed{s}"] = round(per_seed_avg[display][i], 4)
        r["mean"] = round(float(np.mean(per_seed_avg[display])), 4)
        r["sd"] = round(float(np.std(per_seed_avg[display])), 4)
        table14.append(r)
    write_csv(table14, out / "table14_per_seed.csv")

    # ---------------- Table 6 -------------------------------------------
    write_csv([{"model": k, "source": "run",
                **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                   for kk, vv in v.items()}}
               for k, v in dynamics.items()],
              out / "table6_training_dynamics.csv")

    # ---------------- Table 5 -------------------------------------------
    table5 = []
    pairs = [("SURGeLLM-FULL-RoBERTa", "SURGeLLM-IWN-RoBERTa", "RoBERTa"),
             ("SURGeLLM-S-BERT", "SURGeLLM-IWN-BERT", "BERT")]
    for no_iwn, with_iwn, bk in pairs:
        if no_iwn not in per_seed or with_iwn not in per_seed:
            continue
        for label, name in ((no_iwn, no_iwn), (with_iwn, with_iwn)):
            row = {"variant": name, "source": "run", "backbone": bk}
            for tid in TASK_ORDER:
                row[tid] = round(float(np.mean(per_seed[name][tid])), 4)
            row.update({k: (round(v, 3) if v is not None else None)
                        for k, v in d4_breakdown.get(name, {}).items()})
            table5.append(row)
        delta = {"variant": f"Delta ({bk})", "source": "run", "backbone": bk}
        for tid in TASK_ORDER:
            delta[tid] = round(float(np.mean(per_seed[with_iwn][tid])
                                     - np.mean(per_seed[no_iwn][tid])), 4)
        a, b = d4_breakdown.get(with_iwn, {}), d4_breakdown.get(no_iwn, {})
        for k in ("hum_p", "hum_r", "llm_p", "llm_r"):
            if a.get(k) is not None and b.get(k) is not None:
                delta[k] = round(a[k] - b[k], 3)
        table5.append(delta)
    write_csv(table5, out / "table5_iwn_ablation.csv")

    # ---------------- parameters + efficiency ---------------------------
    write_csv(param_rows, out / "parameter_report.csv")

    base_t = float(np.mean(times.get("Baseline-DistilBERT",
                                     times.get(grid[0][0], [1.0]))))
    eff_rows = [
        efficiency_row(
            display,
            BACKBONE_REGISTRY[backbone].reported_params,
            float(np.mean(times[display])),
            float(np.mean(per_seed_avg[display])),
            baseline_seconds=base_t,
        )
        for display, _, backbone in grid
    ]
    eff_rows = pareto_frontier(eff_rows)
    for r in eff_rows:
        r["source"] = "run"
        r["note"] = ("parameter count is the paper's backbone size; wall-clock "
                     "is this run's offline backbone" if demo else "")
    write_csv(eff_rows, out / "table12_efficiency.csv")

    write_json({
        "per_seed_task_f1": per_seed,
        "per_seed_avg_f1": per_seed_avg,
        "wall_clock_s": times,
        "mode": args.mode, "provider": provider, "scale": args.scale,
        "seeds": args.seeds, "reference_model": REFERENCE_MODEL,
        "total_minutes": round((time.time() - t_start) / 60, 2),
    }, logs / "per_seed_raw.json")

    print(f"\n[02] done in {(time.time() - t_start) / 60:.1f} min")
    best = max(table3, key=lambda r: r["avg_f1"])
    print(f"[02] best model: {best['model']}  avg F1 = {best['avg_f1']:.4f} "
          f"(Delta vs {REFERENCE_MODEL} = {best['delta_vs_baseline_roberta']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
