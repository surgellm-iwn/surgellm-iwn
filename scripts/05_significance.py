#!/usr/bin/env python3
"""Step 5 - statistical significance (§6.2, Table 4).

Paired Welch t-tests across seeds for each SURGeLLM variant against its
same-backbone baseline, Benjamini-Hochberg FDR correction at q = 0.05 over
the 4 x 4 = 16 task-variant comparisons, plus bootstrap confidence
intervals (B = 2,000, seed 0) for every model's Avg F1.

Reads ``results/logs/per_seed_raw.json`` from Step 2.

Emits
-----
``table4_significance.csv``      every comparison, raw and BH-corrected p
``bootstrap_ci.csv``             95% CI per model, and CI overlap with the
                                 baseline (the Abstract-revision claim of
                                 Appendix L.4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from surgellm.data.schema import TASK_ORDER
from surgellm.eval.significance import (BOOTSTRAP_B, FDR_Q, bootstrap_ci,
                                        paired_comparison_table)
from surgellm.utils.io import ensure_dir, read_json, write_csv

#: §6.2 compares each SURGeLLM variant with its *same-backbone* baseline,
#: plus the two cross-variant comparisons the paper highlights.
COMPARISONS: List[Tuple[str, str]] = [
    ("SURGeLLM-S-DistilBERT", "Baseline-DistilBERT"),
    ("SURGeLLM-S-BERT", "Baseline-BERT"),
    ("SURGeLLM-IWN-BERT", "Baseline-BERT"),
    ("SURGeLLM-G-RoBERTa", "Baseline-RoBERTa"),
    ("SURGeLLM-FULL-RoBERTa", "Baseline-RoBERTa"),
    ("SURGeLLM-IWN-RoBERTa", "Baseline-RoBERTa"),
    ("SURGeLLM-FULL-ALBERT", "Baseline-RoBERTa"),
    ("SURGeLLM-IWN-RoBERTa", "SURGeLLM-FULL-RoBERTa"),
    ("T5-base", "Baseline-RoBERTa"),
]

BASELINE = "Baseline-RoBERTa"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(ROOT / "results" / "logs" / "per_seed_raw.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--q", type=float, default=FDR_Q)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"[05] missing {raw_path}; run scripts/02_main_table.py first")
        return 1

    blob = read_json(raw_path)
    per_task: Dict[str, Dict[str, List[float]]] = blob["per_seed_task_f1"]
    per_avg: Dict[str, List[float]] = blob["per_seed_avg_f1"]
    out = ensure_dir(Path(args.out))

    available = [(a, b) for a, b in COMPARISONS if a in per_task and b in per_task]
    rows = paired_comparison_table(per_task, available, TASK_ORDER, q=args.q)
    for r in rows:
        r["source"] = "run"
    write_csv(rows, out / "table4_significance.csv")

    n_sig = sum(1 for r in rows if r["significant"])
    print(f"[05] {len(rows)} comparisons in the BH family "
          f"(q = {args.q}), {n_sig} significant")
    for r in sorted(rows, key=lambda r: r["p_bh"])[:10]:
        flag = "*" if r["significant"] else " "
        print(f"   {flag} {r['comparison']:<48s} {r['task']}  "
              f"d={r['delta']:+.4f}  p_BH={r['p_bh']:.4g}")

    # ---- bootstrap CIs on Avg F1 ---------------------------------------
    ci_rows: List[Dict] = []
    base = per_avg.get(BASELINE, [])
    base_ci = bootstrap_ci(base, B=BOOTSTRAP_B, seed=0) if base else (np.nan,) * 3
    for model, vals in per_avg.items():
        lo, hi, width = bootstrap_ci(vals, B=BOOTSTRAP_B, seed=0)
        overlaps = (not np.isnan(base_ci[0])) and not (lo > base_ci[1] or hi < base_ci[0])
        ci_rows.append({
            "model": model, "source": "run",
            "mean_avg_f1": round(float(np.mean(vals)), 4),
            "sd": round(float(np.std(vals)), 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "ci_width": round(width, 4),
            "n_seeds": len(vals), "B": BOOTSTRAP_B,
            f"ci_overlaps_{BASELINE}": bool(overlaps),
        })
    ci_rows.sort(key=lambda r: -r["mean_avg_f1"])
    write_csv(ci_rows, out / "bootstrap_ci.csv")
    print(f"\n[05] bootstrap CIs (B={BOOTSTRAP_B}) -> {out / 'bootstrap_ci.csv'}")
    for r in ci_rows[:4]:
        print(f"   {r['model']:<24s} {r['mean_avg_f1']:.4f} "
              f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
