#!/usr/bin/env python3
"""Step 11 - claim crosswalk.

For each headline claim in the paper, reads the artifact this repository
produced and reports what the run actually shows.

The output is deliberately blunt about what a demo run can and cannot
establish.  Three verdicts are used:

``supported``
    The run reproduces the claim's *direction and shape* (e.g. IWN improves
    D4, random vocabulary hurts, the gate degenerates at rho = 0).
``not_supported``
    The run contradicts the direction.
``not_tested``
    The artifact is missing, or the claim is about absolute numbers on the
    real corpora, which a demo run cannot speak to.

Absolute paper values are never reported as "reproduced" by a demo run.
Emits ``results/tables/claim_crosswalk.csv`` and prints a summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from surgellm.paper_reference import HEADLINE_CLAIMS
from surgellm.utils.io import ensure_dir, write_csv


def _read(tables: Path, name: str) -> Optional[pd.DataFrame]:
    p = tables / name
    if p.exists() and p.stat().st_size > 0:
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def _verdict(observed: Optional[float], paper: float, kind: str) -> str:
    if observed is None or (isinstance(observed, float) and np.isnan(observed)):
        return "not_tested"
    if kind in ("delta", "absolute", "ratio", "fraction"):
        # Direction / shape, not magnitude.
        if kind == "delta":
            return "supported" if np.sign(observed) == np.sign(paper) else "not_supported"
        if kind == "ratio":
            return "supported" if observed >= 0.95 else "not_supported"
        if kind == "fraction":
            return "supported" if observed >= paper * 0.95 else "not_supported"
        return "supported" if observed > 0 else "not_supported"
    if kind == "boolean":
        return "supported" if bool(observed) else "not_supported"
    return "not_tested"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    args = ap.parse_args()

    tables = Path(args.tables)
    out = ensure_dir(tables)
    demo = args.mode == "demo"

    t3 = _read(tables, "table3_main_results.csv")
    t5 = _read(tables, "table5_iwn_ablation.csv")
    t7 = _read(tables, "table7_component_ablation.csv")
    t9 = _read(tables, "table9_vocab_control.csv")
    t10 = _read(tables, "table10_surface_ablation.csv")
    t11 = _read(tables, "table11_leave_one_out.csv")
    t15 = _read(tables, "table15_rho_estimates.csv")
    params = _read(tables, "parameter_report.csv")
    prop2 = _read(tables, "proposition2_degeneracy.csv")

    observed: Dict[str, Optional[float]] = {k: None for k in
                                            (c["id"] for c in HEADLINE_CLAIMS)}
    notes: Dict[str, str] = {}

    if t3 is not None and "avg_f1" in t3:
        best = t3.loc[t3["avg_f1"].idxmax()]
        observed["C1"] = float(best["avg_f1"])
        notes["C1"] = f"best model in this run: {best['model']}"
        iwn = t3[t3["model"].astype(str).str.contains("IWN")]
        non = t3[~t3["model"].astype(str).str.contains("IWN")]
        if len(iwn) and len(non):
            observed["C2"] = float(iwn["avg_f1"].max() - non["avg_f1"].max())
            notes["C2"] = (f"best IWN {iwn['avg_f1'].max():.4f} vs best non-IWN "
                           f"{non['avg_f1'].max():.4f}")

    if t3 is not None and "D4" in t3:
        iwn = t3[t3["model"] == "SURGeLLM-IWN-RoBERTa"]
        base = t3[t3["model"] == "Baseline-RoBERTa"]
        if len(iwn) and len(base):
            observed["C3"] = float(iwn["D4"].iloc[0] - base["D4"].iloc[0])
            notes["C3"] = (f"D4 {base['D4'].iloc[0]:.4f} -> {iwn['D4'].iloc[0]:.4f}")

    if t7 is not None and "delta_D4" in t7.columns:
        row = t7[t7["model"] == "SURGeLLM-FULL-RoBERTa"]
        if len(row) and pd.notna(row["delta_D4"].iloc[0]):
            observed["C4"] = float(row["delta_D4"].iloc[0])
            notes["C4"] = "gate+extended without IWN, D4 delta vs same-backbone baseline"

    if t9 is not None and "avg" in t9.columns:
        d = t9[t9["vocab"] == "Delta Random"]
        if len(d):
            observed["C5"] = float(d["avg"].iloc[0])
        rec = t9[t9["vocab"].astype(str).str.startswith("Auto / Curated")]
        if len(rec):
            observed["C6"] = float(rec["avg"].iloc[0])
            notes["C6"] = "auto-extracted avg F1 / curated avg F1"

    if t10 is not None and "D4" in t10.columns:
        d = t10[t10["config"] == "Delta no-S"]
        if len(d):
            observed["C7"] = float(d["D4"].iloc[0])

    if t15 is not None:
        d4 = t15[t15["task"] == "D4"]
        if len(d4):
            observed["C8"] = float(d4["rho_increase"].iloc[0])
            notes["C8"] = (f"rho_4 {d4['rho_no_iwn'].iloc[0]:.3f} -> "
                           f"{d4['rho_iwn'].iloc[0]:.3f}")

    if params is not None and "shared_fraction" in params.columns:
        observed["C9"] = float(params["shared_fraction"].max())
        notes["C9"] = ("tiny stand-in backbones have proportionally larger heads "
                       "than a real 125M encoder" if demo else "")
    if params is not None and "surgical_overhead_fraction" in params.columns:
        observed["C11"] = float(params["surgical_overhead_fraction"].min())
        notes["C11"] = "gate + task embedding as a fraction of backbone parameters"

    if prop2 is not None and "check" in prop2.columns:
        row = prop2[prop2["check"] == "all_directional_checks_pass"]
        if len(row):
            v = str(row["value"].iloc[0]).strip().lower()
            observed["C10"] = 1.0 if v in ("true", "1", "1.0") else 0.0

    if t11 is not None and "dominant_for" in t11.columns:
        tasks_covered = set()
        for s in t11["dominant_for"].dropna().astype(str):
            tasks_covered |= {t for t in s.split(",") if t}
        observed["C12"] = 1.0 if len(tasks_covered) >= 3 else 0.0
        notes["C12"] = f"tasks with a distinct dominant group: {sorted(tasks_covered)}"

    rows: List[Dict] = []
    for claim in HEADLINE_CLAIMS:
        cid = claim["id"]
        obs = observed.get(cid)
        verdict = _verdict(obs, claim["paper_value"], claim["kind"])
        if demo and claim["kind"] == "absolute" and verdict == "supported":
            verdict = "shape_only"
        rows.append({
            "id": cid, "claim": claim["claim"],
            "paper_value": claim["paper_value"], "kind": claim["kind"],
            "observed_value": (None if obs is None else round(float(obs), 4)),
            "verdict": verdict, "artifact": claim["artifact"],
            "note": notes.get(cid, ""),
            "run_mode": args.mode,
        })

    write_csv(rows, out / "claim_crosswalk.csv")

    width = max(len(r["claim"]) for r in rows)
    print(f"\n{'ID':<4} {'claim':<{width}}  {'paper':>8}  {'observed':>9}  verdict")
    print("-" * (width + 42))
    for r in rows:
        obs = "-" if r["observed_value"] is None else f"{r['observed_value']:+.4f}"
        print(f"{r['id']:<4} {r['claim']:<{width}}  {r['paper_value']:>8}  "
              f"{obs:>9}  {r['verdict']}")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if demo:
        print("\nNote: this is a demo run on surrogate corpora with offline "
              "stand-in backbones.\nIt can support the *direction and mechanism* "
              "of a claim, never its absolute value\non SST-2 / HotPotQA / LLM-7 "
              "/ HumLLM.  Run with --mode paper for those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
