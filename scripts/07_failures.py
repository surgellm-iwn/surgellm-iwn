#!/usr/bin/env python3
"""Step 7 - failure-case analysis (§7.5).

Trains SURGeLLM-IWN, collects its test-set errors, and classifies up to 50
misclassified examples per task into the paper's diagnostic modes.

Emits
-----
``failure_modes.csv``        per-task mode counts and fractions
``failure_cases.csv``        the inspected examples with their modes
``failure_summary.json``     machine-readable rollup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from surgellm.data.registry import load_all_tasks
from surgellm.data.schema import TASK_ORDER, TASKS
from surgellm.eval.failure_analysis import N_INSPECTED, analyse_failures
from surgellm.experiment import RunSpec, run_single
from surgellm.utils.io import ensure_dir, write_csv, write_json


@torch.no_grad()
def _predict(model, split, task_index: int, batch_size: int = 64):
    device = next(model.parameters()).device
    model.eval()
    preds, probs = [], []
    for i in range(0, len(split), batch_size):
        out = model(
            input_ids=split.input_ids[i:i + batch_size].to(device),
            attention_mask=split.attention_mask[i:i + batch_size].to(device),
            task_ids=torch.full((len(split.labels[i:i + batch_size]),), task_index,
                                dtype=torch.long, device=device),
            s=split.features[i:i + batch_size].to(device),
        )
        logits = out["logits"]
        n_c = int((logits[0] > float("-inf")).sum())
        logits = logits[:, :n_c]
        preds.append(logits.argmax(-1).cpu().numpy())
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--provider", default=None)
    ap.add_argument("--variant", default="surgellm-iwn")
    ap.add_argument("--backbone", default="roberta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.20)
    ap.add_argument("--n-inspect", type=int, default=N_INSPECTED)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    demo = args.mode == "demo"
    provider = args.provider or ("synthetic" if demo else "hf")
    out = ensure_dir(Path(args.out) / "tables")
    logs = ensure_dir(Path(args.out) / "logs")

    tasks = load_all_tasks(provider=provider, seed=args.seed, scale=args.scale)
    art = run_single(RunSpec(
        variant=args.variant, backbone=args.backbone, seed=args.seed,
        provider=provider, offline_backbone=demo, scale=args.scale,
        profile="demo" if demo else "paper", tag="failures",
    ), tasks=tasks, verbose=False)
    print(f"[07] {args.variant}-{args.backbone} avg F1 = {art.avg_f1:.4f}")

    mode_rows: List[Dict] = []
    case_rows: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for i, tid in enumerate(TASK_ORDER):
        split = art.tensors["test"].get(tid)
        if split is None:
            continue
        y_pred, probs = _predict(art.model, split, i)
        rep = analyse_failures(
            tid, split.texts, split.labels.numpy(), y_pred, probs,
            extractor=art.extractor, n_inspect=args.n_inspect, seed=args.seed,
        )
        summary[tid] = {k: v for k, v in rep.items() if k != "cases"}
        case_rows.extend(rep["cases"])
        for mode, count in sorted(rep["mode_counts"].items(), key=lambda kv: -kv[1]):
            mode_rows.append({
                "task": tid, "source": "run", "corpus": TASKS[tid].source,
                "mode": mode, "count": count,
                "fraction_of_inspected": round(rep["mode_fractions"][mode], 4),
                "n_inspected": rep["n_inspected"], "n_errors": rep["n_errors"],
                "error_rate": round(rep["error_rate"], 4),
                "median_error_length": rep["median_error_length"],
                "frac_errors_short": round(rep["frac_errors_short"], 4),
            })
        top = max(rep["mode_counts"].items(), key=lambda kv: kv[1], default=("none", 0))
        print(f"   {tid} ({TASKS[tid].source}): {rep['n_errors']} errors "
              f"({rep['error_rate']:.3f}), inspected {rep['n_inspected']}, "
              f"dominant mode = {top[0]} ({top[1]})")

    write_csv(mode_rows, out / "failure_modes.csv")
    write_csv(case_rows, out / "failure_cases.csv")
    write_json(summary, logs / "failure_summary.json")
    print(f"[07] -> {out / 'failure_modes.csv'}, {out / 'failure_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
