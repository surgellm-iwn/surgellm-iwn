#!/usr/bin/env python3
"""Step 1 - train a single configuration.

Examples
--------
Offline smoke run (no downloads, ~2 minutes on CPU)::

    python scripts/01_train.py --variant surgellm-iwn --backbone roberta \\
        --mode demo --scale 0.2 --seed 0

Paper configuration (needs ``transformers``/``datasets`` and a GPU)::

    python scripts/01_train.py --variant surgellm-iwn --backbone roberta \\
        --mode paper --seed 0 --device cuda

Writes a checkpoint to ``checkpoints/<run_id>.pt`` with the model weights,
the full config and the run's metrics, plus a JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from surgellm.experiment import RunSpec, run_single
from surgellm.models.surgellm import VARIANT_TABLE
from surgellm.models.backbone import BACKBONE_REGISTRY
from surgellm.utils.io import ensure_dir, write_json
from surgellm.utils.seeding import determinism_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="surgellm-iwn", choices=sorted(VARIANT_TABLE))
    ap.add_argument("--backbone", default="roberta", choices=sorted(BACKBONE_REGISTRY))
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--provider", default=None, choices=[None, "synthetic", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    demo = args.mode == "demo"

    spec = RunSpec(
        variant=args.variant, backbone=args.backbone, seed=args.seed,
        provider=args.provider or ("synthetic" if demo else "hf"),
        offline_backbone=demo, scale=args.scale, device=args.device,
        max_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, cache_dir=args.cache_dir,
        checkpoint_dir=ensure_dir(args.checkpoint_dir).as_posix(),
        profile="demo" if demo else "paper",
    )

    art = run_single(spec, verbose=not args.quiet)
    res = art.result

    print("\n" + "=" * 68)
    print(f"  {spec.run_id}")
    print("=" * 68)
    for tid, m in res.test_metrics.items():
        extras = " ".join(
            f"{k.split('_', 1)[1]}_{k.split('_', 1)[0][:1].upper()}={m[k]:.3f}"
            for k in m if k.startswith(("precision_", "recall_"))
        )
        print(f"  {tid}  macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"AUC {m.get('roc_auc', float('nan')):.4f}")
        if extras:
            print(f"        {extras}")
    print(f"  Avg macro-F1 : {res.avg_test_macro_f1:.4f}")
    print(f"  best epoch   : {res.best_epoch} / {res.epochs_run}"
          f"{'  (early stopped)' if res.early_stopped else ''}")
    print(f"  loss         : {res.initial_loss:.4f} -> {res.final_loss:.4f} "
          f"(delta {res.loss_reduction:.4f})")
    print(f"  wall clock   : {res.wall_clock_s:.1f}s")
    pr = res.parameter_report
    print(f"  parameters   : {int(pr['total_parameters']):,} total, "
          f"{pr['shared_fraction'] * 100:.2f}% shared, "
          f"{int(pr['surgical_overhead_parameters']):,} surgical overhead")
    print("=" * 68)

    write_json({"spec": spec.to_dict(), "result": res.to_dict(),
                "determinism": determinism_report()},
               ROOT / "results" / "logs" / f"run_{spec.run_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
