#!/usr/bin/env python3
"""Step 0 - corpora, vocabulary and feature statistics.

Emits
-----
``results/tables/table2_corpus_statistics.csv``   Table 2 reproduction
``results/tables/feature_statistics.csv``         per-dimension s(x) stats
``results/tables/normalization_contamination.csv``  Eq. (9) diagnostics
``configs/surgical_vocabulary.json``              Appendix D as data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from surgellm.data.registry import corpus_report, load_all_tasks
from surgellm.data.schema import TASKS, TOTAL_EXAMPLES
from surgellm.features import SurgicalFeatureExtractor, feature_summary
from surgellm.normalization import FeatureNormalizer, prior_contamination
from surgellm.utils.io import write_csv, write_json
from surgellm.vocab import dump_default_vocabulary, default_vocabulary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--provider", default=None, choices=[None, "synthetic", "hf"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
    args = ap.parse_args()

    if args.provider is None:
        args.provider = "synthetic" if args.mode == "demo" else "hf"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vocab_path = dump_default_vocabulary(ROOT / "configs" / "surgical_vocabulary.json")
    print(f"[00] surgical vocabulary -> {vocab_path}")

    print(f"[00] loading corpora (provider={args.provider}, scale={args.scale})")
    tasks = load_all_tasks(provider=args.provider, seed=args.seed, scale=args.scale)

    rows = corpus_report(tasks)
    rows.append({"task": "Total", "id": "", "n": sum(r["n"] for r in rows),
                 "source": "", "provider": args.provider})
    write_csv(rows, out / "table2_corpus_statistics.csv")
    print(f"[00] Table 2 -> {out / 'table2_corpus_statistics.csv'} "
          f"(paper total: {TOTAL_EXAMPLES})")

    extractor = SurgicalFeatureExtractor(default_vocabulary())
    feat_rows, contam_rows = [], []
    for tid, td in tasks.items():
        pool = td.normalizer_pool
        s = extractor.extract_batch(pool.texts)
        for name, stats in feature_summary(s, extractor.feature_names()).items():
            feat_rows.append({"task": tid, "feature": name, **{
                k: round(v, 4) for k, v in stats.items()}})

        std = FeatureNormalizer("standard").fit(s, pool.labels, pool.n_classes)
        iwn = FeatureNormalizer("iwn").fit(s, pool.labels, pool.n_classes)
        contam = prior_contamination(std)
        z_std, z_iwn = std.transform(s), iwn.transform(s)
        contam_rows.append({
            "task": tid, "source": TASKS[tid].source,
            "pool_n": len(pool), "pool_skew": round(pool.skew(), 2),
            "paper_raw_skew": TASKS[tid].raw_skew,
            **{k: round(v, 4) for k, v in contam.items()},
            "max_abs_z_standard": round(float(np.abs(z_std).max()), 2),
            "max_abs_z_iwn": round(float(np.abs(z_iwn).max()), 2),
            "mean_scale_standard": round(float(std.stats.std.mean()), 4),
            "mean_scale_iwn": round(float(iwn.stats.std.mean()), 4),
            "iwn_learnable_parameters": iwn.n_parameters,
        })

    write_csv(feat_rows, out / "feature_statistics.csv")
    write_csv(contam_rows, out / "normalization_contamination.csv")
    write_json({"provider": args.provider, "seed": args.seed, "scale": args.scale,
                "tasks": {t: td.splits.sizes() for t, td in tasks.items()}},
               out.parent / "logs" / "00_prepare_data.json")

    print(f"[00] feature statistics -> {out / 'feature_statistics.csv'}")
    print(f"[00] contamination      -> {out / 'normalization_contamination.csv'}")
    for r in contam_rows:
        print(f"     {r['task']}: pool skew {r['pool_skew']:>5}  "
              f"majority prior {r['majority_prior']:.3f}  "
              f"normalised mean gap {r['normalized_mean_gap']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
