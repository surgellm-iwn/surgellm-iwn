#!/usr/bin/env python3
"""Step 3 - §7.1-7.2 ablations.

``table7_component_ablation.csv``   P / G / E / I grid across backbones
``table8_group_count.csv``          |V| in {0, 5, 10, 15, 20}
``table9_vocab_control.csv``        none / random / auto-extracted / curated
``table10_surface_ablation.csv``    G+S / G only / S only
``table11_leave_one_out.csv``       per-group F1 drop
``autoextract_report.json``         Appendix E extraction details

Table 7 is assembled from the Step 2 runs when ``per_seed_raw.json`` is
available (the rows are a subset of Table 3), so it costs nothing extra;
pass ``--retrain-table7`` to run it independently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from surgellm.autoextract import auto_extract_vocabulary, random_control_vocabulary
from surgellm.data.registry import load_all_tasks
from surgellm.data.schema import TASK_ORDER
from surgellm.experiment import RunSpec, run_single
from surgellm.models.surgellm import variant_flags
from surgellm.utils.io import ensure_dir, read_json, write_csv, write_json
from surgellm.vocab import GROUP_ORDER, SurgicalVocabulary, default_vocabulary

#: §7.2 uses SURGeLLM-G-RoBERTa as the reference configuration.
REF_VARIANT = "surgellm-g"
REF_BACKBONE = "roberta"


def _mean_over_seeds(runs) -> Dict[str, float]:
    per_task: Dict[str, List[float]] = {t: [] for t in TASK_ORDER}
    avgs: List[float] = []
    for art in runs:
        for tid, m in art.result.test_metrics.items():
            per_task[tid].append(float(m["macro_f1"]))
        avgs.append(float(art.avg_f1))
    out = {t: (round(float(np.mean(v)), 4) if v else float("nan"))
           for t, v in per_task.items()}
    out["avg"] = round(float(np.mean(avgs)), 4) if avgs else float("nan")
    out["avg_sd"] = round(float(np.std(avgs)), 4) if avgs else float("nan")
    return out


def _run_config(tasks_by_seed, seeds, *, variant=REF_VARIANT, backbone=REF_BACKBONE,
                vocabulary: Optional[SurgicalVocabulary] = None,
                use_surface=True, use_groups=True, demo=True, scale=0.20,
                provider="synthetic", tag="") -> Dict[str, float]:
    runs = []
    for seed in seeds:
        spec = RunSpec(
            variant=variant, backbone=backbone, seed=seed, provider=provider,
            offline_backbone=demo, scale=scale,
            profile="demo" if demo else "paper",
            use_surface_features=use_surface, use_lexical_groups=use_groups,
            tag=tag,
        )
        runs.append(run_single(spec, tasks=tasks_by_seed[seed],
                               vocabulary=vocabulary, verbose=False))
    return _mean_over_seeds(runs)


def _table7_from_step2(path: Path) -> List[Dict]:
    """Assemble Table 7 from the Step 2 per-seed JSON."""
    if not path.exists():
        return []
    blob = read_json(path)
    per_task = blob["per_seed_task_f1"]
    per_avg = blob["per_seed_avg_f1"]

    layout = [
        ("Baseline-RoBERTa", "RoBERTa", "baseline"),
        ("SURGeLLM-G-RoBERTa", "RoBERTa", "surgellm-g"),
        ("SURGeLLM-FULL-RoBERTa", "RoBERTa", "surgellm-full"),
        ("SURGeLLM-IWN-RoBERTa", "RoBERTa", "surgellm-iwn"),
        ("Baseline-BERT", "BERT", "baseline"),
        ("SURGeLLM-S-BERT", "BERT", "surgellm-s"),
        ("SURGeLLM-IWN-BERT", "BERT", "surgellm-iwn"),
        ("Baseline-DistilBERT", "DistilBERT", "baseline"),
        ("SURGeLLM-S-DistilBERT", "DistilBERT", "surgellm-s"),
        ("SURGeLLM-FULL-ALBERT", "ALBERT", "surgellm-full"),
    ]
    base_by_backbone = {
        "RoBERTa": "Baseline-RoBERTa", "BERT": "Baseline-BERT",
        "DistilBERT": "Baseline-DistilBERT", "ALBERT": None,
    }
    rows: List[Dict] = []
    for display, bk, variant in layout:
        if display not in per_avg:
            continue
        f = variant_flags(variant)
        row = {"model": display, "source": "run", "backbone": bk,
               "P": int(f["prefix"]), "G": int(f["gate"]),
               "E": int(f["extended"]), "I": int(f["iwn"])}
        for tid in TASK_ORDER:
            vals = per_task[display].get(tid, [])
            row[tid] = round(float(np.mean(vals)), 4) if vals else float("nan")
        row["avg"] = round(float(np.mean(per_avg[display])), 4)
        base = base_by_backbone.get(bk)
        row["delta"] = (round(row["avg"] - float(np.mean(per_avg[base])), 4)
                        if base and base in per_avg and base != display else None)
        for tid in TASK_ORDER:
            if base and base in per_task and base != display:
                b = per_task[base].get(tid, [])
                row[f"delta_{tid}"] = (round(row[tid] - float(np.mean(b)), 4)
                                       if b else None)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--provider", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--loo-seeds", type=int, nargs="+", default=[0],
                    help="leave-one-out is 10 extra runs per seed")
    ap.add_argument("--scale", type=float, default=0.20)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["table7", "table8", "table9", "table10", "table11"])
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    demo = args.mode == "demo"
    provider = args.provider or ("synthetic" if demo else "hf")
    out = ensure_dir(Path(args.out) / "tables")
    logs = ensure_dir(Path(args.out) / "logs")
    t0 = time.time()

    all_seeds = sorted(set(args.seeds) | set(args.loo_seeds))
    tasks_by_seed = {s: load_all_tasks(provider=provider, seed=s, scale=args.scale)
                     for s in all_seeds}
    common = dict(demo=demo, scale=args.scale, provider=provider)

    # ---------------- Table 7 -------------------------------------------
    if "table7" not in args.skip:
        rows = _table7_from_step2(Path(args.out) / "logs" / "per_seed_raw.json")
        if rows:
            write_csv(rows, out / "table7_component_ablation.csv")
            print(f"[03] Table 7 assembled from Step 2 ({len(rows)} rows)")
        else:
            print("[03] Table 7 skipped: run scripts/02_main_table.py first")

    curated = default_vocabulary()

    # ---------------- Table 8: group-count sweep -------------------------
    if "table8" not in args.skip:
        print("[03] Table 8: group-count sweep |V| in {0,5,10,15,20}")
        t8: List[Dict] = []
        for n in [0, 5, 10, 15, 20]:
            if n == 0:
                vocab = SurgicalVocabulary([], name="none")
                res = _run_config(tasks_by_seed, args.seeds, variant="baseline",
                                  vocabulary=vocab, tag="V0", **common)
                note = "none, baseline"
            elif n <= 10:
                # Reducing: retain the most discriminative groups.  The paper
                # ranks by chi-squared on training data; GROUP_ORDER already
                # lists the curated groups in curation order, and the per-task
                # dominant groups of Table 11 are all inside the first ten.
                vocab = curated.subset(GROUP_ORDER[:n])
                res = _run_config(tasks_by_seed, args.seeds, vocabulary=vocab,
                                  tag=f"V{n}", **common)
                note = "ours" if n == 10 else ""
            else:
                # Increasing: add semantically redundant thesaurus-style
                # variants by duplicating existing groups under new names.
                extra = {}
                for i in range(n - 10):
                    src = curated.groups[i % len(curated.groups)]
                    extra[f"{src.name}_redundant{i}"] = list(src.terms)
                vocab = SurgicalVocabulary.from_dict(
                    {**curated.to_dict(), **extra}, name=f"curated+{n - 10}")
                res = _run_config(tasks_by_seed, args.seeds, vocabulary=vocab,
                                  tag=f"V{n}", **common)
                note = ""
            t8.append({"n_groups": n, "source": "run", "note": note, **res})
            print(f"     |V|={n:>2}  avg={res['avg']:.4f}")
        write_csv(t8, out / "table8_group_count.csv")

    # ---------------- Table 9: vocabulary control ------------------------
    if "table9" not in args.skip:
        print("[03] Table 9: random / auto-extracted / curated vocabulary")
        ref_tasks = tasks_by_seed[args.seeds[0]]
        all_texts = [t for td in ref_tasks.values() for t in td.splits.train.texts]

        rand_vocab = random_control_vocabulary(all_texts, curated, seed=0)
        auto = auto_extract_vocabulary(
            {tid: td.splits.train.texts for tid, td in ref_tasks.items()},
            {tid: td.splits.train.labels for tid, td in ref_tasks.items()},
            {tid: td.n_classes for tid, td in ref_tasks.items()},
            seed=0, use_sbert=True,
        )
        write_json({
            "embedding_backend": auto.embedding_backend,
            "n_candidates": auto.n_candidates,
            "cluster_sizes": auto.cluster_sizes,
            "groups": auto.vocabulary.to_dict(),
            "top_terms_per_class": auto.top_terms_per_class,
        }, logs / "autoextract_report.json")
        print(f"     auto-extraction: {auto.n_candidates} candidates, "
              f"backend={auto.embedding_backend}")

        t9: List[Dict] = []
        configs = [
            ("None (Baseline)", SurgicalVocabulary([], "none"), "baseline"),
            ("Random", rand_vocab, REF_VARIANT),
            ("Auto-extracted", auto.vocabulary, REF_VARIANT),
            ("Curated", curated, REF_VARIANT),
        ]
        by_name: Dict[str, Dict] = {}
        for name, vocab, variant in configs:
            res = _run_config(tasks_by_seed, args.seeds, variant=variant,
                              vocabulary=vocab, tag=name.split()[0], **common)
            by_name[name] = res
            t9.append({"vocab": name, "source": "run", **res})
            print(f"     {name:18s} avg={res['avg']:.4f}")

        cur = by_name["Curated"]
        for label, other in (("Delta Random", "Random"), ("Delta Auto", "Auto-extracted")):
            row = {"vocab": label, "source": "run"}
            for tid in TASK_ORDER + ["avg"]:
                row[tid] = round(by_name[other][tid] - cur[tid], 4)
            t9.append(row)
        t9.append({
            "vocab": "Auto / Curated recovery",
            "source": "run",
            "avg": round(by_name["Auto-extracted"]["avg"] / max(cur["avg"], 1e-9), 4),
        })
        write_csv(t9, out / "table9_vocab_control.csv")

    # ---------------- Table 10: surface-features ablation ----------------
    if "table10" not in args.skip:
        print("[03] Table 10: surface vs lexical features")
        t10: List[Dict] = []
        cfgs = [("G + S (full)", True, True), ("G only", False, True),
                ("S only", True, False)]
        by_cfg: Dict[str, Dict] = {}
        for name, surf, groups in cfgs:
            res = _run_config(tasks_by_seed, args.seeds, use_surface=surf,
                              use_groups=groups, tag=name.replace(" ", ""), **common)
            by_cfg[name] = res
            t10.append({"config": name, "source": "run", **res})
            print(f"     {name:14s} avg={res['avg']:.4f}")
        full = by_cfg["G + S (full)"]
        for label, other in (("Delta no-S", "G only"), ("Delta no-G", "S only")):
            row = {"config": label, "source": "run"}
            for tid in TASK_ORDER + ["avg"]:
                row[tid] = round(by_cfg[other][tid] - full[tid], 4)
            t10.append(row)
        write_csv(t10, out / "table10_surface_ablation.csv")

    # ---------------- Table 11: leave-one-out ----------------------------
    if "table11" not in args.skip:
        print("[03] Table 11: per-group leave-one-out (10 runs/seed)")
        full = _run_config(tasks_by_seed, args.loo_seeds, tag="LOOfull", **common)
        t11: List[Dict] = []
        for g in GROUP_ORDER:
            res = _run_config(tasks_by_seed, args.loo_seeds,
                              vocabulary=curated.drop(g), tag=f"LOO-{g}", **common)
            row = {"group_removed": g, "source": "run"}
            for tid in TASK_ORDER:
                row[tid] = round(res[tid] - full[tid], 4)
            row["avg"] = round(res["avg"] - full["avg"], 4)
            t11.append(row)
            print(f"     -{g:14s} " +
                  "  ".join(f"{tid}:{row[tid]:+.3f}" for tid in TASK_ORDER))

        # "Most important group per task" - the paper's bolded cells.
        dominant = {tid: min(t11, key=lambda r: r[tid])["group_removed"]
                    for tid in TASK_ORDER}
        for r in t11:
            r["dominant_for"] = ",".join(t for t, g in dominant.items()
                                         if g == r["group_removed"])
        write_csv(t11, out / "table11_leave_one_out.csv")
        write_json({"full_reference": full, "dominant_group_per_task": dominant},
                   logs / "leave_one_out_reference.json")
        print(f"     dominant groups: {dominant}")

    print(f"\n[03] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
