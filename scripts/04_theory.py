#!/usr/bin/env python3
"""Step 4 - theory verification.

``table15_rho_estimates.csv``       Definition 1 / Appendix G: rho_k with and
                                    without IWN, per task
``theorem1_bound.csv``              Eq. (16) terms and whether the bound holds
``lemma4_lipschitz.csv``            composed Lipschitz constant vs empirical
``proposition2_degeneracy.csv``     the four conclusions, before vs after
``proposition2_decay_trace.csv``    closed-form ``W_s <- (1-lambda*eta) W_s``
``corollary3_safety.csv``           gate cannot hurt at rho_k = 0
``imbalance_loophole.csv``          §A.3 empirical caveat
``alignment_vs_gain.csv``           does the gain ordering track rho_k?

The Proposition 2 experiment constructs its own hypothesis: a task whose
surgical features are label-independent noise, i.e. rho_k = 0 by
construction.  That is the only theorem here whose premise can be
manufactured, and therefore the only one that is verified rather than
merely measured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from surgellm.data.registry import load_all_tasks
from surgellm.data.schema import TASK_ORDER, TASKS
from surgellm.experiment import RunSpec, run_single
from surgellm.theory.alignment import (alignment_gain_correlation,
                                       estimate_rho_all_tasks)
from surgellm.theory.bound import estimate_lipschitz, evaluate_bound
from surgellm.theory.degeneracy import (check_proposition2, degeneracy_diagnostics,
                                        imbalance_loophole_check,
                                        run_decay_simulation)
from surgellm.utils.io import ensure_dir, write_csv, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--provider", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=0.20)
    ap.add_argument("--mc-examples", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    demo = args.mode == "demo"
    provider = args.provider or ("synthetic" if demo else "hf")
    out = ensure_dir(Path(args.out) / "tables")
    logs = ensure_dir(Path(args.out) / "logs")
    t0 = time.time()

    tasks = load_all_tasks(provider=provider, seed=args.seed, scale=args.scale)
    common = dict(backbone="roberta", seed=args.seed, provider=provider,
                  offline_backbone=demo, scale=args.scale,
                  profile="demo" if demo else "paper")

    # ================= rho_k with and without IWN =======================
    print("[04] training SURGeLLM-FULL (no IWN) and SURGeLLM-IWN for rho_k")
    art_no_iwn = run_single(RunSpec(variant="surgellm-full", tag="theory", **common),
                            tasks=tasks, verbose=False)
    art_iwn = run_single(RunSpec(variant="surgellm-iwn", tag="theory", **common),
                         tasks=tasks, verbose=False)

    rho_no = estimate_rho_all_tasks(art_no_iwn.model, art_no_iwn.tensors["val"],
                                    TASK_ORDER, n_examples=args.mc_examples)
    rho_iwn = estimate_rho_all_tasks(art_iwn.model, art_iwn.tensors["val"],
                                     TASK_ORDER, n_examples=args.mc_examples)

    t15: List[Dict] = []
    for tid in TASK_ORDER:
        if tid not in rho_no:
            continue
        t15.append({
            "task": tid, "name": TASKS[tid].source, "source": "run",
            "rho_no_iwn": round(rho_no[tid]["rho"], 4),
            "rho_no_iwn_sem": round(rho_no[tid]["rho_sem"], 4),
            "rho_iwn": round(rho_iwn[tid]["rho"], 4),
            "rho_iwn_sem": round(rho_iwn[tid]["rho_sem"], 4),
            "rho_increase": round(rho_iwn[tid]["rho"] - rho_no[tid]["rho"], 4),
            "f1_no_iwn": round(art_no_iwn.result.test_metrics[tid]["macro_f1"], 4),
            "f1_iwn": round(art_iwn.result.test_metrics[tid]["macro_f1"], 4),
            "f1_gain": round(art_iwn.result.test_metrics[tid]["macro_f1"]
                             - art_no_iwn.result.test_metrics[tid]["macro_f1"], 4),
            "n_mc_examples": rho_no[tid]["n_examples"],
        })
    write_csv(t15, out / "table15_rho_estimates.csv")
    print("[04] rho_k:", {r["task"]: (r["rho_no_iwn"], r["rho_iwn"]) for r in t15})

    corr = alignment_gain_correlation(
        {r["task"]: r["rho_iwn"] for r in t15},
        {r["task"]: r["f1_gain"] for r in t15},
    )
    ranked_rho = sorted(t15, key=lambda r: -r["rho_iwn"])
    write_csv([{
        "source": "run", **corr,
        "rho_rank_order": " > ".join(r["task"] for r in ranked_rho),
        "gain_rank_order": " > ".join(
            r["task"] for r in sorted(t15, key=lambda r: -r["f1_gain"])),
        "orderings_identical": (
            [r["task"] for r in ranked_rho]
            == [r["task"] for r in sorted(t15, key=lambda r: -r["f1_gain"])]
        ),
    }], out / "alignment_vs_gain.csv")

    # ================= Theorem 1 ========================================
    print("[04] Theorem 1: evaluating Eq. (16) terms")
    lip = estimate_lipschitz(art_iwn.model, art_iwn.tensors["test"]["D1"], 0)
    write_csv([{"source": "run", "model": "SURGeLLM-IWN", **lip.to_dict()}],
              out / "lemma4_lipschitz.csv")

    t1_rows: List[Dict] = []
    for i, tid in enumerate(TASK_ORDER):
        if tid not in art_iwn.tensors["test"]:
            continue
        acc = art_iwn.result.test_metrics[tid]["accuracy"]
        bt = evaluate_bound(
            art_iwn.model, art_iwn.tensors["test"][tid], tid, i,
            n_train=len(art_iwn.tensors["train"][tid]),
            test_risk=1.0 - acc, bayes_risk_proxy=0.0,
        )
        t1_rows.append({"source": "run", **{
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in bt.to_dict().items()}})
        print(f"     {tid}: excess={bt.excess_risk_proxy:.4f} "
              f"bound={bt.bound_total:.4f} holds={bt.bound_holds}")
    write_csv(t1_rows, out / "theorem1_bound.csv")

    # ================= Proposition 2 ====================================
    print("[04] Proposition 2: closed-form weight-decay trace")
    trace = run_decay_simulation(d=64, feature_dim=16, lr=1e-2,
                                 weight_decay=1e-2, steps=400)
    write_csv(trace.to_rows()[::10], out / "proposition2_decay_trace.csv")
    print(f"     max |observed - predicted| = {trace.max_abs_error:.3e} "
          f"(half-life {trace.half_life_steps:.0f} steps)")

    print("[04] Proposition 2: zero-alignment training run")
    # Construct the rho_k = 0 hypothesis: replace every surgical feature with
    # label-independent Gaussian noise, so s' carries no information about y.
    rng = np.random.default_rng(args.seed)
    art_zero = run_single(RunSpec(variant="surgellm-iwn", tag="zeroalign", **common),
                          tasks=tasks, verbose=False)
    zero_tensors = art_zero.tensors
    for split in zero_tensors.values():
        for sp in split.values():
            sp.features = torch.tensor(
                rng.normal(size=tuple(sp.features.shape)), dtype=torch.float32)

    from surgellm.train import TrainConfig, Trainer
    from surgellm.experiment import hparams_for
    hp = hparams_for("surgellm-iwn", "roberta", profile="demo" if demo else "paper")

    before = degeneracy_diagnostics(art_zero.model, zero_tensors["test"]["D1"], 0)
    rho_before = estimate_rho_all_tasks(art_zero.model, {"D1": zero_tensors["val"]["D1"]},
                                        TASK_ORDER, n_examples=256)["D1"]["rho"]

    # Train with a deliberately strong weight decay: Proposition 2 is a
    # statement about the limit of training under weight decay lambda > 0,
    # and a finite run needs a visible decay rate to show the direction.
    cfg = TrainConfig(learning_rate=hp["lr"], max_epochs=hp["ep"],
                      batch_size=hp["bs"], grad_accumulation=hp["ga"],
                      weight_decay=0.2, seed=args.seed, device="cpu")
    Trainer(art_zero.model, cfg, task_order=TASK_ORDER).fit(zero_tensors, verbose=False)
    after = degeneracy_diagnostics(art_zero.model, zero_tensors["test"]["D1"], 0)
    rho_after = estimate_rho_all_tasks(art_zero.model, {"D1": zero_tensors["val"]["D1"]},
                                       TASK_ORDER, n_examples=256)["D1"]["rho"]

    checks = check_proposition2(before, after)
    rows = [{"source": "run", "phase": "before", "rho_D1": round(rho_before, 5),
             **{k: round(v, 6) for k, v in before.items()}},
            {"source": "run", "phase": "after", "rho_D1": round(rho_after, 5),
             **{k: round(v, 6) for k, v in after.items()}}]
    for k, v in checks.items():
        rows.append({"source": "run", "phase": "check", "check": k, "value": v})
    write_csv(rows, out / "proposition2_degeneracy.csv")
    print(f"     ||W_s||: {before['Ws_fro_norm']:.4f} -> {after['Ws_fro_norm']:.4f}")
    print(f"     mean gate g: {before['gate_mean']:.4f} -> {after['gate_mean']:.4f}")
    print(f"     ||hhat - LN(htilde)|| (rel): {before['identity_gap_rel']:.4f} "
          f"-> {after['identity_gap_rel']:.4f}")
    print(f"     all directional checks pass: {checks['all_directional_checks_pass']}")

    # ================= Corollary 3 ======================================
    # "Adding the gate cannot increase the minimum achievable empirical risk"
    # at rho_k = 0: compare a gated model against its own identity path.
    corollary = {
        "source": "run",
        "gate_mean_after_decay": round(after["gate_mean"], 4),
        "identity_gap_rel_after_decay": round(after["identity_gap_rel"], 4),
        "s_proj_norm_after_decay": round(after["s_proj_norm_mean"], 4),
        "Ws_norm_after_decay": round(after["Ws_fro_norm"], 4),
        "degenerates_toward_identity": bool(checks["all_directional_checks_pass"]),
        "note": ("Corollary 3 is asymptotic; a finite run is checked for "
                 "movement toward the identity map, not exact convergence."),
    }
    write_csv([corollary], out / "corollary3_safety.csv")

    # ================= imbalance loophole (§A.3) ========================
    d4 = next((r for r in t15 if r["task"] == "D4"), None)
    if d4:
        loop = imbalance_loophole_check(d4["rho_no_iwn"], d4["rho_iwn"],
                                        d4["f1_no_iwn"], d4["f1_iwn"])
        write_csv([{"source": "run", "task": "D4", **loop}],
                  out / "imbalance_loophole.csv")
        print(f"[04] imbalance loophole signature on D4: {loop['loophole_signature']} "
              f"(rho {loop['rho_pre_iwn']:.2f} -> {loop['rho_post_iwn']:.2f}, "
              f"F1 {loop['f1_increase']:+.3f})")

    write_json({"decay_max_abs_error": trace.max_abs_error,
                "decay_half_life_steps": trace.half_life_steps,
                "proposition2_checks": checks,
                "alignment_vs_gain": corr,
                "minutes": round((time.time() - t0) / 60, 2)},
               logs / "04_theory.json")
    print(f"\n[04] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
