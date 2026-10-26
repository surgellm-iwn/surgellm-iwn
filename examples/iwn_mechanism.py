#!/usr/bin/env python3
"""The IWN mechanism in isolation — no model, no training, ~2 seconds.

This is the shortest honest demonstration of what §3.5 claims. It takes a
skewed corpus, standardises its surgical features both ways, and shows what
changes.

    python examples/iwn_mechanism.py

What you should see:

1. The marginal mean of Eq. (8) sits almost on top of the majority class's
   mean (Eq. 9), while the balanced mean of Eq. (10) sits between the two.
2. Under Eq. (8) the minority class is pushed into the tail; under Eq. (11)
   both classes land at comparable distances from the origin.
3. A *converged* linear probe is unaffected — an affine model absorbs an
   affine change of input. IWN is not about linear separability.
4. The gate is where it bites: under Eq. (8) the minority's projected
   features enter the gate at a different magnitude from the majority's, so
   the per-dimension mix the gate learns is tuned to the majority profile.
   Eq. (11) equalises them.
5. On a *balanced* corpus the two schemes coincide, so IWN costs nothing
   where it is not needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from surgellm.data.registry import load_task
from surgellm.eval.metrics import classification_metrics
from surgellm.features import SurgicalFeatureExtractor
from surgellm.normalization import FeatureNormalizer, prior_contamination


def _logistic_probe(X_tr, y_tr, X_te, steps=600, lr=0.3, seed=0):
    """A tiny logistic regression, so the only variable is the input scale."""
    rng = np.random.default_rng(seed)
    X_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    X_te = np.hstack([X_te, np.ones((len(X_te), 1))])
    w = rng.normal(0, 0.01, size=X_tr.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_tr @ w, -30, 30)))
        w -= lr * (X_tr.T @ (p - y_tr)) / len(y_tr)
    return (1.0 / (1.0 + np.exp(-np.clip(X_te @ w, -30, 30))) > 0.5).astype(int)


def main() -> int:
    print(__doc__.split("What you should see:")[0].strip())
    print("\n" + "=" * 72)

    td = load_task("D4", provider="synthetic", seed=0, scale=0.25)
    pool = td.normalizer_pool
    ex = SurgicalFeatureExtractor()
    s_pool = ex.extract_batch(pool.texts)

    std = FeatureNormalizer("standard").fit(s_pool, pool.labels, 2)
    iwn = FeatureNormalizer("iwn").fit(s_pool, pool.labels, 2)

    # ---- 1. Eq. (9) -----------------------------------------------------
    contam = prior_contamination(std)
    maj, mino = std.stats.per_class_mean[0], std.stats.per_class_mean[1]
    print(f"\n1. Prior contamination of the marginal moments  (Eq. 9)")
    print(f"   source pool            : {len(pool):,} texts, "
          f"skew {pool.skew():.2f}:1, majority prior {contam['majority_prior']:.3f}")
    print(f"   ||marginal - majority|| : "
          f"{np.linalg.norm(std.stats.mean - maj):8.4f}   <- Eq. (8) tracks the majority")
    print(f"   ||balanced - majority|| : "
          f"{np.linalg.norm(iwn.stats.mean - maj):8.4f}   <- Eq. (10) sits between classes")
    print(f"   ||balanced - minority|| : "
          f"{np.linalg.norm(iwn.stats.mean - mino):8.4f}")
    print(f"   normalised mean gap     : {contam['normalized_mean_gap']:8.4f}")

    # ---- 2. where each class lands -------------------------------------
    z_std, z_iwn = std.transform(s_pool), iwn.transform(s_pool)
    y = pool.labels
    print(f"\n2. Mean distance from the origin after standardisation")
    print(f"   {'':22s} {'majority':>10s} {'minority':>10s} {'ratio':>8s}")
    for name, z in (("Eq. (8)  standard", z_std), ("Eq. (11) IWN", z_iwn)):
        a = float(np.abs(z[y == 0]).mean())
        b = float(np.abs(z[y == 1]).mean())
        print(f"   {name:22s} {a:10.3f} {b:10.3f} {b / max(a, 1e-9):8.2f}")
    print("   (a ratio far from 1 means one class is treated as an outlier)")

    # ---- 3. the control: a converged linear probe sees no difference ----
    tr, te = td.splits.train, td.splits.test
    s_tr, s_te = ex.extract_batch(tr.texts), ex.extract_batch(te.texts)
    print(f"\n3. Control — a converged linear probe "
          f"({len(tr)} train / {len(te)} test, balanced)")
    print(f"   {'':22s} {'macro-F1':>9s} {'human R':>9s} {'llm R':>9s}")
    for name, norm in (("Eq. (8)  standard", std), ("Eq. (11) IWN", iwn)):
        pred = _logistic_probe(norm.transform(s_tr), tr.labels, norm.transform(s_te))
        m = classification_metrics(te.labels, pred, n_classes=2,
                                   class_names=("human", "llm"))
        print(f"   {name:22s} {m['macro_f1']:9.3f} "
              f"{m['recall_human']:9.3f} {m['recall_llm']:9.3f}")
    print("   These should be identical, and that is the point: a converged")
    print("   affine model absorbs any affine change of input. So whatever IWN")
    print("   does, it is not making the features linearly separable.")

    # ---- 4. where the effect actually lives: the gate -------------------
    import torch
    from surgellm.models.gate import SurgicalFeatureGate

    torch.manual_seed(0)
    d = 128
    gate = SurgicalFeatureGate(d, feature_dim=s_pool.shape[1])
    h = torch.randn(len(s_pool), d)        # encoder-like activations

    print("\n4. The same features through the gate of Eqs. (5)-(7), at init")
    print("   " + " " * 22 + f"{'|s_proj| maj':>13s}{'|s_proj| min':>13s}"
          f"{'ratio':>8s}{'gate gap':>10s}")
    for name, norm in (("Eq. (8)  standard", std), ("Eq. (11) IWN", iwn)):
        z = torch.tensor(norm.transform(s_pool), dtype=torch.float32)
        with torch.no_grad():
            s_proj = gate.project(z)                                  # Eq. (5)
            g = torch.sigmoid(gate.gate_proj(torch.cat([h, s_proj], -1)))  # Eq. (6)
        maj = float(s_proj[y == 0].norm(dim=-1).mean())
        mino = float(s_proj[y == 1].norm(dim=-1).mean())
        gap = float((g[y == 1].mean(0) - g[y == 0].mean(0)).abs().mean())
        print(f"   {name:22s} {maj:13.3f}{mino:13.3f}"
              f"{mino / max(maj, 1e-9):8.3f}{gap:10.4f}")
    print("   'ratio' is the minority/majority magnitude of s' entering the gate;")
    print("   'gate gap' is how differently the gate treats the two classes.")
    print("   Eq. (8) feeds the gate an off-centre, class-asymmetric input, so the")
    print("   per-dimension mix it learns is tuned to the majority profile.")
    print("   Eq. (11) equalises both, which is all §3.5 claims IWN does.")
    print("   (This is initialisation-time geometry. The downstream F1 effect")
    print("    needs training: see scripts/04_theory.py and table5_iwn_ablation.csv.)")

    # ---- 5. the balanced case ------------------------------------------
    td_bal = load_task("D1", provider="synthetic", seed=0, scale=0.25)
    s_bal = ex.extract_batch(td_bal.normalizer_pool.texts)
    yb = td_bal.normalizer_pool.labels
    a = FeatureNormalizer("standard").fit(s_bal, yb, 2)
    b = FeatureNormalizer("iwn").fit(s_bal, yb, 2)
    print(f"\n5. On a balanced corpus (D1, skew {td_bal.normalizer_pool.skew():.2f}:1)")
    print(f"   ||mean_standard - mean_iwn|| : {np.linalg.norm(a.stats.mean - b.stats.mean):.4f}")
    print(f"   mean |z| standard / IWN      : "
          f"{np.abs(a.transform(s_bal)).mean():.3f} / {np.abs(b.transform(s_bal)).mean():.3f}")
    print("   IWN is a strict generalisation: on balanced data it costs nothing.")

    print(f"\n   learnable parameters added by IWN: {iwn.n_parameters}")
    print(f"   labels required at inference     : none")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
