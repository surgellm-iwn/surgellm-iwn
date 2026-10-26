"""Proposition 2 (gate degeneracy under zero alignment) and Corollary 3.

Proposition 2.  If rho_k = 0 then at any local minimum of the regularised
loss with weight decay lambda > 0:

    1.  ||W_s|| -> 0 as training proceeds
    2.  s'(x)  -> 0 for all x
    3.  g*_i   -> 1 for all i in {1..d}
    4.  hhat   -> LN(htilde)

Corollary 3.  For any task with rho_k = 0, adding the gate cannot increase
the minimum achievable empirical risk: it strictly improves (rho_k > 0) or
degenerates to identity (rho_k = 0).

This is the one theorem in the paper that is *directly* verifiable by
simulation, because its hypothesis can be constructed: feed the gate
features that carry no label information and check all four conclusions.

Two experiments are provided:

``run_decay_simulation``
    The exact dynamics of the proof: under ``E[grad_{W_s} L] = 0`` and SGD
    with weight decay the update reduces to ``W_s <- (1 - lambda*eta) W_s``,
    pure exponential decay.  This is checked in closed form.

``run_zero_alignment_training``
    The empirical version: train the full model on a task whose surgical
    features are pure noise (label-independent), and track ``||W_s||``,
    ``||s'||``, ``mean(g)`` and ``||hhat - LN(htilde)||`` over epochs.

``imbalance_loophole_check`` documents the empirical caveat of §A.3: under
severe class skew, the *measured* rho_k on the prior-contaminated
distribution can be misleadingly low even when the underlying feature
signal is informative, which is the D4-without-IWN failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.cache import TokenizedSplit
from ..models.surgellm import SurgeLLM

__all__ = ["DecayTrace", "run_decay_simulation", "degeneracy_diagnostics",
           "check_proposition2", "imbalance_loophole_check"]


@dataclass
class DecayTrace:
    steps: np.ndarray
    norms: np.ndarray
    predicted: np.ndarray
    max_abs_error: float
    half_life_steps: float

    def to_rows(self) -> List[Dict]:
        return [
            {"step": int(s), "Ws_norm": float(n), "predicted_norm": float(p)}
            for s, n, p in zip(self.steps, self.norms, self.predicted)
        ]


def run_decay_simulation(d: int = 64, feature_dim: int = 16, lr: float = 1e-2,
                         weight_decay: float = 1e-2, steps: int = 400,
                         seed: int = 0) -> DecayTrace:
    """Closed-form check of proof step 1: ``W_s <- (1 - lambda*eta) W_s``.

    We drive the optimiser with a loss whose gradient w.r.t. ``W_s`` is
    exactly zero in expectation (the rho_k = 0 hypothesis), leaving weight
    decay as the only force, and compare the observed norm trajectory with
    the predicted geometric decay ``||W_s(0)|| (1 - lambda*eta)^t``.
    """
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(d, feature_dim) * 0.1)
    opt = torch.optim.SGD([W], lr=lr, weight_decay=weight_decay)

    n0 = float(W.detach().norm())
    norms, preds = [], []
    for t in range(steps):
        opt.zero_grad()
        # Zero-alignment: the data term contributes no gradient.
        (0.0 * W.sum()).backward()
        opt.step()
        norms.append(float(W.detach().norm()))
        preds.append(n0 * (1.0 - weight_decay * lr) ** (t + 1))

    norms_a, preds_a = np.array(norms), np.array(preds)
    rate = 1.0 - weight_decay * lr
    half_life = float(np.log(0.5) / np.log(rate)) if 0 < rate < 1 else float("inf")
    return DecayTrace(
        steps=np.arange(1, steps + 1), norms=norms_a, predicted=preds_a,
        max_abs_error=float(np.max(np.abs(norms_a - preds_a))),
        half_life_steps=half_life,
    )


@torch.no_grad()
def degeneracy_diagnostics(model: SurgeLLM, split: TokenizedSplit,
                           task_index: int, n_examples: int = 256,
                           device: Optional[torch.device] = None) -> Dict[str, float]:
    """Measure all four Proposition 2 conclusions on a trained model."""
    device = device or next(model.parameters()).device
    n = min(int(n_examples), len(split))
    sel = torch.arange(n)
    ids = split.input_ids[sel].to(device)
    am = split.attention_mask[sel].to(device)
    s = split.features[sel].to(device)
    t = torch.full((n,), task_index, dtype=torch.long, device=device)

    h = model.backbone.encode(ids, am)
    h_tilde = model.task_embedding(h, t)
    out = model.gate(h_tilde, s)
    identity = model.gate.post_norm(h_tilde)

    return {
        # (1) ||W_s||
        "Ws_fro_norm": float(model.gate.feature_proj.weight.norm()),
        "Ws_spectral_norm": model.gate.spectral_norm_Ws(),
        # (2) s'(x) -> 0
        "s_proj_norm_mean": float(out.projected.norm(dim=-1).mean()),
        "s_proj_norm_max": float(out.projected.norm(dim=-1).max()),
        "s_proj_rel_to_h": float(
            (out.projected.norm(dim=-1) / h_tilde.norm(dim=-1).clamp(min=1e-9)).mean()
        ),
        # (3) g_i -> 1
        "gate_mean": float(out.gate.mean()),
        "gate_min": float(out.gate.min()),
        "gate_frac_above_0.9": float((out.gate > 0.9).float().mean()),
        "gate_frac_above_0.99": float((out.gate > 0.99).float().mean()),
        # (4) hhat -> LN(htilde)
        "identity_gap_abs": float((out.fused - identity).norm(dim=-1).mean()),
        "identity_gap_rel": float(
            ((out.fused - identity).norm(dim=-1)
             / identity.norm(dim=-1).clamp(min=1e-9)).mean()
        ),
        "n_examples": float(n),
    }


def check_proposition2(before: Dict[str, float], after: Dict[str, float],
                       tol_gate: float = 0.75, tol_rel_gap: float = 0.5
                       ) -> Dict[str, object]:
    """Turn the four conclusions into pass/fail assertions.

    The thresholds are deliberately loose: Proposition 2 is an asymptotic
    statement about the limit of training, so a finite run is checked for
    *movement in the predicted direction*, not for exact convergence.
    """
    checks = {
        "c1_Ws_norm_decreased": after["Ws_fro_norm"] < before["Ws_fro_norm"],
        "c2_s_proj_shrank": after["s_proj_norm_mean"] < before["s_proj_norm_mean"],
        "c3_gate_moved_toward_one": after["gate_mean"] > before["gate_mean"],
        "c3_gate_above_tol": after["gate_mean"] > tol_gate,
        "c4_identity_gap_shrank": after["identity_gap_rel"] < before["identity_gap_rel"],
        "c4_identity_gap_small": after["identity_gap_rel"] < tol_rel_gap,
    }

    # Conclusions 1 and 2 are the load-bearing ones: they are what the proof
    # actually derives (zero expected gradient + weight decay => exponential
    # decay of W_s, hence s' -> 0), and they are what Corollary 3 needs.
    #
    # Conclusion 3 (g_i -> 1) is weaker in practice.  Weight decay shrinks
    # W_g too, so the gate pre-activation decays and g -> sigma(0) = 0.5
    # rather than 1.  Once s' = 0 there is no loss gradient pushing g
    # anywhere: every g yields an equally usable LN(g (*) htilde) for the
    # head, so g is unconstrained rather than driven to 1.  We therefore
    # report conclusions 1-2 and 3-4 separately instead of collapsing them
    # into a single verdict.
    strong = bool(checks["c1_Ws_norm_decreased"] and checks["c2_s_proj_shrank"])
    shrink_ratio = after["s_proj_norm_mean"] / max(before["s_proj_norm_mean"], 1e-12)

    return {
        **{k: bool(v) for k, v in checks.items()},
        "all_directional_checks_pass": bool(
            checks["c1_Ws_norm_decreased"] and checks["c2_s_proj_shrank"]
            and checks["c3_gate_moved_toward_one"] and checks["c4_identity_gap_shrank"]
        ),
        "core_conclusions_1_2_pass": strong,
        "s_proj_shrink_ratio": float(shrink_ratio),
        "Ws_shrink_ratio": float(
            after["Ws_fro_norm"] / max(before["Ws_fro_norm"], 1e-12)),
        "delta_Ws_norm": after["Ws_fro_norm"] - before["Ws_fro_norm"],
        "delta_gate_mean": after["gate_mean"] - before["gate_mean"],
        "delta_identity_gap_rel": after["identity_gap_rel"] - before["identity_gap_rel"],
        "note_c3": ("g -> sigma(0) = 0.5 when weight decay also shrinks W_g; "
                    "once s' = 0 the loss does not constrain g, and Corollary 3 "
                    "follows from conclusion 2 alone"),
    }


def imbalance_loophole_check(rho_pre_iwn: float, rho_post_iwn: float,
                             f1_pre_iwn: float, f1_post_iwn: float
                             ) -> Dict[str, object]:
    """The §A.3 "empirical caveat: the imbalance loophole".

    Corollary 3 assumes rho_k faithfully captures gradient-feature
    alignment *under the distribution the gate sees*.  Under severe skew,
    Eq. (8) feeds the gate prior-contaminated features, so the measured
    rho_k is low even when the underlying signal is informative - and the
    gate then *hurts*, apparently violating the corollary.  IWN restores
    the conditions.  The diagnostic signature is: rho rises and F1 rises
    together, with the pre-IWN rho low.
    """
    return {
        "rho_pre_iwn": float(rho_pre_iwn),
        "rho_post_iwn": float(rho_post_iwn),
        "rho_increase": float(rho_post_iwn - rho_pre_iwn),
        "rho_ratio": float(rho_post_iwn / max(rho_pre_iwn, 1e-9)),
        "f1_increase": float(f1_post_iwn - f1_pre_iwn),
        "loophole_signature": bool(
            rho_pre_iwn < rho_post_iwn and f1_post_iwn > f1_pre_iwn
        ),
        "interpretation": (
            "IWN raises measured alignment and F1 together: the gate was "
            "architecturally sound but fed prior-contaminated features."
        ),
    }
