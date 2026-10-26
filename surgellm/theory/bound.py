"""Theorem 1 (gate approximation bound) and Lemma 4 (Lipschitz composition).

Theorem 1.  With probability >= 1 - delta over the draw of D^tr_k,

    R(f_theta) - R(f*)  <=     C / sqrt(N_k)                       [generalization]
                             + lambda_max(W_g^T W_g) / 2 * ||s' - s*||^2   [approximation]

with ``C = O(L_phi L_head rho sqrt(log(1/delta)))``.

Lemma 4.  ``L_theta <= L_phi * L_G * L_head``.

What is and is not checkable
----------------------------
The bound is an *upper* bound on an unobservable quantity (the Bayes risk
``R(f*)`` is unknown, and ``s*`` is an oracle).  So this module does three
honest things rather than pretending to "verify" the theorem:

1. **Computes every term that is computable** - the Rademacher-style
   generalization term from measured Lipschitz constants, and the
   approximation term for a surrogate oracle ``s*`` obtained by ridge
   regression of the score direction onto the feature space.
2. **Checks the inequality against an empirical risk proxy**, using the
   test risk of the best achievable head as a stand-in for ``R(f*)``, and
   reports whether the bound holds and how loose it is.
3. **Checks the monotonicity the theorem predicts**: the approximation
   term scales with ``lambda_max(W_g^T W_g)`` and with ``||s' - s*||^2``,
   and the generalization term with ``1/sqrt(N_k)``.  These are testable
   by construction and are what ``tests/test_theory_bound.py`` asserts.

The paper also notes the bound is standard Rademacher complexity and may
be loose for modern transformers (Limitations, §8); the slack we report
quantifies exactly that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.cache import TokenizedSplit
from ..models.surgellm import SurgeLLM

__all__ = ["LipschitzReport", "BoundTerms", "estimate_lipschitz",
           "oracle_feature_vector", "evaluate_bound", "sigmoid_sup_derivative"]

#: sup_z sigma'(z) = 1/4, used when propagating through the bilinear gate.
SIGMOID_SUP_DERIVATIVE = 0.25


def sigmoid_sup_derivative() -> float:
    return SIGMOID_SUP_DERIVATIVE


@dataclass
class LipschitzReport:
    """Lemma 4 constants, measured on the trained model."""

    L_encoder: float
    L_gate: float
    L_head: float
    L_composed: float
    empirical_L_composed: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "L_encoder": self.L_encoder, "L_gate": self.L_gate,
            "L_head": self.L_head, "L_composed_bound": self.L_composed,
            "L_composed_empirical": (
                float("nan") if self.empirical_L_composed is None
                else self.empirical_L_composed
            ),
            "bound_holds": (
                True if self.empirical_L_composed is None
                else bool(self.empirical_L_composed <= self.L_composed + 1e-6)
            ),
        }


@torch.no_grad()
def _head_lipschitz(model: SurgeLLM, task_index: int) -> float:
    head = model.heads.heads[task_index]
    s1 = float(torch.linalg.svdvals(head.fc1.weight.float())[0])
    s2 = float(torch.linalg.svdvals(head.fc2.weight.float())[0])
    return s1 * s2          # GELU is 1-Lipschitz up to a constant ~1.13; use 1


@torch.no_grad()
def _encoder_lipschitz_proxy(model: SurgeLLM) -> float:
    """Spectral-norm product proxy for ``L_phi``.

    An exact Lipschitz constant for a transformer is intractable; the
    standard proxy is the product of layer spectral norms, which is what
    the Rademacher argument of Appendix C.1 effectively uses.  Reported as
    a log-sum to avoid overflow, then exponentiated with a cap.
    """
    log_prod = 0.0
    for name, p in model.backbone.named_parameters():
        if p.ndim == 2 and "embed" not in name:
            log_prod += float(torch.linalg.svdvals(p.detach().float())[0].clamp(min=1e-8).log())
    return float(math.exp(min(log_prod, 50.0)))


def estimate_lipschitz(model: SurgeLLM, split: Optional[TokenizedSplit] = None,
                       task_index: int = 0, n_pairs: int = 200,
                       seed: int = 0) -> LipschitzReport:
    """Lemma 4 constants plus an empirical finite-difference lower bound.

    The empirical constant is the max observed ratio
    ``||yhat - yhat'|| / ||hhat - hhat'||`` over random example pairs,
    taken in representation space (input space has no useful metric for
    token ids).  A bound that is violated here would falsify Lemma 4's
    composition for this model.
    """
    L_phi = _encoder_lipschitz_proxy(model)
    L_gate = model.gate.lipschitz_constant()
    L_head = _head_lipschitz(model, task_index)
    composed = L_phi * L_gate * L_head

    empirical = None
    if split is not None and len(split) >= 2:
        device = next(model.parameters()).device
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(split), size=(n_pairs, 2))
        ratios = []
        with torch.no_grad():
            for a, b in idx:
                if a == b:
                    continue
                sel = torch.tensor([int(a), int(b)])
                ids = split.input_ids[sel].to(device)
                am = split.attention_mask[sel].to(device)
                s = split.features[sel].to(device)
                t = torch.full((2,), task_index, dtype=torch.long, device=device)
                out = model.encode(ids, am, t, s)
                logits = model.heads(out.fused, t)
                logits = torch.nan_to_num(logits, neginf=0.0)
                dh = float((out.fused[0] - out.fused[1]).norm())
                dy = float((logits[0] - logits[1]).norm())
                if dh > 1e-6:
                    ratios.append(dy / dh)
        if ratios:
            empirical = float(np.max(ratios))

    return LipschitzReport(L_phi, L_gate, L_head, composed, empirical)


@dataclass
class BoundTerms:
    task: str
    n_train: int
    generalization_term: float
    approximation_term: float
    bound_total: float
    excess_risk_proxy: float
    bound_holds: bool
    slack: float
    lambda_max_WgTWg: float
    feature_gap_sq: float
    C_constant: float
    delta: float

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@torch.no_grad()
def oracle_feature_vector(model: SurgeLLM, split: TokenizedSplit, task_index: int,
                          ridge: float = 1e-2, n_examples: int = 512,
                          device: Optional[torch.device] = None) -> Dict[str, float]:
    """A tractable surrogate for the oracle ``s*`` of Theorem 1.

    ``s*`` is defined as the projected feature vector minimising the gate
    approximation error.  We approximate it per task by the ridge-regression
    solution that best predicts the *score direction* from the standardised
    features, then report ``||s' - s*||^2`` averaged over examples.  This is
    a surrogate, not the true oracle, so the reported approximation term is
    an estimate of the theorem's second term, not a certificate.
    """
    device = device or next(model.parameters()).device
    n = min(int(n_examples), len(split))
    sel = torch.arange(n)
    ids = split.input_ids[sel].to(device)
    am = split.attention_mask[sel].to(device)
    s = split.features[sel].to(device)
    y = split.labels[sel].to(device)
    t = torch.full((n,), task_index, dtype=torch.long, device=device)

    h = model.backbone.encode(ids, am)
    h_tilde = model.task_embedding(h, t)
    s_proj = model.gate.project(s)                               # s'

    with torch.enable_grad():
        h_hat = model.gate(h_tilde, s).fused.detach().clone().requires_grad_(True)
        logits = model.heads(h_hat, t)
        n_c = int((logits[0] > float("-inf")).sum())
        lp = F.log_softmax(logits[:, :n_c].float(), -1).gather(1, y.view(-1, 1)).sum()
        target = torch.autograd.grad(lp, h_hat)[0]               # score direction

    # Ridge: find the linear map of s onto the score direction; its image is
    # the best feature-expressible approximation of the useful direction.
    X = s.float()
    Y = target.float()
    XtX = X.T @ X + ridge * torch.eye(X.size(1), device=device)
    W = torch.linalg.solve(XtX, X.T @ Y)                         # (16, d)
    s_star = X @ W
    # Scale-match: the gate only needs the direction up to the gate's own
    # magnitude, so compare after matching the mean norm of s'.
    scale = (s_proj.norm(dim=-1).mean() / s_star.norm(dim=-1).mean().clamp(min=1e-9))
    s_star = s_star * scale

    gap = ((s_proj.float() - s_star) ** 2).sum(-1)
    return {
        "feature_gap_sq_mean": float(gap.mean()),
        "feature_gap_sq_max": float(gap.max()),
        "s_proj_norm_mean": float(s_proj.norm(dim=-1).mean()),
        "s_star_norm_mean": float(s_star.norm(dim=-1).mean()),
        "n_examples": n,
    }


def evaluate_bound(model: SurgeLLM, split: TokenizedSplit, task: str,
                   task_index: int, n_train: int, test_risk: float,
                   bayes_risk_proxy: float = 0.0, delta: float = 0.05,
                   lipschitz: Optional[LipschitzReport] = None,
                   rho_loss_lipschitz: float = 1.0) -> BoundTerms:
    """Assemble and check Eq. (16) for one task.

    ``test_risk`` is the measured 0-1 risk of ``f_theta``; ``bayes_risk_proxy``
    stands in for ``R(f*)`` (0 by default, which makes the check
    conservative: a bound that holds against ``R(f_theta) - 0`` certainly
    holds against ``R(f_theta) - R(f*) >= 0``).
    """
    lip = lipschitz or estimate_lipschitz(model, split, task_index)
    oracle = oracle_feature_vector(model, split, task_index)
    lam = model.gate.spectral_norm_Wg()

    # C = O(L_phi L_head rho sqrt(log(1/delta))); we take the O(.) as 1.
    C = lip.L_encoder * lip.L_head * rho_loss_lipschitz * math.sqrt(math.log(1.0 / delta))
    gen = C / math.sqrt(max(n_train, 1))
    approx = 0.5 * lam * oracle["feature_gap_sq_mean"]
    total = gen + approx
    excess = max(test_risk - bayes_risk_proxy, 0.0)

    return BoundTerms(
        task=task, n_train=int(n_train),
        generalization_term=float(gen), approximation_term=float(approx),
        bound_total=float(total), excess_risk_proxy=float(excess),
        bound_holds=bool(excess <= total), slack=float(total - excess),
        lambda_max_WgTWg=float(lam),
        feature_gap_sq=float(oracle["feature_gap_sq_mean"]),
        C_constant=float(C), delta=float(delta),
    )
