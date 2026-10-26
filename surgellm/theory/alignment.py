"""Definition 1: surgical feature alignment rho_k, and its Monte Carlo
estimator (Appendix G).

    rho_k = E_{(x,y) ~ P_{t_k}} [ | <s'(x), grad_{hhat} log p(y | hhat)> | ]

Interpretation (§A.1): rho_k measures how much the lexical-feature
direction ``s'`` contributes useful gradient signal.  rho_k = 0 means
``s'`` is orthogonal in expectation to the score function, which is the
hypothesis of Proposition 2.

Appendix G estimates it by Monte Carlo on 1,000 validation examples per
task.  The paper's reported estimates (Table 15):

    ==========  ===============  ===========
    Task        rho_k (no IWN)   rho_k (IWN)
    ==========  ===============  ===========
    D1 SST-2            1.42         1.39
    D2 HotPot           3.71         3.68
    D3 LLM-7            1.83         1.85
    D4 HumLLM           0.61         2.13
    ==========  ===============  ===========

The claim this module makes testable is the *ordering*, not the absolute
magnitudes: the task with the highest rho_k gains most from the gate, and
D4's rho_k rises sharply once IWN removes the prior contamination.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..data.cache import TokenizedSplit
from ..models.surgellm import SurgeLLM

__all__ = ["estimate_rho", "estimate_rho_all_tasks", "MC_EXAMPLES"]

MC_EXAMPLES = 1000


def estimate_rho(model: SurgeLLM, split: TokenizedSplit, task_index: int,
                 n_examples: int = MC_EXAMPLES, batch_size: int = 32,
                 device: Optional[torch.device] = None,
                 seed: int = 0) -> Dict[str, float]:
    """Monte Carlo estimate of rho_k on one task split.

    For each example we backpropagate ``log p(y | hhat)`` to ``hhat``,
    obtaining the score-function gradient, and take the absolute inner
    product with the projected feature vector ``s'(x)``.
    """
    device = device or next(model.parameters()).device
    model.eval()

    n = min(int(n_examples), len(split))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(split))[:n]

    values: List[float] = []
    cos_values: List[float] = []
    for start in range(0, n, batch_size):
        sel = torch.tensor(idx[start:start + batch_size], dtype=torch.long)
        input_ids = split.input_ids[sel].to(device)
        attn = split.attention_mask[sel].to(device)
        s = split.features[sel].to(device)
        y = split.labels[sel].to(device)
        tids = torch.full((len(sel),), task_index, dtype=torch.long, device=device)

        with torch.no_grad():
            h = model.backbone.encode(input_ids, attn)
            h_tilde = model.task_embedding(h, tids)
            s_proj = model.gate.project(s)                     # s'(x), Eq. (5)
            gate_out = model.gate(h_tilde, s)

        # grad_{hhat} log p(y | hhat): hhat is the only leaf we differentiate.
        h_hat = gate_out.fused.detach().clone().requires_grad_(True)
        logits = model.heads(h_hat, tids)
        n_c = int((logits[0] > float("-inf")).sum())
        log_p = F.log_softmax(logits[:, :n_c].float(), dim=-1)
        selected = log_p.gather(1, y.view(-1, 1)).sum()
        grad = torch.autograd.grad(selected, h_hat)[0]         # (B, d)

        inner = (s_proj.detach().float() * grad.float()).sum(-1).abs()
        values.extend(inner.cpu().numpy().tolist())

        denom = (s_proj.detach().float().norm(dim=-1) * grad.float().norm(dim=-1))
        cos = (s_proj.detach().float() * grad.float()).sum(-1).abs() / denom.clamp(min=1e-12)
        cos_values.extend(cos.cpu().numpy().tolist())

    arr = np.asarray(values, dtype=float)
    return {
        "rho": float(arr.mean()),
        "rho_std": float(arr.std()),
        "rho_sem": float(arr.std() / max(np.sqrt(len(arr)), 1.0)),
        "mean_abs_cosine": float(np.mean(cos_values)) if cos_values else float("nan"),
        "n_examples": int(len(arr)),
    }


def estimate_rho_all_tasks(model: SurgeLLM, splits: Dict[str, TokenizedSplit],
                           task_order: Sequence[str], n_examples: int = MC_EXAMPLES,
                           seed: int = 0) -> Dict[str, Dict[str, float]]:
    order = list(task_order)
    return {
        tid: estimate_rho(model, sp, order.index(tid), n_examples=n_examples, seed=seed)
        for tid, sp in splits.items() if tid in order
    }


def alignment_gain_correlation(rho_by_task: Dict[str, float],
                               gain_by_task: Dict[str, float]) -> Dict[str, float]:
    """Does the empirical gain ordering track the alignment ordering?

    §8 claims it does "exactly".  We report both Pearson r and Spearman
    rho over the four tasks; with n = 4 the rank statistic is the honest
    summary and the p-value is not meaningful.
    """
    keys = [k for k in rho_by_task if k in gain_by_task]
    if len(keys) < 2:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "n": len(keys)}
    x = np.array([rho_by_task[k] for k in keys], dtype=float)
    y = np.array([gain_by_task[k] for k in keys], dtype=float)

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a * b).sum() / d) if d > 0 else float("nan")

    return {
        "pearson_r": _corr(x, y),
        "spearman_rho": _corr(_rank(x), _rank(y)),
        "n": len(keys),
        "tasks": ",".join(keys),
    }
