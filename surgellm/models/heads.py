"""Task-specific classification heads (§3.7, Eqs. 13-14).

    u_k    = GELU(W_{1,k} hhat + b_{1,k}),  W_{1,k} in R^{(d/2) x d}
    yhat_k = softmax(W_{2,k} u_k + b_{2,k}), W_{2,k} in R^{n_{c,k} x (d/2)}

Dropout p = 0.1 before ``W_1``, p = 0.05 before ``W_2``.  In a forward pass
samples are routed to their head by a task-integer mask and the per-task
cross-entropy losses are summed (Eq. 1).

The heads are the *only* task-specific parameters; §3.1 claims the shared
parameters are >99% of the total, which :func:`parameter_share_report`
checks numerically.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TaskHead", "MultiTaskHeads", "parameter_share_report"]


class TaskHead(nn.Module):
    """Two-layer MLP head for a single task."""

    def __init__(self, hidden_size: int, n_classes: int,
                 dropout1: float = 0.1, dropout2: float = 0.05):
        super().__init__()
        bottleneck = max(hidden_size // 2, 8)
        self.drop1 = nn.Dropout(dropout1)
        self.fc1 = nn.Linear(hidden_size, bottleneck)
        self.drop2 = nn.Dropout(dropout2)
        self.fc2 = nn.Linear(bottleneck, n_classes)
        self.n_classes = n_classes

    def forward(self, h_hat: torch.Tensor) -> torch.Tensor:
        """Returns **logits**; the softmax of Eq. (14) is folded into the
        cross-entropy loss for numerical stability."""
        u = F.gelu(self.fc1(self.drop1(h_hat)))      # Eq. (13)
        return self.fc2(self.drop2(u))               # Eq. (14), pre-softmax


class MultiTaskHeads(nn.Module):
    """A bank of per-task heads with task-integer-mask routing."""

    def __init__(self, hidden_size: int, n_classes_per_task: Sequence[int],
                 dropout1: float = 0.1, dropout2: float = 0.05):
        super().__init__()
        self.heads = nn.ModuleList([
            TaskHead(hidden_size, int(n_c), dropout1, dropout2)
            for n_c in n_classes_per_task
        ])
        self.n_classes_per_task = [int(c) for c in n_classes_per_task]
        self.max_classes = max(self.n_classes_per_task)

    def forward(self, h_hat: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Route each row to its head; return ``(B, max_classes)`` logits with
        invalid class slots masked to ``-inf`` so they never win an argmax."""
        B = h_hat.size(0)
        out = h_hat.new_full((B, self.max_classes), float("-inf"))
        for t, head in enumerate(self.heads):
            mask = task_ids == t
            if not bool(mask.any()):
                continue
            logits = head(h_hat[mask])
            out[mask, : logits.size(-1)] = logits.to(out.dtype)
        return out

    def loss(self, h_hat: torch.Tensor, task_ids: torch.Tensor, y: torch.Tensor,
             task_weights: Sequence[float] | None = None) -> Dict[str, torch.Tensor]:
        """Eq. (1) with ``w_k = 1``: per-task CE, summed over tasks present."""
        total = h_hat.new_zeros(())
        per_task: Dict[int, torch.Tensor] = {}
        for t, head in enumerate(self.heads):
            mask = task_ids == t
            if not bool(mask.any()):
                continue
            logits = head(h_hat[mask])
            l = F.cross_entropy(logits, y[mask])
            w = 1.0 if task_weights is None else float(task_weights[t])
            per_task[t] = l.detach()
            total = total + w * l
        return {"loss": total, "per_task": per_task}


def parameter_share_report(model: nn.Module, head_module_name: str = "heads") -> Dict[str, float]:
    """Verify the >99% shared-parameter claim of §3.1."""
    total = sum(p.numel() for p in model.parameters())
    task_specific = sum(
        p.numel() for n, p in model.named_parameters() if n.startswith(head_module_name)
    )
    shared = total - task_specific
    return {
        "total_parameters": float(total),
        "shared_parameters": float(shared),
        "task_specific_parameters": float(task_specific),
        "shared_fraction": float(shared / total) if total else 0.0,
    }
