"""The surgical feature gate (§3.4, Eqs. 5-7) and the task embedding (Eq. 3).

Step 1 (Eq. 5)   s' = ReLU(W_s s + b_s),                W_s in R^{d x 16}
Step 2 (Eq. 6)   g  = sigma(W_g [htilde ; s'] + b_g),   W_g in R^{d x 2d}
Step 3 (Eq. 7)   hhat = LN(g . htilde + (1 - g) . s')

Design choices recorded in code because they are claims in the paper:

* **Sigmoid, not softmax.** ``g in (0,1)^d`` is a free per-dimension
  interpolation weight; a softmax would impose a unit budget across
  dimensions, which is the wrong constraint for dimension-wise modality
  fusion.
* **Per-dimension, not scalar.** A scalar gate forces every hidden unit to
  use the same modality mix.
* **Post-fusion LayerNorm.** Keeps the fused representation in the same
  statistical regime as the unfused encoder output.

The module also exposes the quantities the theory needs:
``spectral_norm_Wg`` for the ``lambda_max(W_g^T W_g)`` factor in Theorem 1,
``lipschitz_constant`` for Lemma 4, and ``gate_statistics`` for the
Proposition 2 degeneracy check (``||W_s|| -> 0``, ``g_i -> 1``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TaskEmbedding", "SurgicalFeatureGate", "GateOutput"]


class TaskEmbedding(nn.Module):
    """Eq. (3):  htilde = h + alpha * E_{t_k},  alpha = 0.1.

    The small mixing coefficient follows the residual-norm-preservation
    argument of He et al. (2016): at initialisation the perturbation
    ``alpha ||E_{t_k}||`` is small relative to ``||h|| ~ sqrt(d) sigma_h``.
    Empirically ``alpha in [0.05, 0.2]`` is stable; ``alpha = 1`` lets the
    task embedding dominate early training.
    """

    def __init__(self, n_tasks: int, hidden_size: int, alpha: float = 0.1,
                 init_std: float = 0.02):
        super().__init__()
        self.alpha = float(alpha)
        self.embedding = nn.Embedding(n_tasks, hidden_size)
        nn.init.normal_(self.embedding.weight, std=init_std)

    def forward(self, h: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        return h + self.alpha * self.embedding(task_ids)

    @torch.no_grad()
    def perturbation_ratio(self, h: torch.Tensor, task_ids: torch.Tensor) -> float:
        """``alpha ||E|| / ||h||`` - the quantity the §3.2 argument bounds."""
        e = self.embedding(task_ids)
        return float((self.alpha * e.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-9)).mean())


@dataclass
class GateOutput:
    fused: torch.Tensor            # hhat, (B, d)
    gate: torch.Tensor             # g,    (B, d)
    projected: torch.Tensor        # s',   (B, d)
    task_conditioned: torch.Tensor # htilde, (B, d)


class SurgicalFeatureGate(nn.Module):
    """Per-dimension, instance-conditioned cross-modal gate."""

    def __init__(self, hidden_size: int, feature_dim: int = 16,
                 gate_bias_init: float = 0.0, enabled: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.enabled = enabled

        self.feature_proj = nn.Linear(feature_dim, hidden_size)      # W_s, b_s
        self.gate_proj = nn.Linear(2 * hidden_size, hidden_size)     # W_g, b_g
        self.post_norm = nn.LayerNorm(hidden_size)                   # LN in Eq. (7)

        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        # A positive gate bias starts the model near the unfused encoder
        # (g ~ 1), i.e. near the identity map of Proposition 2.
        nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    # -- forward ----------------------------------------------------------
    def project(self, s: torch.Tensor) -> torch.Tensor:
        """Eq. (5).  ReLU keeps ``s'`` in the positive orthant, the same
        orthant as a typical post-LayerNorm encoder activation."""
        return F.relu(self.feature_proj(s))

    def forward(self, h_tilde: torch.Tensor, s: torch.Tensor) -> GateOutput:
        if not self.enabled:
            # Gate ablated: the model is the plain task-conditioned encoder.
            ones = torch.ones_like(h_tilde)
            return GateOutput(self.post_norm(h_tilde), ones,
                              torch.zeros_like(h_tilde), h_tilde)

        s_proj = self.project(s)                                       # Eq. (5)
        g = torch.sigmoid(self.gate_proj(torch.cat([h_tilde, s_proj], dim=-1)))  # Eq. (6)
        fused = self.post_norm(g * h_tilde + (1.0 - g) * s_proj)       # Eq. (7)
        return GateOutput(fused, g, s_proj, h_tilde)

    # -- theory hooks -----------------------------------------------------
    @torch.no_grad()
    def spectral_norm_Wg(self) -> float:
        """``lambda_max(W_g^T W_g)`` - the Theorem 1 approximation factor."""
        sv = torch.linalg.svdvals(self.gate_proj.weight.detach().float())
        return float(sv[0] ** 2)

    @torch.no_grad()
    def spectral_norm_Ws(self) -> float:
        sv = torch.linalg.svdvals(self.feature_proj.weight.detach().float())
        return float(sv[0])

    @torch.no_grad()
    def lipschitz_constant(self) -> float:
        """Upper bound on ``L_G`` used in Lemma 4.

        For ``hhat = LN(g . htilde + (1-g) . s')`` with ``g = sigma(W_g z)``,
        a crude but valid bound composes: the convex combination is
        1-Lipschitz in its arguments, the gate contributes
        ``sup_z sigma'(z) * ||W_g|| = ||W_g||/4`` times the magnitude of the
        gated signals, the projection contributes ``||W_s||`` (ReLU is
        1-Lipschitz), and LayerNorm is taken as 1-Lipschitz up to its gain.
        """
        w_g = float(torch.linalg.svdvals(self.gate_proj.weight.detach().float())[0])
        w_s = self.spectral_norm_Ws()
        ln_gain = float(self.post_norm.weight.detach().abs().max())
        return float(ln_gain * (1.0 + max(w_s, 1.0) + 0.25 * w_g))

    @torch.no_grad()
    def gate_statistics(self, h_tilde: torch.Tensor, s: torch.Tensor) -> Dict[str, float]:
        """Diagnostics for the Proposition 2 / Corollary 3 safety check."""
        out = self.forward(h_tilde, s)
        return {
            "gate_mean": float(out.gate.mean()),
            "gate_min": float(out.gate.min()),
            "gate_frac_above_0.99": float((out.gate > 0.99).float().mean()),
            "Ws_fro_norm": float(self.feature_proj.weight.norm()),
            "Ws_spectral_norm": self.spectral_norm_Ws(),
            "s_proj_mean_norm": float(out.projected.norm(dim=-1).mean()),
            "identity_gap": float(
                (out.fused - self.post_norm(h_tilde)).norm(dim=-1).mean()
            ),
        }
