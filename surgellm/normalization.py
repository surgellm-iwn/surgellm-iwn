"""Feature standardisation and Instance-Weighted Normalization (IWN).

Implements §3.5 of the paper:

* Eq. (8)  standard (marginal) standardisation
      shat(x) = (s(x) - sbar_k) / (sigma_k + eps)
* Eq. (9)  the pathology:  sbar_k = sum_c pi_c sbar_{c,k} -> sbar_{c*,k}
           as the majority prior pi_{c*} -> 1
* Eq. (10) class-balanced moments
      sbar^bal_k = (1/n_c) sum_c sbar_{c,k},  sigma^bal_k = (1/n_c) sum_c sigma_{c,k}
* Eq. (11) IWN standardisation
      stilde(x) = (s(x) - sbar^bal_k) / (sigma^bal_k + eps)

Four properties claimed in the paper are *testable* against this module and
are exercised by ``tests/test_normalization.py``:

1. **Parameter-free** - no learnable parameters, only constants change.
2. **Test-time class-agnostic** - ``transform`` never sees a label.
3. **Reduces to standard normalization on balanced corpora** - with equal
   class priors the two estimators coincide up to the weighted/unweighted
   variance difference.
4. **Compositional** - orthogonal to loss-side (focal) and sampling-side
   (SMOTE/oversampling) remedies; nothing here touches the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

__all__ = ["FeatureNormalizer", "NormalizerStats", "prior_contamination"]

EPS = 1e-6


@dataclass
class NormalizerStats:
    """Serialisable fitted state of a :class:`FeatureNormalizer`."""

    mode: str                      # "standard" | "iwn"
    mean: np.ndarray               # (d,)
    std: np.ndarray                # (d,)
    per_class_mean: Optional[np.ndarray] = None   # (n_c, d)
    per_class_std: Optional[np.ndarray] = None    # (n_c, d)
    class_priors: Optional[np.ndarray] = None     # (n_c,)
    n_classes: int = 0
    n_examples: int = 0

    def to_json(self) -> Dict:
        def _l(a):
            return None if a is None else np.asarray(a).tolist()
        return {
            "mode": self.mode,
            "mean": _l(self.mean),
            "std": _l(self.std),
            "per_class_mean": _l(self.per_class_mean),
            "per_class_std": _l(self.per_class_std),
            "class_priors": _l(self.class_priors),
            "n_classes": self.n_classes,
            "n_examples": self.n_examples,
        }

    @staticmethod
    def from_json(payload: Dict) -> "NormalizerStats":
        def _a(x):
            return None if x is None else np.asarray(x, dtype=np.float64)
        return NormalizerStats(
            mode=payload["mode"],
            mean=_a(payload["mean"]),
            std=_a(payload["std"]),
            per_class_mean=_a(payload.get("per_class_mean")),
            per_class_std=_a(payload.get("per_class_std")),
            class_priors=_a(payload.get("class_priors")),
            n_classes=int(payload.get("n_classes", 0)),
            n_examples=int(payload.get("n_examples", 0)),
        )


class FeatureNormalizer:
    """Standardises surgical features, optionally with IWN.

    Parameters
    ----------
    mode:
        ``"standard"`` -> Eq. (8);  ``"iwn"`` -> Eq. (11).
    eps:
        Numerical floor added to the scale, as in the paper's ``+ eps``.

    Notes
    -----
    ``fit`` is the **only** place labels are used.  After fitting, the
    object stores two constant vectors; ``transform`` is label-free, which
    is exactly the "test-time class-agnostic" property.
    """

    def __init__(self, mode: str = "standard", eps: float = EPS):
        if mode not in {"standard", "iwn"}:
            raise ValueError(f"unknown normalization mode: {mode!r}")
        self.mode = mode
        self.eps = float(eps)
        self.stats: Optional[NormalizerStats] = None

    # -- fitting ----------------------------------------------------------
    def fit(self, s: np.ndarray, y: Optional[np.ndarray] = None,
            n_classes: Optional[int] = None) -> "FeatureNormalizer":
        """Estimate normalisation constants on the *training partition only*.

        ``s`` is ``(N, d)``; ``y`` is ``(N,)`` and is required for IWN.
        """
        s = np.asarray(s, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError("s must be (N, d)")
        n, d = s.shape

        marg_mean = s.mean(axis=0)
        marg_std = s.std(axis=0)

        per_class_mean = per_class_std = priors = None
        k = 0
        if y is not None:
            y = np.asarray(y).astype(int).ravel()
            k = int(n_classes if n_classes is not None else (y.max() + 1 if y.size else 0))
            per_class_mean = np.zeros((k, d), dtype=np.float64)
            per_class_std = np.zeros((k, d), dtype=np.float64)
            priors = np.zeros(k, dtype=np.float64)
            for c in range(k):
                m = y == c
                priors[c] = m.mean() if n else 0.0
                if m.sum() == 0:
                    # Empty class: fall back to marginal moments so the
                    # balanced average stays well defined.
                    per_class_mean[c] = marg_mean
                    per_class_std[c] = marg_std
                else:
                    per_class_mean[c] = s[m].mean(axis=0)
                    per_class_std[c] = s[m].std(axis=0)

        if self.mode == "iwn":
            if per_class_mean is None:
                raise ValueError("IWN requires training labels (Eq. 10)")
            mean = per_class_mean.mean(axis=0)   # Eq. (10), sbar^bal_k
            std = per_class_std.mean(axis=0)     # Eq. (10), sigma^bal_k
        else:
            mean, std = marg_mean, marg_std      # Eq. (8)

        self.stats = NormalizerStats(
            mode=self.mode, mean=mean, std=std,
            per_class_mean=per_class_mean, per_class_std=per_class_std,
            class_priors=priors, n_classes=k, n_examples=n,
        )
        return self

    # -- application ------------------------------------------------------
    def transform(self, s: np.ndarray) -> np.ndarray:
        """Label-free standardisation; Eq. (8) or Eq. (11) by ``mode``."""
        if self.stats is None:
            raise RuntimeError("FeatureNormalizer.fit must be called first")
        s = np.asarray(s, dtype=np.float64)
        out = (s - self.stats.mean) / (self.stats.std + self.eps)
        return out.astype(np.float32)

    def fit_transform(self, s: np.ndarray, y: Optional[np.ndarray] = None,
                      n_classes: Optional[int] = None) -> np.ndarray:
        return self.fit(s, y, n_classes).transform(s)

    # -- introspection ----------------------------------------------------
    @property
    def n_parameters(self) -> int:
        """Learnable parameters introduced by normalisation: zero.

        Backs the "parameter-free" property in §3.5.
        """
        return 0

    def state_dict(self) -> Dict:
        if self.stats is None:
            raise RuntimeError("not fitted")
        return {"eps": self.eps, **self.stats.to_json()}

    @staticmethod
    def from_state_dict(payload: Dict) -> "FeatureNormalizer":
        norm = FeatureNormalizer(mode=payload["mode"], eps=payload.get("eps", EPS))
        norm.stats = NormalizerStats.from_json(payload)
        return norm


def prior_contamination(normalizer: FeatureNormalizer) -> Dict[str, float]:
    """Quantify Eq. (9): how far the marginal mean sits from the balanced one.

    Returns the per-dimension L2 gap ``||sbar_k - sbar^bal_k||`` normalised by
    the balanced scale, plus the majority prior.  Under severe skew this is
    large, which is precisely the condition under which IWN helps; on a
    balanced corpus it goes to zero, matching the "reduces to standard
    normalization" property.
    """
    st = normalizer.stats
    if st is None or st.per_class_mean is None or st.class_priors is None:
        raise ValueError("normalizer must be fitted with labels")
    marginal = (st.class_priors[:, None] * st.per_class_mean).sum(axis=0)
    balanced = st.per_class_mean.mean(axis=0)
    scale = st.per_class_std.mean(axis=0) + EPS
    gap = np.linalg.norm((marginal - balanced) / scale)
    return {
        "majority_prior": float(st.class_priors.max()),
        "skew_ratio": float(st.class_priors.max() / max(st.class_priors.min(), 1e-12)),
        "normalized_mean_gap": float(gap),
        "max_dim_gap": float(np.max(np.abs((marginal - balanced) / scale))),
    }
