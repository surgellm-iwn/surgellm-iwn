"""Statistical testing (§5, §6.2, Table 4).

* Welch's t-test across seeds (unequal-variance, Welch-Satterthwaite df).
* Benjamini-Hochberg FDR correction at q = 0.05 over the 4 x 4 = 16
  task-variant comparisons.
* Bootstrap confidence intervals, B = 2,000, seed 0.

``scipy`` is used when available; otherwise the survival function of the
t-distribution falls back to a continued-fraction incomplete beta, so the
module has no hard dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["welch_t_test", "benjamini_hochberg", "bootstrap_ci",
           "paired_comparison_table", "TestResult", "BOOTSTRAP_B", "FDR_Q"]

BOOTSTRAP_B = 2000
FDR_Q = 0.05


# ---------------------------------------------------------------------------
# t-distribution tail
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided p-value for Student's t."""
    if df <= 0 or not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats  # type: ignore
        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        return float(_betai(0.5 * df, 0.5, df / (df + t * t)))


@dataclass
class TestResult:
    comparison: str
    task: str
    mean_a: float
    mean_b: float
    delta: float
    t_stat: float
    df: float
    p_raw: float
    p_bh: float = float("nan")
    significant: bool = False
    n_a: int = 0
    n_b: int = 0

    def as_row(self) -> Dict:
        return {
            "comparison": self.comparison, "task": self.task,
            "mean_a": round(self.mean_a, 4), "mean_b": round(self.mean_b, 4),
            "delta": round(self.delta, 4), "t": round(self.t_stat, 4),
            "df": round(self.df, 2), "p_raw": self.p_raw, "p_bh": self.p_bh,
            "significant": self.significant, "n_a": self.n_a, "n_b": self.n_b,
        }


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Return ``(t, df, p_two_sided)`` for unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    if se2 <= 0:
        # Identical constant samples: no evidence of a difference.
        return 0.0, float(na + nb - 2), 1.0
    t = (a.mean() - b.mean()) / math.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(df), _t_sf_two_sided(t, df)


def benjamini_hochberg(pvals: Sequence[float], q: float = FDR_Q
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """BH step-up.  Returns ``(adjusted_p, reject)``."""
    p = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    if not finite.any():
        return adj, np.zeros_like(p, dtype=bool)
    pf = p[finite]
    m = len(pf)
    order = np.argsort(pf)
    ranked = pf[order]
    adj_sorted = ranked * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]   # monotone
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    out = np.empty(m)
    out[order] = adj_sorted
    adj[finite] = out
    reject = np.zeros_like(p, dtype=bool)
    reject[finite] = out < q
    return adj, reject


def bootstrap_ci(values: Sequence[float], B: int = BOOTSTRAP_B, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI of the mean.  Returns ``(lo, hi, width)``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(B, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(hi - lo)


def paired_comparison_table(
    per_seed: Dict[str, Dict[str, List[float]]],
    comparisons: Sequence[Tuple[str, str]],
    tasks: Sequence[str],
    q: float = FDR_Q,
) -> List[Dict]:
    """Build the Table 4 rows.

    ``per_seed[model][task] = [f1_seed0, f1_seed1, ...]``.  Every
    ``(comparison, task)`` pair enters one family for BH correction, which
    is the paper's "4 x 4 = 16 task-variant comparisons".
    """
    results: List[TestResult] = []
    for model_a, model_b in comparisons:
        for task in tasks:
            a = per_seed.get(model_a, {}).get(task, [])
            b = per_seed.get(model_b, {}).get(task, [])
            if not a or not b:
                continue
            t, df, p = welch_t_test(a, b)
            results.append(TestResult(
                comparison=f"{model_a} vs. {model_b}", task=task,
                mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                delta=float(np.mean(a) - np.mean(b)),
                t_stat=t, df=df, p_raw=p, n_a=len(a), n_b=len(b),
            ))
    adj, reject = benjamini_hochberg([r.p_raw for r in results], q=q)
    for r, pa, rej in zip(results, adj, reject):
        r.p_bh = float(pa)
        r.significant = bool(rej)
    return [r.as_row() for r in results]
