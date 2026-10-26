"""Stratified capping and 70/15/15 stratified splits (§4).

Two operations, both seeded and both label-stratified:

``stratified_cap``
    Down-sample a corpus to ``n`` examples while matching a target class
    balance.  Used to produce the Table 2 capped sizes, including the
    50/50 cap on D4 whose source corpus is 9.3:1.

``stratified_split``
    70/15/15 train/val/test, stratified by label, with label reindexing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .schema import Corpus, SplitBundle

__all__ = ["stratified_cap", "stratified_split", "reindex_labels"]


def reindex_labels(labels: Sequence) -> Tuple[np.ndarray, List]:
    """Map arbitrary label values onto ``0..n_c-1`` in sorted order."""
    uniq = sorted(set(labels))
    lut = {v: i for i, v in enumerate(uniq)}
    return np.array([lut[v] for v in labels], dtype=np.int64), uniq


def stratified_cap(corpus: Corpus, n_target: int, seed: int = 0,
                   target_priors: Optional[Sequence[float]] = None) -> Corpus:
    """Cap ``corpus`` to ``n_target`` examples, stratified by label.

    ``target_priors=None`` preserves the existing class balance; passing
    ``[0.5, 0.5]`` produces the balanced cap used for D4.
    """
    rng = np.random.default_rng(seed)
    labels = corpus.labels.astype(int)
    k = corpus.n_classes
    counts = np.bincount(labels, minlength=k)

    if target_priors is None:
        priors = counts / max(counts.sum(), 1)
    else:
        priors = np.asarray(target_priors, dtype=float)
        priors = priors / priors.sum()

    n_target = min(int(n_target), len(corpus))
    # Largest-remainder apportionment, clipped by class availability.
    raw = priors * n_target
    take = np.floor(raw).astype(int)
    take = np.minimum(take, counts)
    # Distribute the shortfall. Two sources of shortfall exist: the floor()
    # above, and classes whose availability is below their quota (asking a
    # 9.3:1 pool for a 50/50 cap exhausts the minority). The loop therefore
    # runs until the remainder is placed or no class can absorb more, rather
    # than for a fixed number of passes.
    remainder = int(n_target - take.sum())
    order = list(np.argsort(-(raw - np.floor(raw))))
    i = 0
    while remainder > 0:
        headroom = [c for c in order if take[c] < counts[c]]
        if not headroom:
            break                      # the corpus simply has no more examples
        c = headroom[i % len(headroom)]
        take[c] += 1
        remainder -= 1
        i += 1

    keep: List[int] = []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        keep.extend(idx[: take[c]].tolist())
    keep_arr = np.array(sorted(keep), dtype=int)
    rng.shuffle(keep_arr)

    return Corpus(
        task_id=corpus.task_id,
        texts=[corpus.texts[i] for i in keep_arr],
        labels=labels[keep_arr],
        n_classes=k,
    )


def stratified_split(corpus: Corpus, seed: int = 0,
                     fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15)
                     ) -> SplitBundle:
    """70/15/15 stratified split with per-class apportionment."""
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    labels = corpus.labels.astype(int)
    tr: List[int] = []
    va: List[int] = []
    te: List[int] = []

    for c in range(corpus.n_classes):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fractions[0] * n))
        n_va = int(round(fractions[1] * n))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        tr.extend(idx[:n_tr].tolist())
        va.extend(idx[n_tr:n_tr + n_va].tolist())
        te.extend(idx[n_tr + n_va:].tolist())

    def _sub(ix: List[int]) -> Corpus:
        arr = np.array(sorted(ix), dtype=int)
        rng.shuffle(arr)
        return Corpus(corpus.task_id, [corpus.texts[i] for i in arr],
                      labels[arr], corpus.n_classes)

    return SplitBundle(
        task_id=corpus.task_id, train=_sub(tr), val=_sub(va), test=_sub(te),
        meta={"seed": seed, "fractions": list(fractions), "n_total": len(corpus)},
    )
