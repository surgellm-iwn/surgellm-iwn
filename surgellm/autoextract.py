"""Auto-extracted vocabulary - the transfer recipe of Appendix E.

Procedure (verbatim from the paper):

1. For each task ``t_k`` and class ``c``, compute the log-odds ratio with
   an informative Dirichlet prior on word frequencies (Monroe et al.,
   2008, "Fightin' Words").
2. Rank words by absolute log-odds; retain the top ``K = 50`` per class.
3. Embed the union of retained words using SBERT.
4. Run ``k``-means with ``k = 10`` on the embedding matrix.
5. Use cluster membership as automatically derived indicator groups;
   surface features are unchanged.

Result: 0.903 avg. F1 versus 0.906 for manual curation - a 0.3% relative
gap - so manual curation is a convenience, not a requirement.

SBERT is optional here: if ``sentence-transformers`` is unavailable we fall
back to a deterministic co-occurrence embedding (PPMI + truncated SVD),
which keeps the recipe runnable offline.  The embedding backend is recorded
in the returned metadata so a result is never silently attributed to SBERT.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import WORD_RE
from .vocab import SurgicalVocabulary, IndicatorGroup

__all__ = ["monroe_log_odds", "auto_extract_vocabulary", "random_control_vocabulary",
           "TOP_K", "N_CLUSTERS"]

TOP_K = 50
N_CLUSTERS = 10

#: Function words are excluded from *candidate* extraction only; the curated
#: vocabulary deliberately keeps some ("i", "my", "we") as personal markers.
_STOP = set("""
a an the of to in and or is are was were be been being it its this that these
those for on with as at by from not no so if then than there here he she they
them his her their you your
""".split())


def _counts(texts: Sequence[str], min_len: int = 3) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for w in WORD_RE.findall(t.lower()):
            if len(w) >= min_len and w not in _STOP and not w.isdigit():
                c[w] += 1
    return c


def monroe_log_odds(texts: Sequence[str], labels: np.ndarray, n_classes: int,
                    alpha0: float = 1000.0, min_count: int = 3
                    ) -> Dict[int, List[Tuple[str, float]]]:
    """Class-conditional log-odds with an informative Dirichlet prior.

    For class ``c`` with counts ``y_i^c``, background counts ``alpha_i``
    (the corpus frequency scaled to total mass ``alpha0``), and
    ``n^c = sum_i y_i^c``:

        delta_i^c = log( (y_i^c + alpha_i) / (n^c + alpha_0 - y_i^c - alpha_i) )
                  - log( (y_i^rest + alpha_i) / (n^rest + alpha_0 - y_i^rest - alpha_i) )

    with variance ``1/(y_i^c + alpha_i) + 1/(y_i^rest + alpha_i)``; the
    returned score is the z-score ``delta / sqrt(var)``, which is the
    quantity Monroe et al. recommend ranking on.
    """
    labels = np.asarray(labels).astype(int)
    overall = _counts(texts)
    vocab = [w for w, c in overall.items() if c >= min_count]
    if not vocab:
        return {c: [] for c in range(n_classes)}
    total = sum(overall[w] for w in vocab)
    alpha = {w: alpha0 * overall[w] / total for w in vocab}

    per_class = {}
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        per_class[c] = _counts([texts[i] for i in idx])

    out: Dict[int, List[Tuple[str, float]]] = {}
    for c in range(n_classes):
        yc = per_class[c]
        yr: Counter = Counter()
        for other in range(n_classes):
            if other != c:
                yr.update(per_class[other])
        nc = sum(yc[w] for w in vocab)
        nr = sum(yr[w] for w in vocab)
        scored: List[Tuple[str, float]] = []
        for w in vocab:
            a = alpha[w]
            num_c = yc[w] + a
            num_r = yr[w] + a
            den_c = nc + alpha0 - num_c
            den_r = nr + alpha0 - num_r
            if den_c <= 0 or den_r <= 0:
                continue
            delta = math.log(num_c / den_c) - math.log(num_r / den_r)
            var = 1.0 / num_c + 1.0 / num_r
            scored.append((w, delta / math.sqrt(var)))
        scored.sort(key=lambda kv: -abs(kv[1]))
        out[c] = scored
    return out


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _sbert_embed(words: Sequence[str], model_name: str = "all-MiniLM-L6-v2"
                 ) -> Optional[np.ndarray]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return None
    try:
        model = SentenceTransformer(model_name)
        return np.asarray(model.encode(list(words), show_progress_bar=False))
    except Exception:
        return None


def _ppmi_svd_embed(words: Sequence[str], texts: Sequence[str], dim: int = 64,
                    window: int = 6, seed: int = 0) -> np.ndarray:
    """Deterministic offline embedding: windowed PPMI + truncated SVD."""
    index = {w: i for i, w in enumerate(words)}
    n = len(words)
    co = np.zeros((n, n), dtype=np.float64)
    for text in texts:
        toks = [t for t in WORD_RE.findall(text.lower()) if t in index]
        for i, a in enumerate(toks):
            for b in toks[i + 1: i + 1 + window]:
                ia, ib = index[a], index[b]
                co[ia, ib] += 1.0
                co[ib, ia] += 1.0
    total = co.sum()
    if total <= 0:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, min(dim, max(n, 2))))
    row = co.sum(1, keepdims=True)
    col = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / np.maximum(row * col, 1e-12))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0))
    k = int(min(dim, max(min(ppmi.shape) - 1, 1)))
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    return u[:, :k] * s[:k]


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    """k-means++ initialisation, Lloyd iterations; no sklearn dependency."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = max(1, min(k, n))
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / max(d2.sum(), 1e-12)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels


@dataclass
class AutoExtractReport:
    vocabulary: SurgicalVocabulary
    embedding_backend: str
    n_candidates: int
    top_terms_per_class: Dict[str, List[str]]
    cluster_sizes: List[int]


def auto_extract_vocabulary(texts_by_task: Dict[str, Sequence[str]],
                            labels_by_task: Dict[str, np.ndarray],
                            n_classes_by_task: Dict[str, int],
                            top_k: int = TOP_K, n_clusters: int = N_CLUSTERS,
                            seed: int = 0, use_sbert: bool = True,
                            sbert_model: str = "all-MiniLM-L6-v2") -> AutoExtractReport:
    """Run the five-step Appendix E recipe across all tasks."""
    candidates: List[str] = []
    top_terms: Dict[str, List[str]] = {}
    all_texts: List[str] = []

    for tid, texts in texts_by_task.items():
        all_texts.extend(texts)
        scored = monroe_log_odds(texts, labels_by_task[tid], n_classes_by_task[tid])
        for c, ranked in scored.items():
            words = [w for w, _ in ranked[:top_k]]       # step 2
            top_terms[f"{tid}_c{c}"] = words
            candidates.extend(words)

    union = sorted(set(candidates))                       # step 3 (union)
    if not union:
        return AutoExtractReport(SurgicalVocabulary([], "auto-empty"), "none", 0, {}, [])

    emb = _sbert_embed(union, sbert_model) if use_sbert else None
    backend = "sbert"
    if emb is None:
        emb = _ppmi_svd_embed(union, all_texts, seed=seed)
        backend = "ppmi-svd (offline fallback)"

    assign = _kmeans(np.asarray(emb, dtype=float), n_clusters, seed=seed)  # step 4

    groups: List[IndicatorGroup] = []
    sizes: List[int] = []
    for j in range(n_clusters):                                            # step 5
        members = [union[i] for i in np.flatnonzero(assign == j)]
        sizes.append(len(members))
        if members:
            groups.append(IndicatorGroup(f"auto_{j:02d}", members))
    # Preserve a stable group count so the feature dimension stays 16.
    while len(groups) < n_clusters:
        groups.append(IndicatorGroup(f"auto_{len(groups):02d}", []))

    return AutoExtractReport(
        vocabulary=SurgicalVocabulary(groups, name="auto-extracted"),
        embedding_backend=backend, n_candidates=len(union),
        top_terms_per_class=top_terms, cluster_sizes=sizes,
    )


def random_control_vocabulary(texts: Sequence[str], reference: SurgicalVocabulary,
                              seed: int = 0, min_count: int = 5) -> SurgicalVocabulary:
    """Random-vocabulary control of §7.2.2.

    Each curated group is replaced by a **same-cardinality** random sample of
    high-frequency content words.  The paper draws from the BNC; offline we
    draw from the corpus's own high-frequency content words, which serves the
    same purpose: same parameter count, no curated semantics.  If gains were
    parametric rather than lexical, this control would match the curated
    vocabulary; the paper measures -0.028 avg F1 instead.
    """
    rng = np.random.default_rng(seed)
    pool = [w for w, c in _counts(texts).items() if c >= min_count and len(w) >= 4]
    if len(pool) < 20:
        pool = [w for w, _ in _counts(texts).most_common(200)] or ["word"]
    groups = []
    for g in reference.groups:
        k = min(g.cardinality(), len(pool))
        picks = rng.choice(len(pool), size=k, replace=False)
        groups.append(IndicatorGroup(f"rand_{g.name}", [pool[int(i)] for i in picks]))
    return SurgicalVocabulary(groups, name="random-control")
