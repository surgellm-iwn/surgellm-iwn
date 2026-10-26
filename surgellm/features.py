"""Surgical feature extraction, s(x) in R^16_{>=0}  (paper §3.3, Eq. 4).

Two properties matter for the paper's claims and are enforced here:

1. **Untruncated computation.**  Features are computed on the *full* text,
   not on the encoder's first ``L in {96,128}`` tokens.  This is the
   "truncation loss" argument of Appendix L.2 / §7.2.3 and is the reason
   surface statistics are not redundant with the encoder.
2. **Determinism.**  ``extract`` is a pure function of the string, so the
   feature cache is safe to reuse across variants and seeds.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np

from .vocab import FEATURE_DIM, SurgicalVocabulary, default_vocabulary

__all__ = ["SurgicalFeatureExtractor", "WORD_RE", "SENT_SPLIT_RE"]

WORD_RE = re.compile(r"[a-z0-9']+")
SENT_SPLIT_RE = re.compile(r"[.!?]")
DIGIT_RE = re.compile(r"\d")


class SurgicalFeatureExtractor:
    """Maps raw text to the surgical feature vector.

    Layout of the returned vector (``n_groups = 10`` for the curated
    vocabulary, giving ``FEATURE_DIM = 16``):

    =========  ==========================================
    index      meaning
    =========  ==========================================
    0..G-1     lexical indicator-group counts  (Eq. 4)
    G+0        s_11  total word count
    G+1        s_12  mean word length (characters)
    G+2        s_13  sentence count (split on ``.!?``)
    G+3        s_14  question-mark count
    G+4        s_15  exclamation-mark count
    G+5        s_16  1[any digit present]
    =========  ==========================================
    """

    def __init__(self, vocabulary: SurgicalVocabulary | None = None,
                 use_surface: bool = True, use_groups: bool = True):
        self.vocabulary = vocabulary or default_vocabulary()
        self.use_surface = use_surface
        self.use_groups = use_groups

    # -- geometry ---------------------------------------------------------
    @property
    def dim(self) -> int:
        d = 0
        if self.use_groups:
            d += self.vocabulary.n_groups
        if self.use_surface:
            d += 6
        return d

    def feature_names(self) -> List[str]:
        names: List[str] = []
        if self.use_groups:
            names += self.vocabulary.group_names
        if self.use_surface:
            names += [
                "word_count", "mean_word_length", "sentence_count",
                "question_marks", "exclamation_marks", "has_digit",
            ]
        return names

    # -- extraction -------------------------------------------------------
    def extract(self, text: str) -> np.ndarray:
        """Return ``s(x)`` for one document (float32, non-negative)."""
        lowered = text.lower()
        tokens = WORD_RE.findall(lowered)
        out: List[float] = []

        if self.use_groups:
            out.extend(float(g.count(lowered, tokens)) for g in self.vocabulary.groups)

        if self.use_surface:
            n_words = len(tokens)
            mean_len = (sum(len(t) for t in tokens) / n_words) if n_words else 0.0
            # sentence count: non-empty fragments after splitting on .!?
            n_sent = float(sum(1 for frag in SENT_SPLIT_RE.split(text) if frag.strip()))
            out.extend([
                float(n_words),
                float(mean_len),
                n_sent,
                float(text.count("?")),
                float(text.count("!")),
                1.0 if DIGIT_RE.search(text) else 0.0,
            ])

        vec = np.asarray(out, dtype=np.float32)
        assert vec.shape == (self.dim,), f"expected {self.dim}, got {vec.shape}"
        assert np.all(vec >= 0.0), "surgical features must lie in R^d_{>=0}"
        return vec

    def extract_batch(self, texts: Sequence[str]) -> np.ndarray:
        """``(N, dim)`` matrix of features."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.extract(t) for t in texts], axis=0)

    # -- prefix string (Eq. 12) ------------------------------------------
    def prefix_string(self, task_name: str, s: np.ndarray) -> str:
        """Build the task-conditioned surgical prefix of Eq. (12).

        ``[TASK:t_k | F1:v1 | ... | F16:v16]`` with ``v_j = floor(s_j)``.
        The prefix is a *string*: it is tokenised together with the input so
        that its representations are co-attended to by every layer
        (§3.6), and it is interpretable and deterministic, unlike soft
        prompts / prefix tuning.
        """
        vals = "|".join(f"F{i + 1}:{int(np.floor(v))}" for i, v in enumerate(s))
        return f"[TASK:{task_name}|{vals}]"


def feature_summary(matrix: np.ndarray, names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-dimension descriptive statistics, used for corpus reports."""
    return {
        name: {
            "mean": float(matrix[:, i].mean()),
            "std": float(matrix[:, i].std()),
            "min": float(matrix[:, i].min()),
            "max": float(matrix[:, i].max()),
            "nonzero_rate": float((matrix[:, i] > 0).mean()),
        }
        for i, name in enumerate(names)
    }


assert FEATURE_DIM == 16  # sanity: paper states s in R^16
