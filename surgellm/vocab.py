"""Surgical vocabulary (Appendix D of the paper).

Ten case-insensitive lexical indicator groups.  Entries ending in ``*`` use
prefix matching so that inflectional families collapse to a single indicator
(``oscillat*`` matches *oscillation, oscillates, oscillating*).

The vocabulary is *data*, not code: it is also mirrored to
``configs/surgical_vocabulary.json`` by :func:`dump_default_vocabulary` so the
released artifact can be inspected without importing Python.

Reference: paper, Appendix D; §3.3 (Surgical Feature Extraction).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

__all__ = [
    "IndicatorGroup",
    "SurgicalVocabulary",
    "DEFAULT_GROUPS",
    "default_vocabulary",
    "dump_default_vocabulary",
    "GROUP_ORDER",
]

# --------------------------------------------------------------------------
# Appendix D: the ten curated indicator groups, verbatim.
# --------------------------------------------------------------------------
DEFAULT_GROUPS: Dict[str, List[str]] = {
    "sst_pos": [
        "great", "excellent", "brilliant", "terrific", "wonderful",
        "masterpiece", "captivat*", "impressive", "delightful", "superb",
    ],
    "sst_neg": [
        "terrible", "awful", "dreadful", "unwatchable", "boring", "dull",
        "mediocre", "disappoint*", "worst", "painful",
    ],
    "llm_stat": [
        "empirically", "statistically", "demonstrated", "observed",
        "evidenced", "indicate*", "suggest*", "results show", "data show",
    ],
    "llm_formal": [
        "moreover", "furthermore", "additionally", "consequently",
        "therefore", "in conclusion", "in summary", "to summarize",
    ],
    "llm_list": [
        "firstly", "secondly", "thirdly", "finally", "in addition",
        "on the other hand", "(1)", "(2)", "(3)",
    ],
    "human_pers": [
        "i", "my", "we", "our", "personally", "i think", "i believe",
        "i feel",
    ],
    "human_hedge": [
        "maybe", "perhaps", "possibly", "kind of", "sort of", "i guess",
        "probably", "somewhat", "arguably",
    ],
    "human_emo": [
        "love", "hate", "amazing", "awesome", "terrible", "awful",
        "fantastic", "horrible", "sad", "happy",
    ],
    "retrieval": [
        "according to", "as stated in", "the article reports",
        "the text states", "multi-hop", "supporting context", "in the passage",
    ],
    "prompt_cot": [
        "step by step", "let us think", "first", "then", "next", "reasoning",
        "the chain of thought", "walk through",
    ],
}

#: Canonical ordering of the ten groups -> feature indices s_1 .. s_10.
GROUP_ORDER: List[str] = list(DEFAULT_GROUPS.keys())

#: Names of the six appended surface features, s_11 .. s_16 (§3.3).
SURFACE_FEATURE_NAMES: List[str] = [
    "word_count",          # s_11  total word count
    "mean_word_length",    # s_12  mean word length in characters
    "sentence_count",      # s_13  sentence count (split on .!?)
    "question_marks",      # s_14  question-mark count
    "exclamation_marks",   # s_15  exclamation-mark count
    "has_digit",           # s_16  1[any digit in x]
]

#: Full 16-dim feature name vector, s in R^16_{>=0}.
FEATURE_NAMES: List[str] = GROUP_ORDER + SURFACE_FEATURE_NAMES
FEATURE_DIM: int = len(FEATURE_NAMES)
assert FEATURE_DIM == 16


@dataclass
class IndicatorGroup:
    """One lexical indicator group.

    Parameters
    ----------
    name:
        Group identifier, e.g. ``"sst_pos"``.
    terms:
        Surface terms.  A trailing ``*`` denotes prefix matching.
    """

    name: str
    terms: List[str]
    _exact: List[str] = field(default_factory=list, repr=False)
    _prefix: List[str] = field(default_factory=list, repr=False)
    _multiword: List[str] = field(default_factory=list, repr=False)
    _symbolic: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._exact, self._prefix, self._multiword, self._symbolic = [], [], [], []
        for raw in self.terms:
            term = raw.strip().lower()
            if not term:
                continue
            if term.endswith("*"):
                self._prefix.append(term[:-1])
            elif not re.search(r"[a-z]", term):
                # Purely symbolic patterns such as "(1)" cannot be tokenised
                # by the word splitter and are matched as raw substrings.
                self._symbolic.append(term)
            elif " " in term:
                self._multiword.append(term)
            else:
                self._exact.append(term)

    # -- matching ---------------------------------------------------------
    def count(self, text_lower: str, tokens: Sequence[str]) -> int:
        """Count of group hits in ``text_lower``.

        Implements the indicator sum of Eq. (4),

        .. math:: s_j = \\sum_{w \\in v_j} \\mathbf{1}[w \\in \\tilde{x}]

        i.e. the number of *distinct group members* that occur at least once
        (``1[w in x]`` is an indicator per vocabulary entry, not a term
        frequency).  Single words are matched on whitespace/punctuation
        tokens, multi-word entries and symbolic entries on the raw string,
        prefix entries (``foo*``) on token prefixes.
        """
        token_set = set(tokens)
        hits = 0
        for term in self._exact:
            if term in token_set:
                hits += 1
        for pref in self._prefix:
            if any(tok.startswith(pref) for tok in token_set):
                hits += 1
        for phrase in self._multiword:
            if phrase in text_lower:
                hits += 1
        for sym in self._symbolic:
            if sym in text_lower:
                hits += 1
        return hits

    def cardinality(self) -> int:
        return len(self.terms)


class SurgicalVocabulary:
    """An ordered collection of :class:`IndicatorGroup` objects.

    The vocabulary is a *swappable* component: the curated default, a
    random control (``random_control``), and an auto-extracted vocabulary
    (:mod:`surgellm.autoextract`) all present this interface, which is what
    makes the §7.2 sensitivity studies a one-line configuration change.
    """

    def __init__(self, groups: Iterable[IndicatorGroup], name: str = "curated"):
        self.groups: List[IndicatorGroup] = list(groups)
        self.name = name

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_dict(cls, mapping: Dict[str, Sequence[str]], name: str = "custom") -> "SurgicalVocabulary":
        return cls([IndicatorGroup(k, list(v)) for k, v in mapping.items()], name=name)

    @classmethod
    def from_json(cls, path: str | Path) -> "SurgicalVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload["groups"], name=payload.get("name", "json"))

    # -- properties -------------------------------------------------------
    @property
    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def feature_names(self) -> List[str]:
        return self.group_names + SURFACE_FEATURE_NAMES

    def subset(self, names: Sequence[str]) -> "SurgicalVocabulary":
        keep = [g for g in self.groups if g.name in set(names)]
        return SurgicalVocabulary(keep, name=f"{self.name}|subset{len(keep)}")

    def drop(self, name: str) -> "SurgicalVocabulary":
        """Leave-one-out vocabulary used by Table 11."""
        return SurgicalVocabulary(
            [g for g in self.groups if g.name != name], name=f"{self.name}|-{name}"
        )

    def to_dict(self) -> Dict[str, List[str]]:
        return {g.name: list(g.terms) for g in self.groups}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.groups)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SurgicalVocabulary(name={self.name!r}, n_groups={self.n_groups})"


def default_vocabulary() -> SurgicalVocabulary:
    """The ten curated groups of Appendix D."""
    return SurgicalVocabulary.from_dict(DEFAULT_GROUPS, name="curated")


def dump_default_vocabulary(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "curated",
                "source": "paper Appendix D",
                "n_groups": len(DEFAULT_GROUPS),
                "surface_features": SURFACE_FEATURE_NAMES,
                "groups": DEFAULT_GROUPS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
