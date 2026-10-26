"""Task schema and corpus containers (§4, Table 2).

Table 2 (corpus statistics after stratified capping):

    ====  ====  =======  =====  =======  ==========
    Task  ID    n        n_c    % min.   Source
    ====  ====  =======  =====  =======  ==========
    Sent  D1      7,666    2     49.5    SST-2
    Retr  D2      2,000    2     49.0    HotPotQA
    Gen   D3      3,164    2     50.0    LLM-7
    Auth  D4      5,000    2     50.0    HumLLM
    Total        17,830
    ====  ====  =======  =====  =======  ==========

D4 is *capped* to 50/50, but its source corpus carries a 9.3:1 skew and the
per-class feature moments stay shifted after capping - which is why IWN
still matters there (§4, Discussion).  ``raw_skew`` records that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["TaskSpec", "TASKS", "Corpus", "SplitBundle", "TOTAL_EXAMPLES"]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str            # "D1"
    name: str               # "sentiment"
    n_classes: int
    capped_n: int
    minority_pct: float
    source: str
    raw_skew: float = 1.0   # majority:minority in the *source* corpus
    class_names: Sequence[str] = ("class_0", "class_1")
    dominant_group: str = ""   # Table 11 "most important group"


TASKS: Dict[str, TaskSpec] = {
    "D1": TaskSpec("D1", "sentiment", 2, 7_666, 49.5, "SST-2", 1.0,
                   ("negative", "positive"), "sst_pos"),
    "D2": TaskSpec("D2", "retrieval", 2, 2_000, 49.0, "HotPotQA", 1.0,
                   ("easy", "medium_hard"), "retrieval"),
    "D3": TaskSpec("D3", "generation", 2, 3_164, 50.0, "LLM-7", 11.8,
                   ("human", "llm"), "prompt_cot"),
    "D4": TaskSpec("D4", "authorship", 2, 5_000, 50.0, "HumLLM", 9.3,
                   ("human", "llm"), "llm_stat"),
}

TASK_ORDER: List[str] = ["D1", "D2", "D3", "D4"]
TOTAL_EXAMPLES: int = sum(t.capped_n for t in TASKS.values())
assert TOTAL_EXAMPLES == 17_830, TOTAL_EXAMPLES


@dataclass
class Corpus:
    """One task's examples."""

    task_id: str
    texts: List[str]
    labels: np.ndarray
    n_classes: int

    def __len__(self) -> int:
        return len(self.texts)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels.astype(int), minlength=self.n_classes)

    def class_priors(self) -> np.ndarray:
        c = self.class_counts().astype(float)
        return c / max(c.sum(), 1.0)

    def minority_pct(self) -> float:
        p = self.class_priors()
        return float(100.0 * p.min())

    def skew(self) -> float:
        c = self.class_counts().astype(float)
        return float(c.max() / max(c.min(), 1.0))


@dataclass
class SplitBundle:
    """70/15/15 stratified train/val/test for one task."""

    task_id: str
    train: Corpus
    val: Corpus
    test: Corpus
    meta: Dict = field(default_factory=dict)

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}
