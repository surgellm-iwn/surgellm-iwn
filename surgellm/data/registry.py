"""Corpus loading for the four tasks (§4).

Two providers:

``provider="hf"``
    The real corpora used for the paper's numbers.
      * D1 SST-2 (GLUE): standard train split (67,349), official validation
        (872) as the test set, stratified 10% of train held out internally,
        train capped to 7,666 stratified by label.
      * D2 HotPotQA: validation split; ``x = [Q] q [CTX] c_{:300}``; binary
        label from the three-tier difficulty annotation, easy -> 0 and
        medium/hard -> 1; stratified sample of 2,000.
      * D3 LLM-7: 14,877 essays at ~11.8:1 human skew, stratified-capped to
        3,164.
      * D4 HumLLM: 788,922 texts at 9.3:1 skew, 5,000 balanced examples.

``provider="synthetic"``
    The offline surrogates of :mod:`surgellm.data.synthetic`.

Both return the same object, so every downstream script is
provider-agnostic.  The D3/D4 loaders also return the *uncapped source
pool*: §4 states that feature normalisation uses the full training data,
which is why IWN still corrects a 50/50-capped D4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import Corpus, SplitBundle, TASKS, TASK_ORDER
from .splits import stratified_cap, stratified_split

__all__ = ["TaskData", "load_task", "load_all_tasks", "corpus_report"]


@dataclass
class TaskData:
    """Everything a task contributes to a run."""

    task_id: str
    splits: SplitBundle
    #: Uncapped training pool used to fit the feature normaliser.  For a
    #: balanced task this is just the capped training split.
    normalizer_pool: Corpus
    provider: str

    @property
    def n_classes(self) -> int:
        return self.splits.train.n_classes


# ---------------------------------------------------------------------------
# HuggingFace providers
# ---------------------------------------------------------------------------
def _load_hf_sst2(seed: int, cache_dir: Optional[str]) -> Tuple[Corpus, Corpus]:
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", cache_dir=cache_dir)
    tr_texts = list(ds["train"]["sentence"])
    tr_labels = np.array(ds["train"]["label"], dtype=np.int64)
    pool = Corpus("D1", tr_texts, tr_labels, 2)
    capped = stratified_cap(pool, TASKS["D1"].capped_n, seed=seed)
    # Official validation split is used as the test set (§4.1).
    test = Corpus("D1", list(ds["validation"]["sentence"]),
                  np.array(ds["validation"]["label"], dtype=np.int64), 2)
    capped.__dict__["_external_test"] = test
    return pool, capped


def _load_hf_hotpotqa(seed: int, cache_dir: Optional[str]) -> Tuple[Corpus, Corpus]:
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", cache_dir=cache_dir)["validation"]
    texts, labels = [], []
    for row in ds:
        q = row["question"]
        sentences: List[str] = []
        for group in row["context"]["sentences"]:
            sentences.extend(group)
        ctx = " ".join(" ".join(sentences).split()[:300])      # c_{:300}
        texts.append(f"[Q] {q} [CTX] {ctx}")
        labels.append(0 if row["level"] == "easy" else 1)      # easy->0, med/hard->1
    pool = Corpus("D2", texts, np.array(labels, dtype=np.int64), 2)
    return pool, stratified_cap(pool, TASKS["D2"].capped_n, seed=seed)


def _load_hf_generic(task_id: str, hf_name: str, text_col: str, label_col: str,
                     seed: int, cache_dir: Optional[str],
                     balanced: bool) -> Tuple[Corpus, Corpus]:
    from datasets import load_dataset
    ds = load_dataset(hf_name, cache_dir=cache_dir)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = ds[split]
    labels_raw = list(rows[label_col])
    uniq = sorted(set(labels_raw))
    lut = {v: i for i, v in enumerate(uniq)}
    pool = Corpus(task_id, [str(t) for t in rows[text_col]],
                  np.array([lut[v] for v in labels_raw], dtype=np.int64), len(uniq))
    capped = stratified_cap(pool, TASKS[task_id].capped_n, seed=seed,
                            target_priors=[1.0 / len(uniq)] * len(uniq) if balanced else None)
    return pool, capped


#: Dataset identifiers for D3/D4.  The paper's citations for LLM-7 and
#: HumLLM are flagged "Author-check required" (canonical URL/DOI pending),
#: so these are configurable rather than hard-coded.
DEFAULT_HF_SOURCES: Dict[str, Dict[str, str]] = {
    "D3": {"hf_name": "", "text_col": "text", "label_col": "label"},
    "D4": {"hf_name": "", "text_col": "text", "label_col": "label"},
}


def load_task(task_id: str, provider: str = "synthetic", seed: int = 0,
              cache_dir: Optional[str] = None, scale: float = 1.0,
              hf_sources: Optional[Dict[str, Dict[str, str]]] = None) -> TaskData:
    """Load one task under the requested provider."""
    if task_id not in TASKS:
        raise KeyError(f"unknown task {task_id!r}")

    if provider == "synthetic":
        from .synthetic import generate_task, _spec
        spec = _spec(task_id)
        pool, capped = generate_task(
            task_id, seed=seed, n_source=max(int(spec.n_source * scale), 200)
        )
        if scale < 1.0:
            tspec = TASKS[task_id]
            capped = stratified_cap(
                pool, max(int(tspec.capped_n * scale), 120), seed=seed,
                target_priors=[0.5, 0.5] if abs(tspec.minority_pct - 50.0) < 1.0 else None,
            )
        splits = stratified_split(capped, seed=seed)
        return TaskData(task_id, splits, pool, provider)

    if provider != "hf":
        raise ValueError(f"unknown provider {provider!r}")

    src = {**DEFAULT_HF_SOURCES, **(hf_sources or {})}
    if task_id == "D1":
        pool, capped = _load_hf_sst2(seed, cache_dir)
    elif task_id == "D2":
        pool, capped = _load_hf_hotpotqa(seed, cache_dir)
    else:
        cfg = src[task_id]
        if not cfg.get("hf_name"):
            raise RuntimeError(
                f"{task_id}: no HuggingFace identifier configured. The paper's "
                f"citation for this corpus is marked 'Author-check required'. "
                f"Set it in configs/datasets.yaml (hf_sources.{task_id}.hf_name) "
                f"or run with --provider synthetic."
            )
        pool, capped = _load_hf_generic(
            task_id, cfg["hf_name"], cfg["text_col"], cfg["label_col"],
            seed, cache_dir, balanced=abs(TASKS[task_id].minority_pct - 50.0) < 1.0,
        )

    splits = stratified_split(capped, seed=seed)
    external = capped.__dict__.get("_external_test")
    if external is not None:
        splits.test = external
        splits.meta["test_source"] = "official validation split"
    return TaskData(task_id, splits, pool, provider)


def load_all_tasks(provider: str = "synthetic", seed: int = 0, scale: float = 1.0,
                   cache_dir: Optional[str] = None,
                   hf_sources: Optional[Dict] = None) -> Dict[str, TaskData]:
    return {
        tid: load_task(tid, provider=provider, seed=seed, scale=scale,
                       cache_dir=cache_dir, hf_sources=hf_sources)
        for tid in TASK_ORDER
    }


def corpus_report(tasks: Dict[str, TaskData]) -> List[Dict]:
    """Rows for the Table 2 reproduction CSV."""
    rows = []
    for tid, td in tasks.items():
        spec = TASKS[tid]
        c = td.splits.train
        total = len(td.splits.train) + len(td.splits.val) + len(td.splits.test)
        rows.append({
            "task": spec.name, "id": tid, "n": total, "n_c": spec.n_classes,
            "pct_minority": round(
                100.0 * min(
                    np.bincount(np.concatenate([
                        td.splits.train.labels, td.splits.val.labels, td.splits.test.labels
                    ]), minlength=spec.n_classes) / max(total, 1)
                ), 2),
            "source": spec.source,
            "source_pool_n": len(td.normalizer_pool),
            "source_pool_skew": round(td.normalizer_pool.skew(), 2),
            "n_train": len(td.splits.train), "n_val": len(td.splits.val),
            "n_test": len(td.splits.test), "provider": td.provider,
        })
    return rows
