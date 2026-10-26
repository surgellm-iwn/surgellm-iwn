"""Offline surrogate corpora for the four tasks.

Why this module exists
----------------------
The paper's numbers come from SST-2, HotPotQA, LLM-7 and HumLLM.  Those
require network access and a few GPU-hours.  To make *every* claim in the
paper executable on a laptop with no downloads, this module generates four
surrogate corpora that preserve the structural properties the claims
actually depend on:

* **D1 (sentiment)** - polarity vocabulary (``sst_pos`` / ``sst_neg``)
  carries the signal; balanced; moderate alignment.
* **D2 (retrieval)** - attribution phrases (``retrieval`` group) are rare in
  questions and frequent in context, giving a clean, highly aligned signal.
  This is the task with the largest rho_k in Appendix G.
* **D3 (generation)** - prompt-scaffolding markers (``prompt_cot``,
  ``llm_list``) on longer texts; moderate alignment.
* **D4 (authorship)** - the hard case.  The *source* pool is drawn at the
  paper's 9.3:1 human:LLM skew; the evaluation subset is then capped to
  50/50, but the normaliser is fitted on the full skewed pool, so the
  marginal moments of Eq. (8) stay contaminated by the majority class.
  Per-class feature moments differ in both mean *and* variance (LLM text is
  more enumerative; human text more personal), which is exactly the
  pathology IWN is designed to remove.

Every generator is seeded, so the corpora are byte-identical across runs.
Artifacts produced from them are labelled ``demo`` and are never presented
as reproductions of the published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..vocab import DEFAULT_GROUPS
from .schema import Corpus, TASKS

__all__ = ["SyntheticSpec", "generate_task", "generate_all", "FILLER_WORDS"]


FILLER_WORDS: List[str] = """
film scene camera script actor pacing dialogue story plot role cast director
system method result value table model region layer signal window segment
report study paper record source passage section chapter column entry field
morning evening city river garden window letter number matter reason moment
project version library dataset sample feature measure outcome baseline note
""".split()


@dataclass
class SyntheticSpec:
    """Class-conditional generative parameters for one task.

    ``group_rates[c][g]`` is the expected number of *distinct* members of
    group ``g`` that appear in a class-``c`` document.  ``length_mu`` and
    ``length_sigma`` control word count per class (log-normal), and
    ``punct_rate`` the per-class question/exclamation intensity.  Differing
    the per-class *variance* - not only the mean - is what makes the
    class-balanced scale of Eq. (10) differ from the pooled scale of
    Eq. (8).
    """

    task_id: str
    group_rates: Dict[int, Dict[str, float]]
    length_mu: Dict[int, float]
    length_sigma: Dict[int, float]
    punct_rate: Dict[int, Tuple[float, float]]   # (question, exclamation)
    digit_prob: Dict[int, float]
    source_skew: float = 1.0                      # majority:minority in the pool
    n_source: int = 20_000
    #: Fraction of documents rendered under the *opposite* class conditional.
    #: This is the surrogate for genuine task difficulty: without it the
    #: classes separate perfectly and every variant saturates at F1 = 1.000,
    #: reproducing exactly the flaw the reviewers flagged in the original D1
    #: (Appendix L.3).  Non-saturation is a precondition for any ablation to
    #: be readable.
    confusion_rate: float = 0.0


def _spec(task_id: str) -> SyntheticSpec:
    """Hand-tuned specs; the asymmetries are the point, not the magnitudes."""
    if task_id == "D1":
        # Sentiment: polarity words carry the signal, but both polarities
        # appear in mixed reviews, and negation/sarcasm are unmodelled -
        # hence a sizeable confusion rate (cf. the §7.5 D1 failure modes).
        return SyntheticSpec(
            task_id="D1",
            group_rates={
                0: {"sst_neg": 2.2, "sst_pos": 0.7, "human_emo": 1.1,
                    "human_pers": 0.8, "human_hedge": 0.5},
                1: {"sst_pos": 2.2, "sst_neg": 0.7, "human_emo": 1.3,
                    "human_pers": 0.9, "human_hedge": 0.4},
            },
            length_mu={0: 3.0, 1: 3.0}, length_sigma={0: 0.45, 1: 0.45},
            punct_rate={0: (0.15, 0.25), 1: (0.15, 0.35)},
            digit_prob={0: 0.10, 1: 0.10}, source_skew=1.0, n_source=12_000,
            confusion_rate=0.04,
        )
    if task_id == "D2":
        # Retrieval: attribution phrases are the cleanest signal in the
        # suite, which is why D2 carries the highest rho_k in Appendix G.
        return SyntheticSpec(
            task_id="D2",
            group_rates={
                0: {"retrieval": 0.8, "llm_formal": 0.4, "llm_stat": 0.3},
                1: {"retrieval": 2.6, "llm_formal": 1.1, "llm_stat": 0.9},
            },
            length_mu={0: 4.2, 1: 4.5}, length_sigma={0: 0.35, 1: 0.40},
            punct_rate={0: (1.0, 0.05), 1: (1.0, 0.05)},
            digit_prob={0: 0.55, 1: 0.70}, source_skew=1.0, n_source=6_000,
            confusion_rate=0.04,
        )
    if task_id == "D3":
        # Generation attribution on longer, prompt-structured texts.
        return SyntheticSpec(
            task_id="D3",
            group_rates={
                0: {"human_pers": 1.8, "human_hedge": 1.3, "human_emo": 1.0,
                    "prompt_cot": 0.7, "llm_list": 0.5},
                1: {"prompt_cot": 2.0, "llm_list": 1.7, "llm_formal": 1.5,
                    "llm_stat": 1.0, "human_pers": 0.7},
            },
            length_mu={0: 4.6, 1: 4.9}, length_sigma={0: 0.50, 1: 0.30},
            punct_rate={0: (0.30, 0.40), 1: (0.10, 0.05)},
            digit_prob={0: 0.25, 1: 0.55}, source_skew=11.8, n_source=14_877,
            confusion_rate=0.03,
        )
    if task_id == "D4":
        # The hard case.  Class 0 = human = majority of the source pool.
        #
        # Two properties are engineered on purpose, because they are the
        # mechanism IWN addresses:
        #
        #  (a) *Near-degenerate majority variance on the minority's own
        #      indicators.*  Human text almost never fires llm_stat /
        #      llm_formal / llm_list, so at a 9.3:1 pool skew the pooled
        #      sigma_k of Eq. (8) on those dimensions is tiny.  Standardising
        #      LLM examples by that scale throws them far into the tail, the
        #      gate's sigmoid saturates, and the fusion degrades.  The
        #      class-balanced sigma^bal_k of Eq. (10) averages in the LLM
        #      class's own (large) variance and restores a usable scale.
        #
        #  (b) *Heteroscedastic length.*  Human text is far more variable in
        #      length than LLM text, so the pooled scale describes neither
        #      class well.
        #
        #  Overlapping group rates keep the task genuinely hard: base models
        #  reach < 0.80 macro-F1 without IWN, as in the paper.
        return SyntheticSpec(
            task_id="D4",
            group_rates={
                0: {"human_pers": 1.7, "human_hedge": 1.3, "human_emo": 1.2,
                    "llm_stat": 0.05, "llm_formal": 0.05, "llm_list": 0.04},
                1: {"llm_stat": 1.1, "llm_formal": 1.0, "llm_list": 0.9,
                    "human_pers": 1.3, "human_hedge": 1.0, "human_emo": 0.9},
            },
            length_mu={0: 3.9, 1: 4.2}, length_sigma={0: 0.80, 1: 0.20},
            punct_rate={0: (0.45, 0.55), 1: (0.20, 0.15)},
            digit_prob={0: 0.35, 1: 0.50}, source_skew=9.3, n_source=30_000,
            confusion_rate=0.12,
        )
    raise KeyError(task_id)


def _render(rng: np.random.Generator, spec: SyntheticSpec, c: int) -> str:
    """Render one document under the class-``c`` conditional."""
    n_words = int(np.clip(rng.lognormal(spec.length_mu[c], spec.length_sigma[c]), 6, 600))
    rates = spec.group_rates.get(c, {})

    tokens: List[str] = []
    for group, rate in rates.items():
        members = DEFAULT_GROUPS[group]
        k = int(rng.poisson(rate))
        if k <= 0:
            continue
        pick = rng.choice(len(members), size=min(k, len(members)), replace=False)
        for i in pick:
            term = members[int(i)]
            # Realise prefix entries as a concrete inflection.
            tokens.append(term[:-1] + rng.choice(["ion", "es", "ing", "ed"])
                          if term.endswith("*") else term)

    n_filler = max(n_words - len(tokens), 0)
    tokens.extend(rng.choice(FILLER_WORDS, size=n_filler, replace=True).tolist())
    rng.shuffle(tokens)

    if rng.random() < spec.digit_prob[c]:
        tokens.insert(int(rng.integers(0, max(len(tokens), 1))), str(int(rng.integers(1, 9999))))

    # Assemble into sentences, then append punctuation per the class rates.
    out: List[str] = []
    i = 0
    while i < len(tokens):
        step = int(np.clip(rng.normal(14, 5), 4, 40))
        out.append(" ".join(tokens[i:i + step]) + ".")
        i += step
    text = " ".join(out) if out else "empty."
    q, e = spec.punct_rate[c]
    text += "?" * int(rng.poisson(q)) + "!" * int(rng.poisson(e))
    return text


def generate_task(task_id: str, seed: int = 0, n_source: Optional[int] = None
                  ) -> Tuple[Corpus, Corpus]:
    """Return ``(source_pool, capped_corpus)`` for one task.

    ``source_pool`` carries the task's raw skew and is what the normaliser is
    fitted on (the paper: "feature normalization uses the full training
    data").  ``capped_corpus`` is the Table 2 evaluation subset.
    """
    from .splits import stratified_cap

    spec = _spec(task_id)
    tspec = TASKS[task_id]
    rng = np.random.default_rng(abs(hash((task_id, seed))) % (2**32))
    n = int(n_source or spec.n_source)

    # Draw labels at the *source* skew (majority = class 0).
    p_major = spec.source_skew / (spec.source_skew + 1.0)
    labels = (rng.random(n) > p_major).astype(np.int64)
    # A ``confusion_rate`` fraction of documents is rendered under the other
    # class's conditional while keeping its label: irreducible Bayes error.
    flip = rng.random(n) < spec.confusion_rate
    render_class = np.where(flip, 1 - labels, labels)
    texts = [_render(rng, spec, int(c)) for c in render_class]
    pool = Corpus(task_id, texts, labels, tspec.n_classes)

    target = [0.5, 0.5] if abs(tspec.minority_pct - 50.0) < 1.0 else None
    if target is None:
        p = tspec.minority_pct / 100.0
        target = [1.0 - p, p]
    capped = stratified_cap(pool, tspec.capped_n, seed=seed, target_priors=target)
    return pool, capped


def generate_all(seed: int = 0, scale: float = 1.0
                 ) -> Dict[str, Tuple[Corpus, Corpus]]:
    """All four tasks.  ``scale < 1`` shrinks every corpus proportionally for
    fast smoke runs while preserving class balance and skew."""
    out: Dict[str, Tuple[Corpus, Corpus]] = {}
    for tid in ["D1", "D2", "D3", "D4"]:
        spec = _spec(tid)
        pool, capped = generate_task(tid, seed=seed,
                                     n_source=max(int(spec.n_source * scale), 200))
        if scale < 1.0:
            from .splits import stratified_cap
            tspec = TASKS[tid]
            capped = stratified_cap(
                pool, max(int(tspec.capped_n * scale), 120), seed=seed,
                target_priors=[0.5, 0.5] if abs(tspec.minority_pct - 50.0) < 1.0 else None,
            )
        out[tid] = (pool, capped)
    return out
