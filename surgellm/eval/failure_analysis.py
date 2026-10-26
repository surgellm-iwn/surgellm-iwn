"""Failure-case analysis (§7.5) and efficiency accounting (§7.4, Table 12).

§7.5 inspects 50 misclassified examples per task on SURGeLLM-IWN-RoBERTa
and reports four diagnostic modes:

* **D1 (SST-2)** negation scope ("not bad"), sarcasm, mixed sentiment -
  the gate cannot help because polarity vocabulary fires on both sides.
* **D2 (HotPot)** implicit multi-hop chains with no explicit attribution
  cues, so the ``retrieval`` group cannot fire.
* **D3 (LLM-7)** human essays mimicking LLM scaffolding, and LLM essays
  edited to remove enumerative markers.
* **D4 (HumLLM)** short texts (< 30 words) where surgical-feature counts
  are unreliable.

Each mode is operationalised as a detector over the misclassified set, so
"short texts dominate the residual D4 errors" becomes a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..features import SurgicalFeatureExtractor, WORD_RE

__all__ = ["FailureCase", "analyse_failures", "efficiency_row",
           "SHORT_TEXT_THRESHOLD", "N_INSPECTED"]

SHORT_TEXT_THRESHOLD = 30      # §7.5, D4
N_INSPECTED = 50               # examples inspected per task

_NEGATORS = {"not", "no", "never", "nothing", "hardly", "barely", "without",
             "isn't", "wasn't", "don't", "doesn't", "didn't", "cannot", "can't"}


@dataclass
class FailureCase:
    index: int
    task: str
    true_label: int
    pred_label: int
    confidence: float
    n_words: int
    modes: List[str]
    text_preview: str

    def as_row(self) -> Dict:
        return {
            "task": self.task, "index": self.index, "true": self.true_label,
            "pred": self.pred_label, "confidence": round(self.confidence, 4),
            "n_words": self.n_words, "modes": "|".join(self.modes) or "unclassified",
            "preview": self.text_preview,
        }


def _detect_modes(task: str, text: str, s_raw: np.ndarray,
                  group_names: Sequence[str]) -> List[str]:
    lowered = text.lower()
    tokens = WORD_RE.findall(lowered)
    n_words = len(tokens)
    gidx = {g: i for i, g in enumerate(group_names)}
    modes: List[str] = []

    def g(name: str) -> float:
        i = gidx.get(name)
        return float(s_raw[i]) if i is not None and i < len(s_raw) else 0.0

    if task == "D1":
        if any(t in _NEGATORS for t in tokens):
            modes.append("negation_scope")
        if g("sst_pos") > 0 and g("sst_neg") > 0:
            modes.append("mixed_sentiment_polarity_fires_both_sides")
        if "but " in lowered or "though" in lowered:
            modes.append("contrastive_sarcasm_candidate")
    elif task == "D2":
        if g("retrieval") == 0:
            modes.append("no_attribution_cue_retrieval_group_silent")
        if "[ctx]" in lowered and len(lowered.split("[ctx]")) > 1:
            if len(lowered.split("[ctx]")[1].split()) < 60:
                modes.append("short_context")
    elif task == "D3":
        human_sig = g("human_pers") + g("human_hedge") + g("human_emo")
        llm_sig = g("llm_list") + g("llm_formal") + g("prompt_cot") + g("llm_stat")
        if human_sig > 0 and llm_sig >= human_sig:
            modes.append("human_text_mimics_llm_scaffolding")
        if llm_sig == 0:
            modes.append("enumerative_markers_absent")
    elif task == "D4":
        if n_words < SHORT_TEXT_THRESHOLD:
            modes.append("short_text_unreliable_counts")
        if float(s_raw[: len(group_names)].sum()) == 0.0:
            modes.append("no_indicator_fires")

    if n_words < SHORT_TEXT_THRESHOLD and "short_text_unreliable_counts" not in modes:
        modes.append("short_text")
    return modes


def analyse_failures(task: str, texts: Sequence[str], y_true: np.ndarray,
                     y_pred: np.ndarray, probs: Optional[np.ndarray] = None,
                     extractor: Optional[SurgicalFeatureExtractor] = None,
                     n_inspect: int = N_INSPECTED, seed: int = 0) -> Dict:
    """Classify up to ``n_inspect`` misclassified examples into failure modes."""
    extractor = extractor or SurgicalFeatureExtractor()
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    wrong = np.flatnonzero(y_true != y_pred)

    rng = np.random.default_rng(seed)
    if len(wrong) > n_inspect:
        wrong = rng.permutation(wrong)[:n_inspect]

    group_names = extractor.vocabulary.group_names
    cases: List[FailureCase] = []
    for i in wrong:
        text = texts[int(i)]
        s_raw = extractor.extract(text)
        conf = float(probs[int(i)].max()) if probs is not None else float("nan")
        cases.append(FailureCase(
            index=int(i), task=task, true_label=int(y_true[i]), pred_label=int(y_pred[i]),
            confidence=conf, n_words=len(WORD_RE.findall(text.lower())),
            modes=_detect_modes(task, text, s_raw, group_names),
            text_preview=" ".join(text.split()[:18]),
        ))

    counts: Dict[str, int] = {}
    for c in cases:
        for m in (c.modes or ["unclassified"]):
            counts[m] = counts.get(m, 0) + 1

    lengths = [c.n_words for c in cases]
    return {
        "task": task,
        "n_total": int(y_true.size),
        "n_errors": int((y_true != y_pred).sum()),
        "error_rate": float((y_true != y_pred).mean()) if y_true.size else float("nan"),
        "n_inspected": len(cases),
        "mode_counts": counts,
        "mode_fractions": {k: v / max(len(cases), 1) for k, v in counts.items()},
        "median_error_length": float(np.median(lengths)) if lengths else float("nan"),
        "frac_errors_short": (
            float(np.mean([l < SHORT_TEXT_THRESHOLD for l in lengths])) if lengths else float("nan")
        ),
        "cases": [c.as_row() for c in cases],
    }


def efficiency_row(model_name: str, n_params: int, train_seconds: float,
                   avg_f1: float, baseline_seconds: Optional[float] = None) -> Dict:
    """One row of Table 12.

        F1/min = F1 * 60 / T(s)
        Eff    = F1 * 10^3 / log10(P)
    """
    f1_per_min = avg_f1 * 60.0 / max(train_seconds, 1e-9)
    eff = avg_f1 * 1e3 / max(np.log10(max(n_params, 10)), 1e-9)
    return {
        "model": model_name,
        "parameters": int(n_params),
        "train_seconds": round(float(train_seconds), 1),
        "overhead_x": (round(train_seconds / baseline_seconds, 1)
                       if baseline_seconds else 1.0),
        "avg_f1": round(float(avg_f1), 4),
        "f1_per_min": round(float(f1_per_min), 3),
        "efficiency": round(float(eff), 1),
    }


def pareto_frontier(rows: Sequence[Dict], time_key: str = "train_seconds",
                    score_key: str = "avg_f1") -> List[Dict]:
    """Mark the Pareto-efficient models on the (training time, Avg F1) axes."""
    out = [dict(r) for r in rows]
    for r in out:
        r["pareto_efficient"] = not any(
            (o[time_key] <= r[time_key] and o[score_key] > r[score_key]) or
            (o[time_key] < r[time_key] and o[score_key] >= r[score_key])
            for o in out if o is not r
        )
    return out
