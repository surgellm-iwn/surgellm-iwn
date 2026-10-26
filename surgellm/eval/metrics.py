"""Evaluation metrics (§5).

Accuracy, macro-F1, macro precision/recall, ROC-AUC, per-class
precision/recall (needed for the Table 5 D4 breakdown), and task averages.

Macro-F1 is the headline metric throughout the paper; the per-class
breakdown is what makes the IWN mechanism visible (pre-IWN the model is
biased toward the majority "human" class: LLM recall 0.63 vs human recall
0.79; post-IWN both converge around 0.89).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = ["classification_metrics", "per_class_metrics", "macro_f1",
           "aggregate_metrics", "confusion_matrix"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int) -> Dict[str, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": cm.sum(axis=1).astype(float), "confusion": cm}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(per_class_metrics(y_true, y_pred, n_classes)["f1"].mean())


def _roc_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware; no sklearn dependency."""
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           probs: Optional[np.ndarray] = None,
                           n_classes: int = 2,
                           class_names: Optional[Sequence[str]] = None) -> Dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    pc = per_class_metrics(y_true, y_pred, n_classes)

    out: Dict = {
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else float("nan"),
        "macro_f1": float(pc["f1"].mean()),
        "macro_precision": float(pc["precision"].mean()),
        "macro_recall": float(pc["recall"].mean()),
        "n": int(y_true.size),
        "confusion": pc["confusion"].tolist(),
    }
    names = list(class_names) if class_names else [f"class_{i}" for i in range(n_classes)]
    for i, name in enumerate(names[:n_classes]):
        out[f"precision_{name}"] = float(pc["precision"][i])
        out[f"recall_{name}"] = float(pc["recall"][i])
        out[f"f1_{name}"] = float(pc["f1"][i])
        out[f"support_{name}"] = float(pc["support"][i])

    if probs is not None and n_classes == 2:
        probs = np.asarray(probs)
        score = probs[:, 1] if probs.ndim == 2 else probs
        out["roc_auc"] = _roc_auc_binary(y_true, score)
    return out


def aggregate_metrics(per_task: Dict[str, Dict], key: str = "macro_f1") -> float:
    """Unweighted mean over tasks - the paper's "Avg F1" column."""
    vals = [m[key] for m in per_task.values() if not np.isnan(m.get(key, np.nan))]
    return float(np.mean(vals)) if vals else float("nan")
