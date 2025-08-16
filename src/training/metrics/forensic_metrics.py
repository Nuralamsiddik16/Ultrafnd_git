# src/training/metrics/forensic_metrics.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------
# Safe helpers
# ---------------------------

def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    ROC AUC that never crashes:
    - if only one class is present → returns 0.5 (chance level).
    - if any error occurs → returns 0.5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    try:
        if y_true.size == 0 or np.unique(y_true).size < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return 0.5


def _to_prob_1(y_score: np.ndarray) -> np.ndarray:
    """
    Normalize model outputs to positive-class probabilities with shape (N,).
    Accepts:
      - (N,) already-prob
      - (N,2) softmax probabilities → takes column 1
      - (N,2) logits → applies softmax and takes column 1
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        return y_score
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        row_sum = y_score.sum(axis=1)
        if np.allclose(row_sum, 1.0, atol=1e-3):   # looks like probs
            return y_score[:, 1]
        # logits → softmax
        z = y_score - y_score.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p = ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)
        return p[:, 1]
    # Fallback: max across classes
    return np.max(y_score, axis=1)

# ---------------------------
# Core classification metrics
# ---------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    include_cm: bool = False,
) -> Dict[str, float]:
    """
    Safe, publication-ready set of metrics.
    Returns dict with: accuracy, auc, precision, recall, f1, (and cm_* if requested)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = _to_prob_1(y_score).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    auc = _safe_auc(y_true, y_prob)
    prec = float(precision_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else 0.0

    out: Dict[str, float] = {
        "accuracy": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if include_cm and y_true.size:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({
            "cm_tn": float(tn),
            "cm_fp": float(fp),
            "cm_fn": float(fn),
            "cm_tp": float(tp),
        })
    return out

# ---------------------------
# Forensic metrics (CMCS, DFDR)
# ---------------------------

def compute_cmcs(
    semantic_conflict: np.ndarray,
    temporal_delay: np.ndarray,
) -> float:
    """
    Cross-Modal Consistency Score (CMCS) ∈ [0,1]:
      Higher means more consistent (lower conflict and delay).
      CMCS = 1 - mean( min(1, 0.5*(conflict + delay)) )
    Inputs are expected in [0,1].
    """
    sc = np.asarray(semantic_conflict).astype(float)
    td = np.asarray(temporal_delay).astype(float)
    mix = 0.5 * (sc + td)
    mix = np.clip(mix, 0.0, 1.0)
    return float(1.0 - mix.mean()) if mix.size else 0.0


def compute_dfdr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    DeepFake Detection Rate (DFDR): TPR at a chosen threshold
      DFDR = TP / (TP + FN) on the positive (fake) class.
    Returns 0 if there are no positive samples.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = _to_prob_1(y_score).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    pos = (y_true == 1)
    denom = float(pos.sum())
    if denom < 1.0:
        return 0.0
    tp = float((y_pred[pos] == 1).sum())
    return tp / denom


def aggregate_epoch_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    forensic: Optional[Dict[str, np.ndarray]] = None,
    threshold: float = 0.5,
    include_cm: bool = False,
) -> Dict[str, float]:
    """
    One-stop aggregation for logging per split/epoch.
    - y_true: (N,)
    - y_score: (N,) or (N,2)
    - forensic: optional dict with keys:
        * "semantic_conflict": (N,) in [0,1]
        * "temporal_delay":    (N,) in [0,1]
        * "emotion_intensity": (N,) in [0,1]  (logged as mean)
    """
    cls = compute_classification_metrics(y_true, y_score, threshold=threshold, include_cm=include_cm)
    if forensic:
        sc = forensic.get("semantic_conflict", None)
        td = forensic.get("temporal_delay", None)
        if sc is not None and td is not None:
            cls["cmcs"] = compute_cmcs(sc, td)
        ei = forensic.get("emotion_intensity", None)
        if ei is not None:
            ei = np.asarray(ei).astype(float)
            cls["emotion_intensity_mean"] = float(ei.mean()) if ei.size else 0.0
        cls["dfdr"] = compute_dfdr(y_true, y_score, threshold=threshold)
    return cls


def pretty_print(split: str, m: Dict[str, float]) -> None:
    """Compact, stable printer for logs."""
    ordered = ["accuracy", "auc", "precision", "recall", "f1", "cmcs", "dfdr"]
    extras = [k for k in m.keys() if k not in ordered and not k.startswith("cm_")]
    line = " | ".join([f"{k}:{m[k]:.4f}" for k in ordered if k in m])
    if extras:
        line += " | " + " ".join([f"{k}:{m[k]:.4f}" for k in extras])
    print(f"[{split}] {line}")
