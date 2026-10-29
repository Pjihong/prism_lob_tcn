"""Classification metrics and post-training threshold selection.

Inputs: class labels, prediction probabilities and threshold-selection options.
Outputs: metric dictionaries and validation-selected thresholds.
Scope: stable sorting, NaN handling and legacy flat-class defaults are preserved.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


# ───────────────────────────────────────────────────────────────────
# Classification and transaction metrics
# ───────────────────────────────────────────────────────────────────


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def _f1_macro(logits: torch.Tensor, y: torch.Tensor, n_classes: int = 3) -> float:
    pred = logits.argmax(dim=-1).cpu().numpy()
    tgt = y.cpu().numpy()
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (tgt == c)).sum()
        fp = ((pred == c) & (tgt != c)).sum()
        fn = ((pred != c) & (tgt == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


def _mcc_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int = 3) -> float:
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            C[int(t), int(p)] += 1.0
    N = C.sum()
    if N == 0:
        return 0.0
    t_sum = C.sum(axis=1); p_sum = C.sum(axis=0)
    cov_tp = float((np.diag(C) * N - t_sum * p_sum).sum())
    cov_tt = float(N * N - (t_sum ** 2).sum())
    cov_pp = float(N * N - (p_sum ** 2).sum())
    denom = np.sqrt(cov_tt * cov_pp)
    if denom < 1e-12:
        return 0.0
    return cov_tp / denom


def _per_class_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
) -> Dict[str, float]:
    """Confusion-derived support, prediction count, precision, recall and F1."""
    out: Dict[str, float] = {}
    for c in range(n_classes):
        true_c = y_true == c
        pred_c = y_pred == c
        tp = int((true_c & pred_c).sum())
        support = int(true_c.sum())
        predicted = int(pred_c.sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        out[f"support_class_{c}"] = support
        out[f"predicted_class_{c}"] = predicted
        out[f"precision_class_{c}"] = float(precision)
        out[f"recall_class_{c}"] = float(recall)
        out[f"f1_class_{c}"] = float(f1)
    return out


def _correct_transaction_probability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flat_class: int = 1,
    n_classes: int = 3,
) -> Dict[str, float]:
    """
    CTP — fraction of correct predictions among actionable predictions.

    For multi-class, 'actionable' = pred != flat_class. We also break out per
    non-flat class.
    """
    actionable_classes = [c for c in range(n_classes) if c != flat_class]
    trade_mask = (y_pred != flat_class)
    n_actions = int(trade_mask.sum())

    out = {
        "ctp": float("nan"),
        "coverage": float(trade_mask.mean()) if len(y_pred) else 0.0,
        "n_actions": n_actions,
    }
    # per-actionable-class CTP
    for c in actionable_classes:
        out[f"ctp_class_{c}"] = float("nan")

    if n_actions == 0:
        return out
    correct = (y_pred == y_true) & trade_mask
    out["ctp"] = float(correct.sum() / n_actions)
    for c in actionable_classes:
        mask_c = (y_pred == c)
        n_c = int(mask_c.sum())
        if n_c > 0:
            out[f"ctp_class_{c}"] = float(
                ((y_pred == y_true) & mask_c).sum() / n_c
            )
    return out


# ───────────────────────────────────────────────────────────────────
# Imbalance-aware metrics
# ───────────────────────────────────────────────────────────────────


def _pr_auc_binary(scores: np.ndarray, y: np.ndarray) -> float:
    """
    Area under Precision-Recall curve. Higher = better.
    Robust to imbalance. Returns NaN if no positives.
    """
    y = np.asarray(y, dtype=int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    # Sort by score descending
    order = np.argsort(-s, kind="stable")
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Step-wise PR area in recall order (the original paper implementation).
    # Add a starting point (recall=0, precision=precision_at_top)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 1.0], precision])
    # Compute area
    area = 0.0
    for i in range(1, len(recall)):
        area += (recall[i] - recall[i - 1]) * precision[i]
    return float(area)


def _roc_auc_binary(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC. Tie-aware via average ranks."""
    y = np.asarray(y, dtype=int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # tie handling
    unique_s, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sorted_idx = order
        sorted_s = s[sorted_idx]
        i = 0
        while i < len(s):
            j = i
            while j < len(s) - 1 and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            avg = (i + j + 2) / 2.0
            ranks[sorted_idx[i:j + 1]] = avg
            i = j + 1
    sum_rank_pos = ranks[y == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _lift_at_k(
    scores: np.ndarray, y: np.ndarray, k_frac: float = 0.05,
) -> float:
    """
    Lift at top-k%: precision(top-k%) / base_rate. >1 = better than chance.

    For trading: if we trade only on the top 5% confidence signals, our hit
    rate divided by the natural class prevalence.
    """
    y = np.asarray(y, dtype=int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = len(y)
    if n == 0:
        return float("nan")
    base = float((y == 1).mean())
    if base <= 0:
        return float("nan")
    k = max(1, int(n * k_frac))
    top_idx = np.argpartition(-s, k - 1)[:k]
    prec_topk = float((y[top_idx] == 1).mean())
    return prec_topk / base


def _brier_binary(probs: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def _expected_cost(
    y_true: np.ndarray, y_pred: np.ndarray,
    c_fn: float = 10.0, c_fp: float = 1.0,
) -> float:
    """Asymmetric cost: C_FN per false-negative, C_FP per false-positive,
    averaged over n_samples. Lower = better."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    n = max(1, len(y_true))
    return float((c_fn * fn + c_fp * fp) / n)


def _per_class_pr_auc(
    probs: np.ndarray, y: np.ndarray, n_classes: int,
    flat_class: int = 0,
) -> Dict[str, float]:
    """
    Per-class PR-AUC (one-vs-rest). For multi-class imbalanced tasks (Toxic),
    we macro-average across the non-flat classes — the right thing for
    risk-detection framing where the majority class is by definition uninteresting.
    """
    out = {}
    aucs_non_flat = []
    for c in range(n_classes):
        y_bin = (np.asarray(y, dtype=int) == c).astype(int)
        scores = probs[:, c]
        auc = _pr_auc_binary(scores, y_bin)
        out[f"pr_auc_class_{c}"] = auc
        if c != flat_class:
            if not np.isnan(auc):
                aucs_non_flat.append(auc)
    out["pr_auc_macro_nonflat"] = (
        float(np.mean(aucs_non_flat)) if aucs_non_flat else float("nan")
    )
    return out


def _per_class_roc_auc(
    probs: np.ndarray, y: np.ndarray, n_classes: int,
    flat_class: int = 0,
) -> Dict[str, float]:
    out = {}
    aucs = []
    for c in range(n_classes):
        y_bin = (np.asarray(y, dtype=int) == c).astype(int)
        scores = probs[:, c]
        auc = _roc_auc_binary(scores, y_bin)
        out[f"roc_auc_class_{c}"] = auc
        if c != flat_class:
            if not np.isnan(auc):
                aucs.append(auc)
    out["roc_auc_macro_nonflat"] = (
        float(np.mean(aucs)) if aucs else float("nan")
    )
    return out


# ───────────────────────────────────────────────────────────────────
# Threshold tuning (post-training)
# ───────────────────────────────────────────────────────────────────


def find_threshold_high_precision(
    probs: np.ndarray, y: np.ndarray,
    target_class: int, target_precision: float = 0.5,
) -> Optional[float]:
    """
    Smallest threshold τ on P(y == target_class) such that
        precision(predict y=target_class iff P >= τ) ≥ target_precision.
    Returns None if no threshold meets the target.
    """
    scores = probs[:, target_class]
    y_bin = (np.asarray(y) == target_class).astype(int)
    # candidates: unique scores, sorted descending
    order = np.argsort(-scores, kind="stable")
    y_s = y_bin[order]
    s_s = scores[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    prec = tp / np.maximum(tp + fp, 1e-12)
    # find largest k where prec[k-1] ≥ target
    ok = prec >= target_precision
    if not ok.any():
        return None
    # use the LAST index where ok is still True, then return that score
    last_ok = int(np.where(ok)[0].max())
    return float(s_s[last_ok])


def find_threshold_youden(
    probs: np.ndarray, y: np.ndarray, target_class: int,
) -> float:
    """argmax_τ (TPR(τ) − FPR(τ)). The Youden's J statistic."""
    scores = probs[:, target_class]
    y_bin = (np.asarray(y) == target_class).astype(int)
    order = np.argsort(-scores, kind="stable")
    y_s = y_bin[order]
    s_s = scores[order]
    n_pos = max(1, int(y_bin.sum()))
    n_neg = max(1, len(y_bin) - n_pos)
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    tpr = tp / n_pos
    fpr = fp / n_neg
    j = tpr - fpr
    return float(s_s[int(j.argmax())])


def find_threshold_cost(
    probs: np.ndarray, y: np.ndarray, target_class: int,
    c_fn: float = 10.0, c_fp: float = 1.0,
) -> float:
    """argmin_τ [C_FN · FN(τ) + C_FP · FP(τ)] / N."""
    scores = probs[:, target_class]
    y_bin = (np.asarray(y) == target_class).astype(int)
    n_pos = int(y_bin.sum())
    order = np.argsort(-scores, kind="stable")
    y_s = y_bin[order]
    s_s = scores[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    fn = n_pos - tp
    cost = c_fn * fn + c_fp * fp
    return float(s_s[int(cost.argmin())])
