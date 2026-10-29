"""Statistical helpers for experiment summaries.

Provides permutation tests, Cliff's delta, Benjamini-Hochberg adjustment,
bootstrap intervals, paired Wilcoxon tests and binary AUC.
Inputs: numerical sample arrays.
Outputs: effect sizes, test statistics or intervals.
Scope: callers must justify sampling independence and the resampling unit.
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


# ───────────────────────────────────────────────────────────────────
# Effect sizes
# ───────────────────────────────────────────────────────────────────


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size in [-1, 1].

    delta = (#(a_i > b_j) - #(a_i < b_j)) / (n_a * n_b)

    +1 means group a fully dominates b, -1 means b dominates a.
    Convention: |delta| < 0.147 negligible, < 0.33 small, < 0.474 medium, ≥ medium large.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    # vectorized: count gt, lt via searchsorted on sorted b
    b_sorted = np.sort(b)
    n_b = b.size
    # For each a_i: count of b_j < a_i and count of b_j > a_i
    lt_count = np.searchsorted(b_sorted, a, side="left")  # b_j < a_i
    gt_count = n_b - np.searchsorted(b_sorted, a, side="right")  # b_j > a_i
    n_pairs = a.size * n_b
    return float((lt_count.sum() - gt_count.sum()) / n_pairs)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d, parametric effect size (mean difference / pooled std)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n_a, n_b = a.size, b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    s2 = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    s = np.sqrt(s2)
    if s < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / s)


# ───────────────────────────────────────────────────────────────────
# Permutation tests
# ───────────────────────────────────────────────────────────────────


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 2000,
    statistic: str = "mean_diff",
    seed: Optional[int] = None,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Permutation test for two-group comparison.

    Returns (observed_stat, p_value).

    statistic:
      "mean_diff"   : mean(a) - mean(b)
      "median_diff" : median(a) - median(b)

    alternative: "two-sided" | "greater" (a > b) | "less" (a < b)
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    n_a = a.size

    def stat_fn(x, y):
        if statistic == "mean_diff":
            return x.mean() - y.mean()
        elif statistic == "median_diff":
            return np.median(x) - np.median(y)
        else:
            raise ValueError(f"unknown statistic: {statistic}")

    obs = stat_fn(a, b)

    # Vectorized permutation
    count_extreme = 0
    for _ in range(n_perm):
        idx = rng.permutation(pooled.size)
        x = pooled[idx[:n_a]]
        y = pooled[idx[n_a:]]
        s = stat_fn(x, y)
        if alternative == "two-sided":
            if abs(s) >= abs(obs) - 1e-12:
                count_extreme += 1
        elif alternative == "greater":
            if s >= obs - 1e-12:
                count_extreme += 1
        elif alternative == "less":
            if s <= obs + 1e-12:
                count_extreme += 1
        else:
            raise ValueError(f"unknown alternative: {alternative}")

    # +1 smoothing for valid p-value bound
    p = (count_extreme + 1) / (n_perm + 1)
    return float(obs), float(p)


def wilcoxon_signed_rank(d: np.ndarray, alternative: str = "two-sided") -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired differences (e.g., deletion test:
    d_i = score_after_delete_i - score_baseline_i).

    Returns (W_statistic, p_value approx via normal approximation).
    """
    d = np.asarray(d, dtype=np.float64).ravel()
    d = d[d != 0]  # drop zeros
    n = d.size
    if n < 5:
        return float("nan"), float("nan")
    abs_d = np.abs(d)
    ranks = abs_d.argsort().argsort() + 1.0
    # handle ties via average rank
    unique, inverse, counts = np.unique(abs_d, return_inverse=True, return_counts=True)
    tied = counts > 1
    if tied.any():
        # recompute average ranks for ties
        sorted_idx = np.argsort(abs_d)
        avg_ranks = np.empty(n)
        i = 0
        sorted_d = abs_d[sorted_idx]
        while i < n:
            j = i
            while j < n - 1 and sorted_d[j + 1] == sorted_d[i]:
                j += 1
            avg = (i + j + 2) / 2.0
            avg_ranks[sorted_idx[i:j + 1]] = avg
            i = j + 1
        ranks = avg_ranks

    W = float((np.sign(d) * ranks).sum())
    # normal approximation
    var_w = n * (n + 1) * (2 * n + 1) / 6.0
    z = W / np.sqrt(var_w + 1e-12)
    # two-sided
    from math import erf, sqrt
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    if alternative == "two-sided":
        p = 2 * min(cdf, 1 - cdf)
    elif alternative == "greater":
        p = 1 - cdf
    elif alternative == "less":
        p = cdf
    else:
        raise ValueError(f"unknown alternative: {alternative}")
    return float(W), float(np.clip(p, 0, 1))


# ───────────────────────────────────────────────────────────────────
# Multiple-test correction
# ───────────────────────────────────────────────────────────────────


def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (q_values, reject_mask).
    q_values: adjusted p-values
    reject_mask: True where q < alpha
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return p, np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    # BH q-values
    q_ranked = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (running min from right)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.clip(q_ranked, 0, 1)
    return q, q < alpha


# ───────────────────────────────────────────────────────────────────
# Bootstrap CI
# ───────────────────────────────────────────────────────────────────


def bootstrap_ci(
    x: np.ndarray,
    statistic=np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Returns (point_estimate, lower, upper).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(statistic(x))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        boots[i] = statistic(x[idx])
    a = (1 - confidence) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return point, float(lo), float(hi)


# ───────────────────────────────────────────────────────────────────
# Binary classification AUC (event vs control separability)
# ───────────────────────────────────────────────────────────────────


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC under ROC curve.
    scores: continuous values, higher should indicate positive class
    labels: binary {0, 1}
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if scores.size != labels.size or labels.size == 0:
        return float("nan")
    pos_mask = labels == 1
    n_pos = int(pos_mask.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-based AUC = (sum_rank_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks for ties
    unique_scores, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        # recompute with average ranks for ties
        sorted_idx = scores.argsort()
        sorted_s = scores[sorted_idx]
        i = 0
        while i < scores.size:
            j = i
            while j < scores.size - 1 and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            avg = (i + j + 2) / 2.0
            ranks[sorted_idx[i:j + 1]] = avg
            i = j + 1
    sum_rank_pos = ranks[pos_mask].sum()
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)
