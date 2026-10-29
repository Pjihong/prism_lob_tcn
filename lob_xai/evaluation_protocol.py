"""Deterministic sample-selection rules used by the paper analyses.

The manuscript evaluates explanations only for *correct target-risk alerts*
and, when more than the analysis budget are available, keeps the alerts with
the largest target-class confidence.  Keeping this rule in one NumPy-only
module prevents figure and counterfactual scripts from silently substituting
random or first-in-file samples.
"""
from __future__ import annotations

import numpy as np


def select_top_confidence_alerts(
    probs: np.ndarray,
    y_true: np.ndarray,
    target_class: int,
    n_windows: int | None,
    *,
    require_correct: bool = True,
) -> np.ndarray:
    """Return deterministic indices for the paper's explanation cohort.

    Parameters
    ----------
    probs
        Class probabilities with shape ``(N, C)``.
    y_true
        Integer labels with shape ``(N,)``.
    target_class
        The risk class whose emitted alerts are being audited.
    n_windows
        Maximum cohort size. ``None`` or a non-positive value keeps every
        eligible alert, still sorted by decreasing confidence.
    require_correct
        If true (the manuscript default), require both ``prediction == target``
        and ``truth == target``.  If false, select all alerts emitted for the
        target class; this is a clearly labelled diagnostic mode.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    if p.ndim != 2:
        raise ValueError(f"probs must have shape (N,C), got {p.shape}")
    if y.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true must be one-dimensional and align with probs")
    c = int(target_class)
    if not 0 <= c < p.shape[1]:
        raise ValueError(f"target_class={c} outside [0, {p.shape[1]})")
    if not np.isfinite(p).all():
        raise ValueError("probs contains NaN or infinite values")

    pred = p.argmax(axis=1)
    eligible = pred == c
    if require_correct:
        eligible &= y == c
    idx = np.flatnonzero(eligible)
    # Stable sorting makes equal-confidence ties resolve by dataset order.
    order = np.argsort(-p[idx, c], kind="stable")
    idx = idx[order]
    if n_windows is not None and int(n_windows) > 0:
        idx = idx[: int(n_windows)]
    return idx.astype(np.int64, copy=False)

