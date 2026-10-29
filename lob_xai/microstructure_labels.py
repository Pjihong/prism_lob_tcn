"""
microstructure_labels.py — Compute Toxic-Flow and Liquidity-Vacuum labels
                          from raw FI-2010 features.  (SEGMENT-AWARE)

These labels REPLACE the FI-2010 native 5-horizon mid-price-change labels.
They are derived from the *normalized* (z-scored) raw features X[:, 0..3]:
    col 0 : Ask1_Price
    col 1 : Ask1_Volume
    col 2 : Bid1_Price
    col 3 : Bid1_Volume

Domain motivation: see the paper (Section 3). In a single sentence: a market
maker cares far more about being run-over (toxic flow) or being stranded
(liquidity vacuum) than about a 1-tick mid-price wiggle.

Critical: NO LABEL LEAKAGE, AND NO CROSS-SEGMENT LEAKAGE.
    label[t] = f( X[t+k, 0..3] )  with k=10
    window[t] = X[t-99 : t+1]
    → label depends only on future ticks NOT in the window.

The FI-2010 release is 5 stocks (× multiple days) concatenated into a single
array. The future tick X[t+k] used to build label[t] must live in the SAME
segment (stock/day) as t — otherwise we compute "did THIS stock's price
collapse?" using a DIFFERENT stock's quotes. We therefore pass a per-row
`segment_id`; rows whose t+k falls in a different segment are marked invalid
(and excluded from windows downstream).

If `segment_id` is None the module behaves exactly as the pre-segment version
(single continuous sequence; only the last k rows are invalid).

Tick-size in z-score space: empirically ≈ 0.00098 for the standard FI-2010
release. NOTE: FI-2010 z-scores PER STOCK, so the tick size in normalized
units differs per stock; audit per-segment with audit_tick_size if you want
per-stock thresholds (see note at bottom).
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Sequence
import numpy as np


# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────

TICK_SIZE_DEFAULT = 0.00098
K_DEFAULT = 10

# Toxic Flow: 3 ticks of one-sided price collapse within k events.
N_TICKS_TOXIC = 3

# Liquidity Vacuum: top-of-book volume drop AND spread widening, simultaneously.
VOL_DROP_VACUUM = 1.5
N_TICKS_SPREAD_VACUUM = 1


# ═══════════════════════════════════════════════════════════════════
# Segment utilities  (pure numpy — no torch dependency)
# ═══════════════════════════════════════════════════════════════════


def make_segment_id(n: int, boundaries: Sequence[int]) -> np.ndarray:
    """
    Build a per-row segment id array of length n from a list of internal
    cut indices.

    A boundary value `b` means "a NEW segment starts at row b". So
    boundaries=[100, 250] over n=400 gives:
        rows 0..99   -> segment 0
        rows 100..249-> segment 1
        rows 250..399-> segment 2

    The ids are contiguous & monotonically non-decreasing, which the window
    enumeration below relies on (checking the two endpoints of a window is
    then enough to know the whole window lies in one segment).
    """
    seg = np.zeros(int(n), dtype=np.int32)
    for j, b in enumerate(sorted(int(x) for x in boundaries)):
        if 0 < b < n:
            seg[b:] = j + 1
    return seg


def segment_bounds(segment_id: np.ndarray) -> List[Tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for each contiguous segment."""
    segment_id = np.asarray(segment_id)
    if len(segment_id) == 0:
        return []
    change = np.flatnonzero(np.diff(segment_id)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(segment_id)]))
    return list(zip(starts.tolist(), ends.tolist()))


def segment_aware_starts(
    segment_id: np.ndarray,
    valid_row: np.ndarray,
    seq_len: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Enumerate window start indices such that:
      (1) the whole input window X[s : s+seq_len] lies in a SINGLE segment, and
      (2) the label at the last window position (s+seq_len-1) is well-defined
          (valid_row already guarantees its future tick t+k is in-segment).

    Because segment ids are contiguous & monotone, condition (1) reduces to
    "segment_id at the first and last row of the window are equal".
    """
    segment_id = np.asarray(segment_id)
    N = len(segment_id)
    all_starts = np.arange(0, N - seq_len + 1, stride)
    if len(all_starts) == 0:
        return all_starts.astype(np.int64)
    lp = all_starts + seq_len - 1                       # label positions
    same_seg_window = segment_id[all_starts] == segment_id[lp]
    label_ok = valid_row[lp]
    return all_starts[same_seg_window & label_ok].astype(np.int64)


def split_within_segments(
    X: np.ndarray,
    segment_id: np.ndarray,
    val_ratio: float,
    purge_gap: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/val split done INDEPENDENTLY inside each segment
    (Design A: train = early part of every stock, val = late part of every
    stock). This preserves the "predict the future from the past" property
    per stock and never lets a single stock leak across the train/val cut.

    Returns
    -------
    X_train, seg_train, X_val, seg_val
        Re-indexed contiguous arrays. Each original segment j contributes its
        early rows to X_train (with segment id j) and its late rows to X_val
        (also id j). Order within each split follows segment order.

    NOTE: this is *within-segment* splitting. For *by-segment* splitting
    (some whole stocks → train, others → val), build the train/val arrays by
    selecting whole segments instead and call the dataset directly.

    purge_gap (opt-in, default 0 = frozen-protocol behaviour): drop this many
    rows at the start of each segment's validation part.  Train and validation
    windows never share rows even without it, but adjacent windows across the
    boundary are temporally correlated; a purge gap of about seq_len + k rows
    weakens that dependence.  A non-zero gap changes the realized split, so it
    must be recorded in the run config and stated in the manuscript — never
    mix purged and unpurged runs in one table.
    """
    if purge_gap < 0:
        raise ValueError("purge_gap must be >= 0")
    bounds = segment_bounds(segment_id)
    Xtr, Xva, str_, sva = [], [], [], []
    F = X.shape[1]
    for j, (a, b) in enumerate(bounds):
        L = b - a
        n_val = int(L * val_ratio)
        n_tr = L - n_val
        Xtr.append(X[a:a + n_tr])
        str_.append(np.full(n_tr, j, dtype=np.int32))
        val_start = a + n_tr + min(purge_gap, n_val)
        Xva.append(X[val_start:b])
        sva.append(np.full(b - val_start, j, dtype=np.int32))
    X_train = np.concatenate(Xtr, axis=0) if Xtr else np.empty((0, F), X.dtype)
    X_val = np.concatenate(Xva, axis=0) if Xva else np.empty((0, F), X.dtype)
    seg_train = np.concatenate(str_) if str_ else np.zeros(0, np.int32)
    seg_val = np.concatenate(sva) if sva else np.zeros(0, np.int32)
    return X_train, seg_train, X_val, seg_val


def detect_segment_boundaries(
    X: np.ndarray,
    n_internal: int = 4,
    min_gap: int = 5000,
    price_cols: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    HEURISTIC fallback boundary detector (use only if you cannot recover the
    true per-stock row counts). Picks the `n_internal` largest jumps in the
    average top-of-book price level, enforcing a minimum spacing so a single
    anomalous patch cannot produce a cluster of fake boundaries.

    WARNING: FI-2010 normalizes per stock, which can MASK boundaries. Always
    sanity-check the result with `validate_segments` (per-segment mean/std)
    and prefer passing explicit boundaries when known.
    """
    X = np.asarray(X, dtype=float)
    if price_cols is None:
        price_cols = list(range(0, min(40, X.shape[1]), 2))  # 0,2,...,38 = prices
    level = X[:, list(price_cols)].mean(axis=1)
    d = np.abs(np.diff(level))
    order = np.argsort(d)[::-1]
    picked: List[int] = []
    for i in order:
        i = int(i)
        if all(abs(i - p) >= min_gap for p in picked):
            picked.append(i)
        if len(picked) >= n_internal:
            break
    return sorted(p + 1 for p in picked)


def validate_segments(
    X: np.ndarray,
    boundaries: Sequence[int],
    label: str = "",
    col: int = 0,
) -> None:
    """
    Print per-segment size and (mean, std) of one price column. For a
    per-stock z-scored release each segment's price column should be roughly
    mean≈0, std≈1; large deviations suggest the boundaries are wrong.
    """
    X = np.asarray(X, dtype=float)
    seg = make_segment_id(len(X), boundaries)
    bounds = segment_bounds(seg)
    print(f"[seg] {label}: {len(bounds)} segments, boundaries={list(boundaries)}")
    for j, (a, b) in enumerate(bounds):
        s = X[a:b, col]
        flag = "" if 0.5 <= s.std() <= 1.8 else "  ⚠ std≉1 (경계 의심)"
        print(f"[seg]   seg{j}: rows {a:>8d}-{b:<8d} (n={b-a:>7d})  "
              f"col{col} mean={s.mean():+.3f} std={s.std():.3f}{flag}")


# ───────────────────────────────────────────────────────────────────
# Toxic Flow Label (3-class)
# ───────────────────────────────────────────────────────────────────


def compute_toxic_labels(
    X: np.ndarray,
    k: int = K_DEFAULT,
    tick_size: float = TICK_SIZE_DEFAULT,
    n_ticks: int = N_TICKS_TOXIC,
    segment_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Toxic Flow labels (3-class).

    Class 0 (Stationary): neither up nor down toxic sweep
    Class 1 (Toxic Up):   Ask1_Price[t+k] >= Ask1_Price[t] + n_ticks * tick_size
    Class 2 (Toxic Down): Bid1_Price[t+k] <= Bid1_Price[t] - n_ticks * tick_size

    Returns
    -------
    y     : (N,) int8 in {0,1,2}; invalid rows are forced to 0.
    valid : (N,) bool; True iff t < N-k AND (segment_id[t+k] == segment_id[t]).
    """
    n = X.shape[0]
    threshold = n_ticks * tick_size
    ask1_p = X[:, 0]
    bid1_p = X[:, 2]

    y = np.zeros(n, dtype=np.int8)
    valid = np.ones(n, dtype=bool)
    if n <= k:
        valid[:] = False
        return y, valid

    valid[-k:] = False                                  # no future for last k
    if segment_id is not None:                          # no CROSS-segment future
        segment_id = np.asarray(segment_id)
        same = np.zeros(n, dtype=bool)
        same[:-k] = segment_id[k:] == segment_id[:-k]
        valid &= same

    ask_future = ask1_p[k:]
    bid_future = bid1_p[k:]
    ask_now = ask1_p[:-k]
    bid_now = bid1_p[:-k]

    toxic_up_mask = ask_future >= ask_now + threshold
    toxic_down_mask = bid_future <= bid_now - threshold

    y_part = np.zeros(n - k, dtype=np.int8)
    y_part[toxic_up_mask] = 1
    y_part[toxic_down_mask] = 2
    both = toxic_up_mask & toxic_down_mask
    if both.any():                                      # resolve by larger move
        up_mag = ask_future - ask_now
        dn_mag = bid_now - bid_future
        up_wins = both & (up_mag >= dn_mag)
        y_part[up_wins] = 1

    y[:n - k] = y_part
    y[~valid] = 0                                       # drop cross-segment labels
    return y, valid


# ───────────────────────────────────────────────────────────────────
# Liquidity Vacuum Label (2-class)
# ───────────────────────────────────────────────────────────────────


def compute_vacuum_labels(
    X: np.ndarray,
    k: int = K_DEFAULT,
    tick_size: float = TICK_SIZE_DEFAULT,
    vol_drop: float = VOL_DROP_VACUUM,
    n_ticks_spread: int = N_TICKS_SPREAD_VACUUM,
    segment_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Liquidity Vacuum labels (binary).

    Class 1 (Vacuum):
      ( Ask1_Vol[t+k] <= Ask1_Vol[t] - vol_drop  OR
        Bid1_Vol[t+k] <= Bid1_Vol[t] - vol_drop )
      AND
        ( spread[t+k] >= spread[t] + n_ticks_spread * tick_size )
    Class 0 (Normal): otherwise.

    `vol_drop` is in standard deviations (X is z-score normalized);
    `spread = Ask1_Price - Bid1_Price`.

    Returns y, valid with the same segment-aware semantics as the toxic case.
    """
    n = X.shape[0]
    threshold_spread = n_ticks_spread * tick_size

    ask_p = X[:, 0]; ask_v = X[:, 1]
    bid_p = X[:, 2]; bid_v = X[:, 3]
    spread = ask_p - bid_p

    y = np.zeros(n, dtype=np.int8)
    valid = np.ones(n, dtype=bool)
    if n <= k:
        valid[:] = False
        return y, valid

    valid[-k:] = False
    if segment_id is not None:
        segment_id = np.asarray(segment_id)
        same = np.zeros(n, dtype=bool)
        same[:-k] = segment_id[k:] == segment_id[:-k]
        valid &= same

    ask_v_now = ask_v[:-k]; ask_v_future = ask_v[k:]
    bid_v_now = bid_v[:-k]; bid_v_future = bid_v[k:]
    spread_now = spread[:-k]; spread_future = spread[k:]

    ask_dropped = ask_v_future <= ask_v_now - vol_drop
    bid_dropped = bid_v_future <= bid_v_now - vol_drop
    spread_widened = spread_future >= spread_now + threshold_spread

    vacuum = (ask_dropped | bid_dropped) & spread_widened
    y[:n - k] = vacuum.astype(np.int8)
    y[~valid] = 0
    return y, valid


# ───────────────────────────────────────────────────────────────────
# Convenience wrapper
# ───────────────────────────────────────────────────────────────────


def compute_microstructure_labels(
    X: np.ndarray,
    task: str = "toxic",
    k: int = K_DEFAULT,
    tick_size: float = TICK_SIZE_DEFAULT,
    segment_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Dispatch to the right labeling function (segment-aware).

    Returns
    -------
    y     : (N,) int8 labels
    valid : (N,) bool — well-defined AND in-segment future
    info  : dict (n_classes, flat_class, class_names, prevalence on valid rows)
    """
    if task == "toxic":
        y, valid = compute_toxic_labels(X, k=k, tick_size=tick_size,
                                        segment_id=segment_id)
        info = dict(
            task=task, n_classes=3, flat_class=0,
            class_names=["stationary", "toxic_up", "toxic_down"],
            k=k, tick_size=tick_size, n_ticks_threshold=N_TICKS_TOXIC,
        )
    elif task == "vacuum":
        y, valid = compute_vacuum_labels(X, k=k, tick_size=tick_size,
                                         segment_id=segment_id)
        info = dict(
            task=task, n_classes=2, flat_class=0,
            class_names=["normal", "vacuum"],
            k=k, tick_size=tick_size,
            vol_drop=VOL_DROP_VACUUM, n_ticks_spread=N_TICKS_SPREAD_VACUUM,
        )
    else:
        raise ValueError(f"task must be 'toxic' or 'vacuum', got {task!r}")

    y_v = y[valid]
    counts = np.bincount(y_v.astype(int), minlength=info["n_classes"])
    prev = counts / max(1, len(y_v))
    info["counts"] = {info["class_names"][c]: int(counts[c])
                      for c in range(info["n_classes"])}
    info["prevalence"] = {info["class_names"][c]: float(prev[c])
                          for c in range(info["n_classes"])}
    info["n_valid"] = int(valid.sum())
    info["n_total"] = int(len(X))
    if segment_id is not None:
        info["n_segments"] = int(len(segment_bounds(segment_id)))
    return y, valid, info


# ───────────────────────────────────────────────────────────────────
# Per-window validity (kept for backward compat with the pre-segment path)
# ───────────────────────────────────────────────────────────────────


def mask_valid_labels(valid: np.ndarray, seq_len: int = 100) -> np.ndarray:
    """
    Convert a per-row validity mask to a per-WINDOW validity mask, ignoring
    segment membership. Retained only for callers that do not use segments;
    the segment-aware path uses `segment_aware_starts` instead.
    """
    n = len(valid)
    if n < seq_len:
        return np.zeros(0, dtype=bool)
    last_positions = np.arange(seq_len - 1, n)
    return valid[last_positions]


def audit_tick_size(
    X: np.ndarray,
    n_samples: int = 200_000,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """
    Empirically estimate the tick size in the z-score normalized space, as
    the median of the smallest positive |ΔAsk1_Price| steps. Run this PER
    SEGMENT (slice X to one stock first) to get per-stock tick sizes, since
    FI-2010 z-scores per stock.
    """
    rng = np.random.default_rng(rng_seed)
    n = X.shape[0]
    n_samples = min(n_samples, n - 1)
    idx = rng.choice(n - 1, size=n_samples, replace=False)
    diffs = np.abs(X[idx + 1, 0] - X[idx, 0])
    nz = diffs[diffs > 1e-12]
    out = dict(
        n_zero_steps=int((diffs <= 1e-12).sum()),
        min_step=float(nz.min()) if len(nz) > 0 else float("nan"),
        median_step_smallest_1pct=float(
            np.median(np.sort(nz)[: max(1, len(nz) // 100)])
        ) if len(nz) > 0 else float("nan"),
    )
    out["recommended"] = out["median_step_smallest_1pct"]
    return out


if __name__ == "__main__":
    # smoke test with 3 segments
    rng = np.random.default_rng(0)
    seg_sizes = [500, 700, 400]
    parts = []
    for j, s in enumerate(seg_sizes):
        block = rng.standard_normal((s, 40)).astype(np.float32) * 0.001
        block[:, 0] += j * 2.0  # distinct price level per "stock"
        parts.append(block)
    X = np.concatenate(parts, axis=0)
    X[100:120, 0] += 0.005       # toxic up inside seg0
    X[800:820, 2] -= 0.005       # toxic down inside seg1
    boundaries = [500, 1200]
    seg = make_segment_id(len(X), boundaries)

    y_t, v_t, info_t = compute_microstructure_labels(X, "toxic", segment_id=seg)
    y_v, v_v, info_v = compute_microstructure_labels(X, "vacuum", segment_id=seg)
    print("Toxic :", info_t["prevalence"], "n_valid=", info_t["n_valid"])
    print("Vacuum:", info_v["prevalence"], "n_valid=", info_v["n_valid"])
    print("detected boundaries:", detect_segment_boundaries(X, 2, min_gap=200))