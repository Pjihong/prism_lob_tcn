"""Load FI-2010 features and build chronological microstructure windows.

Microstructure labels are recomputed from the first 40 feature columns.
Explicit segment boundaries constrain both windows and label horizons;
omitting boundaries treats each file as one segment. Train/validation splitting
occurs within each segment, with a default purge gap of zero.
Inputs: indexed, header-bearing CSV files and task/window configuration.
Outputs: datasets and dataloaders for train, validation and test.
Scope: legacy mid-price loaders remain available but use different labels.
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, List, Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .branching import make_branching_scheme, scheme_metadata
from .microstructure_labels import (
    compute_microstructure_labels, mask_valid_labels,
    make_segment_id, segment_bounds, segment_aware_starts,
    split_within_segments, detect_segment_boundaries, validate_segments,
    TICK_SIZE_DEFAULT, K_DEFAULT,
)
from .paper_protocol import DATA, TRAINING


# ───────────────────────────────────────────────────────────────────
# Raw CSV → numpy (legacy, unchanged)
# ───────────────────────────────────────────────────────────────────


def load_fi2010_csv(path: str, n_features: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """FI-2010 CSV 로드 (NaN-safe; 라벨 1/2/3 → 0/1/2 변환)."""
    df = pd.read_csv(path, index_col=0)
    n_orig = len(df)

    label_cols = df.columns[-5:]
    nan_mask = df[label_cols].isna().any(axis=1)
    n_bad = int(nan_mask.sum())
    if n_bad > 0:
        print(f"[data]   warning: dropping {n_bad}/{n_orig} rows with NaN label(s)")
        df = df[~nan_mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No valid rows in {path} after NaN drop")

    arr = df.values
    X = arr[:, :n_features].astype(np.float32)
    Y_raw = np.round(arr[:, -5:].astype(np.float64)).astype(np.int64)

    if Y_raw.min() < 1 or Y_raw.max() > 3:
        bad_vals = np.unique(Y_raw[(Y_raw < 1) | (Y_raw > 3)])[:5]
        raise ValueError(
            f"Found label values outside [1, 3]. "
            f"Sample bad values: {bad_vals.tolist()}. Path: {path}"
        )

    Y = Y_raw - 1  # → 0=up, 1=flat, 2=down
    return X, Y


def load_fi2010_features_only(path: str, n_features: int = 40) -> np.ndarray:
    """
    Load only the first n_features columns from an FI-2010 CSV (drops the
    trailing label columns and any extra engineered features). Used for
    microstructure tasks where labels are computed from X.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] < n_features:
        raise ValueError(
            f"{path} has only {df.shape[1]} data columns after index_col=0; "
            f"expected at least {n_features}. Verify the CSV index/header schema."
        )
    feat_nan = df.iloc[:, :n_features].isna().any(axis=1)
    n_bad = int(feat_nan.sum())
    if n_bad > 0:
        # Removing isolated rows would stitch unrelated event times together
        # and create artificial sliding windows.  Paper runs therefore fail
        # closed; repair the source or provide explicit segment breaks.
        raise ValueError(
            f"{path} contains {n_bad}/{len(df)} rows with NaN features; "
            "refusing to drop rows and alter temporal adjacency"
        )
    return df.iloc[:, :n_features].values.astype(np.float32)


# ───────────────────────────────────────────────────────────────────
# Legacy sliding-window dataset (unchanged)
# ───────────────────────────────────────────────────────────────────


class FI2010WindowDataset(Dataset):
    """FI-2010 sliding window dataset. Each item = (X[i:i+seq_len, :], y_horizon)."""

    def __init__(self, X, Y, seq_len=100, horizon_idx=4, stride=1):
        assert X.shape[0] == Y.shape[0]
        assert 0 <= horizon_idx < Y.shape[1]
        col = Y[:, horizon_idx]
        if col.size > 0 and (col.min() < 0 or col.max() >= 100):
            raise ValueError(
                f"label column has out-of-range values "
                f"(min={col.min()}, max={col.max()}). "
                f"This is likely a NaN→int64 conversion artifact — "
                f"check load_fi2010_csv output."
            )
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(Y[:, horizon_idx]).long()
        self.seq_len = seq_len
        self.stride = stride
        self.starts = np.arange(0, X.shape[0] - seq_len + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.X[s: s + self.seq_len]
        y = self.y[s + self.seq_len - 1]
        return x, y

    def get_all_labels(self):
        last_idx = self.starts + self.seq_len - 1
        return self.y[torch.from_numpy(last_idx).long()].numpy()


# ───────────────────────────────────────────────────────────────────
# Microstructure sliding-window dataset (SEGMENT-AWARE)
# ───────────────────────────────────────────────────────────────────


class FI2010MicrostructureDataset(Dataset):
    """
    Segment-aware microstructure sliding window dataset.

    Labels are computed from raw X at construction time (Toxic or Vacuum)
    using future X[t+k]. Only windows that lie entirely within one segment
    AND whose label position has an in-segment future are exposed.

    Parameters
    ----------
    X          : (N, F) features (z-score normalized FI-2010 raw)
    segment_id : (N,) int per-row segment id. None ⇒ single segment.
    task       : "toxic" or "vacuum"
    seq_len    : window size (default 100)
    k          : prediction horizon for label (default 10)
    tick_size  : tick size in normalized units (default 0.00098)
    stride     : window stride (default 1)
    """

    def __init__(
        self,
        X: np.ndarray,
        segment_id: Optional[np.ndarray] = None,
        task: str = "toxic",
        seq_len: int = 100,
        k: int = K_DEFAULT,
        tick_size: float = TICK_SIZE_DEFAULT,
        stride: int = 1,
    ):
        assert X.ndim == 2
        N = X.shape[0]
        if segment_id is None:
            segment_id = np.zeros(N, dtype=np.int32)
        segment_id = np.asarray(segment_id)
        assert len(segment_id) == N, "segment_id length must match X rows"

        y_full, valid_row, info = compute_microstructure_labels(
            X, task=task, k=k, tick_size=tick_size, segment_id=segment_id,
        )
        starts = segment_aware_starts(segment_id, valid_row, seq_len, stride)

        self.X = torch.from_numpy(X).float()
        self.y_full = torch.from_numpy(y_full).long()
        self.segment_id = segment_id
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.starts = starts
        self.task = task
        self.k = int(k)
        self.tick_size = float(tick_size)
        self.info = info

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        x = self.X[s: s + self.seq_len]
        y = self.y_full[s + self.seq_len - 1]
        return x, y

    def get_all_labels(self):
        if len(self.starts) == 0:
            return np.array([], dtype=np.int64)
        last_idx = self.starts + self.seq_len - 1
        return self.y_full[torch.from_numpy(last_idx).long()].numpy()


# ───────────────────────────────────────────────────────────────────
# Legacy build_dataloaders (unchanged — kept for backward compat)
# ───────────────────────────────────────────────────────────────────


def build_dataloaders(
    train_csv: str,
    test_csv: str,
    seq_len: int = 100,
    horizon_idx: int = 4,
    n_features: int = 40,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    branching_scheme: str = "type4",
    custom_branch_indices: Optional[Dict[str, List[int]]] = None,
) -> dict:
    """Build train / val / test DataLoaders + meta dict (legacy mid-price)."""
    print(f"[data] Loading train CSV: {train_csv}")
    X_train_full, Y_train_full = load_fi2010_csv(train_csv, n_features)
    print(f"[data] train shape: X={X_train_full.shape}, Y={Y_train_full.shape}")

    print(f"[data] Loading test CSV: {test_csv}")
    X_test, Y_test = load_fi2010_csv(test_csv, n_features)
    print(f"[data] test shape: X={X_test.shape}, Y={Y_test.shape}")

    N_full = X_train_full.shape[0]
    n_val = int(N_full * val_ratio)
    n_train = N_full - n_val
    X_train, X_val = X_train_full[:n_train], X_train_full[n_train:]
    Y_train, Y_val = Y_train_full[:n_train], Y_train_full[n_train:]
    print(f"[data] Split: train={n_train}, val={n_val}, test={X_test.shape[0]}")

    ds_train = FI2010WindowDataset(X_train, Y_train, seq_len, horizon_idx)
    ds_val = FI2010WindowDataset(X_val, Y_val, seq_len, horizon_idx)
    ds_test = FI2010WindowDataset(X_test, Y_test, seq_len, horizon_idx)
    print(f"[data] Windows: train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}")

    for name, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        labels = ds.get_all_labels()
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        total = sum(dist.values())
        pct = {k: f"{100*v/total:.1f}%" for k, v in dist.items()}
        print(f"[data]   {name}: {dist}  ({pct})  [0=up, 1=flat, 2=down]")

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=False,
                          generator=train_generator)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)

    branch_indices = make_branching_scheme(
        name=branching_scheme, custom_indices=custom_branch_indices,
        n_features=n_features,
    )
    branch_names = list(branch_indices.keys())
    print(f"[data] Branching scheme: {branching_scheme!r} → {len(branch_names)} branches")
    for bn, idx in branch_indices.items():
        print(f"[data]   {bn:24s}: {len(idx):2d} channels  (cols {idx[:6]}{'...' if len(idx)>6 else ''})")

    meta = dict(
        n_features=n_features, seq_len=seq_len, horizon_idx=horizon_idx,
        n_classes=3, flat_class=1,
        feature_names=[f"f{i}" for i in range(n_features)],
        branching_scheme=branching_scheme, branch_names=branch_names,
        branch_indices=branch_indices,
        scheme_meta=scheme_metadata(branching_scheme, branch_indices),
        n_train=n_train, n_val=n_val, n_test=X_test.shape[0],
        task="midprice", class_names=["up", "flat", "down"],
    )
    return dict(
        ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
        dl_train=dl_train, dl_val=dl_val, dl_test=dl_test, meta=meta,
    )


# ───────────────────────────────────────────────────────────────────
# Segment resolution helper
# ───────────────────────────────────────────────────────────────────


def _resolve_boundaries(
    X: np.ndarray,
    boundaries: Optional[Sequence[int]],
    label: str,
) -> List[int]:
    """Use explicit boundaries; otherwise retain the manuscript's one-segment assumption."""
    if boundaries is not None:
        boundaries = sorted(int(b) for b in boundaries)
        print(f"[data] {label}: using EXPLICIT boundaries {boundaries} "
              f"({len(boundaries)+1} segments)")
    else:
        # 이 FI-2010 릴리스는 경계 복구 불가 → 단일 세그먼트.
        # per-stock z-score 데이터에서 점프 휴리스틱은 신뢰 불가하므로 기본으로 쓰지 않음.
        boundaries = []
        print(f"[data] {label}: no boundaries → SINGLE segment (paper mode).")
    validate_segments(X, boundaries, label=label)
    return boundaries


# ───────────────────────────────────────────────────────────────────
# Microstructure DataLoaders (SEGMENT-AWARE, no embargo)
# ───────────────────────────────────────────────────────────────────


def build_microstructure_dataloaders(
    train_csv: str,
    test_csv: str,
    task: str = "toxic",
    seq_len: int = DATA.seq_len,
    k: int = K_DEFAULT,
    tick_size: float = TICK_SIZE_DEFAULT,
    n_features: int = DATA.n_features,
    batch_size: int = TRAINING.batch_size,
    val_ratio: float = DATA.validation_fraction_within_development_file,
    num_workers: int = 0,
    seed: int = TRAINING.seeds[0],
    branching_scheme: str = "type4",
    custom_branch_indices: Optional[Dict[str, List[int]]] = None,
    # ── segment controls ──────────────────────────────────────────
    train_boundaries: Optional[Sequence[int]] = None,
    test_boundaries: Optional[Sequence[int]] = None,
    include_test: bool = True,
) -> dict:
    """
    Build SEGMENT-AWARE microstructure DataLoaders.

    Segments
    --------
    Pass explicit internal cut indices via `train_boundaries` / `test_boundaries`
    when trustworthy stock/day boundaries are available. If omitted, the file
    is treated as one continuous segment. This is the manuscript's declared
    operating condition; no heuristic recovery is silently applied.

    Split
    -----
    train CSV is split into train/val CHRONOLOGICALLY WITHIN EACH SEGMENT
    (early rows of every stock → train, late rows → val). NO embargo gap.
    The test CSV is its own set of segments.

    Set ``include_test=False`` for architecture/loss selection. In that mode
    the test feature file is not opened and ``ds_test``/``dl_test`` are None.

    Returns the same bundle shape as build_dataloaders.
    """
    print(f"[data] Loading train CSV (features only): {train_csv}")
    X_train_full = load_fi2010_features_only(train_csv, n_features)
    print(f"[data] train shape: X={X_train_full.shape}")

    X_test = None
    if include_test:
        print(f"[data] Loading test CSV (features only): {test_csv}")
        X_test = load_fi2010_features_only(test_csv, n_features)
        print(f"[data] test shape: X={X_test.shape}")
    else:
        print("[data] test split SEALED (features and labels not loaded)")

    # 1) Resolve segment boundaries for each file
    train_bnd = _resolve_boundaries(X_train_full, train_boundaries, "train")
    test_bnd = (
        _resolve_boundaries(X_test, test_boundaries, "test")
        if include_test else sorted(int(x) for x in (test_boundaries or []))
    )

    seg_train_full = make_segment_id(len(X_train_full), train_bnd)
    seg_test = make_segment_id(len(X_test), test_bnd) if include_test else None

    # 2) Chronological split WITHIN each segment (no embargo)
    X_train, seg_train, X_val, seg_val = split_within_segments(
        X_train_full, seg_train_full, val_ratio)
    test_rows_text = str(len(X_test)) if include_test else "SEALED"
    print(f"[data] Within-segment split: train_rows={len(X_train)}, "
          f"val_rows={len(X_val)}, test_rows={test_rows_text}")

    # 3) Build segment-aware datasets
    ds_train = FI2010MicrostructureDataset(
        X_train, seg_train, task=task, seq_len=seq_len, k=k, tick_size=tick_size)
    ds_val = FI2010MicrostructureDataset(
        X_val, seg_val, task=task, seq_len=seq_len, k=k, tick_size=tick_size)
    ds_test = (
        FI2010MicrostructureDataset(
            X_test, seg_test, task=task, seq_len=seq_len, k=k,
            tick_size=tick_size
        ) if include_test else None
    )
    test_windows_text = str(len(ds_test)) if include_test else "SEALED"
    print(f"[data] Windows (intra-segment only): "
          f"train={len(ds_train)}, val={len(ds_val)}, test={test_windows_text}")

    info = ds_train.info
    n_classes = info["n_classes"]
    class_names = info["class_names"]
    flat_class = info["flat_class"]

    named_datasets = [("train", ds_train), ("val", ds_val)]
    if include_test:
        named_datasets.append(("test", ds_test))
    for name, ds in named_datasets:
        labels = ds.get_all_labels()
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k_): int(v) for k_, v in zip(unique, counts)}
        total = max(1, sum(dist.values()))
        pct = {k_: f"{100*v/total:.2f}%" for k_, v in dist.items()}
        named = {class_names[k_]: v for k_, v in dist.items()}
        print(f"[data]   {name}: {named}  ({pct})  [task={task}]")

    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=False,
                          generator=train_generator)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    dl_test = (
        DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers)
        if include_test else None
    )

    branch_indices = make_branching_scheme(
        name=branching_scheme, custom_indices=custom_branch_indices,
        n_features=n_features,
    )
    branch_names = list(branch_indices.keys())
    print(f"[data] Branching scheme: {branching_scheme!r} → {len(branch_names)} branches")
    for bn, idx in branch_indices.items():
        print(f"[data]   {bn:24s}: {len(idx):2d} channels")

    meta = dict(
        n_features=n_features, seq_len=seq_len,
        n_classes=n_classes, flat_class=flat_class, class_names=class_names,
        feature_names=[f"f{i}" for i in range(n_features)],
        branching_scheme=branching_scheme, branch_names=branch_names,
        branch_indices=branch_indices,
        scheme_meta=scheme_metadata(branching_scheme, branch_indices),
        n_train=len(ds_train), n_val=len(ds_val),
        n_test=len(ds_test) if include_test else None,
        n_rows_train=len(X_train), n_rows_val=len(X_val),
        n_rows_test=len(X_test) if include_test else None,
        task=task, k=k, tick_size=tick_size, label_info=info,
        # segment bookkeeping
        split_mode="within_segment", embargo=0,
        train_boundaries=list(train_bnd), test_boundaries=list(test_bnd),
        n_segments_train=len(train_bnd) + 1,
        n_segments_test=(len(test_bnd) + 1) if include_test else None,
        test_split_loaded=bool(include_test),
    )
    return dict(
        ds_train=ds_train, ds_val=ds_val, ds_test=ds_test,
        dl_train=dl_train, dl_val=dl_val, dl_test=dl_test, meta=meta,
    )


# ───────────────────────────────────────────────────────────────────
# Stratified subsampling (unchanged)
# ───────────────────────────────────────────────────────────────────


def stratified_subsample(ds, n_target: int = 5000, seed: int = 42,
                         min_per_class: int = 50) -> Subset:
    """Class-stratified subsample for downstream analysis."""
    rng = np.random.default_rng(seed)
    labels = ds.get_all_labels()
    classes = np.unique(labels)
    per_class = n_target // len(classes)
    keep = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        take = max(min(per_class, len(idx_c)), min(min_per_class, len(idx_c)))
        keep.extend(rng.choice(idx_c, size=take, replace=False).tolist())
    keep = sorted(keep)
    return Subset(ds, keep)
