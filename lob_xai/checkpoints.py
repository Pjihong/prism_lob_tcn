#!/usr/bin/env python3
"""Load checkpoint-aligned data and models for evaluation and microstructure analysis.

Uses task, horizon, sequence length, normalization and segment settings saved
with the checkpoint. Microstructure labels are distinct from legacy mid-price
labels. Model loading checks state-dict compatibility; data loading verifies available provenance.
Inputs: checkpoint and FI-2010 CSV paths.
Outputs: ordered windows, labels, class names and model configuration.
"""
from __future__ import annotations
import numpy as np
import torch

from lob_xai.provenance import verify_file_identity

TASK_CLASSES = {
    "toxic":  ["stationary", "toxic_up", "toxic_down"],
    "vacuum": ["normal", "vacuum"],
}


def _safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_ckpt_cfg(ckpt_path: str) -> dict:
    """Return a merged params dict from a checkpoint (config + meta)."""
    ck = _safe_load(ckpt_path)
    cfg = dict(ck.get("config", {}) or {})
    meta = dict(ck.get("meta", {}) or {})
    out = {**cfg, **meta}
    out.setdefault("seq_len", int(cfg.get("seq_len", meta.get("seq_len", 100))))
    out.setdefault("k", int(cfg.get("k", meta.get("k", 10))))
    out.setdefault("tick_size", float(cfg.get("tick_size", meta.get("tick_size", 0.00098))))
    out.setdefault("task", cfg.get("task", meta.get("task", "toxic")))
    out.setdefault("n_features", int(cfg.get("n_features", meta.get("n_features", 40))))
    out["test_boundaries"] = list(
        meta.get("test_boundaries", cfg.get("test_boundaries", [])) or []
    )
    out["train_boundaries"] = list(
        meta.get("train_boundaries", cfg.get("train_boundaries", [])) or []
    )
    return out


def load_micro_windows_in_order(ckpt_path: str, csv_path: str,
                                task: str | None = None,
                                n_features: int | None = None,
                                which: str = "test",
                                verify_identity: bool = True):
    """Load microstructure windows + labels for a CSV, in dataset order.

    Returns
    -------
    X : (N, seq_len, F) float32 ndarray   — windows (dataset order, shuffle=False)
    y : (N,) int ndarray                  — toxic/vacuum class labels
    class_names : list[str]
    cfg : dict                            — params read from the checkpoint
    """
    from lob_xai.data import load_fi2010_features_only, FI2010MicrostructureDataset
    from lob_xai.microstructure_labels import make_segment_id, split_within_segments

    cfg = read_ckpt_cfg(ckpt_path)
    checkpoint_task = cfg["task"]
    if task is not None and task != checkpoint_task:
        raise ValueError(
            f"checkpoint task={checkpoint_task!r}, requested task={task!r}"
        )
    task = task or checkpoint_task
    nfeat = n_features or cfg["n_features"]
    seq_len, k, tick = cfg["seq_len"], cfg["k"], cfg["tick_size"]
    if which not in {"test", "train", "validation", "train_full"}:
        raise ValueError("which must be test, train, validation, or train_full")
    if verify_identity:
        identities = cfg.get("data_files") or {}
        identity_key = "test" if which == "test" else "development"
        if identity_key not in identities:
            raise ValueError(
                f"checkpoint has no {identity_key!r} data identity; refusing "
                "to compute paper metrics from an unverified CSV"
            )
        verify_file_identity(csv_path, identities[identity_key])
    bnd = cfg["test_boundaries"] if which == "test" else cfg["train_boundaries"]

    Xarr = load_fi2010_features_only(csv_path, n_features=nfeat)
    seg = make_segment_id(len(Xarr), bnd)  # [] -> single segment, matches training
    if which in {"train", "validation"}:
        val_ratio = float(cfg.get("val_ratio", 0.15 / 0.85))
        Xtr, str_, Xva, sva = split_within_segments(Xarr, seg, val_ratio)
        if which == "train":
            Xarr, seg = Xtr, str_
        else:
            Xarr, seg = Xva, sva
    ds = FI2010MicrostructureDataset(Xarr, seg, task=task,
                                     seq_len=seq_len, k=k, tick_size=tick)
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    y = np.array([int(ds[i][1]) for i in range(len(ds))])
    return X, y, TASK_CLASSES[task], cfg


def load_ours_model(ckpt_path: str, device="cpu", strict: bool = True):
    """Build + load an 'ours' LOBBranchModel from a checkpoint.

    Mirrors counterfactual_test.build_ours (proven to load the single-seg
    checkpoints), using .get() fallbacks so it won't KeyError on configs that
    lack legacy keys like cnn_dropout. Returns (model, cfg, meta).
    """
    import torch
    from lob_xai.models import LOBBranchModel
    ck = _safe_load(ckpt_path)
    cfg = dict(ck.get("config", {}) or {})
    meta = dict(ck.get("meta", {}) or {})
    task = cfg.get("task") or meta.get("task") or "toxic"
    if task not in TASK_CLASSES:
        raise ValueError(f"checkpoint has unsupported task={task!r}")
    nc = cfg.get("n_classes") or meta.get("n_classes") or len(TASK_CLASSES[task])
    bidx = meta.get("branch_indices")
    if bidx is None:
        try:
            from lob_xai.branching import make_branching_scheme
            bidx = make_branching_scheme(cfg.get("branching_scheme", "type4"),
                                         n_features=int(cfg.get("n_features", 40)))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "checkpoint meta has no 'branch_indices' and make_branching_scheme "
                f"could not be imported ({e}); cannot rebuild model.") from e
    model = LOBBranchModel(
        n_features=int(cfg.get("n_features", 40)), n_classes=int(nc),
        backbone=cfg.get("backbone", "tcn"), pool=cfg.get("pool", "avg"),
        cnn_channels=tuple(cfg.get("cnn_channels", (32, 32, 64))),
        cnn_kernel=int(cfg.get("cnn_kernel", 3)),
        cnn_dropout=float(cfg.get("cnn_dropout", 0.1)),
        fc_hidden=tuple(cfg.get("fc_hidden", (64,))),
        branch_indices=bidx,
        tcn_dilation_base=cfg.get("tcn_dilation_base", 2),
        input_norm=cfg.get("input_norm", "identity"),
        tcn_norm=cfg.get("tcn_norm", "none"),
        interaction_channels=int(cfg.get("interaction_channels", 0)),
        seq_len=int(cfg.get("seq_len", 100)),
        pool_windows=tuple(cfg.get("pool_windows", (8, 32, 100))),
    ).to(device)
    try:
        model.load_state_dict(ck["model_state"], strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"checkpoint architecture mismatch for {ckpt_path!r}; refusing to "
            "compute paper metrics with partially loaded weights"
        ) from exc
    model.eval()
    # ensure downstream code that reads meta["branch_names"] still works (type4)
    meta.setdefault("branch_names",
                    ["ask_price", "ask_volume", "bid_price", "bid_volume"])
    return model, cfg, meta
