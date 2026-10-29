#!/usr/bin/env python3
"""Plot learned branch-local temporal readout weights.

Read checkpoint weights directly without computing predictions or accessing
the test set. Validate checkpoint settings before comparing branches/seeds.
Outputs: PNG/PDF, weights CSV and summary JSON at the required --out-prefix.
Run with python -m lob_xai_viz.plot_positional_readout --help.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


BRANCH_NAMES = ("ask_price", "ask_volume", "bid_price", "bid_volume")
BRANCH_LABELS = {
    "ask_price": "Ask price",
    "ask_volume": "Ask volume",
    "bid_price": "Bid price",
    "bid_volume": "Bid volume",
}
BRANCH_COLORS = {
    "ask_price": "#2F6B9A",
    "ask_volume": "#D9822B",
    "bid_price": "#2A9D8F",
    "bid_volume": "#B24C63",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _natural_key(text: str) -> List[Any]:
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", text)
    ]


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: checkpoint is not a dictionary")
    return obj


def _state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint  # bare state_dict
    raise KeyError(
        "Could not find model_state/state_dict/model_state_dict in checkpoint"
    )


def _candidate_score(key: str) -> int:
    low = key.lower()
    score = 0
    if "logit" in low:
        score += 20
    if "positional" in low or "position" in low:
        score += 15
    if "pool" in low:
        score += 10
    if "weight" in low:
        score += 3
    if any(name in low for name in BRANCH_NAMES):
        score += 4
    return score


def _candidate_listing(state: Dict[str, torch.Tensor]) -> str:
    rows = []
    for key, value in state.items():
        if torch.is_tensor(value) and (
            "pool" in key.lower()
            or "position" in key.lower()
            or "logit" in key.lower()
        ):
            rows.append(f"  {key}: {tuple(value.shape)}")
    return "\n".join(rows) if rows else "  <no pool/position/logit tensors>"


def _coerce_four_by_time(
    tensors: Sequence[Tuple[str, torch.Tensor]],
) -> Tuple[np.ndarray, List[str]]:
    """Return raw values with shape (4,T) and the source state keys."""
    if len(tensors) == 1:
        key, tensor = tensors[0]
        arr = tensor.detach().cpu().float().numpy().squeeze()
        if arr.ndim == 2 and arr.shape[0] == 4:
            return arr, [key]
        if arr.ndim == 2 and arr.shape[1] == 4:
            return arr.T, [key]
        if arr.ndim == 1 and arr.size % 4 == 0 and arr.size >= 32:
            return arr.reshape(4, -1), [key]
        raise ValueError(f"{key}: cannot interpret shape {arr.shape} as (4,T)")

    if len(tensors) == 4:
        tensors = sorted(tensors, key=lambda item: _natural_key(item[0]))
        arrays = [
            value.detach().cpu().float().numpy().squeeze()
            for _, value in tensors
        ]
        if not all(arr.ndim == 1 for arr in arrays):
            raise ValueError("Expected four one-dimensional positional vectors")
        lengths = {arr.size for arr in arrays}
        if len(lengths) != 1 or next(iter(lengths)) < 8:
            raise ValueError(f"Incompatible positional vector lengths: {lengths}")
        return np.stack(arrays, axis=0), [key for key, _ in tensors]

    raise ValueError(f"Expected one matrix or four vectors, got {len(tensors)}")


def _extract_raw_weights(
    state: Dict[str, torch.Tensor],
    explicit_keys: Sequence[str] | None,
) -> Tuple[np.ndarray, List[str]]:
    if explicit_keys:
        missing = [key for key in explicit_keys if key not in state]
        if missing:
            raise KeyError(f"Requested state keys are missing: {missing}")
        return _coerce_four_by_time([(key, state[key]) for key in explicit_keys])

    candidates: List[Tuple[int, str, torch.Tensor]] = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        score = _candidate_score(key)
        if score <= 0:
            continue
        squeezed = value.squeeze()
        if squeezed.ndim == 2 and 4 in squeezed.shape and squeezed.numel() >= 32:
            candidates.append((score + 8, key, value))
        elif squeezed.ndim == 1 and squeezed.numel() >= 8:
            candidates.append((score, key, value))

    matrix = sorted(
        (item for item in candidates if item[2].squeeze().ndim == 2),
        key=lambda item: (-item[0], _natural_key(item[1])),
    )
    if matrix:
        return _coerce_four_by_time([(matrix[0][1], matrix[0][2])])

    by_length: Dict[int, List[Tuple[int, str, torch.Tensor]]] = defaultdict(list)
    for item in candidates:
        arr = item[2].squeeze()
        if arr.ndim == 1:
            by_length[int(arr.numel())].append(item)

    viable = []
    for length, items in by_length.items():
        if len(items) >= 4:
            chosen = sorted(
                items, key=lambda item: (-item[0], _natural_key(item[1]))
            )[:4]
            viable.append((sum(item[0] for item in chosen), length, chosen))
    if viable:
        _, _, chosen = max(viable, key=lambda item: (item[0], item[1]))
        return _coerce_four_by_time(
            [(key, value) for _, key, value in chosen]
        )

    raise RuntimeError(
        "Could not identify four positional pooling vectors automatically.\n"
        "Candidate state entries:\n"
        f"{_candidate_listing(state)}\n"
        "Rerun with --state-key KEY four times, using the exact state keys."
    )


def _softmax_rows(raw: np.ndarray) -> np.ndarray:
    shifted = raw - raw.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _to_probabilities(
    raw: np.ndarray, source_keys: Sequence[str], stored_values: str
) -> Tuple[np.ndarray, str]:
    if stored_values == "logits":
        return _softmax_rows(raw), "logits"
    if stored_values == "weights":
        if np.any(raw < 0):
            raise ValueError("--stored-values weights but negative values were found")
        sums = raw.sum(axis=1, keepdims=True)
        if np.any(sums <= 0):
            raise ValueError("Pooling weights must have positive row sums")
        return raw / sums, "weights"

    key_text = " ".join(source_keys).lower()
    row_sums = raw.sum(axis=1)
    already_prob = (
        np.all(raw >= -1e-8)
        and np.allclose(row_sums, 1.0, atol=1e-4, rtol=1e-4)
    )
    if "logit" in key_text or not already_prob:
        return _softmax_rows(raw), "logits"
    return np.clip(raw, 0.0, None) / raw.sum(axis=1, keepdims=True), "weights"


def _infer_seed(config: Dict[str, Any], path: Path) -> int:
    if "seed" in config:
        return int(config["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer seed from config or path: {path}")
    return int(match.group(1))


def _verify_config(config: Dict[str, Any], task: str, path: Path) -> None:
    found_task = config.get("task")
    if found_task is not None and str(found_task) != task:
        raise ValueError(f"{path}: task={found_task!r}, expected {task!r}")
    found_pool = config.get("pool")
    if found_pool is not None and str(found_pool) != "positional":
        raise ValueError(f"{path}: pool={found_pool!r}, expected 'positional'")


def _metrics(weights: np.ndarray) -> Dict[str, np.ndarray]:
    # weights shape: (S,4,T); t=0 oldest, t=T-1 most recent
    time_length = weights.shape[-1]
    lag = np.arange(time_length - 1, -1, -1, dtype=np.float64)
    last8 = weights[:, :, -min(8, time_length) :].sum(axis=-1)
    last32 = weights[:, :, -min(32, time_length) :].sum(axis=-1)
    expected_lag = (weights * lag[None, None, :]).sum(axis=-1)
    clipped = np.clip(weights, 1e-12, None)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1) / math.log(time_length)
    return {
        "last8": last8,
        "last32": last32,
        "expected_lag": expected_lag,
        "entropy": entropy,
    }


def _write_csv(
    path: Path, weights: np.ndarray, seeds: Sequence[int]
) -> None:
    time_length = weights.shape[-1]
    gap = 1.0 / time_length
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["seed", "branch", "timestep", "lag", "weight", "weight_over_gap"]
        )
        for si, seed in enumerate(seeds):
            for bi, branch in enumerate(BRANCH_NAMES):
                for timestep in range(time_length):
                    writer.writerow(
                        [
                            seed,
                            branch,
                            timestep,
                            time_length - 1 - timestep,
                            f"{weights[si, bi, timestep]:.12g}",
                            f"{weights[si, bi, timestep] / gap:.12g}",
                        ]
                    )


def _summary_payload(
    task: str,
    checkpoints: Sequence[Path],
    hashes: Sequence[str],
    seeds: Sequence[int],
    source_keys: Sequence[Sequence[str]],
    source_types: Sequence[str],
    weights: np.ndarray,
) -> Dict[str, Any]:
    metrics = _metrics(weights)
    time_length = weights.shape[-1]
    payload: Dict[str, Any] = {
        "task": task,
        "pool": "positional",
        "interpretation": (
            "Input-independent branch-local temporal readout; these weights are "
            "not per-window saliency."
        ),
        "time_length": time_length,
        "gap_weight_per_timestep": 1.0 / time_length,
        "gap_last8": min(8, time_length) / time_length,
        "gap_last32": min(32, time_length) / time_length,
        "gap_expected_lag": (time_length - 1) / 2.0,
        "checkpoints": [],
        "aggregate": {},
    }
    for i, (path, digest, seed) in enumerate(zip(checkpoints, hashes, seeds)):
        seed_rows = {}
        for bi, branch in enumerate(BRANCH_NAMES):
            seed_rows[branch] = {
                "last8": float(metrics["last8"][i, bi]),
                "last32": float(metrics["last32"][i, bi]),
                "expected_lag": float(metrics["expected_lag"][i, bi]),
                "normalized_entropy": float(metrics["entropy"][i, bi]),
                "weights": weights[i, bi].tolist(),
            }
        payload["checkpoints"].append(
            {
                "path": str(path),
                "sha256": digest,
                "seed": int(seed),
                "state_keys": list(source_keys[i]),
                "stored_as": source_types[i],
                "branches": seed_rows,
            }
        )

    for bi, branch in enumerate(BRANCH_NAMES):
        payload["aggregate"][branch] = {
            "weight_mean": weights[:, bi].mean(axis=0).tolist(),
            "weight_sd": weights[:, bi].std(axis=0, ddof=1).tolist(),
        }
        for metric_name, values in metrics.items():
            payload["aggregate"][branch][f"{metric_name}_mean"] = float(
                values[:, bi].mean()
            )
            payload["aggregate"][branch][f"{metric_name}_sd"] = float(
                values[:, bi].std(ddof=1)
            )
    return payload


def _plot(
    out_png: Path,
    out_pdf: Path,
    task: str,
    weights: np.ndarray,
    seeds: Sequence[int],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n_seed, _, time_length = weights.shape
    gap = 1.0 / time_length
    relative = weights / gap
    metrics = _metrics(weights)
    mean = relative.mean(axis=0)
    sd = relative.std(axis=0, ddof=1) if n_seed > 1 else np.zeros_like(mean)
    x = np.arange(time_length)

    fig = plt.figure(figsize=(11.4, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.05, 1.05, 0.95), wspace=0.25, hspace=0.24
    )
    profile_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    ymax = max(1.25, float(np.nanmax(mean + sd)) * 1.10)
    tick_positions = np.linspace(0, time_length - 1, 5).round().astype(int)
    tick_labels = [str(time_length - 1 - int(value)) for value in tick_positions]

    for bi, (axis, branch) in enumerate(zip(profile_axes, BRANCH_NAMES)):
        color = BRANCH_COLORS[branch]
        for si, seed in enumerate(seeds):
            axis.plot(
                x,
                relative[si, bi],
                color=color,
                alpha=0.23,
                linewidth=0.8,
            )
        axis.fill_between(
            x,
            np.maximum(mean[bi] - sd[bi], 0),
            mean[bi] + sd[bi],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(x, mean[bi], color=color, linewidth=2.0)
        axis.axhline(1.0, color="#4B5563", linewidth=1.0, linestyle="--")
        if time_length >= 32:
            axis.axvline(
                time_length - 32 - 0.5,
                color="#9CA3AF",
                linewidth=0.8,
                linestyle=":",
            )
        if time_length >= 8:
            axis.axvline(
                time_length - 8 - 0.5,
                color="#6B7280",
                linewidth=0.8,
                linestyle=":",
            )
        last8_ratio = metrics["last8"][:, bi].mean() / (min(8, time_length) / time_length)
        lag_mean = metrics["expected_lag"][:, bi].mean()
        axis.text(
            0.02,
            0.96,
            f"last-8: {last8_ratio:.2f}× GAP\nE[lag]: {lag_mean:.1f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="#374151",
        )
        axis.set_title(BRANCH_LABELS[branch], loc="left", fontweight="semibold")
        axis.set_ylim(0, ymax)
        axis.set_xlim(0, time_length - 1)
        axis.set_xticks(tick_positions, tick_labels)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        if bi in (2, 3):
            axis.set_xlabel("Lag from prediction time (ticks; 0 = most recent)")
        if bi in (0, 2):
            axis.set_ylabel(r"Relative temporal weight  $\alpha_t/(1/T)$")

    profile_legend = [
        Line2D([0], [0], color="#6B7280", alpha=0.35, linewidth=0.9,
               label="individual seeds"),
        Line2D([0], [0], color="#374151", linewidth=2.0,
               label="3-seed mean"),
        Line2D([0], [0], color="#4B5563", linewidth=1.0, linestyle="--",
               label="GAP"),
    ]
    fig.legend(
        handles=profile_legend,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.37, 0.965),
        ncol=3,
    )
    profile_axes[0].text(
        -0.18,
        1.10,
        "(a)",
        transform=profile_axes[0].transAxes,
        fontweight="bold",
        fontsize=11,
    )

    ax_mass = fig.add_subplot(grid[0, 2])
    y_base = np.arange(len(BRANCH_NAMES))
    offsets = (-0.12, 0.12)
    marker_info = [
        ("Last 8", "o", metrics["last8"], min(8, time_length) / time_length),
        ("Last 32", "s", metrics["last32"], min(32, time_length) / time_length),
    ]
    for offset, (label, marker, values, gap_mass) in zip(offsets, marker_info):
        ratio = values / gap_mass
        means = ratio.mean(axis=0)
        errors = (
            ratio.std(axis=0, ddof=1)
            if n_seed > 1
            else np.zeros(len(BRANCH_NAMES))
        )
        ax_mass.errorbar(
            means,
            y_base + offset,
            xerr=errors,
            fmt=marker,
            markersize=5.2,
            capsize=2.5,
            linewidth=1.1,
            color="#374151" if marker == "s" else "#1F77B4",
            label=label,
        )
    ax_mass.axvline(1.0, color="#4B5563", linewidth=1.0, linestyle="--")
    ax_mass.set_yticks(y_base, [BRANCH_LABELS[name] for name in BRANCH_NAMES])
    ax_mass.invert_yaxis()
    ax_mass.set_xlabel("Recency mass relative to GAP")
    ax_mass.set_title("(b) Recency enrichment", loc="left", fontweight="semibold")
    ax_mass.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    ax_mass.spines[["top", "right", "left"]].set_visible(False)
    ax_mass.legend(frameon=False, loc="upper right")

    ax_lag = fig.add_subplot(grid[1, 2])
    lag_values = metrics["expected_lag"]
    lag_mean = lag_values.mean(axis=0)
    lag_sd = (
        lag_values.std(axis=0, ddof=1)
        if n_seed > 1
        else np.zeros(len(BRANCH_NAMES))
    )
    for bi, branch in enumerate(BRANCH_NAMES):
        ax_lag.errorbar(
            lag_mean[bi],
            bi,
            xerr=lag_sd[bi],
            fmt="o",
            markersize=5.5,
            capsize=2.5,
            linewidth=1.2,
            color=BRANCH_COLORS[branch],
        )
    ax_lag.axvline(
        (time_length - 1) / 2.0,
        color="#4B5563",
        linewidth=1.0,
        linestyle="--",
        label="GAP",
    )
    ax_lag.set_yticks(y_base, [BRANCH_LABELS[name] for name in BRANCH_NAMES])
    ax_lag.invert_yaxis()
    ax_lag.set_xlim(0, time_length - 1)
    ax_lag.set_xlabel("Expected lag (ticks; lower = more recent)")
    ax_lag.set_title("(c) Temporal center of mass", loc="left", fontweight="semibold")
    ax_lag.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.65)
    ax_lag.spines[["top", "right", "left"]].set_visible(False)
    ax_lag.legend(frameon=False, loc="lower left")

    task_label = task.capitalize()
    fig.suptitle(
        f"Branch-local positional readout — {task_label}",
        fontsize=12,
        fontweight="semibold",
        y=1.015,
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("toxic", "vacuum"))
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--out-prefix",
        required=True,
        type=Path,
        help="Output prefix without an extension",
    )
    parser.add_argument(
        "--state-key",
        action="append",
        default=None,
        help=(
            "Exact positional state key; repeat four times for per-branch vectors "
            "or once for a (4,T) tensor. Usually automatic discovery is sufficient."
        ),
    )
    parser.add_argument(
        "--stored-values",
        choices=("auto", "logits", "weights"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoints = [path.expanduser().resolve() for path in args.checkpoints]
    if len(checkpoints) < 2:
        raise ValueError("Use at least two seed checkpoints; three are recommended")
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    records = []
    for path in checkpoints:
        checkpoint = _load_checkpoint(path)
        config = checkpoint.get("config", {})
        if not isinstance(config, dict):
            config = {}
        _verify_config(config, args.task, path)
        seed = _infer_seed(config, path)
        raw, source_keys = _extract_raw_weights(
            _state_dict(checkpoint), args.state_key
        )
        probabilities, source_type = _to_probabilities(
            raw, source_keys, args.stored_values
        )
        if probabilities.shape[0] != len(BRANCH_NAMES):
            raise ValueError(
                f"{path}: expected four branches, got {probabilities.shape}"
            )
        records.append(
            {
                "path": path,
                "seed": seed,
                "weights": probabilities,
                "source_keys": source_keys,
                "source_type": source_type,
                "sha256": _sha256(path),
            }
        )

    records.sort(key=lambda record: record["seed"])
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique, got {seeds}")
    shapes = {record["weights"].shape for record in records}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent positional weight shapes: {shapes}")
    weights = np.stack([record["weights"] for record in records], axis=0)

    out_prefix: Path = args.out_prefix.expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_prefix.with_suffix(".png")
    out_pdf = out_prefix.with_suffix(".pdf")
    out_csv = out_prefix.parent / f"{out_prefix.name}_weights.csv"
    out_json = out_prefix.parent / f"{out_prefix.name}_summary.json"

    _write_csv(out_csv, weights, seeds)
    payload = _summary_payload(
        task=args.task,
        checkpoints=[record["path"] for record in records],
        hashes=[record["sha256"] for record in records],
        seeds=seeds,
        source_keys=[record["source_keys"] for record in records],
        source_types=[record["source_type"] for record in records],
        weights=weights,
    )
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    _plot(out_png, out_pdf, args.task, weights, seeds)

    print(
        f"[verified] task={args.task} pool=positional "
        f"seeds={seeds} shape={tuple(weights.shape)}"
    )
    for si, seed in enumerate(seeds):
        print(f"seed={seed}")
        for bi, branch in enumerate(BRANCH_NAMES):
            row = payload["checkpoints"][si]["branches"][branch]
            print(
                f"  {branch:<12} last8={row['last8']:.3f} "
                f"last32={row['last32']:.3f} "
                f"E[lag]={row['expected_lag']:.1f}"
            )
    print(f"[saved] {out_png}")
    print(f"[saved] {out_pdf}")
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
