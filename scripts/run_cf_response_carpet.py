#!/usr/bin/env python3
"""Generate counterfactual response carpets without modifying the audit runner.

For each non-flat target class, the script:

1. selects correct alerts across the full confidence range (default);
2. computes one saliency map per selected window;
3. runs CAM, random, and anti-CAM conditional-replacement traces;
4. saves every window-by-deletion-fraction probability and intermediate
   predicted class; and
5. renders a shared-scale response carpet with the first class-flip marked.

The script deliberately keeps the original counterfactual implementation as
the source of model loading, data loading, span selection, and conditional
sampling. It is a visualization/export layer, not a new perturbation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


THIS = Path(__file__).resolve()
for candidate in (THIS.parents[2], Path.cwd()):
    if (candidate / "lob_xai").is_dir():
        sys.path.insert(0, str(candidate))
        break

from lob_xai.branching import make_branching_scheme
from lob_xai.counterfactual_test import (
    DeepLOBGradCAMProvider,
    NeighborConditionalSampler,
    OursCAMProvider,
    TASK_CLASSES,
    TASK_FLAT,
    best_span,
    build_dl,
    build_ours,
    load_windows,
    make_demo,
)
from lob_xai.models import LOBBranchModel
from lob_xai.paper_protocol import COUNTERFACTUAL, DATA
from lob_xai.provenance import sha256_file, verify_file_identity


MODES = ("cam", "random", "anti")
MODE_LABELS = {
    "cam": "CAM-selected span",
    "random": "Random span",
    "anti": "Anti-CAM span",
}


def _predict_probs(model, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(
                np.ascontiguousarray(X[start:start + batch_size])
            ).float().to(device)
            parts.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _select_indices(
    probs: np.ndarray,
    labels: np.ndarray,
    target: int,
    n_windows: int,
    cohort: str,
    rng: np.random.Generator,
) -> np.ndarray:
    preds = probs.argmax(axis=1)
    if cohort in {"stratified_correct", "top_correct", "random_correct"}:
        candidates = np.flatnonzero((labels == target) & (preds == target))
    elif cohort == "predicted":
        candidates = np.flatnonzero(preds == target)
    elif cohort == "true":
        candidates = np.flatnonzero(labels == target)
    else:
        raise ValueError(f"unsupported cohort: {cohort}")

    if len(candidates) == 0:
        return candidates

    confidence = probs[candidates, target]
    order = np.argsort(confidence, kind="mergesort")
    candidates = candidates[order]

    if len(candidates) > n_windows:
        if cohort == "top_correct":
            candidates = candidates[-n_windows:]
        elif cohort == "random_correct":
            candidates = np.sort(rng.choice(candidates, n_windows, replace=False))
            candidates = candidates[np.argsort(probs[candidates, target], kind="mergesort")]
        else:
            # Deterministic quantile grid: covers low through high confidence
            # without making model confidence the inclusion criterion.
            positions = np.rint(
                np.linspace(0, len(candidates) - 1, n_windows)
            ).astype(int)
            candidates = candidates[positions]

    # Carpet row order is always low -> high target confidence.
    return candidates[np.argsort(probs[candidates, target], kind="mergesort")]


def _saliency_batches(
    provider,
    X: np.ndarray,
    target: int,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(X[start:start + batch_size])
        ).float()
        parts.append(np.asarray(provider.saliency(xb, target), dtype=np.float64))
    return np.concatenate(parts, axis=0)


def _trace_one_window(
    model,
    sampler: NeighborConditionalSampler,
    device: torch.device,
    window: np.ndarray,
    saliency: np.ndarray,
    original_prob: np.ndarray,
    target: int,
    q_grid: np.ndarray,
    mc_samples: int,
    inference_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean class probabilities, argmaxes, and selected span endpoints.

    Shapes:
      probvec:    (mode, q, class)
      argmax:     (mode, q)
      span_start: (mode, q)
      span_end:   (mode, q)
    """
    n_modes = len(MODES)
    n_q = len(q_grid)
    n_classes = len(original_prob)
    probvec = np.empty((n_modes, n_q, n_classes), dtype=np.float64)
    argmax = np.empty((n_modes, n_q), dtype=np.int16)
    span_start = np.full((n_modes, n_q), -1, dtype=np.int16)
    span_end = np.full((n_modes, n_q), -1, dtype=np.int16)

    probvec[:, 0, :] = original_prob
    argmax[:, 0] = int(original_prob.argmax())

    variants: list[np.ndarray] = []
    keys: list[tuple[int, int, int, int]] = []
    T = int(window.shape[0])

    for mode_index, mode in enumerate(MODES):
        for q_index, q in enumerate(q_grid[1:], start=1):
            length = max(1, int(math.ceil(float(q) * T)))
            start, end = best_span(
                saliency,
                length,
                mode,
                rng=sampler.rng if mode == "random" else None,
            )
            generated = sampler.sample(window, start, end, mc_samples)
            variants.append(np.asarray(generated, dtype=np.float32))
            keys.append((mode_index, q_index, start, end))

    all_variants = np.concatenate(variants, axis=0)
    all_probs = _predict_probs(
        model, all_variants, device=device, batch_size=inference_batch_size
    )

    offset = 0
    for mode_index, q_index, start, end in keys:
        block = all_probs[offset:offset + mc_samples]
        offset += mc_samples
        mean_prob = block.mean(axis=0)
        probvec[mode_index, q_index] = mean_prob
        argmax[mode_index, q_index] = int(mean_prob.argmax())
        span_start[mode_index, q_index] = int(start)
        span_end[mode_index, q_index] = int(end)

    return probvec, argmax, span_start, span_end


def _first_flip_index(argmax: np.ndarray, target: int) -> np.ndarray:
    """First q index leaving target; -1 means no flip in the evaluated grid."""
    result = np.full(argmax.shape[0], -1, dtype=np.int16)
    for row in range(argmax.shape[0]):
        if argmax[row, 0] != target:
            result[row] = 0
            continue
        hits = np.flatnonzero(argmax[row, 1:] != target)
        if len(hits):
            result[row] = int(hits[0] + 1)
    return result


def _plot_carpet(
    output: Path,
    model_label: str,
    task: str,
    class_name: str,
    seed: int,
    cohort: str,
    q_grid: np.ndarray,
    confidence: np.ndarray,
    target_prob: np.ndarray,
    argmax: np.ndarray,
    target: int,
    vmax: float,
) -> None:
    n_windows = len(confidence)
    retention = target_prob / np.maximum(target_prob[:, :, :1], 1e-12)
    clipped_fraction = float((retention > vmax).mean())

    fig, axes = plt.subplots(
        1, 3, figsize=(13.2, 6.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    x = np.arange(len(q_grid))

    for mode_index, (mode, ax) in enumerate(zip(MODES, axes)):
        matrix = retention[:, mode_index, :]
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        first_flip = _first_flip_index(argmax[:, mode_index, :], target)
        rows = np.flatnonzero(first_flip >= 0)
        if len(rows):
            ax.scatter(
                first_flip[rows],
                rows,
                marker="|",
                s=28,
                linewidths=1.1,
                color="white",
                label="first class flip",
            )

        for boundary in (n_windows / 3 - 0.5, 2 * n_windows / 3 - 0.5):
            ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.75)

        qmax_flip = float((argmax[:, mode_index, -1] != target).mean())
        ax.set_title(f"{MODE_LABELS[mode]}\nflip@qmax={qmax_flip:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{100 * q:.0f}" for q in q_grid], rotation=0)
        ax.set_xlabel("replaced fraction of window (%)")
        if len(rows):
            ax.legend(loc="upper right", frameon=True, fontsize=8)

    axes[0].set_ylabel(
        "correct-alert windows\nlow → high target confidence"
        if "correct" in cohort
        else "windows\nlow → high target confidence"
    )
    # Directly label the three confidence strata without adding a second axis.
    if n_windows >= 3:
        for y, label in (
            (n_windows / 6, "low"),
            (n_windows / 2, "mid"),
            (5 * n_windows / 6, "high"),
        ):
            axes[0].text(
                -0.08,
                y,
                label,
                transform=axes[0].get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
            )

    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.022, pad=0.02)
    colorbar.set_label(r"target-probability retention  $p_c(q)/p_c(0)$")
    fig.suptitle(
        f"Counterfactual response carpet — {model_label}, {task}/{class_name}\n"
        f"seed={seed}, cohort={cohort}, n={n_windows}; "
        f"white tick = first predicted-class flip",
        fontsize=13,
    )
    if clipped_fraction > 0:
        fig.text(
            0.995,
            0.01,
            f"{100 * clipped_fraction:.2f}% cells clipped above {vmax:g}",
            ha="right",
            va="bottom",
            fontsize=8,
        )

    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_window_csv(
    output: Path,
    window_indices: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
    original_pred: np.ndarray,
    q_grid: np.ndarray,
    argmax: np.ndarray,
    target: int,
) -> None:
    fieldnames = [
        "row", "window_index", "true_class", "initial_pred", "confidence",
        "qstar_cam", "qstar_random", "qstar_anti",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, window_index in enumerate(window_indices):
            values = {}
            for mode_index, mode in enumerate(MODES):
                first = _first_flip_index(argmax[row:row + 1, mode_index, :], target)[0]
                values[f"qstar_{mode}"] = (
                    "" if first < 0 else f"{float(q_grid[first]):.6g}"
                )
            writer.writerow({
                "row": row,
                "window_index": int(window_index),
                "true_class": int(labels[window_index]),
                "initial_pred": int(original_pred[window_index]),
                "confidence": f"{float(confidence[row]):.9g}",
                **values,
            })


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ours", "deeplob"])
    parser.add_argument("--ckpt")
    parser.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    parser.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    parser.add_argument("--task", required=True, choices=["toxic", "vacuum"])
    parser.add_argument("--method", default="hirescam",
                        choices=["hirescam", "gradcam", "tsr"])
    parser.add_argument(
        "--cohort",
        default="stratified_correct",
        choices=[
            "stratified_correct", "top_correct", "random_correct",
            "predicted", "true",
        ],
        help=(
            "stratified_correct covers the entire correct-alert confidence "
            "range and is the recommended bias-audit cohort"
        ),
    )
    parser.add_argument("--n-windows", type=int, default=120)
    parser.add_argument("--pool-size", type=int, default=30_000)
    parser.add_argument("--mc", type=int, default=10)
    parser.add_argument("--neighbors-k", type=int, default=50)
    parser.add_argument("--q-max", type=float, default=0.5)
    parser.add_argument("--q-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--saliency-batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--color-max", type=float, default=1.25)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-unverified-data", action="store_true")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.demo and not args.ckpt:
        raise SystemExit("--ckpt is required unless --demo is used")
    for name in (
        "n_windows", "pool_size", "mc", "neighbors_k",
        "saliency_batch_size", "inference_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not (0 < args.q_max <= 1 and 0 < args.q_step <= args.q_max):
        raise SystemExit("require 0 < q-step <= q-max <= 1")
    if args.color_max <= 0:
        raise SystemExit("--color-max must be positive")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    classes = TASK_CLASSES[args.task]
    flat = TASK_FLAT[args.task]
    q_grid = np.arange(0, args.q_max + 1e-12, args.q_step, dtype=np.float64)

    if args.demo:
        test_X, test_y = make_demo(
            args.task, DATA.seq_len, n=max(300, args.n_windows * 10), seed=args.seed
        )
        pool_X, _ = make_demo(
            args.task, DATA.seq_len, n=max(args.pool_size, 200), seed=args.seed + 1
        )
        if args.model == "ours":
            model = LOBBranchModel(
                n_features=40,
                n_classes=len(classes),
                backbone="tcn",
                pool="avg",
                cnn_channels=(32, 32, 64),
                cnn_kernel=3,
                branch_indices=make_branching_scheme("type4", 40),
            ).to(device).eval()
            pool_name = "avg-demo"
        else:
            from models.deeplob import build_deeplob

            model = build_deeplob(n_classes=len(classes)).to(device).eval()
            pool_name = None
        checkpoint_hash = "DEMO"
        data_verified = False
        seq_len = DATA.seq_len
        horizon = DATA.toxic_horizon if args.task == "toxic" else DATA.vacuum_horizon
    else:
        checkpoint_path = Path(args.ckpt).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        cfg = {
            **dict(checkpoint.get("config", {}) or {}),
            **dict(checkpoint.get("meta", {}) or {}),
        }
        identities = cfg.get("data_files") or {}
        data_verified = bool(identities)
        if identities:
            verify_file_identity(args.train_csv, identities.get("development", {}))
            verify_file_identity(args.test_csv, identities.get("test", {}))
        elif not args.allow_unverified_data:
            raise SystemExit(
                "checkpoint has no data SHA-256 provenance; use a paper checkpoint "
                "or add --allow-unverified-data for a diagnostic-only run"
            )

        seq_len = int(cfg.get("seq_len", DATA.seq_len))
        default_k = DATA.toxic_horizon if args.task == "toxic" else DATA.vacuum_horizon
        horizon = int(cfg.get("k", default_k))
        tick_size = float(cfg.get("tick_size", DATA.tick_size_z))
        val_ratio = float(
            cfg.get("val_ratio", DATA.validation_fraction_within_development_file)
        )
        train_boundaries = list(cfg.get("train_boundaries", []) or [])
        test_boundaries = list(cfg.get("test_boundaries", []) or [])

        if args.model == "ours":
            model, _ = build_ours(str(checkpoint_path), device, args.task)
            pool_name = str(cfg.get("pool", "unknown"))
        else:
            model, _ = build_dl(str(checkpoint_path), device, args.task)
            pool_name = None

        test_X, test_y = load_windows(
            args.test_csv,
            args.task,
            seq_len,
            horizon,
            tick_size,
            boundaries=test_boundaries,
        )
        pool_X, _ = load_windows(
            args.train_csv,
            args.task,
            seq_len,
            horizon,
            tick_size,
            limit=args.pool_size,
            seed=args.seed,
            boundaries=train_boundaries,
            subset="train",
            val_ratio=val_ratio,
        )
        if len(pool_X) != args.pool_size:
            raise SystemExit(
                f"requested donor pool={args.pool_size}, available={len(pool_X)}"
            )
        checkpoint_hash = sha256_file(checkpoint_path)

    probs = _predict_probs(
        model, test_X, device=device, batch_size=args.inference_batch_size
    )
    preds = probs.argmax(axis=1)
    sampler = NeighborConditionalSampler(
        pool_X, seed=args.seed, n_neighbors=args.neighbors_k
    )
    provider = (
        OursCAMProvider(model, device, args.method)
        if args.model == "ours"
        else DeepLOBGradCAMProvider(model, device, args.method)
    )
    model_label = (
        f"Ours–{pool_name}" if args.model == "ours" else "DeepLOB"
    )

    generated: list[dict[str, object]] = []
    for target in range(len(classes)):
        if target == flat:
            continue
        selected = _select_indices(
            probs,
            test_y,
            target,
            args.n_windows,
            args.cohort,
            rng,
        )
        if len(selected) == 0:
            print(f"[skip] {classes[target]}: no eligible windows")
            continue

        selected_X = np.ascontiguousarray(test_X[selected])
        confidence = probs[selected, target]
        print(
            f"[cohort] class={classes[target]} n={len(selected)} "
            f"confidence=[{confidence.min():.4f},{confidence.max():.4f}]"
        )
        saliency = _saliency_batches(
            provider,
            selected_X,
            target,
            batch_size=args.saliency_batch_size,
        )

        n_windows = len(selected)
        n_modes = len(MODES)
        n_q = len(q_grid)
        n_classes = len(classes)
        probvec = np.empty(
            (n_windows, n_modes, n_q, n_classes), dtype=np.float32
        )
        argmax = np.empty((n_windows, n_modes, n_q), dtype=np.int16)
        span_start = np.empty((n_windows, n_modes, n_q), dtype=np.int16)
        span_end = np.empty((n_windows, n_modes, n_q), dtype=np.int16)

        for row, window_index in enumerate(selected):
            trace = _trace_one_window(
                model=model,
                sampler=sampler,
                device=device,
                window=test_X[window_index],
                saliency=saliency[row],
                original_prob=probs[window_index],
                target=target,
                q_grid=q_grid,
                mc_samples=args.mc,
                inference_batch_size=args.inference_batch_size,
            )
            probvec[row], argmax[row], span_start[row], span_end[row] = trace
            if (row + 1) % 10 == 0 or row + 1 == n_windows:
                print(f"[trace] {classes[target]} {row + 1}/{n_windows}")

        stem = (
            f"cf_response_carpet_{args.model}_{args.task}_{classes[target]}"
            f"_{args.method}_{args.cohort}_seed{args.seed}"
        )
        base = output_dir / stem
        metadata = {
            "schema_version": 1,
            "model": args.model,
            "model_label": model_label,
            "pool": pool_name,
            "task": args.task,
            "class_index": target,
            "class_name": classes[target],
            "method": args.method,
            "cohort": args.cohort,
            "seed": args.seed,
            "n_windows_requested": args.n_windows,
            "n_windows_actual": int(n_windows),
            "pool_size": int(len(pool_X)),
            "mc_samples": args.mc,
            "neighbors_k": args.neighbors_k,
            "seq_len": seq_len,
            "horizon": horizon,
            "checkpoint_sha256": checkpoint_hash,
            "data_provenance_verified": data_verified,
            "modes": list(MODES),
            "q_grid": q_grid.tolist(),
            "color_max": args.color_max,
        }
        np.savez_compressed(
            base.with_suffix(".npz"),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            q_grid=q_grid,
            modes=np.asarray(MODES),
            window_indices=selected.astype(np.int64),
            confidence=confidence.astype(np.float32),
            true_class=test_y[selected].astype(np.int16),
            initial_pred=preds[selected].astype(np.int16),
            saliency=saliency.astype(np.float32),
            probvec=probvec,
            argmax=argmax,
            span_start=span_start,
            span_end=span_end,
        )
        _write_window_csv(
            base.with_suffix(".windows.csv"),
            selected,
            confidence,
            test_y,
            preds,
            q_grid,
            argmax,
            target,
        )
        _plot_carpet(
            base,
            model_label=model_label,
            task=args.task,
            class_name=classes[target],
            seed=args.seed,
            cohort=args.cohort,
            q_grid=q_grid,
            confidence=confidence,
            target_prob=probvec[:, :, :, target],
            argmax=argmax,
            target=target,
            vmax=args.color_max,
        )
        base.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated.append({
            "class": classes[target],
            "base": str(base),
            "n": int(n_windows),
        })
        print(f"[saved] {base}.{{png,pdf,npz,json,windows.csv}}")

    run_manifest = {
        "schema_version": 1,
        "status": "complete",
        "model": args.model,
        "task": args.task,
        "method": args.method,
        "cohort": args.cohort,
        "seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
        "generated": generated,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[done] {output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
