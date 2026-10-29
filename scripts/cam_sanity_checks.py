#!/usr/bin/env python3
"""Sanity checks for the CAM explanations on a frozen checkpoint.

Three checks, all computed on the same evaluation cohort (correct target-class
windows, top confidence, stable ordering — the paper cohort rule):

1. **Method agreement** (diagnostic, not faithfulness): pairwise Spearman and
   top-k Jaccard between |CAM| maps from ``hirescam``, ``gradcam``, ``tsr``.
2. **Input-perturbation stability**: add seeded Gaussian noise in z-space,
   keep only windows whose prediction is preserved, and compare maps.
   High similarity is desirable.
3. **Parameter randomization** (Adebayo et al. 2018 style): re-initialize the
   classifier head (and optionally the final TCN block) and compare maps.
   Similarity SHOULD drop — high similarity after randomization means the map
   reflects the architecture, not the learned evidence.

The per-window map used everywhere is the branch-summed ``|CAM|`` over time
(length ``T``), matching the counterfactual span-selection saliency.

Usage
-----
    PYTHONPATH=. python scripts/cam_sanity_checks.py \
        --ckpt loggers/results/.../model.pt \
        --test-csv data/FI2010/FI2010_test.csv \
        --task vacuum --target 1 --n-windows 100 \
        --out-json figures/cam_sanity_vacuum_seed1.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS = Path(__file__).resolve()
for candidate in (THIS.parents[2], Path.cwd()):
    if (candidate / "lob_xai").is_dir():
        sys.path.insert(0, str(candidate))
        break

from scipy.stats import spearmanr

from lob_xai.gradcam import compute_cam_all_branches
from lob_xai.checkpoints import (
    TASK_CLASSES,
    load_micro_windows_in_order,
    load_ours_model,
)

METHODS = ("hirescam", "gradcam", "tsr")


def topk_jaccard(a: np.ndarray, b: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(np.ceil(len(a) * fraction)))
    top_a = set(np.argpartition(a, -count)[-count:].tolist())
    top_b = set(np.argpartition(b, -count)[-count:].tolist())
    return len(top_a & top_b) / len(top_a | top_b)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    value = float(spearmanr(a, b)[0])  # index works across scipy versions
    return 0.0 if not np.isfinite(value) else value


def pairwise_summary(maps_a: np.ndarray, maps_b: np.ndarray,
                     keep: np.ndarray | None = None) -> dict:
    if keep is None:
        keep = np.ones(len(maps_a), dtype=bool)
    rows = [
        (safe_spearman(a, b), topk_jaccard(a, b))
        for a, b in zip(maps_a[keep], maps_b[keep])
    ]
    if not rows:
        return {"n": 0, "spearman_mean": float("nan"),
                "topk_jaccard_mean": float("nan")}
    spearman, jaccard = zip(*rows)
    return {
        "n": int(len(rows)),
        "spearman_mean": float(np.mean(spearman)),
        "spearman_sd": float(np.std(spearman, ddof=1)) if len(rows) > 1 else 0.0,
        "topk_jaccard_mean": float(np.mean(jaccard)),
    }


def branch_summed_maps(model, X: np.ndarray, device, target: int,
                       method: str, batch_size: int = 64) -> np.ndarray:
    """Return (n_windows, T) branch-summed |CAM| maps."""
    outputs = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[start:start + batch_size])).float()
        cams = compute_cam_all_branches(
            model, xb, device, target_class=int(target), method=method
        )
        total = None
        for _, (_, cam_abs) in cams.items():
            arr = np.asarray(cam_abs, dtype=np.float64)
            total = arr if total is None else total + arr
        outputs.append(total)
    return np.concatenate(outputs, axis=0)


def randomized_copy(model, scope: str, seed: int):
    """Deep-copied model with re-initialized head (and last TCN block)."""
    clone = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)

    def reinit(module: torch.nn.Module) -> None:
        for p in module.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)

    reinit(clone.fc)
    if scope == "head_and_last_block":
        for branch in clone.cnns:
            blocks = [m for m in branch.modules()
                      if m.__class__.__name__ == "TCNBlock"]
            if blocks:
                reinit(blocks[-1])
    clone.eval()
    return clone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--task", required=True, choices=["toxic", "vacuum"])
    ap.add_argument("--target", type=int, required=True,
                    help="risk class index (e.g. toxic_up=1, vacuum=1)")
    ap.add_argument("--n-windows", type=int, default=100)
    ap.add_argument("--noise-sigma", type=float, default=0.05,
                    help="z-space Gaussian noise for the stability check")
    ap.add_argument("--randomize-scope", default="head",
                    choices=["head", "head_and_last_block"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--allow-unverified-data", action="store_true",
                    help="diagnostic only; never for paper results")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, cfg, meta = load_ours_model(args.ckpt, device=device, strict=True)
    X, y, classes, _ = load_micro_windows_in_order(
        args.ckpt, args.test_csv, task=args.task, which="test",
        verify_identity=not args.allow_unverified_data,
    )
    if not 0 <= args.target < len(classes):
        raise SystemExit(f"target must be in [0,{len(classes)}) for {classes}")

    # Paper cohort rule: correct target-class windows, top confidence, stable.
    with torch.no_grad():
        probs = []
        for start in range(0, len(X), 512):
            xb = torch.from_numpy(np.ascontiguousarray(X[start:start + 512])).float()
            probs.append(torch.softmax(model(xb.to(device)), dim=-1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    preds = probs.argmax(axis=1)
    eligible = np.flatnonzero((y == args.target) & (preds == args.target))
    if len(eligible) == 0:
        raise SystemExit("no correct target-class windows found")
    order = np.argsort(probs[eligible, args.target], kind="mergesort")
    cohort = eligible[order][-args.n_windows:]
    Xc = X[cohort]

    rng = np.random.default_rng(args.seed)

    # 1) method agreement
    maps = {m: branch_summed_maps(model, Xc, device, args.target, m,
                                  args.batch_size)
            for m in METHODS}
    agreement = {
        f"{a}_vs_{b}": pairwise_summary(maps[a], maps[b])
        for i, a in enumerate(METHODS) for b in METHODS[i + 1:]
    }

    # 2) input-perturbation stability (prediction-preserved only)
    noisy = Xc + rng.normal(0.0, args.noise_sigma, Xc.shape).astype(np.float32)
    with torch.no_grad():
        noisy_pred = []
        for start in range(0, len(noisy), 512):
            xb = torch.from_numpy(np.ascontiguousarray(noisy[start:start + 512])).float()
            noisy_pred.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
    preserved = np.concatenate(noisy_pred) == args.target
    noisy_maps = branch_summed_maps(model, noisy, device, args.target,
                                    "hirescam", args.batch_size)
    stability = pairwise_summary(maps["hirescam"], noisy_maps, keep=preserved)
    stability["n_prediction_preserved"] = int(preserved.sum())
    stability["noise_sigma"] = args.noise_sigma

    # 3) parameter randomization — similarity SHOULD drop
    random_model = randomized_copy(model, args.randomize_scope, args.seed)
    random_maps = branch_summed_maps(random_model, Xc, device, args.target,
                                     "hirescam", args.batch_size)
    randomization = pairwise_summary(maps["hirescam"], random_maps)
    randomization["scope"] = args.randomize_scope

    payload = {
        "checkpoint": args.ckpt,
        "task": args.task,
        "target_class": classes[args.target],
        "cohort": "top_correct",
        "n_windows": int(len(cohort)),
        "seed": args.seed,
        "pool": cfg.get("pool", "avg"),
        "interpretation": {
            "agreement": "diagnostic consistency between methods; high values "
                         "do not imply faithfulness",
            "stability": "higher is better (prediction-preserved windows only)",
            "randomization": "similarity should DROP after re-initialization; "
                             "high residual similarity is a red flag",
        },
        "method_agreement": agreement,
        "input_stability_hirescam": stability,
        "parameter_randomization_hirescam": randomization,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True),
                   encoding="utf-8")
    print(f"[done] {out}")
    for name, block in agreement.items():
        print(f"  agreement {name}: rho={block['spearman_mean']:.3f} "
              f"jaccard={block['topk_jaccard_mean']:.3f}")
    print(f"  stability: rho={stability['spearman_mean']:.3f} "
          f"(n={stability['n_prediction_preserved']})")
    print(f"  randomization: rho={randomization['spearman_mean']:.3f} "
          f"(should be low)")


if __name__ == "__main__":
    main()
