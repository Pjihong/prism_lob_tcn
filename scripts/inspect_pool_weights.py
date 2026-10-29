#!/usr/bin/env python3
"""Export the learned, branch-local temporal readout weights."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.checkpoints import load_ours_model


def _weight_stats(weights: np.ndarray) -> dict:
    weights = np.asarray(weights, dtype=float)
    length = len(weights)
    positions = np.arange(length)
    lags = (length - 1) - positions
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-15))))
    top = np.argsort(weights)[::-1][: min(10, length)]
    return {
        "sum": float(weights.sum()),
        "normalized_entropy": entropy / math.log(length) if length > 1 else 0.0,
        "expected_lag_from_end": float(np.sum(weights * lags)),
        "mass_last_8": float(weights[-min(8, length):].sum()),
        "mass_last_32": float(weights[-min(32, length):].sum()),
        "top_positions": [
            {
                "tick_from_start_1based": int(i + 1),
                "lag_from_end": int(length - 1 - i),
                "weight": float(weights[i]),
            }
            for i in top
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    results = []
    for checkpoint in args.checkpoints:
        model, cfg, meta = load_ours_model(checkpoint, device="cpu", strict=True)
        if not getattr(model, "branch_local_pool", False):
            raise SystemExit(f"not a branch-local pooling checkpoint: {checkpoint}")
        branches = list(meta.get("branch_names") or model.branch_names)
        record = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "task": cfg.get("task"),
            "seed": cfg.get("seed"),
            "pool": model.pool_kind,
            "n_params": sum(p.numel() for p in model.parameters()),
            "branches": {},
        }
        for name, pool in zip(branches, model.pool.pools):
            effective = pool.temporal_weights(model.seq_len).detach().cpu().numpy()
            detail = {"effective": _weight_stats(effective)}
            if model.pool_kind == "multi_scale":
                mixture = torch.softmax(pool.logits, dim=0).detach().cpu().numpy()
                detail["scale_mixture"] = {
                    str(window): float(mass)
                    for window, mass in zip(pool.windows, mixture)
                }
            record["branches"][name] = detail
        results.append(record)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    for item in results:
        print(f"{item['task']} seed={item['seed']} pool={item['pool']} "
              f"params={item['n_params']:,}")
        for branch, detail in item["branches"].items():
            stats = detail["effective"]
            extra = ""
            if "scale_mixture" in detail:
                extra = f" scales={detail['scale_mixture']}"
            print(f"  {branch:12s} last8={stats['mass_last_8']:.3f} "
                  f"last32={stats['mass_last_32']:.3f} "
                  f"E[lag]={stats['expected_lag_from_end']:.1f}{extra}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
