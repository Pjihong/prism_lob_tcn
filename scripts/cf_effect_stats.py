#!/usr/bin/env python3
"""Paired effect statistics for counterfactual response-carpet outputs.

Reads the ``.npz`` files written by ``run_cf_response_carpet.py`` and, for
each file, computes per-window retention-curve AUCs for the three span
modes (``cam``, ``random``, ``anti``).  It then reports the paired evidence
that CAM-guided replacement is stronger than each control:

- mean paired difference (control AUC - CAM AUC; positive = CAM stronger);
- seeded percentile bootstrap CI of the mean difference;
- one-sided Wilcoxon signed-rank test (H1: CAM AUC < control AUC);
- Cliff's delta on the paired differences;
- flip rate at the maximum deletion fraction per mode.

Statistical caveat (printed and stored): windows overlap in time, so the
paired samples are NOT independent.  Without block-level resampling these
numbers are descriptive audit evidence, not confirmatory p-values; the
manuscript must present them as such or add block/cluster uncertainty.

Usage
-----
    PYTHONPATH=. python scripts/cf_effect_stats.py \
        --npz figures/cf_response_carpet/full/*/*.npz \
        --out-csv figures/cf_effect_stats.csv \
        --out-json figures/cf_effect_stats.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
for candidate in (THIS.parents[2], Path.cwd()):
    if (candidate / "lob_xai").is_dir():
        sys.path.insert(0, str(candidate))
        break

from lob_xai.stats import bootstrap_ci, cliffs_delta, wilcoxon_signed_rank

MODES = ("cam", "random", "anti")
_trapz = getattr(np, "trapezoid", None) or np.trapz


def retention_auc(probvec: np.ndarray, target: int, q_grid: np.ndarray) -> np.ndarray:
    """Trapezoidal AUC of per-window target-probability retention.

    probvec: (n_windows, n_modes, n_q, n_classes); q_grid includes rho=0.
    Returns (n_windows, n_modes).  Lower AUC = stronger deletion effect.
    """
    target_prob = probvec[:, :, :, target]
    base = np.maximum(target_prob[:, :, :1], 1e-12)
    retention = target_prob / base
    auc = np.empty(retention.shape[:2], dtype=np.float64)
    for m in range(retention.shape[1]):
        auc[:, m] = _trapz(retention[:, m, :], q_grid, axis=1)
    return auc / (q_grid[-1] - q_grid[0])


def analyse_one(path: Path, n_boot: int, seed: int) -> dict:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    modes = [str(m) for m in data["modes"]]
    if modes != list(MODES):
        raise ValueError(f"{path}: unexpected mode order {modes}")
    q_grid = np.asarray(data["q_grid"], dtype=np.float64)
    probvec = np.asarray(data["probvec"], dtype=np.float64)
    argmax = np.asarray(data["argmax"])
    target = int(metadata.get("class_index", -1))
    if target < 0:
        true_class = np.asarray(data["true_class"])
        values, counts = np.unique(true_class, return_counts=True)
        target = int(values[counts.argmax()])

    auc = retention_auc(probvec, target, q_grid)  # (n_windows, 3)
    cam = auc[:, 0]
    result = {
        "npz": str(path),
        "model": metadata.get("model", "?"),
        "task": metadata.get("task", "?"),
        "seed": metadata.get("seed", "?"),
        "class_name": metadata.get("class_name", str(target)),
        "cohort": metadata.get("cohort", "?"),
        "n_windows": int(auc.shape[0]),
        "q_max": float(q_grid[-1]),
        "dependence_caveat": (
            "overlapping windows; descriptive unless block-resampled"
        ),
        "auc_mean": {
            mode: float(auc[:, i].mean()) for i, mode in enumerate(MODES)
        },
        "flip_rate_at_qmax": {
            mode: float((argmax[:, i, -1] != target).mean())
            for i, mode in enumerate(MODES)
        },
        "paired": {},
    }
    for i, control in ((1, "random"), (2, "anti")):
        diff = auc[:, i] - cam  # positive = CAM deletes harder
        _, low, high = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        w_stat, w_p = wilcoxon_signed_rank(diff, alternative="greater")
        result["paired"][f"cam_vs_{control}"] = {
            "mean_diff": float(diff.mean()),
            "bootstrap_ci95": [float(low), float(high)],
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_one_sided": float(w_p),
            "cliffs_delta": float(cliffs_delta(auc[:, i], cam)),
            "n_pairs": int(len(diff)),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = [analyse_one(Path(p), args.n_boot, args.seed)
               for p in sorted(args.npz)]

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(results, indent=1, sort_keys=True), encoding="utf-8"
    )
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "task", "seed", "class", "cohort", "n_windows",
            "auc_cam", "auc_random", "auc_anti",
            "diff_random_mean", "diff_random_ci_low", "diff_random_ci_high",
            "diff_random_wilcoxon_p",
            "diff_anti_mean", "diff_anti_ci_low", "diff_anti_ci_high",
            "diff_anti_wilcoxon_p",
            "flip_cam", "flip_random", "flip_anti",
        ])
        for r in results:
            pr, pa = r["paired"]["cam_vs_random"], r["paired"]["cam_vs_anti"]
            writer.writerow([
                r["model"], r["task"], r["seed"], r["class_name"],
                r["cohort"], r["n_windows"],
                f"{r['auc_mean']['cam']:.6f}",
                f"{r['auc_mean']['random']:.6f}",
                f"{r['auc_mean']['anti']:.6f}",
                f"{pr['mean_diff']:.6f}",
                f"{pr['bootstrap_ci95'][0]:.6f}",
                f"{pr['bootstrap_ci95'][1]:.6f}",
                f"{pr['wilcoxon_p_one_sided']:.3e}",
                f"{pa['mean_diff']:.6f}",
                f"{pa['bootstrap_ci95'][0]:.6f}",
                f"{pa['bootstrap_ci95'][1]:.6f}",
                f"{pa['wilcoxon_p_one_sided']:.3e}",
                f"{r['flip_rate_at_qmax']['cam']:.4f}",
                f"{r['flip_rate_at_qmax']['random']:.4f}",
                f"{r['flip_rate_at_qmax']['anti']:.4f}",
            ])

    print(f"[done] {len(results)} npz analysed -> {args.out_csv}")
    print("[caveat] overlapping windows: treat p-values as descriptive "
          "unless block/cluster uncertainty is added")


if __name__ == "__main__":
    main()
