"""
scripts/audit_label_prevalence.py — Verify class prevalence per stock / session
                                     for the new Toxic/Vacuum labels.

Outputs:
  - figures/label_prevalence_toxic.csv
  - figures/label_prevalence_vacuum.csv

Each row = one (split, segment) tuple with per-class counts and percentages.

For the paper, we want to see roughly 2.8% / 2.8% / 94.4% (Toxic) and
10% / 90% (Vacuum) consistently. Strong stock-to-stock drift > 50%
relative would be an issue worth discussing.

Usage
-----
python scripts/audit_label_prevalence.py \\
    --train-csv data/FI2010/FI2010_train.csv \\
    --test-csv  data/FI2010/FI2010_test.csv \\
    --segment-size 200000
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.data import load_fi2010_features_only
from lob_xai.microstructure_labels import (
    compute_microstructure_labels, TICK_SIZE_DEFAULT, K_DEFAULT,
)


def _segment_audit(X, task, segment_size, k, tick_size):
    rows = []
    n = X.shape[0]
    for s in range(0, n, segment_size):
        seg = X[s:s + segment_size]
        if len(seg) < 200:
            continue
        y, valid, info = compute_microstructure_labels(
            seg, task=task, k=k, tick_size=tick_size,
        )
        y_v = y[valid]
        if len(y_v) == 0:
            continue
        counts = np.bincount(y_v.astype(int), minlength=info["n_classes"])
        for c in range(info["n_classes"]):
            rows.append(dict(
                seg_start=int(s), seg_end=int(s + len(seg)),
                n_valid=int(valid.sum()),
                class_idx=int(c),
                class_name=info["class_names"][c],
                count=int(counts[c]),
                pct=float(counts[c]) / max(1, len(y_v)),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv",  default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--segment-size", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--tick-size", type=float, default=TICK_SIZE_DEFAULT)
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for task in ["toxic", "vacuum"]:
        rows_all = []
        for split, csv_path in [("train", args.train_csv),
                                  ("test", args.test_csv)]:
            print(f"\n[audit] === {task} / {split} ===")
            X = load_fi2010_features_only(csv_path, n_features=40)
            print(f"  X.shape = {X.shape}")
            seg_rows = _segment_audit(
                X, task=task, segment_size=args.segment_size,
                k=args.k, tick_size=args.tick_size,
            )
            for r in seg_rows:
                r["split"] = split
            rows_all.extend(seg_rows)

            # print per-task overall prevalence on the whole split
            y, valid, info = compute_microstructure_labels(
                X, task=task, k=args.k, tick_size=args.tick_size,
            )
            print(f"  overall prevalence: {info['prevalence']}")

        out_csv = Path(args.out_dir) / f"label_prevalence_{task}.csv"
        keys = ["split", "seg_start", "seg_end", "n_valid",
                "class_idx", "class_name", "count", "pct"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows_all: w.writerow({k: r[k] for k in keys})
        print(f"[audit] wrote {out_csv}  ({len(rows_all)} rows)")


if __name__ == "__main__":
    main()
