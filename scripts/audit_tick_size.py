"""
scripts/audit_tick_size.py — Empirical audit of the z-score tick size.

Computes the smallest non-zero |ΔAsk1_Price| across the training CSV and
reports several robust estimators. The recommended TICK_SIZE_DEFAULT (0.00098
in microstructure_labels.py) should agree with the median of the smallest
1% of positive steps on the FI-2010 standard release.

If your release / preprocessing differs and the audit shows a different
value, update `microstructure_labels.TICK_SIZE_DEFAULT` or pass
`--tick-size` to all training scripts.

Usage
-----
python scripts/audit_tick_size.py \\
    --train-csv data/FI2010/FI2010_train.csv \\
    --out-csv figures/tick_size_audit.csv
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.data import load_fi2010_features_only
from lob_xai.microstructure_labels import audit_tick_size, TICK_SIZE_DEFAULT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--n-samples", type=int, default=500_000)
    ap.add_argument("--out-csv", default="figures/tick_size_audit.csv")
    ap.add_argument("--columns", type=int, nargs="+", default=[0, 2],
                    help="raw FI-2010 column indices to audit (default: "
                         "0=Ask1_Price, 2=Bid1_Price)")
    args = ap.parse_args()

    print(f"[audit] loading {args.train_csv}")
    X = load_fi2010_features_only(args.train_csv, n_features=40)
    print(f"[audit] X.shape = {X.shape}")

    rows = []
    for col in args.columns:
        print(f"\n[audit] === col {col} ===")
        # adapt: compute on each column rather than just col 0
        X_local = X.copy()
        # swap so audit_tick_size measures col `col` (it hard-codes 0)
        X_local[:, 0] = X[:, col]
        res = audit_tick_size(X_local, n_samples=args.n_samples)
        print(f"  n_zero_steps             = {res['n_zero_steps']}")
        print(f"  min positive step        = {res['min_step']:.6g}")
        print(f"  median(smallest 1%)      = {res['median_step_smallest_1pct']:.6g}")
        print(f"  recommended tick_size    = {res['recommended']:.6g}")
        print(f"  DEFAULT in code          = {TICK_SIZE_DEFAULT:.6g}")
        relative_err = abs(res["recommended"] - TICK_SIZE_DEFAULT) / TICK_SIZE_DEFAULT
        print(f"  |recommended − default|  = {relative_err:.2%} of default")
        rows.append(dict(
            column=col, **res,
            default=TICK_SIZE_DEFAULT,
            relative_err_vs_default=relative_err,
        ))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\n[audit] wrote {args.out_csv}")

    # Final recommendation
    recs = [r["recommended"] for r in rows]
    overall = float(np.median(recs))
    print(f"\n[audit] OVERALL recommended tick_size = {overall:.6g}")
    if max(abs(r - overall) for r in recs) / overall > 0.2:
        print("[audit] WARNING: column-to-column variation > 20% — "
              "consider per-column tick sizes.")


if __name__ == "__main__":
    main()
