#!/usr/bin/env python3
"""Aggregate validation-only branch architecture experiments.

The collector groups runs by the actual saved architecture/loss configuration,
not by directory naming.  By default it rejects runs that already contain test
metrics, which makes accidental test-set model selection visible.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def _signature(cfg: dict) -> tuple:
    return (
        tuple(cfg.get("cnn_channels", (32, 32, 64))),
        int(cfg.get("cnn_kernel", 3)),
        int(cfg.get("tcn_dilation_base", 2)),
        cfg.get("pool", "avg"),
        tuple(cfg.get("pool_windows", (8, 32, 100))),
        cfg.get("tcn_norm", "none"),
        cfg.get("input_norm", "identity"),
        int(cfg.get("interaction_channels", 0)),
        cfg.get("loss", "ce_weighted"),
        float(cfg.get("focal_gamma", 2.0)),
        float(cfg.get("class_weight_power", 1.0)),
        cfg.get("optimizer", "adamw"),
        cfg.get("scheduler", "cosine"),
        float(cfg.get("lr", 1e-3)),
        float(cfg.get("weight_decay", 1e-5)),
        cfg.get("early_stop_metric", "val_f1"),
        tuple(cfg.get("fc_hidden", (64,))),
    )


def _label(sig: tuple) -> str:
    (channels, kernel, dilation, pool, pool_windows, tcn_norm, input_norm,
     interaction, loss, gamma, weight_power, optimizer, scheduler, lr,
     weight_decay, early_metric, hidden) = sig
    ch = "-".join(str(x) for x in channels)
    fc = "-".join(str(x) for x in hidden)
    pw = "-".join(str(x) for x in pool_windows)
    return (
        f"ch{ch}_k{kernel}_d{dilation}_pool-{pool}_pw{pw}_tn-{tcn_norm}_"
        f"in-{input_norm}_mix{interaction}_{loss}-g{gamma:g}-w{weight_power:g}_"
        f"{optimizer}-{scheduler}-lr{lr:g}-wd{weight_decay:g}_"
        f"sel-{early_metric}_fc{fc}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="loggers/results")
    ap.add_argument("--name-contains", default="branchdev_")
    ap.add_argument("--out-csv", default="figures/branch_dev_validation.csv")
    ap.add_argument("--allow-unsealed", action="store_true",
                    help="include runs whose summary contains test metrics")
    args = ap.parse_args()

    rows = []
    rejected = []
    for path in sorted(Path(args.results_dir).glob("*/summary.json")):
        if args.name_contains and args.name_contains not in path.parent.name:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("test") is not None and not args.allow_unsealed:
            rejected.append(str(path))
            continue
        val = doc.get("val") or {}
        cfg = doc.get("config") or {}
        required = ("f1", "mcc", "pr_auc_macro_nonflat")
        if any(k not in val for k in required):
            continue
        flat_class = int(doc.get("flat_class", 0))
        n_classes = int(doc.get("n_classes", 0))
        rare_recalls = [
            float(val.get(f"recall_class_{c}", 0.0))
            for c in range(n_classes) if c != flat_class
        ]
        rows.append(dict(
            run=path.parent.name,
            task=doc.get("task"),
            k=int(cfg.get("k", -1)),
            seed=int(cfg.get("seed", -1)),
            signature=_signature(cfg),
            n_params=int(doc.get("n_params", -1)),
            val_f1=float(val["f1"]),
            val_mcc=float(val["mcc"]),
            val_pr_auc=float(val["pr_auc_macro_nonflat"]),
            zero_rare_recall=any(x <= 0.0 for x in rare_recalls),
        ))

    if rejected:
        print(f"[sealed] rejected {len(rejected)} run(s) containing test metrics")
    if not rows:
        raise SystemExit("no matching validation-only summary.json files found")

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["task"], row["k"], row["signature"])
        groups.setdefault(key, []).append(row)

    output = []
    for (task, k, sig), grp in sorted(groups.items(), key=lambda item: str(item[0])):
        def stats(key: str) -> tuple[float, float]:
            values = [r[key] for r in grp]
            return mean(values), stdev(values) if len(values) > 1 else 0.0

        f1_m, f1_s = stats("val_f1")
        mcc_m, mcc_s = stats("val_mcc")
        pr_m, pr_s = stats("val_pr_auc")
        output.append(dict(
            task=task, k=k, config=_label(sig), n_runs=len(grp),
            seeds=",".join(str(r["seed"]) for r in sorted(grp, key=lambda x: x["seed"])),
            n_params=grp[0]["n_params"],
            val_f1_mean=f1_m, val_f1_sd=f1_s,
            val_mcc_mean=mcc_m, val_mcc_sd=mcc_s,
            val_pr_auc_mean=pr_m, val_pr_auc_sd=pr_s,
            collapsed_runs=sum(int(r["zero_rare_recall"]) for r in grp),
        ))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(f"{'task':8s}  {'k':>3s}  {'config':88s}  {'n':>2s}  {'collapse':>8s}  {'val MCC':>15s}  {'val F1':>15s}  {'val PR':>15s}")
    for row in sorted(output, key=lambda x: x["val_mcc_mean"], reverse=True):
        print(
            f"{str(row['task']):8s}  {row['k']:3d}  {row['config'][:88]:88s}  "
            f"{row['n_runs']:2d}  "
            f"{row['collapsed_runs']:8d}  "
            f"{row['val_mcc_mean']:.4f}±{row['val_mcc_sd']:.4f}  "
            f"{row['val_f1_mean']:.4f}±{row['val_f1_sd']:.4f}  "
            f"{row['val_pr_auc_mean']:.4f}±{row['val_pr_auc_sd']:.4f}"
        )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
