#!/usr/bin/env python3
"""Aggregate exactly one frozen test result for each declared task/seed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


METRICS = {
    "mcc": "MCC",
    "f1": "macro F1",
    "pr_auc_macro_nonflat": "PR-AUC_nf",
    "roc_auc_macro_nonflat": "ROC-AUC_nf",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--require-seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    paths = sorted(Path(args.input_dir).glob("*_seed*.json"))
    if not paths:
        raise SystemExit("no per-seed JSON files found")
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = set(args.require_seeds)
    grouped: dict[str, list[dict]] = {}
    for doc in docs:
        task = str(doc.get("task"))
        seed = int(doc.get("seed"))
        key = (task, seed)
        if any((str(old.get("task")), int(old.get("seed"))) == key
               for old in docs if old is not doc):
            raise SystemExit(f"duplicate result for {task} seed {seed}")
        grouped.setdefault(task, []).append(doc)

    if set(grouped) != {"toxic", "vacuum"}:
        raise SystemExit(f"expected toxic and vacuum, found {sorted(grouped)}")

    rows = []
    for task in ("toxic", "vacuum"):
        group = sorted(grouped[task], key=lambda d: int(d["seed"]))
        got = {int(doc["seed"]) for doc in group}
        if got != expected:
            raise SystemExit(
                f"{task}: expected seeds {sorted(expected)}, found {sorted(got)}"
            )
        row = {"task": task, "n": len(group),
               "seeds": ",".join(str(x) for x in sorted(got))}
        for key in METRICS:
            values = [float(doc["metrics"][key]) for doc in group]
            row[f"{key}_mean"] = mean(values)
            row[f"{key}_sd"] = pstdev(values)
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "standard_deviation": "population (ddof=0)",
        "per_seed": docs,
        "aggregate": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("task      n   MCC mean±sd       F1 mean±sd        PR-AUC_nf mean±sd")
    for row in rows:
        print(
            f"{row['task']:8s}  {row['n']:1d}   "
            f"{row['mcc_mean']:.4f}±{row['mcc_sd']:.4f}    "
            f"{row['f1_mean']:.4f}±{row['f1_sd']:.4f}    "
            f"{row['pr_auc_macro_nonflat_mean']:.4f}±"
            f"{row['pr_auc_macro_nonflat_sd']:.4f}"
        )
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
