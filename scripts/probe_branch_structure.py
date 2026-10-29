#!/usr/bin/env python3
"""
probe_branch_structure.py — discover & VERIFY the ours model's branch structure,
then write branch_config.json that axis-A / axis-B read.

What it does
  1. loads the ours checkpoint and one batch of test windows (same loaders as
     the counterfactual experiments),
  2. prints every module's output shape (so you can see the architecture),
  3. auto-detects the 4 branch encoders (fusion points), a per-branch temporal
     feature (for s_b(t)), and the head,
  4. VERIFIES the detection: zeros each detected branch's fused feature and
     checks the target logit actually moves (if it doesn't, the wrong module was
     found),
  5. writes branch_config.json.

If step 3/4 looks wrong, read the printed tree and either (a) edit
branch_config.json by hand (fusion_modules / temporal_modules are dotted names),
or (b) send the printed tree back and it will be finalized for you.

Usage
  export PYTHONPATH=.
  python scripts/probe_branch_structure.py \
      --task vacuum --seed 1 --k 30 --device cpu
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import os

import numpy as np
import torch

from lob_xai.checkpoints import load_micro_windows_in_order, load_ours_model
from lob_xai.branch_analysis import (discover_branches, BranchAdapter, find_ckpt,
                               BRANCH_NAMES, architecture_signature,
                               validate_run_protocol)
from lob_xai.provenance import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="vacuum")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--results-root", default="loggers/results")
    ap.add_argument("--ckpt", default=None,
                    help="explicit ours model.pt (recommended)")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target", type=int, default=1,
                    help="target class: vacuum=1, toxic->1(up)/2(down)")
    ap.add_argument("--config-out", default="branch_config.json")
    ap.add_argument("--n-probe", type=int, default=64)
    args = ap.parse_args()

    ck = args.ckpt or find_ckpt(args.results_root, "ours", args.task, args.seed, args.k)
    if ck is None:
        raise SystemExit(f"[probe] no ours ckpt for {args.task} seed{args.seed} k{args.k}")
    print(f"[probe] ckpt: {ck}")
    model, _, _ = load_ours_model(ck, args.device)
    model.eval()
    X, y, class_names, cfg = load_micro_windows_in_order(ck, args.test_csv, task=args.task)
    validate_run_protocol(cfg, task=args.task, k=args.k, seed=args.seed)
    if not 0 <= args.target < len(class_names):
        raise SystemExit(f"target={args.target} outside [0,{len(class_names)})")
    print(f"[probe] X.shape={tuple(X.shape)} classes={class_names} target={args.target}")
    Xp = X[:args.n_probe]

    # ---- full module tree with output shapes ----
    cfgd = discover_branches(model, Xp, args.device,
                             n_branches=4, seq_len=X.shape[1])
    print("\n=== module output shapes (name : shape) ===")
    for name, shp in cfgd["module_shapes"].items():
        mark = ""
        if name in cfgd["fusion_modules"]:
            mark = "   <== FUSION (branch encoder)"
        if name in cfgd["temporal_modules"]:
            mark += "   <== TEMPORAL (s_b(t) source)"
        if name == cfgd["head_module"]:
            mark += "   <== HEAD?"
        print(f"  {name:40s} {str(tuple(shp)):20s}{mark}")

    print("\n=== auto-detected ===")
    print("  fusion_modules :", cfgd["fusion_modules"])
    print("  temporal_modules:", cfgd["temporal_modules"])
    print("  head_module    :", cfgd["head_module"])
    print("  seq_len        :", cfgd["seq_len"])

    if len(cfgd["fusion_modules"]) != 4:
        print("\n[!!] Could not auto-detect 4 branch modules. The model is not an "
              "obvious ModuleList-of-4. Read the tree above and fill "
              f"'{args.config_out}' fusion_modules/temporal_modules by hand, "
              "or send the tree back to finalize.")
        cfgd.pop("module_shapes", None)
        with open(args.config_out, "w", encoding="utf-8") as f:
            json.dump(cfgd, f, ensure_ascii=False, indent=2)
        print(f"[probe] partial config written to {args.config_out}")
        return

    # ---- VERIFY: zero each branch's fused feature, target logit must move ----
    print("\n=== verification: zero each branch's fused feature -> Δ target logit ===")
    adapter = BranchAdapter(model, cfgd, args.device)
    adapter._mode = "zero"
    full = set(range(4))
    base = adapter.value(Xp, args.target, full)            # all branches present
    ok = True
    deltas = []
    for i in range(4):
        vi = adapter.value(Xp, args.target, full - {i})    # branch i ablated
        d = float(np.mean(np.abs(base - vi)))
        deltas.append(d)
        flag = "ok" if d > 1e-5 else "  <-- NO EFFECT (suspicious)"
        if d <= 1e-5:
            ok = False
        print(f"  ablate {BRANCH_NAMES[i]:11s}: mean|Δlogit| = {d:.4e} {flag}")

    # also check the temporal hook returns a time axis
    st = adapter.branch_temporal(Xp[:16], args.target)
    tshapes = {BRANCH_NAMES[i]: tuple(st[i].shape) for i in st}
    print("  per-branch s_b(t) shapes:", tshapes)
    adapter.close()

    cfgd.pop("module_shapes", None)
    cfgd["verified"] = bool(ok)
    cfgd["checkpoint"] = os.path.abspath(ck)
    cfgd["checkpoint_sha256"] = sha256_file(ck)
    cfgd["architecture_sha256"] = architecture_signature(model)
    cfgd["task"] = args.task
    cfgd["seed"] = int(args.seed)
    cfgd["k"] = int(args.k)
    cfgd["ablation_delta_logit"] = {BRANCH_NAMES[i]: deltas[i] for i in range(4)}
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(cfgd, f, ensure_ascii=False, indent=2)
    print(f"\n[probe] config written to {args.config_out}  (verified={ok})")
    if not ok:
        print("[!!] At least one branch had no effect when zeroed -> the detected "
              "module is probably not the true branch fusion point. Inspect the "
              "tree and adjust before trusting axis-A/B numbers.")


if __name__ == "__main__":
    main()
