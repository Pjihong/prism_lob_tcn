#!/usr/bin/env python3
"""Measure branch attribution against training-mean feature replacement.

For each selected correct target alert, compute signed drop-one target-logit
changes and per-branch |HiResCAM| shares. Compare their four-branch rankings
with per-window Kendall tau and the predeclared sign test; also report
branch-mean Spearman agreement and branch-wise Wilcoxon tests.

Inputs: explicit checkpoint, verified probe config, FI-2010 train/test CSVs.
The replacement baseline is fitted on training windows, not test windows.
Outputs: axisA_min_* CSV/JSON and diagnostic PNG/PDF. These established
output names are retained for downstream readers.
Run --help for arguments; docs/REPRODUCE.md connects probe and analysis.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import itertools
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lob_xai.checkpoints import load_micro_windows_in_order, load_ours_model
from lob_xai.branch_analysis import (BranchAdapter, architecture_signature, load_config,
                               find_ckpt, validate_run_protocol)
from lob_xai.evaluation_protocol import select_top_confidence_alerts
from lob_xai.provenance import sha256_file

try:
    from scipy.stats import binomtest, wilcoxon, kendalltau, spearmanr
    _HAVE_SCIPY = True
except Exception:                       # scipy is optional
    _HAVE_SCIPY = False

    def spearmanr(a, b):                # numpy fallback (rho only)
        ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
        return np.corrcoef(ar, br)[0, 1], np.nan

TEAL = "#1D9E75"
GRAY = "#888780"


# ---------------------------------------------------------------------------
# small stats helpers (work with or without scipy)
# ---------------------------------------------------------------------------
def kendall_tau(a, b):
    """Kendall tau-b on short vectors; hand impl when scipy is absent."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if _HAVE_SCIPY:
        t = kendalltau(a, b).correlation
        return float(t) if (t is not None and np.isfinite(t)) else np.nan
    n = len(a); nc = nd = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                nc += 1
            elif s < 0:
                nd += 1
    denom = nc + nd
    return (nc - nd) / denom if denom else np.nan


def exact_perm_p_spearman(x, y):
    """Exact permutation p = P(rho_perm >= rho_obs) over all |y|! orderings.
    Intended for n<=8 (here n=4 branches => 24 permutations)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho_obs, _ = spearmanr(x, y)
    cnt = tot = 0
    for p in itertools.permutations(range(len(y))):
        r, _ = spearmanr(x, y[list(p)])
        tot += 1
        if r >= rho_obs - 1e-12:
            cnt += 1
    return float(rho_obs), cnt / tot


def stars(p):
    if not np.isfinite(p):
        return "?"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="toxic")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--results-root", default="loggers/results")
    ap.add_argument("--ckpt", default=None,
                    help="explicit ours model.pt (recommended for paper runs)")
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target", type=int, default=1,
                    help="target class: vacuum=1, toxic up=1 / down=2")
    ap.add_argument("--n-windows", type=int, default=400)
    ap.add_argument("--config", default="branch_config.json")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--baseline", choices=["mean", "zero"], default="mean")
    ap.add_argument("--use-prob", action="store_true",
                    help="value function = target probability instead of logit")
    ap.add_argument("--pred-based", action="store_true",
                    help="diagnostic: audit emitted alerts even when the true class differs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    n = cfg["n_branches"]; bnames = cfg["branch_names"]
    if len(cfg.get("fusion_modules", [])) != n:
        raise SystemExit(f"[axisA-min] {args.config} has no valid fusion_modules; "
                         "run probe_branch_structure.py first.")
    if not cfg.get("verified", False):
        raise SystemExit("[axisA-min] branch_config.json is not verified; run the probe first")

    ck = args.ckpt or find_ckpt(args.results_root, "ours", args.task, args.seed, args.k)
    if ck is None:
        raise SystemExit(f"[axisA-min] no ours ckpt for {args.task} seed{args.seed} k{args.k}")
    model, _, _ = load_ours_model(ck, args.device)
    if cfg.get("architecture_sha256") != architecture_signature(model):
        raise SystemExit(
            "branch_config architecture does not match this checkpoint; rerun the probe"
        )
    X, y, class_names, ck_cfg = load_micro_windows_in_order(
        ck, args.test_csv, task=args.task
    )
    validate_run_protocol(ck_cfg, task=args.task, k=args.k, seed=args.seed)
    if not 0 <= args.target < len(class_names):
        raise SystemExit(f"target={args.target} outside [0,{len(class_names)})")
    X_train, _, _, _ = load_micro_windows_in_order(
        ck, args.train_csv, task=args.task, which="train"
    )
    print(f"[axisA-min] X={tuple(X.shape)} classes={class_names} target={args.target}")

    adapter = BranchAdapter(model, cfg, args.device)
    adapter._mode = args.baseline
    if args.baseline == "mean":
        adapter.fit_baseline(X_train)

    # Paper cohort: correct target alerts, highest target probability first.
    with torch.no_grad():
        probs = []
        for s in range(0, len(X), 256):
            xb = torch.from_numpy(np.asarray(X[s:s + 256])).float().to(args.device)
            probs.append(torch.softmax(model(xb), -1).cpu().numpy())
        probs = np.concatenate(probs)
    sel = select_top_confidence_alerts(
        probs, y, args.target, args.n_windows,
        require_correct=not args.pred_based,
    )
    if len(sel) == 0:
        raise SystemExit("[axisA-min] no eligible target alerts")
    Xs = X[sel]
    print(f"[axisA-min] n target windows = {len(Xs)}")

    # ---- (1) drop-one ablation effect per window: Delta_i = v(N) - v(N\{i}) ----
    full = adapter.value(Xs, args.target, set(range(n)), use_prob=args.use_prob)   # [N]
    drop = np.stack(
        [full - adapter.value(Xs, args.target, set(range(n)) - {i}, use_prob=args.use_prob)
         for i in range(n)], axis=1)                                               # [N, n]

    # ---- (2) per-branch |HiResCAM| share per window (observational) ----
    s0 = adapter.branch_temporal(Xs, args.target)                                  # idx -> [N,T] signed
    cam_mag = np.stack([np.abs(s0[i]).sum(axis=1) for i in range(n)], axis=1)       # [N, n]
    cam_share = cam_mag / (cam_mag.sum(axis=1, keepdims=True) + 1e-12)              # [N, n]
    adapter.close()

    # ---- importance: mean +/- SE + per-branch Wilcoxon (H0: Delta_i = 0) ----
    drop_mean = drop.mean(0); drop_se = drop.std(0) / np.sqrt(max(len(drop), 1))
    cam_mean = cam_share.mean(0)
    pval = np.full(n, np.nan)
    if _HAVE_SCIPY:
        for i in range(n):
            try:
                pval[i] = wilcoxon(drop[:, i], alternative="two-sided").pvalue
            except ValueError:                       # all-zero differences
                pval[i] = np.nan

    # ---- agreement: per-window Kendall tau(cam_share, drop), tested over windows ----
    taus_all = np.array([kendall_tau(cam_share[w], drop[w]) for w in range(len(Xs))])
    taus = taus_all[np.isfinite(taus_all)]
    if len(taus) == 0:
        raise RuntimeError("every per-window Kendall tau is undefined")
    tau_mean = float(np.mean(taus)) if len(taus) else np.nan
    tau_p = np.nan
    if _HAVE_SCIPY and len(taus):
        nonzero = taus[taus != 0]
        if len(nonzero):
            tau_p = binomtest(
                int((nonzero > 0).sum()), len(nonzero), p=0.5,
                alternative="greater",
            ).pvalue
    # descriptive headline on the 4 branch means (+ exact 4!-permutation p)
    rho_mean, rho_perm_p = exact_perm_p_spearman(cam_mean, drop_mean)

    # ---- report ----
    order = np.argsort(-drop_mean)
    vlabel = "prob" if args.use_prob else "logit"
    print(f"\n=== branch importance (drop-one ablation, value = {vlabel}) ===")
    for i in order:
        print(f"  {bnames[i]:11s} drop-one={drop_mean[i]:+.4f} +/- {drop_se[i]:.4f}   "
              f"CAM share={cam_mean[i]:.3f}   Wilcoxon p={pval[i]:.2e} {stars(pval[i])}")
    print("\n  agreement (observational CAM  vs  causal ablation):")
    print(f"    per-window Kendall tau: mean={tau_mean:+.3f}  sign-test(tau>0) p={tau_p:.2e}  "
          f"(n={len(taus)} windows)   <-- headline")
    print(f"    branch-mean Spearman={rho_mean:+.3f}  exact 4!-perm p={rho_perm_p:.3f} "
          f"(descriptive; floor 1/24 ~= 0.042)")

    # ---- CSV ----
    out_csv = os.path.join(args.out_dir, f"axisA_min_{args.task}_k{args.k}_seed{args.seed}.csv")
    checkpoint_hash = sha256_file(ck)
    config_hash = sha256_file(args.config)
    selection_name = ("top_confidence_predicted" if args.pred_based
                      else "top_confidence_correct")
    with open(out_csv, "w", newline="") as f:
        fields = [
            "task", "k", "seed", "target", "value", "baseline", "selection",
            "n_windows", "branch", "dropone_mean", "dropone_se", "cam_share",
            "wilcoxon_p", "kendall_tau_mean", "kendall_sign_test_p",
            "spearman_branchmean", "spearman_perm_p", "checkpoint_sha256",
            "branch_config_sha256",
        ]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i in range(n):
            w.writerow(dict(
                task=args.task, k=args.k, seed=args.seed, target=args.target,
                value=vlabel, baseline=args.baseline, selection=selection_name,
                n_windows=len(taus), branch=bnames[i],
                dropone_mean=f"{drop_mean[i]:.6f}",
                dropone_se=f"{drop_se[i]:.6f}", cam_share=f"{cam_mean[i]:.6f}",
                wilcoxon_p=f"{pval[i]:.3e}", kendall_tau_mean=f"{tau_mean:.6f}",
                kendall_sign_test_p=f"{tau_p:.3e}",
                spearman_branchmean=f"{rho_mean:.6f}",
                spearman_perm_p=f"{rho_perm_p:.6f}",
                checkpoint_sha256=checkpoint_hash,
                branch_config_sha256=config_hash,
            ))

    window_csv = os.path.splitext(out_csv)[0] + ".windows.csv"
    with open(window_csv, "w", newline="") as f:
        fields = [
            "task", "k", "seed", "target", "selection_rank", "window_index",
            "confidence", "kendall_tau", "branch", "dropone", "cam_share",
            "checkpoint_sha256",
        ]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rank, wi in enumerate(sel, 1):
            for i, branch in enumerate(bnames):
                w.writerow(dict(
                    task=args.task, k=args.k, seed=args.seed, target=args.target,
                    selection_rank=rank, window_index=int(wi),
                    confidence=float(probs[wi, args.target]),
                    kendall_tau=float(taus_all[rank - 1]), branch=branch,
                    dropone=float(drop[rank - 1, i]),
                    cam_share=float(cam_share[rank - 1, i]),
                    checkpoint_sha256=checkpoint_hash,
                ))
    print(f"[saved] {out_csv}, {window_csv}")

    # ---- figure: (A) drop-one importance bars + Wilcoxon stars ----
    # ----         (B) CAM-share <-> drop-one slopegraph (rank agreement) ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    cols = [TEAL if i == order[0] else GRAY for i in order]
    ax[0].barh(np.arange(n), drop_mean[order], xerr=drop_se[order],
               color=cols, capsize=3, height=0.6)
    ax[0].set_yticks(np.arange(n)); ax[0].set_yticklabels([bnames[i] for i in order])
    ax[0].invert_yaxis(); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel(("Delta P" if args.use_prob else "Delta logit") + "  (drop-one ablation)")
    ax[0].set_title("branch importance (intervention)")
    for yi, i in enumerate(order):
        if np.isfinite(pval[i]):
            ax[0].text(drop_mean[i] + drop_se[i], yi, " " + stars(pval[i]),
                       va="center", fontsize=9)

    def _share(v):
        v = np.clip(v, 0.0, None); s = v.sum()
        return v / s if s > 0 else v
    cam_s = _share(cam_mean); drp_s = _share(drop_mean)
    for i in range(n):
        c = TEAL if i == order[0] else GRAY
        ax[1].plot([0, 1], [cam_s[i], drp_s[i]], "-o", color=c,
                   lw=(2.6 if i == order[0] else 1.4), ms=6)
        ax[1].annotate(bnames[i], (1, drp_s[i]), textcoords="offset points",
                       xytext=(6, 0), fontsize=8, va="center")
    ax[1].set_xlim(-0.05, 1.35); ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["CAM share\n(observational)", "drop-one\n(causal)"])
    ax[1].set_ylabel("share of branch importance")
    ax[1].set_title("observational = causal\n"
                    f"mean Kendall tau={tau_mean:.2f} (p={tau_p:.1e}), "
                    f"Spearman={rho_mean:.2f}")

    fig.suptitle(f"Axis A (minimal, no Shapley) - {args.task}, k={args.k}, seed{args.seed}; "
                 "DeepLOB: undefined (no branch axis)", fontsize=11)
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, f"axisA_min_{args.task}_k{args.k}_seed{args.seed}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n[saved] {out_csv}\n[saved] {out_png}")


if __name__ == "__main__":
    main()
