#!/usr/bin/env python3
"""Compare signed CAM with the positional-readout factorization.

Inputs: FI-2010 CSVs and a checkpoint, or explicitly synthetic --demo input.
For a representative correct target alert, plot temporal weights, signed
Grad-CAM and direct/reconstructed HiResCAM. Check the reconstruction residual
on the declared cohort, without changing the model or the sample selection.
Outputs: PNG/PDF, residual data and provenance under --out-dir.
Run with python -m lob_xai_viz.plot_factorization --help.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lob_xai.counterfactual_test import (
    TASK_CLASSES, build_ours, load_windows, make_demo,
)
from lob_xai.branching import make_branching_scheme
from lob_xai.evaluation_protocol import select_top_confidence_alerts
from lob_xai.gradcam import compute_cam_all_branches
from lob_xai.models import LOBBranchModel
from lob_xai.paper_protocol import COUNTERFACTUAL, DATA
from lob_xai.provenance import sha256_file, verify_file_identity

BRANCH_COLORS = {
    "ask_price": "#2F6B9A",
    "ask_volume": "#D9822B",
    "bid_price": "#2A9D8F",
    "bid_volume": "#B24C63",
}
BRANCH_LABELS = {
    "ask_price": "Ask price",
    "ask_volume": "Ask volume",
    "bid_price": "Bid price",
    "bid_volume": "Bid volume",
}
INK, MUTED = "#1a1a1a", "#767676"
REL_TOL = 1e-3          # float32 tolerance for the exact identity


def dodge(pairs, min_gap):
    pairs = sorted(pairs, key=lambda p: p[1])
    ys = [p[1] for p in pairs]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    off = np.mean([p[1] for p in pairs]) - np.mean(ys)
    return {p[0]: y + off for p, y in zip(pairs, ys)}


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0", lw=0.8)
    ax.set_axisbelow(True)


def readout_alphas(model, T):
    """alpha_t per branch from checkpoint parameters. Refuses non-linear pools."""
    pool = getattr(model, "pool", None)
    if not getattr(model, "branch_local_pool", False) or pool is None:
        raise SystemExit(
            "factorization requires a branch-local learned readout "
            "(multi_scale / positional); this checkpoint has pool_kind="
            f"{getattr(model, 'pool_kind', '?')!r}. The alpha_t law does not "
            "apply to GAP/attention readouts — refusing to draw.")
    alphas, mixtures = {}, {}
    for name, p in zip(model.branch_names, pool.pools):
        with torch.no_grad():
            alphas[name] = p.temporal_weights(T).detach().cpu().numpy()
        if hasattr(p, "windows"):                      # MultiScaleTailPool1D
            with torch.no_grad():
                mix = torch.softmax(p.logits, dim=0).cpu().numpy()
            mixtures[name] = {"windows": [int(w) for w in p.windows],
                              "mixture": [round(float(v), 6) for v in mix]}
    return alphas, (mixtures or None), pool.kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="toxic", choices=["toxic", "vacuum"])
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--seed", type=int, default=COUNTERFACTUAL.seeds[0],
                    help="checkpoint seed being shown")
    ap.add_argument("--ckpt", default=None, help="ours model.pt (globs allowed)")
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--rank", type=int, default=1,
                    help="representative window = cohort rank (1 = alert-card window)")
    ap.add_argument("--n-windows", type=int, default=400,
                    help="cohort size for the selection rule (same as Axis A)")
    ap.add_argument("--n-verify", type=int, default=128,
                    help="cohort windows used for the identity check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="figure_revision_output/factorization")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic data + untrained model (watermarked)")
    ap.add_argument("--allow-unverified-data", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    classes = TASK_CLASSES[args.task]
    if not 0 <= args.target < len(classes):
        raise SystemExit(f"target={args.target} outside classes {classes}")
    target_name = classes[args.target]

    # ---- model + data (mirrors plot_alert_card.py empirical path) -----------
    checkpoint_sha = None
    if args.demo:
        print("[factorization][demo] synthetic windows + untrained model")
        torch.manual_seed(args.seed)
        test_X, test_y = make_demo(args.task, DATA.seq_len, n=400)
        bidx = make_branching_scheme("type4", 40)
        pool_kind = "multi_scale" if args.task == "toxic" else "positional"
        model = LOBBranchModel(n_features=40, n_classes=len(classes),
                               backbone="tcn", pool=pool_kind,
                               cnn_channels=(32, 32, 64), cnn_kernel=3,
                               branch_indices=bidx,
                               seq_len=DATA.seq_len).to(device).eval()
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required (or use --demo explicitly)")
        hits = sorted(_glob.glob(args.ckpt))
        if not hits:
            raise SystemExit(f"no checkpoint matches {args.ckpt!r}")
        if len(hits) > 1:
            print(f"[factorization][warn] {len(hits)} ckpt matches; using {hits[-1]}")
        args.ckpt = hits[-1]
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_cfg = {**dict(ck.get("config", {}) or {}),
                  **dict(ck.get("meta", {}) or {})}
        identities = ck_cfg.get("data_files") or {}
        if not identities and not args.allow_unverified_data:
            raise SystemExit("checkpoint has no data SHA-256 provenance; "
                             "use --allow-unverified-data for diagnostics only")
        if identities:
            verify_file_identity(args.test_csv, identities.get("test", {}))
        checkpoint_sha = sha256_file(args.ckpt)
        seq_len = int(ck_cfg.get("seq_len", DATA.seq_len))
        k = int(ck_cfg.get("k", DATA.toxic_horizon if args.task == "toxic"
                           else DATA.vacuum_horizon))
        tick = float(ck_cfg.get("tick_size", DATA.tick_size_z))
        model, _ = build_ours(args.ckpt, device, args.task)
        test_X, test_y = load_windows(
            args.test_csv, args.task, seq_len, k, tick,
            boundaries=list(ck_cfg.get("test_boundaries", []) or []))
    model.eval()
    branch_names = list(getattr(
        model, "branch_names",
        ["ask_price", "ask_volume", "bid_price", "bid_volume"]))
    T = int(np.asarray(test_X[0]).shape[0])

    # ---- alpha_t from checkpoint parameters --------------------------------
    alphas, mixtures, pool_kind = readout_alphas(model, T)
    for b in branch_names:
        s = float(alphas[b].sum())
        if abs(s - 1.0) > 1e-4:
            raise SystemExit(f"alpha_t for {b} sums to {s}, expected 1 — "
                             "readout is not a convex combination?")
    boundary_pairs = []                       # [(x, tail_window), ...]
    if mixtures:                              # multi_scale staircase boundaries
        ws = mixtures[branch_names[0]]["windows"]
        boundary_pairs = [(T - w + 1, w)
                          for w in sorted(ws, reverse=True) if w < T]
    boundaries = [p[0] for p in boundary_pairs]

    # ---- cohort selection (same rule as Axis A / alert card) ---------------
    with torch.no_grad():
        probs = []
        for s0 in range(0, len(test_X), 256):
            xb = torch.from_numpy(np.asarray(test_X[s0:s0 + 256])).float().to(device)
            probs.append(torch.softmax(model(xb), -1).cpu().numpy())
        probs = np.concatenate(probs)
    sel = select_top_confidence_alerts(probs, test_y, args.target,
                                       args.n_windows,
                                       require_correct=not args.demo)
    if len(sel) == 0:
        raise SystemExit("no eligible target alerts")
    n_verify = min(args.n_verify, len(sel))
    if not 1 <= args.rank <= n_verify:
        raise SystemExit(f"--rank {args.rank} outside verify cohort {n_verify}")
    verify_idx = np.asarray(sel[:n_verify], dtype=int)
    rep_pos = args.rank - 1
    rep_window = int(verify_idx[rep_pos])
    print(f"[factorization] cohort {len(sel)}; identity check on first "
          f"{n_verify}; representative window index={rep_window} "
          f"(rank {args.rank})")

    # ---- both CAMs through the frozen code path ----------------------------
    Xv = torch.from_numpy(np.asarray(test_X)[verify_idx]).float()
    cams_h = compute_cam_all_branches(model, Xv, device,
                                      target_class=int(args.target),
                                      method="hirescam")
    cams_g = compute_cam_all_branches(model, Xv, device,
                                      target_class=int(args.target),
                                      method="gradcam")

    # ---- identity check: HiResCAM(t) = T * alpha_t * GradCAM(t) ------------
    rows, all_h, all_r = [], [], []
    per_branch_max = {}
    for b in branch_names:
        H = np.asarray(cams_h[b][0], dtype=np.float64)          # signed (N, T)
        G = np.asarray(cams_g[b][0], dtype=np.float64)
        R = T * alphas[b][None, :] * G                           # reconstruction
        scale = np.abs(H).max(axis=1) + 1e-12                    # per window
        rel = np.abs(R - H).max(axis=1) / scale                  # (N,)
        per_branch_max[b] = float(rel.max())
        all_h.append(H.ravel()); all_r.append(R.ravel())
        for j in range(H.shape[0]):
            rows.append((int(verify_idx[j]), b,
                         float(np.abs(R[j] - H[j]).max()), float(rel[j])))
    all_h = np.concatenate(all_h); all_r = np.concatenate(all_r)
    ss_res = float(((all_h - all_r) ** 2).sum())
    ss_tot = float(((all_h - all_h.mean()) ** 2).sum()) + 1e-300
    r2 = 1.0 - ss_res / ss_tot
    max_rel = max(per_branch_max.values())
    med_rel = float(np.median([r[3] for r in rows]))
    verdict = "PASS" if max_rel < REL_TOL else "FAIL"
    print(f"[factorization] identity {verdict}: max rel err {max_rel:.2e} "
          f"(median {med_rel:.2e}), R^2 = {r2:.8f}")
    if verdict == "FAIL":
        print("[factorization][FAIL] per-branch max rel err:")
        for b in branch_names:
            print(f"    {b}: {per_branch_max[b]:.3e}")
        print("  diagnose before use: a nonlinearity/normalization between the "
              "branch output (CAM hook) and the readout breaks linearity.")

    # ---- representative-window slices for panels (b), (c) ------------------
    rep_G = {b: np.asarray(cams_g[b][0])[rep_pos] for b in branch_names}
    rep_H = {b: np.asarray(cams_h[b][0])[rep_pos] for b in branch_names}
    rep_R = {b: T * alphas[b] * rep_G[b] for b in branch_names}

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10.5, "axes.edgecolor": "#cccccc",
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(12.6, 9.8), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 1.0, 1.05], hspace=0.38,
                          left=0.085, right=0.845, top=0.895, bottom=0.075)
    t = np.arange(1, T + 1)

    def draw_boundaries(ax, y_frac=0.90):
        for i, xb in enumerate(boundaries):
            ax.axvline(xb, color="#999999", lw=1, ls=(0, (2, 3)))
        return None

    # (a) READOUT
    ax1 = fig.add_subplot(gs[0])
    amax = max(float(alphas[b].max()) for b in branch_names)
    amin = min(float(alphas[b][alphas[b] > 0].min()) for b in branch_names)
    use_log = amax / max(amin, 1e-12) > 30
    for b in branch_names:
        if mixtures:
            m = mixtures[b]["mixture"]
            lbl = (f"{BRANCH_LABELS[b]}  m=("
                   + ",".join(f"{v:.2f}" for v in m) + ")")
        else:
            lbl = BRANCH_LABELS[b]
        ax1.plot(t, alphas[b], color=BRANCH_COLORS[b], lw=2, label=lbl)
    if use_log:
        ax1.set_yscale("log")
    draw_boundaries(ax1)
    for xb, w in boundary_pairs:
        ax1.text(xb - 0.8, amax * (0.55 if not use_log else 0.45),
                 f"last-{w}", color=MUTED, fontsize=8.5, ha="right",
                 rotation=90, va="top")
    ax1.set_ylabel(r"$\alpha_t$" + ("  (log)" if use_log else ""))
    ax1.set_title(r"(a) Positional weights $\alpha_t$", fontsize=11.5, loc="left", pad=6)
    ax1.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
    style_axis(ax1)

    # (b) RAW EVIDENCE
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for b in branch_names:
        ax2.plot(t, rep_G[b], color=BRANCH_COLORS[b], lw=2)
    ax2.axhline(0, color="#d9d9d9", lw=1)
    draw_boundaries(ax2)
    g_lo = min(float(rep_G[b].min()) for b in branch_names)
    g_hi = max(float(rep_G[b].max()) for b in branch_names)
    ends = dodge([(b, float(rep_G[b][-1])) for b in branch_names],
                 0.10 * (g_hi - g_lo + 1e-12))
    for b in branch_names:
        ax2.annotate(BRANCH_LABELS[b], xy=(T + 0.8, ends[b]), xytext=(2, 0),
                     textcoords="offset points", color=BRANCH_COLORS[b],
                     fontweight="bold", fontsize=10, va="center",
                     annotation_clip=False)
    ax2.set_ylabel(r"Grad-CAM$_b$(t)  (signed)")
    ax2.set_title("(b) Signed Grad-CAM", fontsize=11.5, loc="left", pad=6)
    style_axis(ax2)

    # (c) VERIFICATION
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for b in branch_names:
        ax3.plot(t, rep_H[b], color=BRANCH_COLORS[b], lw=2.6, alpha=0.9)
    for b in branch_names:
        ax3.plot(t, rep_R[b], color=INK, lw=1.1, ls=(0, (4, 3)))
    ax3.axhline(0, color="#d9d9d9", lw=1)
    draw_boundaries(ax3)
    h_lo = min(float(rep_H[b].min()) for b in branch_names)
    h_hi = max(float(rep_H[b].max()) for b in branch_names)
    ends = dodge([(b, float(rep_H[b][-1])) for b in branch_names],
                 0.10 * (h_hi - h_lo + 1e-12))
    for b in branch_names:
        ax3.annotate(BRANCH_LABELS[b], xy=(T + 0.8, ends[b]), xytext=(2, 0),
                     textcoords="offset points", color=BRANCH_COLORS[b],
                     fontweight="bold", fontsize=10, va="center",
                     annotation_clip=False)
    ax3.plot([], [], color=INK, lw=1.1, ls=(0, (4, 3)),
             label=r"reconstruction $T\,\alpha_t\,$Grad-CAM$_b$(t)")
    ax3.legend(loc="upper left", frameon=False, fontsize=9)
    box_col = "#2A7D46" if verdict == "PASS" else "#B3261E"
    ax3.text(0.012, 0.72,
             f"identity check, cohort N={n_verify} × {len(branch_names)} "
             f"branches × T={T}:\n"
             f"max rel. err = {max_rel:.1e}   median = {med_rel:.1e}   "
             f"$R^2$ = {r2:.6f}   →  {verdict}",
             transform=ax3.transAxes, fontsize=9.5, color=box_col,
             va="top", bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                 ec=box_col, lw=1.2))
    ax3.set_ylabel(r"HiResCAM$_b$(t)  (signed)")
    ax3.set_xlabel(f"window tick  (1 → {T} = alert time)")
    ax3.set_title("(c) HiResCAM reconstruction", fontsize=11.5, loc="left", pad=6)
    style_axis(ax3)

    for ax in (ax1, ax2, ax3):
        ax.axvline(T, color="#bbbbbb", lw=1, ls=":")
        ax.set_xlim(1, T * 1.005)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    demo_tag = ("\n[DEMO — synthetic data, untrained model]"
                if args.demo else "")
    fig.suptitle("Readout–evidence factorization — "
                 f"{target_name.replace('_', ' ')} · Seed {args.seed} · Window {rep_window}"
                 f"{demo_tag}",
                 fontsize=13, fontweight="bold", y=0.975)
    if verdict == "FAIL":
        fig.text(0.5, 0.5, "IDENTITY FAIL — diagnose before use",
                 fontsize=30, color="#B3261E", alpha=0.35,
                 ha="center", va="center", rotation=20, fontweight="bold")

    base = os.path.join(args.out_dir,
                        f"factorization_{args.task}_seed{args.seed}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    print(f"[saved] {base}.png / .pdf")

    # ---- CSV ledgers --------------------------------------------------------
    with open(base + "_alpha.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["t"] + [f"alpha_{b}" for b in branch_names])
        for i in range(T):
            wcsv.writerow([i + 1] + [f"{alphas[b][i]:.8e}" for b in branch_names])
    with open(base + "_errors.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["window_index", "branch", "max_abs_err", "rel_err"])
        for r in rows:
            wcsv.writerow([r[0], r[1], f"{r[2]:.6e}", f"{r[3]:.6e}"])
    print(f"[saved] {base}_alpha.csv / _errors.csv")

    prov = {
        "task": args.task, "target": args.target, "target_name": target_name,
        "seed": args.seed, "demo": bool(args.demo),
        "checkpoint": args.ckpt, "checkpoint_sha256": checkpoint_sha,
        "pool_kind": pool_kind,
        "mixtures": mixtures,
        "staircase_boundaries_1based": boundaries,
        "law": "HiResCAM(t) = T * alpha_t * GradCAM(t)",
        "cohort_rule": "top-confidence correct alerts (same as Axis A)",
        "cohort_size": int(len(sel)), "n_verify": int(n_verify),
        "representative_rank": args.rank,
        "representative_window_index": rep_window,
        "max_rel_err": max_rel, "median_rel_err": med_rel,
        "per_branch_max_rel_err": {b: per_branch_max[b] for b in branch_names},
        "r2": r2, "rel_tol": REL_TOL, "verdict": verdict,
    }
    with open(base + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[saved] {base}.provenance.json")


if __name__ == "__main__":
    main()