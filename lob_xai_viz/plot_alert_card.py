#!/usr/bin/env python3
"""Render a representative alert and its temporal replacement audit.

Inputs: FI-2010 train/test CSVs and a checkpoint; --demo uses synthetic data
and an untrained model. Select a correct target alert by confidence/rank.
Show its input, branch |CAM|, and target probabilities after replacing the
CAM-selected or random span with train-derived conditional donors.
Outputs: PNG/PDF, numerical data and provenance under --out-dir.
Run with python -m lob_xai_viz.plot_alert_card --help.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from lob_xai.counterfactual_test import (
    TASK_CLASSES, NeighborConditionalSampler, best_span, build_ours,
    load_windows, make_demo, mc_mean_probs,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="toxic", choices=["toxic", "vacuum"])
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--seed", type=int, default=COUNTERFACTUAL.seeds[0],
                    help="sampler seed AND the checkpoint seed being shown")
    ap.add_argument("--ckpt", default=None,
                    help="ours model.pt (globs allowed)")
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--rank", type=int, default=1,
                    help="cohort rank to display (1 = most confident correct alert)")
    ap.add_argument("--rho", type=float, default=0.25,
                    help="CAM top-span fraction to delete (contiguous, len=ceil(rho*T))")
    ap.add_argument("--mc", type=int, default=COUNTERFACTUAL.mc_samples)
    ap.add_argument("--pool-size", type=int, default=COUNTERFACTUAL.pool_size)
    ap.add_argument("--neighbors-k", type=int,
                    default=COUNTERFACTUAL.neighbor_candidates)
    ap.add_argument("--method", default=COUNTERFACTUAL.cam_method)
    ap.add_argument("--n-windows", type=int, default=400,
                    help="cohort size for the selection rule (same as Axis A)")
    ap.add_argument("--input-branch", default="auto",
                    help="branch shown in the INPUT heatmap "
                         "(auto = largest |CAM| share of this window)")
    ap.add_argument("--readout-steps", type=int, nargs="*", default=None,
                    help="multi_scale tail windows (e.g. 8 32) -> envelope "
                         "boundary lines in the OBSERVATION row")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="figure_revision_output/alert_card")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic data + untrained model (watermarked)")
    ap.add_argument("--allow-unverified-data", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    classes = TASK_CLASSES[args.task]
    if not 0 <= args.target < len(classes):
        raise SystemExit(f"target={args.target} outside classes {classes}")
    if not 0 < args.rho < 1:
        raise SystemExit("require 0 < rho < 1")
    target_name = classes[args.target]

    # ---- model + data (mirrors counterfactual_test.py empirical path) -------
    checkpoint_sha = None
    if args.demo:
        print("[alert-card][demo] synthetic windows + untrained model")
        torch.manual_seed(args.seed)        # deterministic random init
        test_X, test_y = make_demo(args.task, DATA.seq_len, n=400)
        pool_X, _ = make_demo(args.task, DATA.seq_len, n=2000, seed=args.seed + 1)
        bidx = make_branching_scheme("type4", 40)
        model = LOBBranchModel(n_features=40, n_classes=len(classes),
                               backbone="tcn", pool="avg",
                               cnn_channels=(32, 32, 64), cnn_kernel=3,
                               branch_indices=bidx).to(device).eval()
        branch_indices = bidx
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required (or use --demo explicitly)")
        hits = sorted(_glob.glob(args.ckpt))
        if not hits:
            raise SystemExit(f"no checkpoint matches {args.ckpt!r}")
        if len(hits) > 1:
            print(f"[alert-card][warn] {len(hits)} ckpt matches; using {hits[-1]}")
        args.ckpt = hits[-1]
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_cfg = {**dict(ck.get("config", {}) or {}),
                  **dict(ck.get("meta", {}) or {})}
        identities = ck_cfg.get("data_files") or {}
        if not identities and not args.allow_unverified_data:
            raise SystemExit("checkpoint has no data SHA-256 provenance; "
                             "use --allow-unverified-data for diagnostics only")
        if identities:
            verify_file_identity(args.train_csv, identities.get("development", {}))
            verify_file_identity(args.test_csv, identities.get("test", {}))
        checkpoint_sha = sha256_file(args.ckpt)
        seq_len = int(ck_cfg.get("seq_len", DATA.seq_len))
        k = int(ck_cfg.get("k", DATA.toxic_horizon if args.task == "toxic"
                           else DATA.vacuum_horizon))
        tick = float(ck_cfg.get("tick_size", DATA.tick_size_z))
        val_ratio = float(ck_cfg.get(
            "val_ratio", DATA.validation_fraction_within_development_file))
        model, _ = build_ours(args.ckpt, device, args.task)
        test_X, test_y = load_windows(
            args.test_csv, args.task, seq_len, k, tick,
            boundaries=list(ck_cfg.get("test_boundaries", []) or []))
        pool_X, _ = load_windows(
            args.train_csv, args.task, seq_len, k, tick,
            limit=args.pool_size, seed=args.seed,
            boundaries=list(ck_cfg.get("train_boundaries", []) or []),
            subset="train", val_ratio=val_ratio)
        if len(pool_X) != args.pool_size:
            raise SystemExit(f"donor pool has {len(pool_X)} windows, "
                             f"requested {args.pool_size}")
        branch_indices = (ck_cfg.get("branch_indices")
                          or make_branching_scheme(
                              ck_cfg.get("branching_scheme", "type4"), 40))
    branch_names = list(getattr(
        model, "branch_names",
        ["ask_price", "ask_volume", "bid_price", "bid_volume"]))

    # ---- cohort selection (same rule as Axis A) -----------------------------
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
    if not 1 <= args.rank <= len(sel):
        raise SystemExit(f"--rank {args.rank} outside cohort size {len(sel)}")
    wi = int(sel[args.rank - 1])
    window = np.asarray(test_X[wi], dtype=np.float32)          # (T, 40)
    T = window.shape[0]
    p0 = probs[wi]
    print(f"[alert-card] window index={wi} (cohort rank {args.rank}/{len(sel)}) "
          f"p({target_name})={p0[args.target]:.3f}")

    # ---- observation: per-branch |CAM| + frozen-protocol saliency -----------
    X1 = torch.from_numpy(window[None]).float()
    cams = compute_cam_all_branches(model, X1, device,
                                    target_class=int(args.target),
                                    method=args.method)
    cam_abs = {b: np.asarray(cams[b][1])[0] for b in branch_names}   # (T,)
    sal = np.sum([cam_abs[b] for b in branch_names], axis=0)         # (T,)
    shares = {b: float(np.abs(cam_abs[b]).sum()) for b in branch_names}
    tot = sum(shares.values()) + 1e-12
    shares = {b: v / tot for b, v in shares.items()}

    # ---- intervention: CAM span vs random span (frozen sampler) -------------
    sampler = NeighborConditionalSampler(pool_X, seed=args.seed,
                                         n_neighbors=args.neighbors_k)
    L = max(1, int(np.ceil(args.rho * T)))
    s_cam, e_cam = best_span(sal, L, "cam")
    var_cam = sampler.sample(window, s_cam, e_cam, args.mc)
    p_cam = mc_mean_probs(model, var_cam, device)
    s_rnd, e_rnd = best_span(sal, L, "random", rng=sampler.rng)
    var_rnd = sampler.sample(window, s_rnd, e_rnd, args.mc)
    p_rnd = mc_mean_probs(model, var_rnd, device)
    flip_cam = int(np.argmax(p_cam)) != args.target
    flip_rnd = int(np.argmax(p_rnd)) != args.target
    print(f"[alert-card] CAM span [{s_cam + 1},{e_cam}]  "
          f"p {p0[args.target]:.3f} -> {p_cam[args.target]:.3f} "
          f"({'FLIP' if flip_cam else 'no flip'});  random span "
          f"[{s_rnd + 1},{e_rnd}] -> {p_rnd[args.target]:.3f} "
          f"({'FLIP' if flip_rnd else 'no flip'})")

    # ---- input branch for the heatmap ---------------------------------------
    if args.input_branch == "auto":
        show_b = max(shares, key=shares.get)
    else:
        if args.input_branch not in branch_names:
            raise SystemExit(f"--input-branch must be one of {branch_names}")
        show_b = args.input_branch
    # branch_indices is a dict branch_name -> feature columns (order preserved,
    # cf. make_branching_scheme); checkpoints may store it as a list instead.
    if isinstance(branch_indices, dict):
        feat_idx = np.asarray(branch_indices[show_b], dtype=int)
    else:
        feat_idx = np.asarray(branch_indices[branch_names.index(show_b)],
                              dtype=int)
    levels = window[:, feat_idx].T                              # (n_levels, T)

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10.5, "axes.edgecolor": "#cccccc",
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(12.6, 9.8), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.36,
                          left=0.085, right=0.845, top=0.895, bottom=0.075)
    t = np.arange(1, T + 1)

    # INPUT
    ax1 = fig.add_subplot(gs[0])
    # Colour scale: if the window's values straddle zero, use a diverging map
    # centred at 0 (global z-score reading). If they sit on one side and are
    # nearly flat on the global z scale (typical for PRICE branches, whose
    # within-window moves are ~3 ticks = 0.003 z), a 0-centred map saturates
    # into one colour — switch to a robust within-window sequential scale.
    lo_q, hi_q = np.percentile(levels, [2, 98])
    if lo_q < 0.0 < hi_q:
        vmax = float(np.percentile(np.abs(levels), 99)) or 1.0
        im = ax1.imshow(levels, aspect="auto", cmap="RdBu_r",
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                        extent=[1, T, len(feat_idx) + 0.5, 0.5])
        cbar_label = "z-score"
    else:
        if hi_q - lo_q < 1e-6:
            lo_q, hi_q = lo_q - 1e-3, hi_q + 1e-3
        im = ax1.imshow(levels, aspect="auto", cmap="Oranges",
                        vmin=lo_q, vmax=hi_q,
                        extent=[1, T, len(feat_idx) + 0.5, 0.5])
        cbar_label = "z-score (window-range scale)"
    ax1.set_ylabel(f"{BRANCH_LABELS[show_b].lower()} level")
    ax1.set_yticks([1, len(feat_idx) // 2, len(feat_idx)])
    ax1.set_title(f"(a) Input: {BRANCH_LABELS[show_b].lower()}",
                  fontsize=11.5, loc="left", pad=6)
    cax = ax1.inset_axes([1.015, 0.05, 0.013, 0.9])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)

    # OBSERVATION
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    y_max = max(float(cam_abs[b].max()) for b in branch_names) or 1.0
    for b in branch_names:
        ax2.plot(t, cam_abs[b], color=BRANCH_COLORS[b], lw=2)
    ends = dodge([(b, float(cam_abs[b][-1])) for b in branch_names],
                 0.08 * y_max)
    for b in branch_names:
        ax2.annotate(f"{BRANCH_LABELS[b]}  {shares[b]:.2f}",
                     xy=(T + 0.8, ends[b]), xytext=(2, 0),
                     textcoords="offset points", color=BRANCH_COLORS[b],
                     fontweight="bold", fontsize=10, va="center",
                     annotation_clip=False)
    ax2.axvspan(s_cam + 1, e_cam, color="#D9822B", alpha=0.10, lw=0)
    ax2.text((s_cam + 1 + e_cam) / 2, y_max * 1.02,
             f"CAM top-{args.rho:.0%} span", ha="center", color="#a95c14",
             fontsize=9.5)
    if args.readout_steps:
        for si, s_len in enumerate(sorted(args.readout_steps, reverse=True)):
            xb = T - s_len + 1
            ax2.axvline(xb, color="#999999", lw=1, ls=(0, (2, 3)))
            ax2.text(xb - 0.8, y_max * (0.92 - 0.12 * si), f"last-{s_len}",
                     color=MUTED, fontsize=8.5, ha="right")
    ax2.set_ylim(0, y_max * 1.14)
    ax2.set_ylabel(f"|CAM$_b$(t)|  ({args.method})")
    ax2.set_title(r"(b) Branch attribution $|\mathrm{CAM}|$ (labels: window share)",
                  fontsize=11.5, loc="left", pad=6)
    style_axis(ax2)

    # INTERVENTION
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    orig_series = levels.mean(axis=0)                            # (T,)
    cf_series = var_cam[:, :, feat_idx].mean(axis=2)             # (M, T)
    for m in range(cf_series.shape[0]):
        ax3.plot(t, cf_series[m], color="#8a1f1f", lw=0.7, alpha=0.25)
    ax3.plot(t, cf_series.mean(axis=0), color="#8a1f1f", lw=1.8,
             ls="--", dashes=(4, 3), label=f"deleted span (M={args.mc} donors)")
    ax3.plot(t, orig_series, color=INK, lw=2, label="original")
    ax3.axvspan(s_cam + 1, e_cam, color="#D9822B", alpha=0.10, lw=0)
    lo = min(float(orig_series.min()), float(cf_series.min()))
    hi = max(float(orig_series.max()), float(cf_series.max()))
    pad = 0.08 * (hi - lo + 1e-9)
    ax3.set_ylim(lo - pad, hi + pad)
    ax3.set_ylabel(f"{BRANCH_LABELS[show_b].lower()}\n(level mean, z)")
    ax3.set_xlabel(f"window tick  (1 → {T} = alert time)")
    flip_txt = "→ FLIP" if flip_cam else "(no flip)"
    ax3.set_title("(c) Deletion response", fontsize=11.5, loc="left", pad=6)
    ax3.text(0.01, 0.97,
             f"Target probability: original {p0[args.target]:.3f} | "
             f"CAM {p_cam[args.target]:.3f} | random {p_rnd[args.target]:.3f}",
             transform=ax3.transAxes, va="top", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    ay = dodge([("orig", float(orig_series[-1])),
                ("cf", float(cf_series.mean(axis=0)[-1]))],
               0.14 * (hi - lo + 1e-9))
    ax3.annotate("Original",
                 xy=(T + 0.8, ay["orig"]), xytext=(2, 0),
                 textcoords="offset points", color=INK, fontsize=10,
                 fontweight="bold", va="center", annotation_clip=False)
    ax3.annotate("CAM deletion",
                 xy=(T + 0.8, ay["cf"]), xytext=(2, 0),
                 textcoords="offset points", color="#8a1f1f", fontsize=10,
                 fontweight="bold", va="center", annotation_clip=False)
    ax3.legend(loc="best", frameon=False, fontsize=9)
    style_axis(ax3)

    for ax in (ax1, ax2, ax3):
        ax.axvline(T, color="#bbbbbb", lw=1, ls=":")
        ax.set_xlim(1, T * 1.005)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    demo_tag = "  [DEMO — synthetic data, untrained model]" if args.demo else ""
    fig.suptitle(f"{target_name.replace('_', ' ').capitalize()} · Seed {args.seed} · Window {wi}"
                 f" · Rank {args.rank}{demo_tag}",
                 fontsize=14, fontweight="bold", y=0.965)

    base = os.path.join(
        args.out_dir,
        f"alert_card_{args.task}_{target_name}_seed{args.seed}_rank{args.rank}")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    print(f"[saved] {base}.png / .pdf")

    prov = {
        "task": args.task, "target": args.target, "target_name": target_name,
        "seed": args.seed, "rank": args.rank, "window_index": wi,
        "cohort_size": int(len(sel)), "demo": bool(args.demo),
        "checkpoint": args.ckpt, "checkpoint_sha256": checkpoint_sha,
        "method": args.method, "rho": args.rho,
        "mc": args.mc, "pool_size": int(len(pool_X)),
        "neighbors_k": args.neighbors_k,
        "cam_span_ticks_1based": [int(s_cam + 1), int(e_cam)],
        "random_span_ticks_1based": [int(s_rnd + 1), int(e_rnd)],
        "input_branch": show_b,
        "cam_shares": {b: round(v, 4) for b, v in shares.items()},
        "p_original": [round(float(x), 4) for x in p0],
        "p_cam_deleted": [round(float(x), 4) for x in p_cam],
        "p_random_deleted": [round(float(x), 4) for x in p_rnd],
        "flip_cam": bool(flip_cam), "flip_random": bool(flip_rnd),
    }
    with open(base + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[saved] {base}.provenance.json")


if __name__ == "__main__":
    main()