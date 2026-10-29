#!/usr/bin/env python3
"""Render branch evidence, replacement effects and ranking agreement.

Inputs: analyze_branch_importance CSVs and checkpoints with a verified
branch config; --from-cache reuses previously computed temporal CAM arrays.
Panels show cohort-mean |CAM|, attribution/replacement shares, and the
per-window Kendall distribution with its predeclared sign test.
Outputs: PNG/PDF and supporting numerical files under --out-dir.
Run with python -m lob_xai_viz.plot_branch_importance --help.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Established branch palette (validated: CVD warnings mitigated by the direct
# labels this figure always draws — never rely on hue alone).
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


# ---------------------------------------------------------------------------
# panel (a) data: cohort-mean |CAM_b(t)| per seed (checkpoint -> npz cache)
# ---------------------------------------------------------------------------
def compute_cam_time(args, seed: int, cache_path: str) -> dict:
    """Compute cohort-mean |CAM_b(t)| for one seed and cache it as npz."""
    import torch
    from lob_xai.checkpoints import load_micro_windows_in_order, load_ours_model
    from lob_xai.branch_analysis import (BranchAdapter, architecture_signature,
                                   load_config, find_ckpt, validate_run_protocol)
    from lob_xai.evaluation_protocol import select_top_confidence_alerts
    from lob_xai.provenance import sha256_file

    cfg = load_config(args.config)
    if not cfg.get("verified", False):
        raise SystemExit("[branch-imp] branch_config.json is not verified; run the probe first")
    n = cfg["n_branches"]
    bnames = cfg["branch_names"]

    if args.ckpt_template:
        hits = sorted(glob.glob(args.ckpt_template.format(seed=seed)))
        if not hits:
            raise SystemExit(f"[branch-imp] no checkpoint matches "
                             f"{args.ckpt_template.format(seed=seed)!r}")
        if len(hits) > 1:
            print(f"[branch-imp][warn] {len(hits)} matches for seed {seed}; using {hits[-1]}")
        ck = hits[-1]
    else:
        ck = find_ckpt(args.results_root, "ours", args.task, seed, args.k)
        if ck is None:
            raise SystemExit(f"[branch-imp] no ours ckpt for {args.task} seed{seed} k{args.k}")

    model, _, _ = load_ours_model(ck, args.device)
    if cfg.get("architecture_sha256") != architecture_signature(model):
        raise SystemExit("branch_config architecture does not match this checkpoint")

    X, y, class_names, ck_cfg = load_micro_windows_in_order(ck, args.test_csv, task=args.task)
    validate_run_protocol(ck_cfg, task=args.task, k=args.k, seed=seed)

    # Identical cohort rule to axis_a_min.py: correct target alerts, most
    # confident first.
    with torch.no_grad():
        probs = []
        for s in range(0, len(X), 256):
            xb = torch.from_numpy(np.asarray(X[s:s + 256])).float().to(args.device)
            probs.append(torch.softmax(model(xb), -1).cpu().numpy())
        probs = np.concatenate(probs)
    sel = select_top_confidence_alerts(probs, y, args.target, args.n_windows,
                                       require_correct=True)
    if len(sel) == 0:
        raise SystemExit("[branch-imp] no eligible target alerts")
    Xs = X[sel]
    print(f"[branch-imp] seed {seed}: ckpt={ck}  cohort n={len(Xs)}")

    adapter = BranchAdapter(model, cfg, args.device)
    try:
        s0 = adapter.branch_temporal(Xs, args.target)     # idx -> [N, T] signed CAM
    finally:
        adapter.close()

    m = np.stack([np.abs(s0[i]).mean(axis=0) for i in range(n)], axis=0)  # [n, T]
    np.savez(cache_path,
             cam_time=m, branch_names=np.array(bnames), n_cohort=len(Xs),
             checkpoint=ck, checkpoint_sha256=sha256_file(ck),
             task=args.task, k=args.k, seed=seed, target=args.target)
    print(f"[branch-imp] cached -> {cache_path}")
    return {"cam_time": m, "branch_names": list(bnames), "n_cohort": len(Xs),
            "checkpoint": ck, "checkpoint_sha256": sha256_file(ck)}


def load_cam_time(args, seed: int) -> dict:
    cache_path = os.path.join(
        args.out_dir, f"cam_time_{args.task}_k{args.k}_seed{seed}.npz")
    if os.path.exists(cache_path) or args.from_cache:
        if not os.path.exists(cache_path):
            raise SystemExit(f"[branch-imp] --from-cache but {cache_path} is missing")
        z = np.load(cache_path, allow_pickle=False)
        return {"cam_time": z["cam_time"],
                "branch_names": [str(b) for b in z["branch_names"]],
                "n_cohort": int(z["n_cohort"]),
                "checkpoint": str(z["checkpoint"]),
                "checkpoint_sha256": str(z["checkpoint_sha256"])}
    return compute_cam_time(args, seed, cache_path)


# ---------------------------------------------------------------------------
# panels (b)/(c) data: axisA_min CSVs (no model needed)
# ---------------------------------------------------------------------------
def read_axis_a(args, seed: int):
    base = os.path.join(args.axis_a_dir,
                        f"axisA_min_{args.task}_k{args.k}_seed{seed}")
    summary_path, windows_path = base + ".csv", base + ".windows.csv"
    for p in (summary_path, windows_path):
        if not os.path.exists(p):
            raise SystemExit(f"[branch-imp] missing Axis A artifact: {p}")

    cam_share, dropone, meta = {}, {}, {}
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            b = row["branch"]
            cam_share[b] = float(row["cam_share"])
            dropone[b] = float(row["dropone_mean"])
            meta = {"n_windows": int(row["n_windows"]),
                    "tau_mean": float(row["kendall_tau_mean"]),
                    "tau_p": float(row["kendall_sign_test_p"])}

    neg = {b: v for b, v in dropone.items() if v < 0}
    if neg:
        print(f"[branch-imp][warn] seed {seed}: negative drop-one means {neg}; "
              "clipped to 0 before share normalisation (noted in provenance json)")
    clipped = {b: max(v, 0.0) for b, v in dropone.items()}
    tot = sum(clipped.values())
    drop_share = {b: (v / tot if tot > 0 else np.nan) for b, v in clipped.items()}

    taus, seen = [], set()
    with open(windows_path) as f:
        for row in csv.DictReader(f):
            w = row["window_index"]
            if w in seen:
                continue
            seen.add(w)
            t = float(row["kendall_tau"])
            if np.isfinite(t):
                taus.append(t)
    return {"cam_share": cam_share, "drop_share": drop_share,
            "dropone_raw": dropone, "taus": np.asarray(taus), **meta}


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def dodge(pairs, min_gap):
    """pairs: [(key, y)] -> {key: dodged_y}, order preserved, mean preserved."""
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


def fmt_p(p):
    if not np.isfinite(p):
        return "p = n/a"
    if p == 0 or p < 1e-300:
        return "p < 1e-300"
    e = int(np.floor(np.log10(p)))
    return f"p = {p / 10 ** e:.1f}e{e}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="toxic")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--axis-a-dir", default="figures/axis_a_final")
    ap.add_argument("--out-dir", default="figure_revision_output/branch_importance")
    # panel (a) compute options
    ap.add_argument("--ckpt-template", default=None,
                    help="glob with {seed}, e.g. "
                         "'loggers/results/final_toxic_multiscale_seed{seed}_*/model.pt'")
    ap.add_argument("--results-root", default="loggers/results")
    ap.add_argument("--config", default="branch_config.json")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-windows", type=int, default=400)
    ap.add_argument("--from-cache", action="store_true",
                    help="require the npz cache; never touch checkpoints")
    ap.add_argument("--smooth", type=int, default=0,
                    help="centered moving-average half-width for panel (a) "
                         "display only (0 = raw)")
    ap.add_argument("--readout-steps", type=int, nargs="*", default=None,
                    help="tail-window lengths of a multi_scale readout (e.g. "
                         "8 32); draws the alpha_t envelope boundaries at "
                         "t = T-s+1 in panel (a) so the staircase reads as "
                         "the learned recency envelope, not an artifact. "
                         "Omit for positional/GAP readouts.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target_name = {("toxic", 1): "toxic-up", ("toxic", 2): "toxic-down",
                   ("vacuum", 1): "vacuum"}.get((args.task, args.target),
                                                f"{args.task}/class{args.target}")

    # ---- gather data --------------------------------------------------------
    cam_time = {s: load_cam_time(args, s) for s in args.seeds}
    axis_a = {s: read_axis_a(args, s) for s in args.seeds}

    branches = cam_time[args.seeds[0]]["branch_names"]
    for s in args.seeds:
        if cam_time[s]["branch_names"] != branches:
            raise SystemExit("branch name order differs across seeds")
        if set(axis_a[s]["cam_share"]) != set(branches):
            raise SystemExit(f"axis A branches {set(axis_a[s]['cam_share'])} "
                             f"!= checkpoint branches {set(branches)}")

    # per-seed scale normalisation for (a): divide by mean total magnitude so
    # seeds are comparable; the curve SHAPE (which branch, when) is untouched.
    curves = {}
    for s in args.seeds:
        m = cam_time[s]["cam_time"].astype(float)          # [n, T]
        scale = m.sum(axis=0).mean()
        curves[s] = m / (scale + 1e-12)
    T = curves[args.seeds[0]].shape[1]
    t = np.arange(1, T + 1)

    if args.smooth > 0:
        w = 2 * args.smooth + 1
        kern = np.ones(w) / w
        for s in args.seeds:
            curves[s] = np.stack(
                [np.convolve(c, kern, mode="same") for c in curves[s]], axis=0)

    # ---- figure -------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cccccc",
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(13.6, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42,
                          wspace=0.30, left=0.07, right=0.965,
                          top=0.90, bottom=0.075)

    # ---------------- (a) branch-by-time evidence map ------------------------
    ax = fig.add_subplot(gs[0, :])
    stack = {b: np.stack([curves[s][i] for s in args.seeds])
             for i, b in enumerate(branches)}                    # b -> [S, T]
    y_max = max(float(stack[b].max()) for b in branches)
    for b in branches:
        ax.fill_between(t, stack[b].min(0), stack[b].max(0),
                        color=BRANCH_COLORS[b], alpha=0.14, lw=0)
        ax.plot(t, stack[b].mean(0), color=BRANCH_COLORS[b], lw=2)
    ends = dodge([(b, float(stack[b].mean(0)[-1])) for b in branches],
                 0.055 * y_max)
    for b in branches:
        ax.annotate(BRANCH_LABELS.get(b, b), xy=(T + 0.8, ends[b]),
                    xytext=(2, 0), textcoords="offset points",
                    color=BRANCH_COLORS[b], fontweight="bold", fontsize=11.5,
                    va="center", annotation_clip=False)
    ax.axvline(T, color="#bbbbbb", lw=1, ls=":")
    ax.text(T - 0.8, y_max * 1.02, "alert emitted", color=MUTED,
            fontsize=9.5, ha="right")
    if args.readout_steps:
        # The staircase is the learned recency envelope (alpha_t law):
        # HiResCAM(t) = T * alpha_t * Grad-CAM(t); a multi_scale readout makes
        # alpha_t a step function with jumps at t = T - s + 1.
        for si, s_len in enumerate(sorted(args.readout_steps, reverse=True)):
            xb = T - s_len + 1
            ax.axvline(xb, color="#999999", lw=1, ls=(0, (2, 3)))
            ax.text(xb - 0.8, y_max * (0.95 - 0.09 * si),
                    f"last-{s_len} readout window", color=MUTED, fontsize=9,
                    ha="right")
        ax.text(0.015, 0.955,
                "step envelope = learned readout weights "
                "($\\alpha_t$ law: HiResCAM$(t) = T\\,\\alpha_t\\,$Grad-CAM$(t)$)",
                transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")
    ax.set_xlim(1, T * 1.08)
    ax.set_ylim(0, y_max * 1.12)
    ax.set_xlabel(f"window tick  (1 → {T} = alert time)")
    ax.set_ylabel("cohort-mean |CAM$_b$(t)|  (per-seed scale normalised)")
    n_str = "/".join(str(cam_time[s]["n_cohort"]) for s in args.seeds)
    ax.set_title(f"(a) Which branch, when — branch-by-time evidence map "
                 f"({len(args.seeds)}-seed mean ± range, cohort n={n_str})",
                 fontsize=12.5, pad=8)
    style_axis(ax)

    # ---------------- (b) observation vs intervention slopegraph -------------
    ax = fig.add_subplot(gs[1, 0])
    x0, x1 = 0.0, 1.0
    cam_m = {b: float(np.mean([axis_a[s]["cam_share"][b] for s in args.seeds]))
             for b in branches}
    drp_m = {b: float(np.nanmean([axis_a[s]["drop_share"][b] for s in args.seeds]))
             for b in branches}
    for b in branches:
        ax.plot([x0, x1], [cam_m[b], drp_m[b]], color=BRANCH_COLORS[b],
                lw=2.6, zorder=3)
        ax.scatter([x0, x1], [cam_m[b], drp_m[b]], s=64,
                   color=BRANCH_COLORS[b], zorder=4)
        ax.scatter([x0 - 0.045] * len(args.seeds),
                   [axis_a[s]["cam_share"][b] for s in args.seeds],
                   s=18, color=BRANCH_COLORS[b], alpha=0.45, zorder=2)
        ax.scatter([x1 + 0.045] * len(args.seeds),
                   [axis_a[s]["drop_share"][b] for s in args.seeds],
                   s=18, color=BRANCH_COLORS[b], alpha=0.45, zorder=2)
    ly = dodge([(b, cam_m[b]) for b in branches], 0.024)
    for b in branches:
        ax.text(x0 - 0.10, ly[b], f"{cam_m[b]:.2f}", color=BRANCH_COLORS[b],
                ha="right", va="center", fontsize=10.5)
    ry = dodge([(b, drp_m[b]) for b in branches], 0.030)
    for b in branches:
        ax.text(x1 + 0.10, ry[b], f"{BRANCH_LABELS.get(b, b)}  {drp_m[b]:.2f}",
                color=BRANCH_COLORS[b], ha="left", va="center",
                fontsize=11.5, fontweight="bold")
    top = max(max(cam_m.values()), max(drp_m.values()),
              max(axis_a[s]["drop_share"][b] for s in args.seeds for b in branches))
    ax.set_xlim(-0.42, 1.95)
    ax.set_ylim(0.0, top * 1.15)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Observational\n|CAM| share", "Interventional\ndrop-one share"])
    ax.set_ylabel("branch share of evidence (sums to 1)")
    ax.set_title(f"(b) Observation vs intervention agree on the same axis\n"
                 f"({target_name}, {len(args.seeds)} seeds: small dots = seeds, "
                 f"large = mean)", fontsize=11.5, pad=8)
    style_axis(ax)

    # ---------------- (c) per-window Kendall tau distribution ----------------
    ax = fig.add_subplot(gs[1, 1])
    # With 4 branches, per-window Kendall tau is DISCRETE (±1, ±2/3, ±1/3, 0,
    # plus tie-adjusted values). Draw one horizontal mark per observed level,
    # its width proportional to the window count — an honest discrete
    # distribution instead of a jitter cloud that fakes continuity.
    all_taus = []
    max_count = 1
    level_counts = {}
    for s in args.seeds:
        vals, cnts = np.unique(np.round(axis_a[s]["taus"], 3),
                               return_counts=True)
        level_counts[s] = (vals, cnts)
        max_count = max(max_count, int(cnts.max()))
        all_taus.append(axis_a[s]["taus"])
    for j, s in enumerate(args.seeds, start=1):
        vals, cnts = level_counts[s]
        for v, c in zip(vals, cnts):
            half = 0.42 * (c / max_count) / 2
            ax.plot([j - half, j + half], [v, v], color="#2F6B9A",
                    lw=3.4, alpha=0.55, solid_capstyle="butt", zorder=2)
        ax.scatter([j], [axis_a[s]["taus"].mean()], s=130, marker="D",
                   color="#2F6B9A", edgecolor="white", lw=1.4, zorder=4)
    all_taus = np.concatenate(all_taus)
    mtau = float(all_taus.mean())
    ax.axhline(0, color="#c9c9c9", lw=1)
    ax.axhline(mtau, color="#2F6B9A", lw=1.2, ls="--", alpha=0.8)
    ax.text(0.94, mtau + 0.03, f"mean τ = {mtau:.2f}",
            color="#2F6B9A", fontsize=10.5, ha="right", va="bottom")
    ax.text(0.985, 0.02, "mark width ∝ number of windows at that τ level\n"
            "(τ over 4 branches is discrete by construction)",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
            ha="right", va="bottom")
    worst_p = max(axis_a[s]["tau_p"] for s in args.seeds)
    ax.set_xlim(0.5, len(args.seeds) + 0.5)
    ax.set_ylim(min(-0.6, float(all_taus.min()) - 0.05), 1.12)
    ax.set_xticks(range(1, len(args.seeds) + 1))
    ax.set_xticklabels([f"seed {s}\n(n={axis_a[s]['n_windows']})"
                        for s in args.seeds])
    ax.set_ylabel("per-window Kendall τ  (CAM rank vs ablation rank)")
    ax.set_title(f"(c) Convergence holds window-by-window\n"
                 f"sign-test τ>0: {fmt_p(worst_p)} or smaller in every seed",
                 fontsize=11.5, pad=8)
    style_axis(ax)

    fig.suptitle(f"Intervention-verified decomposition in time — {target_name} "
                 f"(final checkpoints)", fontsize=15, fontweight="bold", y=0.97)

    base = os.path.join(args.out_dir,
                        f"branch_importance_{args.task}_k{args.k}")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    print(f"[saved] {base}.png / .pdf")

    # ---- provenance ---------------------------------------------------------
    prov = {
        "task": args.task, "k": args.k, "target": args.target,
        "seeds": args.seeds,
        "axis_a_dir": args.axis_a_dir,
        "checkpoints": {s: {"path": cam_time[s]["checkpoint"],
                            "sha256": cam_time[s]["checkpoint_sha256"],
                            "cohort_n": cam_time[s]["n_cohort"]}
                        for s in args.seeds},
        "negative_dropone_clipped": {
            s: {b: v for b, v in axis_a[s]["dropone_raw"].items() if v < 0}
            for s in args.seeds},
        "smooth": args.smooth,
        "panel_a_normalisation":
            "per seed: |CAM_b(t)| cohort mean divided by mean over t of the "
            "branch-summed magnitude (shape preserved, scale removed)",
    }
    with open(base + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"[saved] {base}.provenance.json")


if __name__ == "__main__":
    main()