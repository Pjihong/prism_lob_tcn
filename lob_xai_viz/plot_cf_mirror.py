#!/usr/bin/env python3
"""Render deletion/insertion curves from existing per-window CF results.

Rows distinguish replacement (deletion) and restoration (insertion);
columns show vacuum, toxic-up and toxic-down. Lines are equal-weight means
of seed means; CAM bands show seed min/max, not confidence intervals.
Deletion and insertion may use different cohorts. Missing insertion cells
are explicitly marked rather than silently dropped.
Inputs: response-carpet NPZ or deletion/insertion curve templates.
Outputs: PNG/PDF, sample counts and provenance under --out-dir.
Run with python -m lob_xai_viz.plot_cf_mirror --help.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#1a1a1a", "#767676"
MODEL_COLOR = {"ours": "#1a1a1a", "deeplob": "#9a9a9a"}
MODEL_LABEL = {"ours": "Prism-LOB", "deeplob": "DeepLOB"}
MODE_STYLE = {
    "cam": dict(ls="-", lw=2.4),
    "random": dict(ls=(0, (1, 2)), lw=1.8),
    "anti": dict(ls=(0, (4, 2, 1, 2)), lw=1.6),
}
# (task, class, display name) — the paper's three headline cells
CELLS = [("vacuum", "vacuum", "Liquidity vacuum"),
         ("toxic", "toxic_up", "Toxic-up"),
         ("toxic", "toxic_down", "Toxic-down")]


def load_carpet_deletion(root, model, task, cls, seed):
    """-> (q_grid, {mode: mean curve (Q,)}, meta) or None."""
    pat = os.path.join(
        root, "**", f"cf_response_carpet_{model}_{task}_{cls}_*_seed{seed}.npz")
    hits = sorted(glob.glob(pat, recursive=True))
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(f"Ambiguous deletion inputs: {hits}")
    z = np.load(hits[-1], allow_pickle=False)
    meta = json.loads(str(z["metadata_json"]))
    ci = int(meta["class_index"])
    modes = [str(m) for m in z["modes"]]
    probvec = z["probvec"]                     # (n, modes, Q, C)
    curves = {m: probvec[:, mi, :, ci].mean(axis=0)
              for mi, m in enumerate(modes)}
    return np.asarray(z["q_grid"], dtype=float), curves, {
        "file": hits[-1], "n": int(probvec.shape[0]),
        "checkpoint_sha256": meta.get("checkpoint_sha256")}


def load_curves_npz(path, cls):
    """-> (q_grid, {mode: mean curve}, meta) or None. counterfactual .curves.npz."""
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    classes = [str(c) for c in z["classes"]]
    if cls not in classes:
        print(f"[mirror][warn] class {cls!r} not in {path} ({classes})")
        return None
    curves = {m: z[f"{cls}__{m}"].mean(axis=0)
              for m in ("cam", "random", "anti")}
    return np.asarray(z["q_grid"], dtype=float), curves, {
        "file": path, "n": int(z[f"{cls}__cam"].shape[0]),
        "checkpoint_sha256": str(z["checkpoint_sha256"])}


def gather(loader_per_seed, seeds):
    """Aggregate per-seed mean curves -> (q, {mode: (mean, lo, hi)}, metas)."""
    per_seed, metas, q_grid = [], [], None
    for s in seeds:
        got = loader_per_seed(s)
        if got is None:
            raise SystemExit(f"Missing curve for seed {s}; refusing incomplete aggregation")
        q, curves, meta = got
        if q_grid is None:
            q_grid = q
        elif len(q) != len(q_grid) or not np.allclose(q, q_grid):
            raise SystemExit(f"[mirror] q_grid mismatch in {meta['file']}")
        per_seed.append(curves)
        metas.append({"seed": s, **meta})
    if not per_seed:
        return None
    agg = {}
    for m in ("cam", "random", "anti"):
        stack = np.stack([c[m] for c in per_seed])       # (S, Q)
        agg[m] = (stack.mean(0), stack.min(0), stack.max(0))
    return q_grid, agg, metas


def draw_cell(ax, q_grid, agg, direction):
    x = q_grid * 100.0
    ends = {}
    for model in ("deeplob", "ours"):                    # ours drawn on top
        if model not in agg or agg[model] is None:
            continue
        _, curves, _ = agg[model]
        mean, lo, hi = curves["cam"]
        ax.fill_between(x, lo, hi, color=MODEL_COLOR[model], alpha=0.10, lw=0)
        for mode in ("anti", "random", "cam"):
            mean_m = curves[mode][0]
            ax.plot(x, mean_m, color=MODEL_COLOR[model], **MODE_STYLE[mode])
        ends[model] = float(curves["cam"][0][-1])
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0", lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["ours", "deeplob"])
    ap.add_argument("--carpet-root", default=None,
                    help="deletion source: response-carpet output root "
                         "(searched recursively)")
    ap.add_argument("--deletion-template", default=None,
                    help="deletion source: counterfactual .curves.npz with "
                         "{model}/{task}/{seed} placeholders (overrides "
                         "--carpet-root)")
    ap.add_argument("--insertion-template", default=None,
                    help="counterfactual *_insertion.curves.npz with "
                         "{model}/{task}/{seed} placeholders")
    ap.add_argument("--out-dir", default="figure_revision_output/cf_mirror")
    args = ap.parse_args()
    if not args.carpet_root and not args.deletion_template:
        raise SystemExit("need --carpet-root or --deletion-template")
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- gather ------------------------------------------------------------
    data = {}          # (direction, task, cls) -> {model: gather() or None}
    prov = {"sources": []}
    for task, cls, _disp in CELLS:
        for model in args.models:
            if args.deletion_template:
                def del_loader(s, m=model, t=task, c=cls):
                    return load_curves_npz(
                        args.deletion_template.format(model=m, task=t, seed=s), c)
            else:
                def del_loader(s, m=model, t=task, c=cls):
                    return load_carpet_deletion(args.carpet_root, m, t, c, s)
            got = gather(del_loader, args.seeds)
            data.setdefault(("deletion", task, cls), {})[model] = got
            if got:
                prov["sources"] += [{"direction": "deletion", "task": task,
                                     "cls": cls, "model": model, **m}
                                    for m in got[2]]
            if args.insertion_template:
                def ins_loader(s, m=model, t=task, c=cls):
                    return load_curves_npz(
                        args.insertion_template.format(model=m, task=t, seed=s), c)
                got_i = gather(ins_loader, args.seeds)
            else:
                got_i = None
            data.setdefault(("insertion", task, cls), {})[model] = got_i
            if got_i:
                prov["sources"] += [{"direction": "insertion", "task": task,
                                     "cls": cls, "model": model, **m}
                                    for m in got_i[2]]

    # ---- figure ------------------------------------------------------------
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10.5, "axes.edgecolor": "#cccccc",
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, len(CELLS), figsize=(11.8, 7.4),
                             sharex="col", facecolor="white",
                             gridspec_kw=dict(hspace=0.43, wspace=0.22,
                                              left=0.065, right=0.94,
                                              top=0.83, bottom=0.10))
    row_titles = ["(a) Deletion", "(b) Insertion"]
    for col, (task, cls, disp) in enumerate(CELLS):
        for row, direction in enumerate(("deletion", "insertion")):
            ax = axes[row][col]
            cell = data[(direction, task, cls)]
            if all(v is None for v in cell.values()):
                ax.set_xlim(0, 50); ax.set_ylim(0, 1)
                ax.text(0.5, 0.5,
                        "insertion curves not found —\nre-run counterfactual_test"
                        "\n(--direction insertion, v2.4 dump)"
                        if direction == "insertion" else "deletion data not found",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#a33",
                        bbox=dict(fc="#fff4f2", ec="#e0b4ac",
                                  boxstyle="round,pad=0.4"))
                ax.spines[["top", "right"]].set_visible(False)
            else:
                q_ref = next(v[0] for v in cell.values() if v is not None)
                draw_cell(ax, q_ref, cell, direction)
            if row == 0:
                ax.set_title(disp.replace("-", " "), fontsize=11.5, pad=8)
            if col == 0:
                ax.set_ylabel("mean target-class probability")
            if row == 1:
                ax.set_xlabel("span fraction ρ of window (%)")
    fig.text(0.065, 0.875, row_titles[0], fontsize=11, fontweight="bold")
    axes[1][0].text(0, 1.13, row_titles[1],
                    transform=axes[1][0].transAxes, fontsize=11,
                    fontweight="bold", va="bottom")

    handles = [Line2D([], [], color=INK, **MODE_STYLE["cam"],
                      label="CAM-selected span"),
               Line2D([], [], color=INK, **MODE_STYLE["random"],
                      label="random span"),
               Line2D([], [], color=INK, **MODE_STYLE["anti"],
                      label="anti-CAM span"),
               Line2D([], [], color=MODEL_COLOR["ours"], lw=2.4, label="Prism-LOB"),
               Line2D([], [], color=MODEL_COLOR["deeplob"], lw=2.4,
                      label="DeepLOB")]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.06, 0.95))
    fig.suptitle("Counterfactual response", fontsize=14,
                 fontweight="bold", x=0.065, ha="left", y=0.98)
    fig.text(0.065, 0.012,
             "Lines: mean across seeds. CAM shading: min–max of seed means.",
             fontsize=9, color=MUTED)

    base = os.path.join(args.out_dir, "cf_mirror")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    print(f"[saved] {base}.png / .pdf")
    with open(base + ".provenance.json", "w") as f:
        json.dump({"seeds": args.seeds, "models": args.models,
                   "carpet_root": args.carpet_root,
                   "deletion_template": args.deletion_template,
                   "insertion_template": args.insertion_template,
                   **prov}, f, indent=2)
    print(f"[saved] {base}.provenance.json")
    import csv
    with open(base + ".sample_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "task", "class", "model", "seed", "n"])
        for src in prov["sources"]:
            w.writerow([src["direction"], src["task"], src["cls"],
                        MODEL_LABEL[src["model"]], src["seed"], src["n"]])
    with open(base + ".aggregated_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["direction", "task", "class", "model", "mode", "rho", "mean", "seed_min", "seed_max"])
        for (direction, task, cls), cell in data.items():
            for model, got in cell.items():
                if got is None:
                    continue
                q, curves, _ = got
                for mode, (mean, lo, hi) in curves.items():
                    for vals in zip(q, mean, lo, hi):
                        w.writerow([direction, task, cls, model, mode, *vals])


if __name__ == "__main__":
    main()