"""Render correct/incorrect alert examples and cross-seed representative alerts.

Inputs: three task-matched checkpoints and FI-2010 test CSV.
Rank correct target alerts within each seed by confidence, preserving each
model's predictions and cohort identity. Save representative window indices
and caption statistics alongside the PNG/PDF figures.
Run with python -m lob_xai_viz.gallery_qualitative --help.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import sys

from lob_xai.counterfactual_test import (
    TASK_CLASSES, TASK_FLAT, build_ours, load_windows, best_span, _interp_to_T,
)
from lob_xai.gradcam import TimeSeriesGradCAM
from lob_xai.branching import make_branching_scheme
from lob_xai.evaluation_protocol import select_top_confidence_alerts
from lob_xai.provenance import verify_file_identity
from lob_xai.paper_protocol import DATA

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})

DISPLAY = {  # 논문 표기
    "toxic_up": ("UP", "#2ca02c"), "stationary": ("FLAT", "#7f7f7f"),
    "toxic_down": ("DOWN", "#d62728"), "vacuum": ("VACUUM", "#7b3fb3"),
    "normal": ("NORMAL", "#7f7f7f"),
}
SHORT = {"toxic_up": "up", "stationary": "flat", "toxic_down": "down",
         "vacuum": "vacuum", "normal": "normal"}


# ────────────────────────────────────────────────────────────────────
def predict_probs(model, X, device, batch=512):
    ps = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.as_tensor(X[s:s + batch], dtype=torch.float32, device=device)
            ps.append(F.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(ps)


def branch_cams(model, device, X, target_class, method="hirescam", batch=64):
    """(n_branch, N, T) 절댓값 CAM. counterfactual_test.OursCAMProvider와 동일 공식."""
    nb = len(model.cnns)
    N, T = X.shape[0], X.shape[1]
    out = np.zeros((nb, N, T), dtype=np.float64)
    for s in range(0, N, batch):
        xb = torch.as_tensor(np.asarray(X[s:s + batch]), dtype=torch.float32)
        for bi in range(nb):
            eng = TimeSeriesGradCAM(model, device, method=method)
            try:
                _, cam_abs = eng.generate_batch(xb, branch_idx=bi,
                                                target_class=int(target_class))
                if cam_abs.shape[1] != T:
                    cam_abs = _interp_to_T(cam_abs, T)
                out[bi, s:s + batch] = cam_abs
            finally:
                eng.cleanup()
    return out


def branch_names_from_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {}) or {}
    cfg = ck.get("config", {}) or {}
    bidx = meta.get("branch_indices") or make_branching_scheme(
        cfg.get("branching_scheme", "type4"), n_features=40)
    return list(bidx.keys())


def mid_price(win):
    """정규화 공간 중간가격: (ask_price L1 + bid_price L1)/2 = (col0 + col2)/2."""
    return 0.5 * (win[:, 0] + win[:, 2])


# ────────────────────────────────────────────────────────────────────
def draw_case(ax_price, ax_cam, win, sal, color, title, rho=0.25, title_fs=9.5):
    T = len(sal)
    mp = mid_price(win)
    srng = sal.max() - sal.min()
    snorm = (sal - sal.min()) / (srng if srng > 0 else 1.0)  # 창 내 상대 귀속
    ylo, yhi = mp.min(), mp.max()
    pad = 0.08 * (yhi - ylo + 1e-9)
    ylo, yhi = ylo - pad, yhi + pad
    # |CAM| 히트맵 배경 (min-max 정규화 — 구 그림과 동일한 창 내 상대 표현)
    ax_price.imshow(snorm[None, :], aspect="auto", cmap="YlOrRd",
                    extent=[1, T, ylo, yhi], alpha=0.45, zorder=1,
                    vmin=0.0, vmax=1.0)
    # top-ρ 연속 CAM 구간 (= deletion 표적) 음영
    L = int(math.ceil(rho * T))
    s0, e0 = best_span(sal, L, "cam")
    ax_price.axvspan(s0 + 1, e0, color="#d62728", alpha=0.16, zorder=2)
    # 중간가격
    ax_price.plot(np.arange(1, T + 1), mp, color=color, lw=1.8, zorder=3)
    # 봉우리
    tp = int(sal.argmax())
    ax_price.axvline(tp + 1, ls="--", color="k", lw=0.9, zorder=4)
    ax_price.annotate(f"peak: {tp + 1}", xy=(tp + 1, ylo + 0.86 * (yhi - ylo)),
                      xytext=(min(tp + 4, T - 25), ylo + 0.86 * (yhi - ylo)),
                      fontsize=8, zorder=5,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                ec="0.4", alpha=0.9))
    ax_price.set_xlim(1, T)
    ax_price.set_ylim(ylo, yhi)
    ax_price.set_ylabel("mid-price\n(normalized)", fontsize=8)
    ax_price.tick_params(labelsize=8, labelbottom=False)
    ax_price.set_title(title, fontsize=title_fs, fontweight="bold")
    try:
        ax_price.yaxis.get_major_formatter().set_useOffset(False)
    except Exception:
        pass
    # 하단 |CAM|
    ax_cam.fill_between(np.arange(1, T + 1), sal, color="#ff7f0e", alpha=0.75, lw=0)
    ax_cam.set_xlim(1, T)
    ax_cam.set_ylabel("|CAM|", fontsize=8)
    ax_cam.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax_price.spines[sp].set_visible(False)
        ax_cam.spines[sp].set_visible(False)


# ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="toxic", choices=["toxic", "vacuum"])
    ap.add_argument("--ckpt-s1", required=True)
    ap.add_argument("--ckpt-s2", required=True)
    ap.add_argument("--ckpt-s3", required=True)
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="figure_revision_output/qual_gallery")
    ap.add_argument("--model-name", default="Prism-LOB")
    ap.add_argument("--n-cohort", type=int, default=400)
    ap.add_argument("--rho", type=float, default=0.25)
    ap.add_argument("--method", default="hirescam")
    ap.add_argument("--allow-unverified-data", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = TASK_CLASSES[args.task]
    flat = TASK_FLAT[args.task]
    risk_classes = [i for i in range(len(classes)) if i != flat]
    ckpts = {1: args.ckpt_s1, 2: args.ckpt_s2, 3: args.ckpt_s3}

    # ── 데이터: seed1 체크포인트의 config로 창 규약을 고정 (CF와 동일 절차)
    ck = torch.load(ckpts[1], map_location="cpu", weights_only=False)
    ck_cfg = {**dict(ck.get("config", {}) or {}), **dict(ck.get("meta", {}) or {})}
    identities = ck_cfg.get("data_files") or {}
    if identities:
        verify_file_identity(args.test_csv, identities.get("test", {}))
        print("[ok] test CSV SHA-256 provenance 일치")
    elif not args.allow_unverified_data:
        raise SystemExit("[FAIL] 체크포인트에 data SHA 근거 없음 — "
                         "--allow-unverified-data는 진단용으로만 사용")
    seq_len = int(ck_cfg.get("seq_len", DATA.seq_len))
    k = int(ck_cfg.get("k", DATA.toxic_horizon if args.task == "toxic"
                       else DATA.vacuum_horizon))
    tick = float(ck_cfg.get("tick_size", DATA.tick_size_z))
    test_boundaries = list(ck_cfg.get("test_boundaries", []) or [])
    bnames = branch_names_from_ckpt(ckpts[1])
    iv_ask = bnames.index("ask_volume") if "ask_volume" in bnames else 1
    print(f"[cfg] task={args.task} k={k} seq_len={seq_len} tick={tick} "
          f"branches={bnames}")

    print("[load] 전체 테스트 윈도우 적재 중 ...")
    test_X, test_y = load_windows(args.test_csv, args.task, seq_len, k, tick,
                                  boundaries=test_boundaries)
    T = test_X.shape[1]
    print(f"[load] windows={len(test_X)}  T={T}")

    probs = {}
    models = {}
    for s in (1, 2, 3):
        m, _ = build_ours(ckpts[s], device, args.task)
        models[s] = m
        probs[s] = predict_probs(m, test_X, device)
        acc = float((probs[s].argmax(1) == test_y).mean())
        print(f"[seed{s}] probs 계산 완료 (전체 정확도 {acc:.4f})")

    stats = {"task": args.task, "k": k, "rho": args.rho,
             "model_name": args.model_name, "branches": bnames,
             "fig00": {}, "per_seed": {}}

    # ════════════════════════════════════════════════════════════════
    # Figure A — fig00 재생성: 전체 풀, 예측 클래스별 최고신뢰 ✓/✗ (seed 1)
    # ════════════════════════════════════════════════════════════════
    s = 1
    p1 = probs[s]
    pred1 = p1.argmax(1)
    if args.task == "toxic":
        row_order = [classes.index("toxic_up"), flat, classes.index("toxic_down")]
    else:
        row_order = [classes.index("vacuum"), flat]
    nrow = len(row_order)
    fig = plt.figure(figsize=(11, 3.2 * nrow))
    gs = fig.add_gridspec(nrow, 2, hspace=0.40, wspace=0.28, top=0.91, bottom=0.06)
    for r, c in enumerate(row_order):
        disp, color = DISPLAY[classes[c]]
        for j, ok in enumerate((True, False)):
            mask = (pred1 == c) & ((test_y == c) if ok else (test_y != c))
            inner = gs[r, j].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            axp = fig.add_subplot(inner[0])
            axc = fig.add_subplot(inner[1], sharex=axp)
            if not mask.any():
                axp.text(0.5, 0.5, f"no {'correct' if ok else 'wrong'} "
                         f"prediction for {disp}", ha="center", va="center")
                axp.set_axis_off(); axc.set_axis_off()
                continue
            idxs = np.flatnonzero(mask)
            w = int(idxs[np.argsort(-p1[idxs, c], kind="stable")][0])
            conf = float(p1[w, c])
            cams = branch_cams(models[s], device, test_X[w:w + 1], c,
                               method=args.method)
            sal = cams.sum(0)[0]
            tl, _ = DISPLAY[classes[int(test_y[w])]]
            mark = "✓" if ok else "✗"
            title = (f"{disp} | True={SHORT[classes[int(test_y[w])]]}, "
                     f"Pred={SHORT[classes[c]]} {mark} | Conf={conf:.2f}")
            draw_case(axp, axc, test_X[w], sal, color, title, rho=args.rho)
            if r == nrow - 1:
                axc.set_xlabel("Window tick (1–100)", fontsize=9)
            stats["fig00"][f"{classes[c]}_{'correct' if ok else 'wrong'}"] = {
                "window": w, "conf": round(conf, 4),
                "true": classes[int(test_y[w])], "pred": classes[c],
                "peak_t": int(sal.argmax()),
                "askvol_share": round(float(cams[iv_ask, 0].sum() / max(sal.sum(), 1e-12)), 4),
            }
            print(f"[fig00] {disp} {'✓' if ok else '✗'}: window #{w} conf={conf:.3f}")
    fig.suptitle(f"{args.model_name} · Highest-confidence correct and incorrect cases\n"
                 f"{args.task.capitalize()} · Seed 1", fontsize=12, y=0.99)
    f1 = out / "fig00_qual_grounding_positional.png"
    fig.savefig(f1, bbox_inches="tight")
    fig.savefig(f1.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {f1}")

    # ════════════════════════════════════════════════════════════════
    # Figure B — fig11 계승: 경보 코호트 rank-1 정타 경보, 행=클래스 × 열=seed
    #            + 코호트 전형성 통계 (캡션용)
    # ════════════════════════════════════════════════════════════════
    nrow = len(risk_classes)
    fig = plt.figure(figsize=(12.8, 3.4 * nrow))
    gs = fig.add_gridspec(nrow, 3, hspace=0.42, wspace=0.32, top=0.86, bottom=0.08)
    q_last = T - int(math.ceil(T / 4))          # 후반 25% 경계 (T=100 → t>=75)
    for s in (1, 2, 3):
        ps = probs[s]
        stats["per_seed"][s] = {}
        for r, c in enumerate(risk_classes):
            disp, color = DISPLAY[classes[c]]
            cohort = select_top_confidence_alerts(ps, test_y, c, args.n_cohort)
            if len(cohort) == 0:
                print(f"[warn] seed{s} {classes[c]}: 정타 경보 없음")
                continue
            cams = branch_cams(models[s], device, test_X[cohort], c,
                               method=args.method)          # (nb, n, T)
            sal_all = cams.sum(0)                            # (n, T)
            mass = cams.sum(2)                               # (nb, n)
            dom_share = float((mass.argmax(0) == iv_ask).mean())
            peak_share = float((sal_all.argmax(1) >= q_last).mean())
            w = int(cohort[0]); conf = float(ps[w, c])
            av_share = float(mass[iv_ask, 0] / max(mass[:, 0].sum(), 1e-12))
            stats["per_seed"][s][classes[c]] = {
                "n_alerts": int(len(cohort)),
                "rank1_window": w, "rank1_conf": round(conf, 4),
                "rank1_askvol_share": round(av_share, 4),
                "dominant_askvol_share": round(dom_share, 4),
                "peak_last25_share": round(peak_share, 4),
            }
            inner = gs[r, s - 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
            axp = fig.add_subplot(inner[0])
            axc = fig.add_subplot(inner[1], sharex=axp)
            title = (f"{disp} | seed {s} | #{w}\n"
                     f"Conf {conf:.3f} \u00b7 ask-vol {av_share:.2f}")
            draw_case(axp, axc, test_X[w], sal_all[0], color, title,
                      rho=args.rho, title_fs=8.5)
            if r == nrow - 1:
                axc.set_xlabel("Window tick (1–100)", fontsize=9)
            print(f"[fig11] seed{s} {disp}: rank-1 #{w} conf={conf:.3f} "
                  f"n={len(cohort)} dom(ask-vol)={dom_share:.2%} "
                  f"peak-last25={peak_share:.2%}")
    fig.suptitle(f"{args.model_name} · Highest-confidence alerts across seeds\n"
                 f"{args.task.capitalize()} · Rank 1 in each alert cohort", fontsize=12, y=0.99)
    f2 = out / "fig11_alert_typicality_positional.png"
    fig.savefig(f2, bbox_inches="tight")
    fig.savefig(f2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {f2}")

    # ── 일관성 자동 검증: seed1 toxic_up rank-1 == 본문 알림 카드(#20152) 기대
    if args.task == "toxic":
        got = stats["per_seed"][1].get("toxic_up", {}).get("rank1_window")
        if got == 20152:
            print("[check] seed1 toxic_up rank-1 = #20152 — v3 알림 카드와 동일 ✓")
        else:
            print(f"[check][주의] seed1 toxic_up rank-1 = #{got} (알림 카드는 "
                  "#20152). 코호트 규칙/체크포인트가 다르면 값이 달라질 수 있음 — "
                  "확인 필요")

    # ── 캡션용 집계 문장
    def rng_pct(key, cls):
        vals = [stats["per_seed"][s][cls][key] * 100 for s in (1, 2, 3)
                if cls in stats["per_seed"][s]]
        return (min(vals), max(vals)) if vals else (float("nan"),) * 2

    stats["caption"] = {}
    have_all = (args.task == "toxic" and all(
        cls in stats["per_seed"][s] for s in (1, 2, 3)
        for cls in ("toxic_up", "toxic_down")))
    if have_all:
        du = rng_pct("dominant_askvol_share", "toxic_up")
        dd = rng_pct("dominant_askvol_share", "toxic_down")
        pu = rng_pct("peak_last25_share", "toxic_up")
        pd_ = rng_pct("peak_last25_share", "toxic_down")
        nu = [stats["per_seed"][s]["toxic_up"]["n_alerts"] for s in (1, 2, 3)]
        nd = [stats["per_seed"][s]["toxic_down"]["n_alerts"] for s in (1, 2, 3)]
        cap = (f"정타 경보 코호트(시드당 up n={min(nu)}–{max(nu)}, "
               f"down n={min(nd)}–{max(nd)})에서 매도 잔량(ask-volume)이 "
               f"최대 |CAM| 질량 branch인 경보 비율은 up "
               f"{du[0]:.0f}–{du[1]:.0f}%, down {dd[0]:.0f}–{dd[1]:.0f}%"
               f"였고, S(t) 봉우리가 창 후반 25%에 놓인 비율은 up "
               f"{pu[0]:.0f}–{pu[1]:.0f}%, down {pd_[0]:.0f}–{pd_[1]:.0f}%"
               "에 그쳐 증거가 창 전반에 분산되어 있다.")
        stats["caption"]["fig11_ko"] = cap
        print("\n[caption] " + cap)
    elif args.task == "toxic":
        print("[warn] 일부 seed×클래스 코호트가 비어 캡션 집계 생략")

    with open(out / "caption_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out/'caption_stats.json'}")
    print(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()