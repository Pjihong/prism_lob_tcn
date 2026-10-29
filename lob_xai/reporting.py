"""Write classification reports and branch-CAM diagnostics.

Inputs: trained model, history, predictions and task-specific class names.
Outputs: history.csv, test_predictions.csv, confusion matrices, learning curves,
CAM diagnostics and summary.json in the run directory. Probability columns use
p_<class_name>; confusion-matrix dimensions follow the task's class count.
Scope: per-window diagnostics do not establish population-level uncertainty.
"""
from __future__ import annotations
from typing import Dict, Optional
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# matplotlib은 optional (서버에 없을 수도 있음)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from .gradcam import TimeSeriesGradCAM
from .training import evaluate


# ───────────────────────────────────────────────────────────────────
# 학습 결과 저장 (history, predictions, confusion)
# ───────────────────────────────────────────────────────────────────


def save_history_csv(history: Dict[str, list], save_dir: str) -> str:
    """history dict → DataFrame → CSV."""
    df = pd.DataFrame(history)
    path = os.path.join(save_dir, "history.csv")
    df.to_csv(path, index=False)
    print(f"[report] saved {path}")
    return path


def save_predictions_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    save_dir: str,
    class_names: tuple = ("up", "flat", "down"),
) -> str:
    """test_predictions.csv: 각 윈도우의 정답, 예측, 클래스별 확률."""
    df = pd.DataFrame({
        "window_idx": np.arange(len(y_true)),
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for i, name in enumerate(class_names):
        df[f"p_{name}"] = probs[:, i]
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    path = os.path.join(save_dir, "test_predictions.csv")
    df.to_csv(path, index=False)
    print(f"[report] saved {path}")
    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    class_names: tuple = ("up", "flat", "down"),
    n_classes: int = 3,
) -> dict:
    """confusion_matrix.csv + .png 저장."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[int(t), int(p)] += 1

    df = pd.DataFrame(cm, index=[f"actual_{n}" for n in class_names],
                       columns=[f"pred_{n}" for n in class_names])
    csv_path = os.path.join(save_dir, "confusion_matrix.csv")
    df.to_csv(csv_path)
    print(f"[report] saved {csv_path}")

    # per-class precision/recall
    per_class = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per_class.append(dict(
            class_name=class_names[c],
            n_true=int(cm[c, :].sum()),
            n_pred=int(cm[:, c].sum()),
            precision=float(prec), recall=float(rec), f1=float(f1),
        ))
    pc_path = os.path.join(save_dir, "per_class_metrics.csv")
    pd.DataFrame(per_class).to_csv(pc_path, index=False)
    print(f"[report] saved {pc_path}")

    if HAS_MPL:
        png_path = os.path.join(save_dir, "confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(n_classes)); ax.set_xticklabels([f"pred {n}" for n in class_names])
        ax.set_yticks(range(n_classes)); ax.set_yticklabels([f"actual {n}" for n in class_names])
        # text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=11)
        ax.set_title("Confusion Matrix (test set)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        print(f"[report] saved {png_path}")

    return dict(matrix=cm.tolist(), per_class=per_class)


def save_learning_curves(history: Dict[str, list], save_dir: str) -> Optional[str]:
    """학습 곡선 PNG (5 subplot: loss / acc / f1 / mcc / ctp)."""
    if not HAS_MPL:
        return None
    epochs = history["epoch"]
    has_mcc = "val_mcc" in history and len(history["val_mcc"]) > 0
    has_ctp = "val_ctp" in history and len(history["val_ctp"]) > 0
    n_panels = 3 + int(has_mcc) + int(has_ctp)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 4))
    if n_panels == 1: axes = [axes]
    # loss
    axes[0].plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="val",   marker="s", markersize=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    # acc
    axes[1].plot(epochs, history["train_acc"], label="train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="val",   marker="s", markersize=3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    # f1
    axes[2].plot(epochs, history["val_f1"], label="val_f1", marker="s",
                  markersize=3, color="C2")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("macro F1")
    axes[2].set_title("Validation F1"); axes[2].legend(); axes[2].grid(alpha=0.3)
    idx = 3
    if has_mcc:
        axes[idx].plot(epochs, history["val_mcc"], label="val_mcc", marker="d",
                        markersize=3, color="#9467bd")
        axes[idx].axhline(0, color="gray", lw=0.8, alpha=0.5)
        axes[idx].set_xlabel("epoch"); axes[idx].set_ylabel("MCC")
        axes[idx].set_title("Validation MCC"); axes[idx].legend(); axes[idx].grid(alpha=0.3)
        idx += 1
    if has_ctp:
        axes[idx].plot(epochs, history["val_ctp"], label="val_ctp", marker="^",
                        markersize=3, color="#e377c2")
        if "val_coverage" in history:
            ax2 = axes[idx].twinx()
            ax2.plot(epochs, history["val_coverage"], label="coverage",
                      marker="x", markersize=3, color="#7f7f7f", alpha=0.6)
            ax2.set_ylabel("coverage", color="#7f7f7f")
            ax2.tick_params(axis="y", labelcolor="#7f7f7f")
            ax2.set_ylim(0, 1.05)
        axes[idx].axhline(0.5, color="red", lw=0.8, alpha=0.4, ls="--")
        axes[idx].set_xlabel("epoch"); axes[idx].set_ylabel("CTP")
        axes[idx].set_title("Correct Transaction Probability\n(P(correct | pred≠flat))")
        axes[idx].legend(loc="upper left"); axes[idx].grid(alpha=0.3)
        axes[idx].set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(save_dir, "learning_curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] saved {path}")
    return path


# ───────────────────────────────────────────────────────────────────
# CAM 진단 (sparseness 검증)
# ───────────────────────────────────────────────────────────────────


def cam_diagnostic(
    model,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    n_windows: int = 500,
    target_class: str = "predicted",
    method: str = "hirescam",
    seq_len: int = 100,
) -> Dict[str, dict]:
    """
    Sparseness 진단: branch별 CAM peak position 분포 + |CAM| magnitude.

    저장:
      - cam_peak_diagnostic.csv  : (window_idx, branch, peak_pos, cam_max)
      - cam_branch_summary.csv   : branch별 mean/std/range
      - cam_peak_diagnostic.png  : branch별 peak 분포 histogram

    Returns
    -------
    dict[branch_name] = {peak_mean, peak_std, peak_min, peak_max, cam_max}
    """
    model.eval()
    n_branches = len(model.cnns)
    branch_names = list(model.branch_names)

    # collect a single tensor of windows (limit to n_windows)
    xs_chunks, ys_chunks = [], []
    n_collected = 0
    for x, y in loader:
        xs_chunks.append(x); ys_chunks.append(y)
        n_collected += x.shape[0]
        if n_collected >= n_windows:
            break
    if n_collected == 0:
        print("[report] no windows for CAM diagnostic — skipping")
        return {}
    xs = torch.cat(xs_chunks)[:n_windows]
    print(f"[report] CAM diagnostic on {len(xs)} windows × {n_branches} branches")

    rows = []
    summary = {}
    cam_peaks_per_branch = {}  # for plotting

    for bidx, name in enumerate(branch_names):
        eng = TimeSeriesGradCAM(model, device, method=method)
        try:
            cam_s, cam_a = eng.generate_batch(
                xs, branch_idx=bidx, target_class=target_class)
            peaks = cam_a.argmax(axis=-1)               # (N,)
            cam_max_per_window = cam_a.max(axis=-1)     # (N,)

            cam_peaks_per_branch[name] = peaks
            for w in range(len(peaks)):
                rows.append(dict(
                    window_idx=int(w),
                    branch=name,
                    peak_pos=int(peaks[w]),
                    cam_max=float(cam_max_per_window[w]),
                ))
            summary[name] = dict(
                peak_mean=float(peaks.mean()),
                peak_std=float(peaks.std()),
                peak_min=int(peaks.min()),
                peak_max=int(peaks.max()),
                cam_max=float(cam_a.max()),
                cam_mean=float(cam_a.mean()),
            )
        finally:
            eng.cleanup()

    # CSV: per-window per-branch
    df_full = pd.DataFrame(rows)
    full_path = os.path.join(save_dir, "cam_peak_diagnostic.csv")
    df_full.to_csv(full_path, index=False)
    print(f"[report] saved {full_path}")

    # CSV: per-branch summary
    df_sum = pd.DataFrame([dict(branch=name, **stats)
                            for name, stats in summary.items()])
    sum_path = os.path.join(save_dir, "cam_branch_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"[report] saved {sum_path}")

    # PNG: histogram per branch
    if HAS_MPL and len(cam_peaks_per_branch) > 0:
        fig, axes = plt.subplots(1, len(branch_names),
                                  figsize=(4 * len(branch_names), 4),
                                  sharey=True)
        if len(branch_names) == 1:
            axes = [axes]
        for ax, name in zip(axes, branch_names):
            peaks = cam_peaks_per_branch[name]
            ax.hist(peaks, bins=40, range=(0, seq_len), color="steelblue",
                     edgecolor="black", alpha=0.85)
            ax.axvline(seq_len - 1, color="red", linestyle="--", alpha=0.6,
                       label="last timestep")
            ax.set_title(f"{name}\n(std={summary[name]['peak_std']:.2f}, "
                          f"max|CAM|={summary[name]['cam_max']:.3f})")
            ax.set_xlabel(f"peak timestep (T={seq_len})")
            ax.set_xlim(0, seq_len)
            ax.legend(loc="upper left", fontsize=8)
        axes[0].set_ylabel("# windows")
        fig.suptitle(f"CAM peak position distribution per branch  "
                      f"(n={len(xs)}, target_class={target_class})", y=1.02)
        fig.tight_layout()
        png_path = os.path.join(save_dir, "cam_peak_diagnostic.png")
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[report] saved {png_path}")

    return summary


# ───────────────────────────────────────────────────────────────────
# Summary JSON (한눈에 보기)
# ───────────────────────────────────────────────────────────────────


def save_summary_json(
    save_dir: str,
    *,
    model_id: str,
    config: dict,
    test_metrics: dict,
    history: Dict[str, list],
    cam_summary: Optional[Dict[str, dict]] = None,
    extra: Optional[dict] = None,
) -> str:
    """summary.json: 핵심 metric을 한눈에 볼 수 있게."""
    h = history
    best_val_f1_idx = int(np.argmax(h["val_f1"])) if h["val_f1"] else 0
    best_val_loss_idx = int(np.argmin(h["val_loss"])) if h["val_loss"] else 0

    out = dict(
        model_id=model_id,
        backbone=config.get("backbone", "cnn"),
        pool=config.get("pool", "attention"),
        n_epochs_trained=len(h["epoch"]),
        test=test_metrics,
        train_summary=dict(
            final_train_acc=float(h["train_acc"][-1]) if h["train_acc"] else float("nan"),
            best_val_acc=float(max(h["val_acc"])) if h["val_acc"] else float("nan"),
            best_val_f1=float(max(h["val_f1"])) if h["val_f1"] else float("nan"),
            best_val_f1_at_epoch=int(h["epoch"][best_val_f1_idx]) if h["epoch"] else 0,
            best_val_loss=float(min(h["val_loss"])) if h["val_loss"] else float("nan"),
            best_val_loss_at_epoch=int(h["epoch"][best_val_loss_idx]) if h["epoch"] else 0,
            best_val_mcc=(float(max(h["val_mcc"])) if h.get("val_mcc")
                          else float("nan")),
            best_val_ctp=(float(max(h["val_ctp"])) if h.get("val_ctp")
                          else float("nan")),
            train_val_gap=(float(h["train_acc"][-1] - h["val_acc"][-1])
                           if h["train_acc"] and h["val_acc"] else float("nan")),
        ),
    )
    if cam_summary is not None:
        out["cam_diagnostic"] = cam_summary
        # 가장 sparse한 branch 찾기 (peak std 가장 작은 branch)
        if cam_summary:
            most_sparse = min(cam_summary.items(),
                               key=lambda kv: kv[1]["peak_std"])
            out["cam_diagnostic_alert"] = dict(
                most_sparse_branch=most_sparse[0],
                peak_std=most_sparse[1]["peak_std"],
                interpretation=(
                    "OK: peak distributed" if most_sparse[1]["peak_std"] > 5.0
                    else "WARNING: CAM sparseness — peak concentrated on few timesteps"
                    if most_sparse[1]["peak_std"] > 1.0
                    else "CRITICAL: CAM collapsed to single timestep — like VIX failure mode"
                ),
            )
    if extra:
        out["extra"] = extra

    path = os.path.join(save_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[report] saved {path}")
    return path


# ───────────────────────────────────────────────────────────────────
# All-in-one: 학습 후 호출
# ───────────────────────────────────────────────────────────────────


def generate_full_report(
    model,
    save_dir: str,
    *,
    model_id: str,
    config: dict,
    history: Dict[str, list],
    test_metrics: dict,
    dl_test: DataLoader,
    device: torch.device,
    n_classes: int = 3,
    seq_len: int = 100,
    cam_n_windows: int = 500,
    cam_target_class: str = "predicted",
) -> Dict[str, str]:
    """학습 직후 호출 — 모든 진단 파일 생성.

    Returns dict of generated file paths (key = file role).
    """
    print("\n" + "=" * 60)
    print(f"[report] Generating full report → {save_dir}")
    print("=" * 60)

    paths = {}
    paths["history"] = save_history_csv(history, save_dir)
    paths["learning_curves"] = save_learning_curves(history, save_dir)

    # test predictions + confusion
    print("[report] running test set with predictions...")
    res = evaluate(model, dl_test, device, n_classes, return_predictions=True)

    # Derive class names dynamically — supports legacy 3-class mid-price as well
    # as new microstructure tasks (toxic=3, vacuum=2). Falls back to indices.
    task = config.get("task") if isinstance(config, dict) else None
    if task == "toxic":
        class_names = ("stationary", "toxic_up", "toxic_down")
    elif task == "vacuum":
        class_names = ("normal", "vacuum")
    elif n_classes == 3:
        class_names = ("up", "flat", "down")
    else:
        class_names = tuple(f"class_{i}" for i in range(n_classes))

    paths["predictions"] = save_predictions_csv(
        res["y_true"], res["y_pred"], res["probs"], save_dir,
        class_names=class_names)
    cm_info = save_confusion_matrix(res["y_true"], res["y_pred"], save_dir,
                                      n_classes=n_classes,
                                      class_names=class_names)

    # CAM diagnostic
    cam_summary = cam_diagnostic(
        model, dl_test, device, save_dir,
        n_windows=cam_n_windows, target_class=cam_target_class,
        seq_len=seq_len,
    )

    # summary.json
    paths["summary"] = save_summary_json(
        save_dir, model_id=model_id, config=config,
        test_metrics=test_metrics, history=history,
        cam_summary=cam_summary,
        extra=dict(per_class_test=cm_info["per_class"]),
    )

    print("=" * 60)
    print(f"[report] DONE — {len(paths)} files in {save_dir}")
    print("=" * 60)
    return paths
