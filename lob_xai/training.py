"""Training, loss functions, evaluation, and early stopping.

Metric implementations live in metrics.py; imports below preserve the
existing training-module API used by analysis scripts.
"""
from __future__ import annotations
from typing import Dict, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .loggers import logger as yaml_logger
from .metrics import (
    _accuracy,
    _f1_macro,
    _mcc_multiclass,
    _per_class_classification_metrics,
    _correct_transaction_probability,
    _pr_auc_binary,
    _roc_auc_binary,
    _lift_at_k,
    _brier_binary,
    _expected_cost,
    _per_class_pr_auc,
    _per_class_roc_auc,
    find_threshold_high_precision,
    find_threshold_youden,
    find_threshold_cost,
)


# ───────────────────────────────────────────────────────────────────
# Loss functions
# ───────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017): FL(p_t) = − α_t · (1 − p_t)^γ · log(p_t)

    For multi-class with `alpha` as a per-class tensor (length n_classes).
    Equivalent to CrossEntropyLoss when alpha=1 and gamma=0.

    Parameters
    ----------
    alpha : (n_classes,) tensor of per-class weights (or None = uniform).
            Pass inverse-frequency × n_classes for the standard recipe.
    gamma : focusing parameter (default 2.0). Higher = more emphasis on hard
            examples.
    reduction : "mean" | "sum" | "none"
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C); target: (B,) long
        log_probs = F.log_softmax(logits, dim=-1)              # (B, C)
        probs = log_probs.exp()                                 # (B, C)
        # Gather per-sample log_prob and prob for the true class
        target = target.long()
        tgt_log_p = log_probs.gather(1, target[:, None]).squeeze(1)   # (B,)
        tgt_p = probs.gather(1, target[:, None]).squeeze(1)            # (B,)
        focal_w = (1.0 - tgt_p) ** self.gamma
        loss = -focal_w * tgt_log_p
        if self.alpha is not None:
            at = self.alpha[target]
            loss = at * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_alpha_inverse_freq(
    loader: DataLoader,
    n_classes: int,
    smoothing: float = 1e-3,
    cap: float = 50.0,
    power: float = 1.0,
) -> torch.Tensor:
    """
    Per-class inverse-frequency weights, normalized so mean(α) = 1.
    Clipped at `cap` to avoid pathological gradient scales for extreme rarity.
    """
    counts = torch.zeros(n_classes, dtype=torch.float64)
    for _, y in loader:
        for c in range(n_classes):
            counts[c] += (y == c).sum().double()
    total = counts.sum().clamp_min(1.0)
    freq = counts / total
    if power < 0:
        raise ValueError("class-weight power must be non-negative")
    alpha = (1.0 / (freq + smoothing)) ** float(power)
    alpha = alpha / alpha.mean()
    alpha = alpha.clamp(max=cap)
    return alpha.float()


# ───────────────────────────────────────────────────────────────────
# evaluate()
# ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int = 3,
    return_predictions: bool = False,
    flat_class: int = 1,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Evaluate model on loader.

    Returns dict with:
      loss, acc, f1, mcc                              (always)
      pr_auc_class_*, pr_auc_macro_nonflat            (imbalance-aware)
      roc_auc_class_*, roc_auc_macro_nonflat
      lift_at_5, lift_at_10 (per non-flat class)
      brier (per non-flat class)
      ctp, ctp_class_*, coverage, n_actions           (legacy)

    If return_predictions=True, also returns y_true, y_pred, probs.
    """
    model.eval()
    crit = criterion if criterion is not None else nn.CrossEntropyLoss()
    losses, all_logits, all_y = [], [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        losses.append(float(loss.item()))
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    if not losses:
        out = dict(loss=float("nan"), acc=float("nan"), f1=float("nan"),
                    mcc=float("nan"), ctp=float("nan"),
                    coverage=0.0, n_actions=0, pr_auc_macro_nonflat=float("nan"),
                    roc_auc_macro_nonflat=float("nan"))
        if return_predictions:
            out.update(y_true=np.array([]), y_pred=np.array([]),
                        probs=np.zeros((0, n_classes)))
        return out

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    y_true_np = y.numpy()
    y_pred_np = logits.argmax(dim=-1).numpy()
    probs_np = torch.softmax(logits, dim=-1).numpy()

    out = dict(
        loss=float(np.mean(losses)),
        acc=_accuracy(logits, y),
        f1=_f1_macro(logits, y, n_classes),
        mcc=_mcc_multiclass(y_true_np, y_pred_np, n_classes),
    )
    out.update(_per_class_classification_metrics(
        y_true_np, y_pred_np, n_classes
    ))
    out.update(_correct_transaction_probability(y_true_np, y_pred_np,
                                                  flat_class, n_classes))
    out.update(_per_class_pr_auc(probs_np, y_true_np, n_classes, flat_class))
    out.update(_per_class_roc_auc(probs_np, y_true_np, n_classes, flat_class))

    # Lift @ 5% and 10% — for each non-flat class, score = P(class)
    for c in range(n_classes):
        if c == flat_class:
            continue
        y_bin = (y_true_np == c).astype(int)
        scores = probs_np[:, c]
        out[f"lift5_class_{c}"] = _lift_at_k(scores, y_bin, 0.05)
        out[f"lift10_class_{c}"] = _lift_at_k(scores, y_bin, 0.10)
        out[f"brier_class_{c}"] = _brier_binary(probs_np[:, c],
                                                  (y_true_np == c).astype(int))

    if return_predictions:
        out.update(y_true=y_true_np, y_pred=y_pred_np, probs=probs_np)
    return out


# ───────────────────────────────────────────────────────────────────
# train()
# ───────────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
    early_stop_patience: int = 10,
    n_classes: int = 3,
    model_id: str = "model",
    use_class_weights: bool = True,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    early_stop_metric: str = "val_f1",
    flat_class: int = 1,
    optimizer_name: str = "adamw",
    scheduler_name: str = "cosine",
    class_weight_power: float = 1.0,
    grad_clip_norm: Optional[float] = None,
) -> Dict:
    """
    Standard training loop with early stopping. Tracks PR-AUC + MCC + CTP
    per epoch.

    Loss options:
      use_focal=True   → FocalLoss(α=inverse_freq, γ=focal_gamma)
      use_class_weights=True (default) and use_focal=False → CE(weight=inverse_freq)
      both False → plain CrossEntropyLoss

    early_stop_metric ∈ {"val_loss", "val_f1", "val_pr_auc", "val_mcc"}
        The manuscript fixes this to ``val_f1`` so reporting metrics do not
        influence model selection.
    """
    assert early_stop_metric in ("val_loss", "val_f1", "val_pr_auc", "val_mcc")

    if use_focal:
        alpha = compute_alpha_inverse_freq(
            dl_train, n_classes, power=class_weight_power
        ).to(device)
        print(f"[train] FocalLoss α = {alpha.tolist()}  γ = {focal_gamma}")
        crit = FocalLoss(alpha=alpha, gamma=focal_gamma).to(device)
    elif use_class_weights:
        # Use the same alpha but as a CE weight
        cw = compute_alpha_inverse_freq(
            dl_train, n_classes, power=class_weight_power
        ).to(device)
        print(f"[train] class_weights = {cw.tolist()}")
        crit = nn.CrossEntropyLoss(weight=cw)
    else:
        crit = nn.CrossEntropyLoss()

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'adamw' or 'adam'")

    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(epochs))
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError("scheduler_name must be 'cosine' or 'none'")
    model = model.to(device)

    is_loss_metric = early_stop_metric == "val_loss"
    best_score = float("inf") if is_loss_metric else -float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_left = early_stop_patience
    history = dict(epoch=[], train_loss=[], train_acc=[], train_f1=[],
                    val_loss=[], val_acc=[], val_f1=[],
                    val_mcc=[], val_ctp=[], val_coverage=[],
                    val_pr_auc=[], lr=[])

    print(f"[train] starting training for {epochs} epochs on {device}")
    print(f"[train] optimizer={optimizer_name} scheduler={scheduler_name}")
    print(f"[train] early stopping on {early_stop_metric}, patience={early_stop_patience}, flat_class={flat_class}")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        ep_losses, ep_logits, ep_y = [], [], []
        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            if grad_clip_norm is not None:
                if grad_clip_norm <= 0:
                    raise ValueError("grad_clip_norm must be positive or None")
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()
            constrain = getattr(model, "constrain_parameters_", None)
            if callable(constrain):
                constrain()
            ep_losses.append(float(loss.item()))
            ep_logits.append(logits.detach().cpu())
            ep_y.append(y.detach().cpu())
        tr_loss = float(np.mean(ep_losses))
        tr_logits = torch.cat(ep_logits)
        tr_y = torch.cat(ep_y)
        tr_acc = _accuracy(tr_logits, tr_y)
        tr_f1 = _f1_macro(tr_logits, tr_y, n_classes)

        val = evaluate(model, dl_val, device, n_classes,
                        flat_class=flat_class, criterion=crit)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["val_f1"].append(val["f1"])
        history["val_mcc"].append(val["mcc"])
        history["val_ctp"].append(val["ctp"])
        history["val_coverage"].append(val["coverage"])
        history["val_pr_auc"].append(val.get("pr_auc_macro_nonflat", float("nan")))
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        elapsed = time.time() - t0
        print(f"[ep {epoch+1:3d}/{epochs}] tr_loss={tr_loss:.4f} tr_f1={tr_f1:.3f} | "
              f"val_loss={val['loss']:.4f} f1={val['f1']:.3f} mcc={val['mcc']:.3f} "
              f"pr_auc={val.get('pr_auc_macro_nonflat', float('nan')):.3f} "
              f"ctp={val['ctp']:.3f} ({elapsed:.0f}s)")

        # determine new score per chosen metric
        if early_stop_metric == "val_loss":
            new_score = val["loss"]
            improved = new_score < best_score
        elif early_stop_metric == "val_f1":
            new_score = val["f1"]
            improved = new_score > best_score
        elif early_stop_metric == "val_pr_auc":
            new_score = val.get("pr_auc_macro_nonflat", float("nan"))
            # never improve if NaN
            improved = (not np.isnan(new_score)) and new_score > best_score
        elif early_stop_metric == "val_mcc":
            new_score = val["mcc"]
            improved = new_score > best_score
        else:
            raise RuntimeError(f"unhandled early_stop_metric: {early_stop_metric}")
        if improved:
            best_score = new_score
            best_epoch = epoch + 1
            # cache best state on CPU
            best_state_dict = {k_: v.detach().cpu().clone()
                                for k_, v in model.state_dict().items()}
            patience_left = early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] early stop at epoch {epoch+1} "
                      f"(best={best_score:.4f} at ep {best_epoch})")
                break

        if scheduler is not None:
            scheduler.step()

    # Restore best state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[train] restored best checkpoint at epoch {best_epoch} "
              f"({early_stop_metric}={best_score:.4f})")

    yaml_logger(model_id, "training_history", history)
    return history


# ───────────────────────────────────────────────────────────────────
# Legacy class weighting
# ───────────────────────────────────────────────────────────────────


def compute_class_weights(loader: DataLoader, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced data (LEGACY)."""
    counts = torch.zeros(n_classes)
    for _, y in loader:
        for c in range(n_classes):
            counts[c] += (y == c).sum().float()
    total = counts.sum()
    weights = 1.0 - counts / total
    weights = torch.clamp(weights, min=1e-3)
    return weights
