"""
scripts/train_baseline_microstructure.py — DeepLOB/TLOB on the new
                                           microstructure tasks.

Same baselines as train_baseline.py, same FULL native capacity, same
data split and class-weight protocol — BUT trained on Toxic/Vacuum labels
instead of mid-price labels.

Differences vs train_baseline.py:
  - Labels come from microstructure_labels (not Y[:, 4])
  - n_classes is 3 for Toxic, 2 for Vacuum (build_deeplob/build_tlob
    are reconstructed with the right output dim)
  - Loss defaults to class-weighted CE, matching the paper protocol
  - Saves val + test predictions for threshold tuning

Usage:
  python scripts/train_baseline_microstructure.py \\
      --model deeplob --task toxic --name deeplob_toxic_seed1_k10
  python scripts/train_baseline_microstructure.py \\
      --model tlob --task vacuum --name tlob_vacuum_seed1_k30 \\
      --tlob-hidden 128 --tlob-layers 4 --tlob-heads 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.data import (
    build_microstructure_dataloaders,
)
from lob_xai.training import (
    evaluate, FocalLoss, compute_alpha_inverse_freq,
)
from lob_xai.microstructure_labels import TICK_SIZE_DEFAULT
from lob_xai.paper_protocol import (
    DATA, TRAINING, assert_paper_split, baseline_parameter_count,
    parse_boundaries, split_fraction_report,
)
from lob_xai.provenance import file_identity
from lob_xai.reproducibility import configure_determinism

def _device(spec):
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _as_TF(x):
    """Return a validated (B,T,F=40) tensor."""
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 window batch, got shape={tuple(x.shape)}")
    if x.shape[-1] == 40:
        return x
    if x.shape[1] == 40:
        return x.transpose(1, 2)
    raise ValueError(f"cannot locate the 40-feature axis in shape={tuple(x.shape)}")


def _count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["deeplob", "tlob", "bin_ctabl"])
    ap.add_argument("--task", required=True, choices=["toxic", "vacuum"])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tick-size", type=float, default=TICK_SIZE_DEFAULT)
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv",  default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--epochs", type=int, default=TRAINING.max_epochs)
    ap.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    ap.add_argument("--lr", type=float, default=TRAINING.learning_rate)
    ap.add_argument("--weight-decay", type=float, default=TRAINING.weight_decay)
    ap.add_argument("--optimizer", default=TRAINING.optimizer, choices=["adam", "adamw"])
    ap.add_argument("--scheduler", default=TRAINING.scheduler, choices=["none", "cosine"])
    ap.add_argument("--early-stop-patience", type=int, default=TRAINING.early_stop_patience)
    ap.add_argument("--early-stop-metric", default=TRAINING.checkpoint_metric,
                    choices=["val_loss", "val_f1", "val_pr_auc", "val_mcc"])
    ap.add_argument("--loss", default=TRAINING.loss,
                    choices=["ce", "ce_weighted", "focal"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--val-ratio", type=float,
                    default=DATA.validation_fraction_within_development_file)
    ap.add_argument("--seq-len", type=int, default=DATA.seq_len)
    ap.add_argument("--seed", type=int, default=TRAINING.seeds[0])
    ap.add_argument("--train-boundaries", default=None)
    ap.add_argument("--test-boundaries", default=None)
    split_group = ap.add_mutually_exclusive_group()
    split_group.add_argument("--enforce-paper-split", dest="enforce_paper_split",
                             action="store_true", default=True)
    split_group.add_argument("--allow-split-mismatch", dest="enforce_paper_split",
                             action="store_false",
                             help="diagnostic only: continue with a non-paper split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default=None)
    # TLOB capacity knobs (defaults = FI-2010 native config)
    ap.add_argument("--tlob-hidden", type=int, default=128)
    ap.add_argument("--tlob-layers", type=int, default=4)
    ap.add_argument("--tlob-heads", type=int, default=1)
    args = ap.parse_args()
    if args.k is None:
        args.k = DATA.toxic_horizon if args.task == "toxic" else DATA.vacuum_horizon
    train_boundaries = parse_boundaries(args.train_boundaries)
    test_boundaries = parse_boundaries(args.test_boundaries)

    configure_determinism(args.seed)
    device = _device(args.device)

    name = args.name or f"{args.model}_{args.task}"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = PROJECT_ROOT / "loggers" / "results" / f"{name}_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] model    = {args.model}")
    print(f"[main] task     = {args.task}")
    print(f"[main] device   = {device}")
    print(f"[main] loss     = {args.loss}  (γ={args.focal_gamma})")
    print(f"[main] save_dir = {save_dir}")
    print("[main] hashing input data for provenance ...")
    data_files = {
        "development": file_identity(args.train_csv),
        "test": file_identity(args.test_csv),
    }

    # ── data: exactly the same segment-aware loader as the proposed model ──
    bundle = build_microstructure_dataloaders(
        train_csv=args.train_csv, test_csv=args.test_csv,
        task=args.task, seq_len=args.seq_len, k=args.k,
        tick_size=args.tick_size, n_features=DATA.n_features,
        batch_size=args.batch_size, val_ratio=args.val_ratio,
        branching_scheme="type4", train_boundaries=train_boundaries,
        test_boundaries=test_boundaries, seed=args.seed,
    )
    ds_tr, ds_va, ds_te = bundle["ds_train"], bundle["ds_val"], bundle["ds_test"]
    dl_tr, dl_va, dl_te = bundle["dl_train"], bundle["dl_val"], bundle["dl_test"]
    meta = bundle["meta"]
    info = meta["label_info"]
    n_classes = info["n_classes"]; flat_class = info["flat_class"]
    print(f"[main] n_classes={n_classes}  flat_class={flat_class}")
    print(f"[main] train prevalence: {info['prevalence']}")

    realised_split = split_fraction_report(
        meta["n_rows_train"], meta["n_rows_val"], meta["n_rows_test"]
    )
    print("[main] realised row split = "
          + ", ".join(f"{k}={v:.3f}" for k, v in realised_split.items()))
    if args.enforce_paper_split:
        assert_paper_split(meta["n_rows_train"], meta["n_rows_val"], meta["n_rows_test"])

    # ── model: exact, explicit builders; no runtime head surgery ──
    if args.model == "deeplob":
        from models.deeplob import build_deeplob
        model_config = dict(n_classes=n_classes, n_features=DATA.n_features)
        model = build_deeplob(**model_config)
        expected_params = baseline_parameter_count(args.model, n_classes)
    elif args.model == "tlob":
        from models.tlob import build_tlob
        model_config = dict(
            seq_size=args.seq_len, num_features=40,
            hidden_dim=args.tlob_hidden,
            num_layers=args.tlob_layers,
            num_heads=args.tlob_heads,
            n_classes=n_classes,
        )
        model = build_tlob(**model_config)
        expected_params = baseline_parameter_count(
            args.model, n_classes, seq_len=args.seq_len,
            n_features=DATA.n_features, hidden_dim=args.tlob_hidden,
            num_layers=args.tlob_layers, num_heads=args.tlob_heads,
        )
    elif args.model == "bin_ctabl":
        # BiN + C(TABL) — small attention-bilinear model (Tran et al. 2018).
        from models.bin_ctabl import build_bin_ctabl
        model_config = dict(
            seq_len=args.seq_len, num_features=40,
            n_classes=n_classes,
        )
        model = build_bin_ctabl(**model_config)
        expected_params = baseline_parameter_count(
            args.model, n_classes, seq_len=args.seq_len,
            n_features=DATA.n_features,
        )
    else:
        raise ValueError(f"unknown --model: {args.model}")
    model = model.to(device)
    # A builder can accept n_classes yet ignore it. Probe the actual output
    # before training rather than discovering the mismatch in the loss call.
    probe_x, _ = next(iter(dl_tr))
    with torch.no_grad():
        probe_logits = model(_as_TF(probe_x[:2]).to(device).float())
    if probe_logits.ndim != 2 or probe_logits.shape[1] != n_classes:
        raise RuntimeError(
            f"{args.model} output shape {tuple(probe_logits.shape)} does not "
            f"match n_classes={n_classes}"
        )
    n_params = _count_params(model)
    if n_params != expected_params:
        raise RuntimeError(
            f"{args.model} parameter count={n_params:,} differs from the "
            f"declared architecture count={expected_params:,}"
        )
    print(f"[main] model params = {n_params:,}  (FULL native capacity)")
    model_source = file_identity(PROJECT_ROOT / "models" / f"{args.model}.py")

    # ── loss ──
    if args.loss == "focal":
        alpha = compute_alpha_inverse_freq(dl_tr, n_classes).to(device)
        print(f"[train] FocalLoss α = {alpha.tolist()}  γ = {args.focal_gamma}")
        crit = FocalLoss(alpha=alpha, gamma=args.focal_gamma).to(device)
    elif args.loss == "ce_weighted":
        cw = compute_alpha_inverse_freq(dl_tr, n_classes).to(device)
        print(f"[train] class_weights = {cw.tolist()}")
        crit = nn.CrossEntropyLoss(weight=cw)
    else:
        crit = nn.CrossEntropyLoss()
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs)
    ) if args.scheduler == "cosine" else None)

    # ── train loop ──
    is_loss_metric = args.early_stop_metric == "val_loss"
    best_val = float("inf") if is_loss_metric else -float("inf")
    best_epoch = 0; best_state = None
    patience = 0
    history = dict(epoch=[], train_loss=[], val_loss=[], val_f1=[],
                    val_pr_auc=[], val_mcc=[], val_ctp=[], lr=[])

    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = _as_TF(xb).to(device).float()
            yb = yb.to(device).long()
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            constrain = getattr(model, "constrain_parameters_", None)
            if callable(constrain):
                constrain()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, len(ds_tr))

        val = evaluate(model, _WrapTF(dl_va), device, n_classes,
                        flat_class=flat_class, criterion=crit)

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val["loss"])
        history["val_f1"].append(val["f1"])
        history["val_pr_auc"].append(val.get("pr_auc_macro_nonflat", float("nan")))
        history["val_mcc"].append(val["mcc"])
        history["val_ctp"].append(val["ctp"])
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        print(f"[ep {ep:3d}/{args.epochs}] tr_loss={tr_loss:.4f} | "
              f"val_loss={val['loss']:.4f} f1={val['f1']:.3f} "
              f"pr_auc={val.get('pr_auc_macro_nonflat',float('nan')):.3f} "
              f"mcc={val['mcc']:.3f} ({time.time()-t0:.0f}s)")

        if args.early_stop_metric == "val_loss":
            new_score = val["loss"]; improved = new_score < best_val
        elif args.early_stop_metric == "val_f1":
            new_score = val["f1"]; improved = new_score > best_val
        elif args.early_stop_metric == "val_pr_auc":
            new_score = val.get("pr_auc_macro_nonflat", float("nan"))
            improved = (not np.isnan(new_score)) and new_score > best_val
        else:
            new_score = val["mcc"]; improved = new_score > best_val

        if improved:
            best_val = new_score; best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"[train] early stop at ep {ep} (best={best_val:.4f} at ep {best_epoch})")
                break
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── final test ──
    test = evaluate(model, _WrapTF(dl_te), device, n_classes,
                     flat_class=flat_class, return_predictions=True)
    val_full = evaluate(model, _WrapTF(dl_va), device, n_classes,
                         flat_class=flat_class, return_predictions=True)
    print(f"\n[main] test: f1={test['f1']:.4f} acc={test['acc']:.4f} "
          f"mcc={test['mcc']:.4f} pr_auc={test['pr_auc_macro_nonflat']:.4f} "
          f"ctp={test['ctp']:.4f}")

    test_log = {k: v for k, v in test.items()
                  if k not in ("y_true", "y_pred", "probs")}
    val_log = {k: v for k, v in val_full.items()
                 if k not in ("y_true", "y_pred", "probs")}

    cfg = dict(
        model=args.model, task=args.task, n_classes=n_classes,
        n_features=DATA.n_features,
        flat_class=flat_class, k=args.k, tick_size=args.tick_size,
        seq_len=args.seq_len, val_ratio=args.val_ratio,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        loss=args.loss, focal_gamma=args.focal_gamma,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        optimizer=args.optimizer, scheduler=args.scheduler,
        train_csv=args.train_csv, test_csv=args.test_csv,
        train_boundaries=train_boundaries, test_boundaries=test_boundaries,
        realised_row_split=realised_split,
        data_files=data_files,
        model_config=model_config,
        model_source=model_source,
        expected_n_params=expected_params,
        tlob_cfg=dict(hidden=args.tlob_hidden, layers=args.tlob_layers,
                       heads=args.tlob_heads) if args.model == "tlob" else None,
    )

    torch.save(dict(
        model_state=model.state_dict(), arch=args.model, n_params=n_params,
        config=cfg, meta=meta, history=history, test_metrics=test_log,
    ), save_dir / "model.pt")
    np.savez_compressed(save_dir / "test_predictions.npz",
                          y_true=test["y_true"], y_pred=test["y_pred"],
                          probs=test["probs"])
    np.savez_compressed(save_dir / "val_predictions.npz",
                          y_true=val_full["y_true"], y_pred=val_full["y_pred"],
                          probs=val_full["probs"])

    with open(save_dir / "summary.json", "w") as f:
        json.dump(dict(
            model=args.model, name=name, task=args.task,
            n_params=int(n_params),
            config=cfg, test=test_log, val=val_log,
            epochs_trained=len(history["epoch"]),
            seed=args.seed,
        ), f, indent=2, default=float)
    print(f"[main] saved → {save_dir}")


class _WrapTF:
    """Wrap a DataLoader so evaluate() receives (B,T,40) tensors."""
    def __init__(self, dl): self.dl = dl
    def __iter__(self):
        for xb, yb in self.dl:
            yield _as_TF(xb), yb
    def __len__(self): return len(self.dl)


if __name__ == "__main__":
    main()
