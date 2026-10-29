"""Train a four-branch model for toxic-flow or liquidity-vacuum proxies.

Run: python scripts/train_microstructure.py "@configs/paper/toxic.args" --seed 1
Inputs: indexed FI-2010 development/test CSVs and CLI or @file arguments.
Outputs: checkpoint, configuration and metrics in loggers/results/.
Scope: paper presets use positional/identity and validation-only selection.
Without a preset, GAP remains the compatibility default. See docs/REPRODUCE.md.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("LOB_XAI_ROOT", str(PROJECT_ROOT))

import numpy as np
import torch

from lob_xai.config import Config
from lob_xai.data import build_microstructure_dataloaders
from lob_xai.models import LOBBranchModel
from lob_xai.training import train, evaluate
from lob_xai.loggers import generate_id, find_save_path, logger as yaml_logger
from lob_xai.reporting import generate_full_report
from lob_xai.microstructure_labels import TICK_SIZE_DEFAULT, K_DEFAULT
from lob_xai.paper_protocol import (
    DATA, TRAINING, assert_paper_split, parse_boundaries, split_fraction_report,
)
from lob_xai.provenance import file_identity
from lob_xai.reproducibility import configure_determinism


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def build_parser():
    """Build the CLI; @files contain one argument token per line."""
    ap = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    # Task
    ap.add_argument("--task", required=True, choices=["toxic", "vacuum"],
                    help="microstructure task to train on")
    ap.add_argument("--k", type=int, default=None,
                    help="prediction horizon (default: toxic=10, vacuum=30)")
    ap.add_argument("--tick-size", type=float, default=TICK_SIZE_DEFAULT,
                    help="tick size in z-score units (default 0.00098)")

    # Data
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv",  default="data/FI2010/FI2010_test.csv")
    ap.add_argument(
        "--val-ratio", type=float,
        default=DATA.validation_fraction_within_development_file,
        help="validation fraction inside the development CSV (paper: 0.15/0.85)",
    )
    ap.add_argument("--train-boundaries", default=None,
                    help="comma-separated internal segment boundaries")
    ap.add_argument("--test-boundaries", default=None,
                    help="comma-separated internal segment boundaries")
    split_group = ap.add_mutually_exclusive_group()
    split_group.add_argument("--enforce-paper-split", dest="enforce_paper_split",
                             action="store_true", default=True,
                             help="check development split 70:15 and a nonempty separate test file")
    split_group.add_argument("--allow-split-mismatch", dest="enforce_paper_split",
                             action="store_false",
                             help="diagnostic only: continue after reporting a non-paper split")

    # Model — ALWAYS type4. topdeep8 is dropped.
    ap.add_argument("--backbone", default="tcn", choices=["cnn", "tcn"])
    ap.add_argument("--pool", default=TRAINING.pool,
                    choices=["attention", "avg", "max", "avg_last",
                             "multi_scale", "positional"])
    ap.add_argument("--pool-windows", type=int, nargs="+", default=[8, 32, 100],
                    help="tail windows for branch-local multi_scale pooling")
    ap.add_argument("--cnn-channels", type=int, nargs="+",
                    default=list(TRAINING.channels))
    ap.add_argument("--cnn-kernel", type=int, default=TRAINING.kernel_size)
    ap.add_argument("--cnn-dropout", type=float, default=TRAINING.dropout)
    ap.add_argument("--fc-hidden", type=int, nargs="+",
                    default=list(TRAINING.fc_hidden))
    ap.add_argument("--tcn-dilation-base", type=int, default=TRAINING.dilation_base)
    ap.add_argument("--tcn-norm", default="none",
                    choices=["none", "layer", "batch"])
    ap.add_argument("--input-norm", default="identity",
                    choices=["identity", "dain", "revin_c"])
    ap.add_argument("--interaction-channels", type=int, default=0,
                    help="0 preserves strict four-branch late fusion; a small "
                         "value such as 64 adds an explicit interaction stream")

    # Loss / optimization
    ap.add_argument("--loss", default=TRAINING.loss,
                    choices=["ce", "ce_weighted", "focal"],
                    help=f"loss function (paper default: {TRAINING.loss})")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=TRAINING.max_epochs)
    ap.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    ap.add_argument("--lr", type=float, default=TRAINING.learning_rate)
    ap.add_argument("--weight-decay", type=float, default=TRAINING.weight_decay)
    ap.add_argument("--optimizer", default=TRAINING.optimizer, choices=["adam", "adamw"])
    ap.add_argument("--scheduler", default=TRAINING.scheduler, choices=["none", "cosine"])
    ap.add_argument("--early-stop-metric", default=TRAINING.checkpoint_metric,
                    choices=["val_loss", "val_f1", "val_pr_auc", "val_mcc"])
    ap.add_argument("--early-stop-patience", type=int, default=TRAINING.early_stop_patience)
    ap.add_argument("--class-weight-power", type=float, default=1.0,
                    help="inverse-frequency exponent; try 0.5 as a less "
                         "aggressive rare-class weighting ablation")
    ap.add_argument("--grad-clip-norm", type=float, default=None)
    ap.add_argument("--seq-len", type=int, default=DATA.seq_len)

    # Misc
    ap.add_argument("--seed", type=int, default=TRAINING.seeds[0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--name", default=None,
                    help="model_id prefix (default = 'tcn_avg_type4_{task}')")
    ap.add_argument("--cam-windows", type=int, default=500)
    ap.add_argument("--cam-target-class", default="predicted",
                    choices=["predicted", "stationary", "toxic_up", "toxic_down",
                             "normal", "vacuum"])
    ap.add_argument("--dev-only", action="store_true",
                    help="train/select on validation and do not evaluate test")
    ap.add_argument("--skip-report", action="store_true",
                    help="skip the expensive CAM/report generation")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if args.k is None:
        args.k = DATA.toxic_horizon if args.task == "toxic" else DATA.vacuum_horizon
    if args.interaction_channels < 0:
        ap.error("--interaction-channels must be >= 0")
    if not args.pool_windows or any(x <= 0 for x in args.pool_windows):
        ap.error("--pool-windows values must be positive")
    if len(set(args.pool_windows)) != len(args.pool_windows):
        ap.error("--pool-windows values must be unique")
    if args.pool in {"multi_scale", "positional"} and args.interaction_channels:
        ap.error("branch-local pooling requires --interaction-channels 0")
    if args.class_weight_power < 0:
        ap.error("--class-weight-power must be >= 0")
    if args.grad_clip_norm is not None and args.grad_clip_norm <= 0:
        ap.error("--grad-clip-norm must be positive")
    train_boundaries = parse_boundaries(args.train_boundaries)
    test_boundaries = parse_boundaries(args.test_boundaries)

    configure_determinism(args.seed)

    device = _resolve_device(args.device)
    print(f"[main] task          = {args.task}")
    print(f"[main] device        = {device}")
    print(f"[main] backbone/pool = {args.backbone} / {args.pool}")
    if args.pool == "multi_scale":
        print(f"[main] pool windows  = {args.pool_windows}")
    print(f"[main] channels/norm = {args.cnn_channels} / {args.tcn_norm}")
    print(f"[main] interaction   = {args.interaction_channels}")
    print(f"[main] branching     = type4 (ALWAYS — topdeep8 dropped)")
    print(f"[main] loss          = {args.loss}  (γ={args.focal_gamma})")
    print(f"[main] early stop on = {args.early_stop_metric}, "
          f"patience={args.early_stop_patience}")
    print(f"[main] k={args.k}  tick_size={args.tick_size}")
    print("[main] hashing input data for provenance ...")
    data_files = {
        "development": file_identity(args.train_csv),
        "test": file_identity(args.test_csv),
    }

    # ── ID + log dir ──────────────────────────────────────────
    if args.name:
        name = args.name
    else:
        name = f"{args.backbone}_{args.pool}_type4_{args.task}"
    model_id = generate_id(name)
    save_dir = find_save_path(model_id)
    print(f"[main] model_id = {model_id}")
    print(f"[main] save_dir = {save_dir}")

    cfg_log = dict(
        task=args.task, k=args.k, tick_size=args.tick_size,
        backbone=args.backbone, pool=args.pool, branching_scheme="type4",
        pool_windows=tuple(args.pool_windows),
        loss=args.loss, focal_gamma=args.focal_gamma,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        train_csv=args.train_csv, test_csv=args.test_csv,
        val_ratio=args.val_ratio, seq_len=args.seq_len,
        cnn_channels=tuple(args.cnn_channels), cnn_kernel=args.cnn_kernel,
        cnn_dropout=args.cnn_dropout, fc_hidden=tuple(args.fc_hidden),
        tcn_dilation_base=args.tcn_dilation_base,
        tcn_norm=args.tcn_norm, input_norm=args.input_norm,
        interaction_channels=args.interaction_channels,
        class_weight_power=args.class_weight_power,
        grad_clip_norm=args.grad_clip_norm,
        dev_only=bool(args.dev_only),
        optimizer=args.optimizer, scheduler=args.scheduler,
        train_boundaries=train_boundaries, test_boundaries=test_boundaries,
        data_files=data_files,
    )
    yaml_logger(model_id, "config", cfg_log)

    # ── data ──────────────────────────────────────────────────
    bundle = build_microstructure_dataloaders(
        train_csv=args.train_csv, test_csv=args.test_csv,
        task=args.task, seq_len=args.seq_len, k=args.k,
        tick_size=args.tick_size,
        n_features=40, batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        branching_scheme="type4",
        train_boundaries=train_boundaries,
        test_boundaries=test_boundaries,
        seed=args.seed,
        include_test=not args.dev_only,
    )
    meta = bundle["meta"]
    n_classes = meta["n_classes"]
    flat_class = meta["flat_class"]
    realised_split = None
    if args.dev_only:
        dev_total = meta["n_rows_train"] + meta["n_rows_val"]
        realised_split = dict(
            train_within_development=meta["n_rows_train"] / dev_total,
            validation_within_development=meta["n_rows_val"] / dev_total,
        )
        print("[main] development-only row split = "
              + ", ".join(f"{k}={v:.3f}" for k, v in realised_split.items()))
    else:
        realised_split = split_fraction_report(
            meta["n_rows_train"], meta["n_rows_val"], meta["n_rows_test"]
        )
        print("[main] realised row split = "
              + ", ".join(f"{k}={v:.3f}" for k, v in realised_split.items()))
        if args.enforce_paper_split:
            assert_paper_split(
                meta["n_rows_train"], meta["n_rows_val"], meta["n_rows_test"]
            )
    cfg_log["realised_row_split"] = realised_split
    # Backfill cfg_log with the resolved n_classes / flat_class so downstream
    # tools that read ckpt["config"] (lob_xai.checkpoints.load_ours_model,
    # tcav_negative_control, etc.) get the right shape without consulting meta.
    cfg_log["n_classes"] = int(n_classes)
    cfg_log["flat_class"] = int(flat_class)
    cfg_log["n_features"] = 40
    cfg_log["cnn_channels"] = tuple(args.cnn_channels)
    cfg_log["cnn_dropout"] = args.cnn_dropout
    cfg_log["fc_hidden"] = tuple(args.fc_hidden)

    if args.quick:
        from torch.utils.data import Subset, DataLoader
        bundle["ds_train"] = Subset(bundle["ds_train"],
                                      list(range(min(2000, len(bundle["ds_train"])))))
        bundle["ds_val"]   = Subset(bundle["ds_val"],
                                      list(range(min(500, len(bundle["ds_val"])))))
        bundle["dl_train"] = DataLoader(bundle["ds_train"], batch_size=args.batch_size,
                                          shuffle=True, drop_last=False,
                                          generator=torch.Generator().manual_seed(args.seed))
        bundle["dl_val"]   = DataLoader(bundle["ds_val"], batch_size=args.batch_size)
        print(f"[main] QUICK mode — train={len(bundle['ds_train'])}, "
              f"val={len(bundle['ds_val'])}")

    yaml_logger(model_id, "data_meta", meta)

    # ── model ─────────────────────────────────────────────────
    model = LOBBranchModel(
        n_features=40, n_classes=n_classes,
        backbone=args.backbone, pool=args.pool,
        cnn_channels=tuple(args.cnn_channels),
        cnn_kernel=args.cnn_kernel, cnn_dropout=args.cnn_dropout,
        fc_hidden=tuple(args.fc_hidden),
        branch_indices=meta["branch_indices"],
        tcn_dilation_base=args.tcn_dilation_base,
        input_norm=args.input_norm,
        tcn_norm=args.tcn_norm,
        interaction_channels=args.interaction_channels,
        seq_len=args.seq_len,
        pool_windows=tuple(args.pool_windows),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] model params = {n_params:,}")
    yaml_logger(model_id, "model_info", dict(
        n_params=int(n_params), backbone=args.backbone, pool=args.pool,
        branching_scheme="type4", branch_names=meta["branch_names"],
        n_classes=n_classes, flat_class=flat_class,
        cnn_channels=list(args.cnn_channels),
        pool_windows=list(args.pool_windows),
        tcn_norm=args.tcn_norm, input_norm=args.input_norm,
        interaction_channels=args.interaction_channels,
    ))

    # ── train ─────────────────────────────────────────────────
    use_focal = (args.loss == "focal")
    use_class_weights = (args.loss == "ce_weighted")
    history = train(
        model, bundle["dl_train"], bundle["dl_val"],
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        device=device, early_stop_patience=args.early_stop_patience,
        n_classes=n_classes, model_id=model_id,
        use_class_weights=use_class_weights,
        use_focal=use_focal, focal_gamma=args.focal_gamma,
        early_stop_metric=args.early_stop_metric,
        flat_class=flat_class,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        class_weight_power=args.class_weight_power,
        grad_clip_norm=args.grad_clip_norm,
    )

    # ── final test eval (with full metric suite) ──────────────
    val_metrics = evaluate(model, bundle["dl_val"], device, n_classes,
                           flat_class=flat_class, return_predictions=True)
    val_log = {k: v for k, v in val_metrics.items()
               if k not in ("y_true", "y_pred", "probs")}
    yaml_logger(model_id, "val_metrics", val_log)
    print("\n[main] restored-checkpoint validation evaluation:")
    print(f"  val f1         = {val_log['f1']:.4f}")
    print(f"  val mcc        = {val_log['mcc']:.4f}")
    print(f"  val pr_auc_nf  = {val_log['pr_auc_macro_nonflat']:.4f}")

    test_metrics = None
    test_log = None
    if args.dev_only:
        print("[main] DEV-ONLY: test inference and test metrics were skipped")
    else:
        print("\n[main] final test set evaluation:")
        test_metrics = evaluate(model, bundle["dl_test"], device, n_classes,
                                flat_class=flat_class,
                                return_predictions=True)
        print(f"  test loss       = {test_metrics['loss']:.4f}")
        print(f"  test acc        = {test_metrics['acc']:.4f}")
        print(f"  test f1         = {test_metrics['f1']:.4f}")
        print(f"  test mcc        = {test_metrics['mcc']:.4f}")
        print(f"  test pr_auc_nf  = {test_metrics['pr_auc_macro_nonflat']:.4f}")
        print(f"  test roc_auc_nf = {test_metrics['roc_auc_macro_nonflat']:.4f}")
        test_log = {k: v for k, v in test_metrics.items()
                    if k not in ("y_true", "y_pred", "probs")}
        yaml_logger(model_id, "test_metrics", test_log)

    # ── save checkpoint (including probs for threshold tuning later) ──
    import json
    ckpt_path = os.path.join(save_dir, "model.pt")
    # Save with the same shape as legacy train.py for compat with audit scripts
    torch.save(dict(
        model_state=model.state_dict(),
        config=cfg_log,
        meta=meta,
        history=history,
        test_metrics=test_log,
    ), ckpt_path)
    print(f"[main] saved checkpoint to {ckpt_path}")

    if test_metrics is not None:
        npz_path = os.path.join(save_dir, "test_predictions.npz")
        np.savez_compressed(
            npz_path,
            y_true=test_metrics["y_true"],
            y_pred=test_metrics["y_pred"],
            probs=test_metrics["probs"],
            class_names=np.array(meta["class_names"]),
        )
        print(f"[main] saved test predictions to {npz_path}")

    # Always save validation predictions for honest model selection.
    val_npz_path = os.path.join(save_dir, "val_predictions.npz")
    np.savez_compressed(
        val_npz_path,
        y_true=val_metrics["y_true"],
        y_pred=val_metrics["y_pred"],
        probs=val_metrics["probs"],
        class_names=np.array(meta["class_names"]),
    )
    print(f"[main] saved val predictions to {val_npz_path}")

    # ── full report (CAM diagnostic, confusion, etc.) ─────────
    # Cast cfg_log to match legacy shape expected by generate_full_report.
    # The reporting module is task-agnostic at the CAM level.
    if not args.dev_only and not args.skip_report:
        cfg_for_report = dict(
            n_features=40, n_classes=n_classes, seq_len=args.seq_len,
            backbone=args.backbone, pool=args.pool,
            pool_windows=tuple(args.pool_windows),
            cnn_channels=tuple(args.cnn_channels),
            cnn_kernel=args.cnn_kernel, cnn_dropout=args.cnn_dropout,
            fc_hidden=tuple(args.fc_hidden),
            tcn_dilation_base=args.tcn_dilation_base,
            input_norm=args.input_norm, tcn_norm=args.tcn_norm,
            interaction_channels=args.interaction_channels,
            horizon_idx=4,  # dummy, unused in reporting
            train_csv=args.train_csv, test_csv=args.test_csv,
        )
        generate_full_report(
            model, save_dir,
            model_id=model_id, config=cfg_for_report,
            history=history, test_metrics=test_log,
            dl_test=bundle["dl_test"], device=device,
            n_classes=n_classes, seq_len=args.seq_len,
            cam_n_windows=args.cam_windows,
            cam_target_class=args.cam_target_class,
        )

    # Drop a JSON summary alongside (used by metrics_summary.py)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(dict(
            model_id=model_id, task=args.task, n_classes=n_classes,
            flat_class=flat_class,
            config=cfg_log,
            val=val_log,
            test=test_log,
            n_params=int(n_params),
            n_epochs_trained=len(history["epoch"]),
        ), f, indent=2, default=float)

    print(f"\n[main] DONE. model_id = {model_id}")
    print(f"[main] all outputs in {save_dir}")


if __name__ == "__main__":
    main()
