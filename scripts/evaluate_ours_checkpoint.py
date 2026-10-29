#!/usr/bin/env python3
"""One-shot evaluation of a selected branched-model checkpoint on test data.

This command deliberately requires ``--confirm-final-test``.  Use the
validation-only training/collector first, freeze one architecture, and then run
this once for each predeclared seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.data import load_fi2010_features_only, FI2010MicrostructureDataset
from lob_xai.microstructure_labels import make_segment_id
from lob_xai.provenance import verify_file_identity
from lob_xai.training import evaluate
from lob_xai.checkpoints import load_ours_model, read_ckpt_cfg


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--allow-unverified-data", action="store_true")
    ap.add_argument("--confirm-final-test", action="store_true",
                    help="required acknowledgement that model selection is frozen")
    args = ap.parse_args()
    if not args.confirm_final_test:
        ap.error("refusing to unseal test; pass --confirm-final-test after freezing the winner")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        ap.error(f"checkpoint not found: {ckpt}")
    cfg = read_ckpt_cfg(str(ckpt))
    identities = cfg.get("data_files") or {}
    if not args.allow_unverified_data:
        if "test" not in identities:
            ap.error("checkpoint has no test-file identity; use --allow-unverified-data only diagnostically")
        verify_file_identity(args.test_csv, identities["test"])

    task = cfg.get("task")
    if task not in {"toxic", "vacuum"}:
        ap.error(f"unsupported checkpoint task: {task!r}")
    n_features = int(cfg.get("n_features", 40))
    X = load_fi2010_features_only(args.test_csv, n_features=n_features)
    boundaries = list(cfg.get("test_boundaries", []) or [])
    segment_id = make_segment_id(len(X), boundaries)
    ds = FI2010MicrostructureDataset(
        X, segment_id=segment_id, task=task,
        seq_len=int(cfg.get("seq_len", 100)),
        k=int(cfg.get("k", 10)),
        tick_size=float(cfg.get("tick_size", 0.00098)),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    device = _device(args.device)
    model, _, meta = load_ours_model(str(ckpt), device=device, strict=True)
    n_classes = int(cfg.get("n_classes", meta.get("n_classes", 3)))
    flat_class = int(cfg.get("flat_class", meta.get("flat_class", 0)))
    metrics = evaluate(
        model, dl, device, n_classes=n_classes, flat_class=flat_class,
        return_predictions=True,
    )
    serial = {k: v for k, v in metrics.items()
              if k not in {"y_true", "y_pred", "probs"}}
    result = dict(
        checkpoint=str(ckpt), task=task, k=int(cfg.get("k", 10)),
        seed=cfg.get("seed"), n_test=len(ds), metrics=serial,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))

    out_json = Path(args.out_json) if args.out_json else ckpt.parent / "final_test_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                   default=float), encoding="utf-8")
    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, y_true=metrics["y_true"],
                            y_pred=metrics["y_pred"], probs=metrics["probs"])
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
