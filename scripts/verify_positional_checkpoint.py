#!/usr/bin/env python3
"""Verify frozen branch-local-pooling checkpoints against the current code.

For every checkpoint this script:

1. loads the file with an explicit ``weights_only=False`` fallback, so it
   also works on PyTorch >= 2.6 where ``torch.load`` defaults to
   ``weights_only=True`` and rejects numpy scalars stored in older configs;
2. rebuilds the model from the saved config via the single authoritative
   loader (``lob_xai.checkpoints.load_ours_model``) and loads the
   state dict with ``strict=True``;
3. checks that the pool kind, branch-local state keys, tensor shapes, and
   the analytic parameter count from ``lob_xai.paper_protocol`` all agree
   with the actual model;
4. optionally verifies each file's SHA-256 against a frozen run manifest
   (``configs/paper/checkpoints.json``).

Any mismatch exits non-zero.  Run this after every code refactor before
trusting downstream XAI results computed from the frozen checkpoints.

Usage
-----
    # explicit checkpoints
    PYTHONPATH=. python scripts/verify_positional_checkpoint.py \
        --checkpoint loggers/results/..._seed1_.../model.pt

    # every checkpoint recorded in a frozen manifest (paths + SHA-256)
    PYTHONPATH=. python scripts/verify_positional_checkpoint.py \
        --manifest configs/paper/checkpoints.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from lob_xai.paper_protocol import branch_variant_parameter_count  # noqa: E402
from lob_xai.checkpoints import load_ours_model  # noqa: E402

BRANCH_NAMES = ("ask_price", "ask_volume", "bid_price", "bid_volume")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_one(path: Path, expected_sha: str | None) -> list[str]:
    """Return a list of human-readable failures (empty = pass)."""
    failures: list[str] = []
    if not path.is_file():
        return [f"missing file: {path}"]

    if expected_sha is not None:
        actual = sha256_file(path)
        if actual != expected_sha:
            failures.append(
                f"SHA-256 mismatch: manifest={expected_sha[:12]}… "
                f"actual={actual[:12]}…"
            )

    try:
        model, cfg, meta = load_ours_model(str(path), device="cpu", strict=True)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"strict model load failed: {exc}")
        return failures

    pool = str(cfg.get("pool", "avg"))
    task = str(cfg.get("task", meta.get("task", "?")))
    n_classes = int(cfg.get("n_classes", 0) or (3 if task == "toxic" else 2))
    seed = cfg.get("seed", meta.get("seed", "?"))
    n_params = sum(p.numel() for p in model.parameters())

    print(f"[ckpt] {path}")
    print(f"       task={task} pool={pool} seed={seed} "
          f"input_norm={cfg.get('input_norm', 'identity')} params={n_params:,}")

    if pool in {"multi_scale", "positional"}:
        state = model.state_dict()
        expected_keys = [f"pool.pools.{i}.logits" for i in range(4)]
        missing = [k for k in expected_keys if k not in state]
        if missing:
            failures.append(f"missing branch-local pool keys: {missing}")
        else:
            seq_len = int(cfg.get("seq_len", 100))
            want = seq_len if pool == "positional" else len(
                tuple(cfg.get("pool_windows", (8, 32, 100)))
            )
            for key in expected_keys:
                got = tuple(state[key].shape)
                if got != (want,):
                    failures.append(f"{key}: shape {got}, expected ({want},)")
            # Report the effective temporal weights per branch.
            for i, name in enumerate(BRANCH_NAMES):
                w = model.pool.pools[i].temporal_weights(seq_len).detach()
                last8 = float(w[-8:].sum())
                lag = float(
                    (w * torch.arange(seq_len - 1, -1, -1, dtype=w.dtype)).sum()
                )
                print(f"       {name:11s} last8={last8:.3f} E[lag]={lag:5.1f}")

        # Analytic parameter count (excludes input-norm parameters).
        try:
            analytic = branch_variant_parameter_count(
                n_classes,
                pool=pool,
                seq_len=int(cfg.get("seq_len", 100)),
                pool_windows=tuple(cfg.get("pool_windows", (8, 32, 100))),
            )
            norm_params = sum(
                p.numel() for p in model.input_norm.parameters()
            ) if hasattr(model, "input_norm") else 0
            if analytic + norm_params != n_params:
                failures.append(
                    f"parameter count mismatch: analytic {analytic} + "
                    f"input-norm {norm_params} != actual {n_params}"
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"analytic parameter count failed: {exc}")

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="checkpoint path (repeatable)")
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest JSON with a 'checkpoints' mapping")
    args = ap.parse_args()

    jobs: list[tuple[Path, str | None]] = [
        (Path(p), None) for p in args.checkpoint
    ]
    if args.manifest:
        doc = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for tag, entry in sorted(doc.get("checkpoints", {}).items()):
            jobs.append((Path(entry["path"]), entry.get("sha256")))
    if not jobs:
        ap.error("give --checkpoint and/or --manifest")

    n_bad = 0
    for path, sha in jobs:
        failures = check_one(path, sha)
        if failures:
            n_bad += 1
            for msg in failures:
                print(f"       FAIL: {msg}")
        else:
            print("       OK: strict load, keys, shapes, parameter count")
    print(f"[done] {len(jobs) - n_bad}/{len(jobs)} checkpoints verified")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
