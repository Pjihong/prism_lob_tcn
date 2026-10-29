#!/usr/bin/env python3
"""Create a machine-readable provenance manifest for a paper run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob_xai.paper_protocol import as_dict
from lob_xai.provenance import file_identity


def source_tree_sha256(root: Path) -> str:
    """Hash active source by relative path and content, excluding caches/backups."""
    files = []
    for pattern in (
        "lob_xai/**/*.py", "lob_xai_viz/**/*.py", "models/**/*.py",
        "scripts/**/*.py",
    ):
        files.extend(root.glob(pattern))
    active = sorted({p for p in files if p.is_file()
                     and "__pycache__" not in p.parts
                     and ".ipynb_checkpoints" not in p.parts
                     and ".bak" not in p.name})
    h = hashlib.sha256()
    for path in active:
        rel = path.relative_to(root).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big")); h.update(rel)
        data = path.read_bytes()
        h.update(len(data).to_bytes(8, "big")); h.update(data)
    return h.hexdigest()


def named_files(values: list[str]) -> dict:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or name in out:
            raise ValueError(f"invalid or duplicate file name: {name!r}")
        out[name] = file_identity(path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--command", required=True,
                    help="exact command that generated the associated metrics")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib",
                 "PyYAML", "statsmodels", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    doc = {
        "generator": "scripts/make_run_manifest.py",
        "command": args.command,
        "code_sha256": source_tree_sha256(ROOT),
        "datasets": named_files(args.data),
        "checkpoints": named_files(args.checkpoint),
        "protocol": as_dict(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
