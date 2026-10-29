"""Small, dependency-free helpers for paper run provenance."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    p = Path(path).expanduser().resolve(strict=True)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: str | Path) -> dict[str, str | int]:
    """Return stable identity fields suitable for a checkpoint config."""
    p = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }


def verify_file_identity(path: str | Path, expected: dict) -> None:
    """Raise if a local file differs from a saved identity record."""
    actual = file_identity(path)
    expected_sha = str(expected.get("sha256", ""))
    if not expected_sha:
        raise ValueError("saved file identity has no sha256")
    if actual["sha256"] != expected_sha:
        raise ValueError(
            f"data SHA-256 mismatch for {actual['path']}: "
            f"got {actual['sha256']}, expected {expected_sha}"
        )

