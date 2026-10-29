"""Verify retained originals and recorded refactors against their SHA-256 hashes."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def check_file(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        return f"Path escapes root: {relative}"
    if not path.is_file():
        return f"Missing: {relative}"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        return f"Changed: {relative}"
    return None


def expected_files(row, include_data=False):
    """Original hashes stay in the ledger even when a source is refactored."""
    if "destination" not in row or row["disposition"] == "removed_by_request":
        return {}
    if row["disposition"] == "local_data" and not include_data:
        return {}
    if row["disposition"] == "refactored":
        return row["refactored_files"]
    return {row["destination"]: row["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    ledger = json.loads((ROOT / "docs/PRESERVATION_MANIFEST.json").read_text(encoding="utf-8"))
    sources = json.loads((ROOT / "docs/SOURCE_FILES.json").read_text(encoding="utf-8"))
    jobs = set()
    for row in ledger:
        for relative, expected in expected_files(row, args.include_data).items():
            jobs.add((ROOT, relative, expected))
    for row in sources:
        if row.get("disposition") == "removed_by_request":
            continue
        if not row["name"].endswith(".zip"):
            jobs.add((ROOT, "manuscripts/reference/" + row["name"], row["sha256"]))
        elif args.archives:
            jobs.add((args.archives.resolve(), row["name"], row["sha256"]))
    failures = []
    for root, relative, expected in sorted(jobs, key=lambda job: str(job[:2])):
        error = check_file(root, relative, expected)
        if error:
            failures.append(error)
    for error in failures:
        print(error)
    print(f"Preservation: {len(jobs) - len(failures)}/{len(jobs)} files match SHA-256.")
    print("Data checked:", args.include_data, "| Source ZIPs checked:", bool(args.archives))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
