"""Check local file links in current reader-facing Markdown, without network access."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
# Historical manuscripts/logs quote obsolete paths; they are explicitly archival.
CURRENT = (
    "README.md", "README.ko.md",
    "docs/README.md", "docs/REPRODUCE.md", "docs/DATA.md",
    "docs/PAPER_MAP.md", "docs/KNOWN_ISSUES.md",
    "configs/paper/README.md", "figures/README.md", "scripts/README.md",
    "legacy/README.md", "lob_xai_viz/README.md",
)


def broken_links(path):
    """Check inline relative link targets; heading anchors and remote URLs are excluded."""
    content = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    errors = []
    for match in re.finditer(r"\]\(([^)\n]+)\)", content):
        target = match.group(1).strip().strip("<>")
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        relative = unquote(url.path)
        if not (path.parent / relative).exists():
            errors.append(f"{path}: missing link target {relative}")
    return errors


def main():
    errors = []
    for name in CURRENT:
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing current document: {name}")
        else:
            errors.extend(broken_links(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Documentation: {len(CURRENT)} current files; local file links pass.")
    print("Remote URLs, heading anchors and archival documents are not checked.")


if __name__ == "__main__":
    main()
