#!/usr/bin/env python3
"""Static audit of the manuscript-to-code contract (no torch/data required)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob_xai.paper_protocol import (COUNTERFACTUAL, DATA, TRAINING,
                                    bin_ctabl_parameter_count,
                                    deeplob_parameter_count,
                                    ours_parameter_count,
                                    tlob_parameter_count)


def issue(items, severity, code, message, **context):
    items.append({"severity": severity, "code": code, "message": message, **context})


def manuscript_audit(path: Path, items: list[dict]):
    text = path.read_text(encoding="utf-8")
    definitions = set(re.findall(r"\\newcommand\{\\([A-Za-z@]+)\}", text))
    families = re.findall(r"\\((?:Case|Ours|DL|AxB)[A-Za-z0-9@]+)", text)
    for name in sorted(set(families) - definitions):
        issue(items, "error", "undefined-macro", f"undefined manuscript macro \\{name}")

    unresolved = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if re.search(r"TODO|TBD|원문 대조|재구성본|확인 권장|확인\]", line, re.I):
            unresolved.append((lineno, line.strip()[:180]))
    for lineno, excerpt in unresolved:
        issue(items, "warning", "unresolved-draft-marker",
              f"unresolved draft marker at line {lineno}", excerpt=excerpt)

    figures = re.findall(r"\\safeincludegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    search_roots = (path.parent, ROOT)
    for rel in figures:
        if not any((base / rel).is_file() for base in search_roots):
            issue(items, "error", "missing-figure", f"missing manuscript figure: {rel}")

    if "0.179/0.186" in text and "& $0.109" in text and "& $0.116" in text:
        issue(items, "error", "detection-number-conflict",
              "draft contains both disputed ours MCC values 0.179/0.186 and table values 0.109/0.116")

    if "$K$개" in text and not re.search(
            r"(?:K\s*\{?=\}?\s*50|\\newcommand\{\\(?:KNN|NeighborK)\}\{50\})", text):
        issue(items, "error", "undefined-neighbor-k",
              f"counterfactual protocol uses K neighbours but never fixes K={COUNTERFACTUAL.neighbor_candidates}")

    if ("정규화 통계 역시 Train 구간에서만 추정" in text
            and "FI-2010이 z-score 정규화" in text):
        issue(items, "error", "normalization-provenance-conflict",
              "draft claims both pre-normalized FI-2010 input and train-only fitted normalization; "
              "the supplied loader performs no normalization fit, so one provenance statement must change")

    if "이상의 연속된 변화를 독성" in text:
        issue(items, "warning", "toxic-label-wording",
              "narrative says consecutive changes, while the equation/code compare only endpoints t and t+k")

    if ("T=100$ 윈도우 전체의 시간적 문맥" in text
            and TRAINING.branch_temporal_receptive_field < DATA.seq_len):
        issue(items, "warning", "tcn-receptive-field-wording",
              "the classifier aggregates all 100 positions through GAP, but each final "
              f"branch timestep has local receptive field "
              f"{TRAINING.branch_temporal_receptive_field}, not {DATA.seq_len}; "
              "word the architectural claim precisely")

    if ("DeepLOB with Grad-CAM" in text
            and "분포 보존 반사실 요약(HiResCAM 기준" in text):
        issue(items, "error", "deeplob-attribution-conflict",
              "abstract says the DeepLOB gate uses Grad-CAM, but the headline counterfactual table says HiResCAM")

    if "전체 윈도우를 이웃 대체로 지운 상태" in text:
        issue(items, "warning", "insertion-base-underspecified",
              "insertion has empty preserved context; state explicitly that rho=0 uses unconditional full-window draws from the training pool")

    if ("seed & take-rate & ours branch acc & ours time acc & DeepLOB time acc" in text
            and "DeepLOB take-rate" not in text):
        issue(items, "warning", "localization-cohort-ambiguity",
              "the localization table reports one take-rate but compares two models; "
              "state whether each model is scored on its own successful injections, "
              "a shared intersection, or an ours-selected cohort, and report both take-rates")

    if "결정 생존 곡선" in text and "t^\\ast=\\arg\\max" in text:
        issue(items, "error", "non-nested-survival-interpretation",
              "argmax spans are reselected independently at each rho and need not be nested; "
              "a monotone survival/first-failure interpretation is unjustified unless a nested deletion path is defined")

    gate = re.search(r"\\begin\{defn\}\[충실성 게이트\].*?\\end\{defn\}", text, re.S)
    if gate and not re.search(r"(?:ratio|배|margin|차이)\s*(?:[>$\\]|\d)", gate.group(0), re.I):
        issue(items, "error", "undefined-g2-threshold",
              "G2 says 'material/multiplicative difference' but gives no numeric threshold")

    expected_tokens = {
        r"\newcommand{\SeqLen}{100}": DATA.seq_len,
        r"\newcommand{\NFeat}{40}": DATA.n_features,
        r"\newcommand{\Horizon}{10}": DATA.toxic_horizon,
        r"\newcommand{\PoolSize}{30{,}000}": COUNTERFACTUAL.pool_size,
        r"\newcommand{\NWin}{400}": COUNTERFACTUAL.n_windows,
        r"\newcommand{\MCM}{10}": COUNTERFACTUAL.mc_samples,
        r"\newcommand{\Seeds}{3}": len(TRAINING.seeds),
    }
    for token, _ in expected_tokens.items():
        if token not in text:
            issue(items, "error", "protocol-token-mismatch",
                  f"canonical protocol token not found verbatim: {token}")


def repository_audit(items: list[dict]):
    required_baselines = {
        "deeplob": (ROOT / "models" / "deeplob.py", "build_deeplob"),
        "tlob": (ROOT / "models" / "tlob.py", "build_tlob"),
        "bin_ctabl": (ROOT / "models" / "bin_ctabl.py", "build_bin_ctabl"),
    }
    baseline_hashes = {}
    for name, (path, builder_name) in required_baselines.items():
        if not path.is_file():
            issue(items, "error", "missing-baseline-module",
                  f"training imports a file that is absent: {path.relative_to(ROOT)}")
            continue
        source = path.read_text(encoding="utf-8")
        baseline_hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source, filename=str(path))
        builder = next(
            (node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == builder_name),
            None,
        )
        if builder is None:
            issue(items, "error", "missing-baseline-builder",
                  f"{path.relative_to(ROOT)} does not expose {builder_name}()")
        elif "n_classes" not in {arg.arg for arg in builder.args.args}:
            issue(items, "error", "fixed-baseline-head",
                  f"{builder_name}() has no explicit n_classes argument")
        if re.search(r"\bnn\.Softmax\s*\(", source):
            issue(items, "error", "baseline-returns-probabilities",
                  f"{path.relative_to(ROOT)} contains nn.Softmax despite CE-logit contract")
    if len(baseline_hashes) == len(required_baselines):
        issue(items, "info", "baseline-sources",
              "all three baseline modules are present and expose explicit class heads",
              sha256=baseline_hashes,
              expected_parameter_counts={
                  "deeplob_3class": deeplob_parameter_count(3),
                  "tlob_3class": tlob_parameter_count(3),
                  "bin_ctabl_3class": bin_ctabl_parameter_count(3),
              })

    expected = {
        "seq_len": DATA.seq_len,
        "n_features": DATA.n_features,
        "toxic_horizon": DATA.toxic_horizon,
        "vacuum_horizon": DATA.vacuum_horizon,
        "batch_size": TRAINING.batch_size,
        "epochs": TRAINING.max_epochs,
        "patience": TRAINING.early_stop_patience,
        "branch_temporal_receptive_field": TRAINING.branch_temporal_receptive_field,
        "ours_params_toxic": ours_parameter_count(3),
        "ours_params_vacuum": ours_parameter_count(2),
        "deeplob_params_toxic": deeplob_parameter_count(3),
        "tlob_params_toxic": tlob_parameter_count(3),
        "bin_ctabl_params_toxic": bin_ctabl_parameter_count(3),
        "pool_size": COUNTERFACTUAL.pool_size,
        "mc": COUNTERFACTUAL.mc_samples,
        "neighbors_k": COUNTERFACTUAL.neighbor_candidates,
        "selection": "top-confidence correct target alerts",
        "donor_subset": "training partition only (validation excluded)",
    }
    issue(items, "info", "canonical-protocol", "loaded canonical protocol", values=expected)

    junk = []
    for pattern in ("**/.ipynb_checkpoints", "**/__pycache__", "**/*.bak*", "**/*.pyc"):
        junk.extend(ROOT.glob(pattern))
    if junk:
        issue(items, "warning", "archive-junk",
              f"source tree contains {len(set(junk))} cache/backup artefacts; exclude them from release")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", type=Path)
    ap.add_argument("--json", dest="json_path", type=Path)
    ap.add_argument("--allow-errors", action="store_true")
    args = ap.parse_args()

    items: list[dict] = []
    repository_audit(items)
    if args.manuscript:
        manuscript_audit(args.manuscript.resolve(), items)

    counts = {s: sum(x["severity"] == s for x in items)
              for s in ("error", "warning", "info")}
    report = {"counts": counts, "issues": items}
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in items:
        print(f"[{item['severity'].upper():7s}] {item['code']}: {item['message']}")
    print(f"\nerrors={counts['error']} warnings={counts['warning']} info={counts['info']}")
    if counts["error"] and not args.allow_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
