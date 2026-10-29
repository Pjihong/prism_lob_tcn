#!/usr/bin/env python3
"""Runtime smoke test for the three manuscript baseline architectures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lob_xai.paper_protocol import baseline_parameter_count
from models.bin_ctabl import build_bin_ctabl
from models.deeplob import build_deeplob
from models.tlob import build_tlob


def build(name: str, n_classes: int):
    if name == "deeplob":
        return build_deeplob(n_classes=n_classes, n_features=40), {}
    if name == "tlob":
        kwargs = dict(
            seq_len=100, n_features=40, hidden_dim=128,
            num_layers=4, num_heads=1,
        )
        model = build_tlob(
            seq_size=kwargs["seq_len"], num_features=kwargs["n_features"],
            hidden_dim=kwargs["hidden_dim"], num_layers=kwargs["num_layers"],
            num_heads=kwargs["num_heads"], n_classes=n_classes,
        )
        return model, kwargs
    if name == "bin_ctabl":
        kwargs = dict(seq_len=100, n_features=40)
        model = build_bin_ctabl(
            seq_len=kwargs["seq_len"], num_features=kwargs["n_features"],
            n_classes=n_classes,
        )
        return model, kwargs
    raise ValueError(name)


def verify(name: str, n_classes: int, device: torch.device) -> dict:
    torch.manual_seed(1234)
    model, count_kwargs = build(name, n_classes)
    model = model.to(device)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = baseline_parameter_count(name, n_classes, **count_kwargs)
    if actual != expected:
        raise RuntimeError(f"{name}: parameter count {actual} != {expected}")

    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for batch in (1, 3):
            logits = model(torch.randn(batch, 100, 40, device=device))
            if tuple(logits.shape) != (batch, n_classes):
                raise RuntimeError(
                    f"{name}: output {tuple(logits.shape)} != {(batch, n_classes)}"
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{name}: non-finite inference logits")
    changed = [
        key for key, value in model.state_dict().items()
        if not torch.equal(value, before[key])
    ]
    if changed:
        raise RuntimeError(f"{name}: eval forward mutated state: {changed}")

    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(torch.randn(3, 100, 40, device=device))
    loss = logits.square().mean()
    loss.backward()
    bad_grad = [
        key for key, value in model.named_parameters()
        if value.grad is not None and not torch.isfinite(value.grad).all()
    ]
    if bad_grad:
        raise RuntimeError(f"{name}: non-finite gradients: {bad_grad}")
    return {
        "model": name,
        "n_classes": n_classes,
        "parameters": actual,
        "batch1_and_batch3_shape_ok": True,
        "eval_state_immutable": True,
        "finite_gradients": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=["deeplob", "tlob", "bin_ctabl"],
        choices=["deeplob", "tlob", "bin_ctabl"],
    )
    ap.add_argument("--classes", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    results = [
        verify(name, n_classes, device)
        for name in args.models for n_classes in args.classes
    ]
    print(json.dumps({"verified": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
