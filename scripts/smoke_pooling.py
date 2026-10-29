#!/usr/bin/env python3
"""Fast architecture/gradient/state-dict smoke test for new pool modules."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lob_xai.models import LOBBranchModel


EXPECTED = {
    ("toxic", "multi_scale"): 142_051,
    ("toxic", "positional"): 142_439,
    ("vacuum", "multi_scale"): 141_902,
    ("vacuum", "positional"): 142_290,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    choices=["multi_scale", "positional"])
    args = ap.parse_args()

    for task, n_classes, input_norm in (
        ("toxic", 3, "revin_c"),
        ("vacuum", 2, "identity"),
    ):
        model = LOBBranchModel(
            n_classes=n_classes, backbone="tcn", pool=args.pool,
            cnn_channels=(32, 32, 64), cnn_kernel=3, cnn_dropout=0.1,
            fc_hidden=(64,), tcn_dilation_base=2, tcn_norm="none",
            input_norm=input_norm, interaction_channels=0, seq_len=100,
            pool_windows=(8, 32, 100),
        )
        params = sum(p.numel() for p in model.parameters())
        assert params == EXPECTED[(task, args.pool)], (task, params)
        assert model.branch_local_pool and len(model.pool.pools) == 4

        x = torch.randn(2, 100, 40, requires_grad=True)
        logits = model(x)
        assert tuple(logits.shape) == (2, n_classes)
        logits.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for pool in model.pool.pools:
            assert pool.logits.grad is not None
            weights = pool.temporal_weights(100).detach()
            assert torch.all(weights >= 0)
            assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

        clone = copy.deepcopy(model)
        clone.load_state_dict(model.state_dict(), strict=True)
        clone.eval()
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), clone(x), atol=1e-6)
        print(f"[ok] {task:6s} pool={args.pool:11s} params={params:,} "
              f"output={(2, n_classes)} strict_reload=yes")


if __name__ == "__main__":
    main()
