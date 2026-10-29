"""Contract tests for the branch-local pooling readouts.

These tests pin down the properties the frozen checkpoints rely on:

- state-dict key layout ``pool.pools.{i}.logits`` with the documented shapes;
- exact-GAP equivalence of ``positional`` at zero initialization;
- convexity of the effective temporal weights (non-negative, sum to one);
- agreement between the analytic parameter count in ``paper_protocol`` and
  the instantiated model;
- the branch-local/interaction exclusivity rule;
- a strict state-dict round trip after rebuild.

If any of these fail after a refactor, the frozen checkpoints can no longer
be trusted to load into the current code.
"""
from __future__ import annotations

import unittest

import torch

from lob_xai.models import LOBBranchModel
from lob_xai.paper_protocol import branch_variant_parameter_count

SEQ_LEN = 100
BRANCH_KEYS = tuple(f"pool.pools.{i}.logits" for i in range(4))


def _build(pool: str, n_classes: int = 3, input_norm: str = "identity"):
    return LOBBranchModel(
        n_classes=n_classes, backbone="tcn", pool=pool,
        cnn_channels=(32, 32, 64), cnn_kernel=3, cnn_dropout=0.1,
        input_norm=input_norm, seq_len=SEQ_LEN,
    )


class TestBranchLocalPooling(unittest.TestCase):
    def test_positional_state_keys_and_shapes(self):
        state = _build("positional").state_dict()
        for key in BRANCH_KEYS:
            self.assertIn(key, state)
            self.assertEqual(tuple(state[key].shape), (SEQ_LEN,))

    def test_multi_scale_state_keys_and_shapes(self):
        state = _build("multi_scale").state_dict()
        for key in BRANCH_KEYS:
            self.assertIn(key, state)
            self.assertEqual(tuple(state[key].shape), (3,))

    def test_positional_zero_init_equals_gap(self):
        torch.manual_seed(0)
        pos = _build("positional").eval()
        gap = _build("avg").eval()
        gap.load_state_dict(
            {k: v for k, v in pos.state_dict().items()
             if k in gap.state_dict()},
            strict=True,
        )
        x = torch.randn(4, SEQ_LEN, 40)
        with torch.no_grad():
            torch.testing.assert_close(pos(x), gap(x), rtol=1e-5, atol=1e-5)

    def test_temporal_weights_are_convex(self):
        for pool in ("positional", "multi_scale"):
            model = _build(pool)
            with torch.no_grad():
                for branch_pool in model.pool.pools:
                    branch_pool.logits.normal_(std=0.5)
                    w = branch_pool.temporal_weights(SEQ_LEN)
                    self.assertGreaterEqual(float(w.min()), 0.0)
                    self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
                    self.assertEqual(w.shape, (SEQ_LEN,))

    def test_parameter_counts_match_protocol(self):
        for pool, n_classes in (("positional", 3), ("positional", 2),
                                ("multi_scale", 3), ("multi_scale", 2)):
            model = _build(pool, n_classes=n_classes)
            actual = sum(p.numel() for p in model.parameters())
            analytic = branch_variant_parameter_count(n_classes, pool=pool)
            self.assertEqual(actual, analytic,
                             f"{pool}/{n_classes}: {actual} != {analytic}")

    def test_branch_local_pool_forbids_interaction(self):
        with self.assertRaises(ValueError):
            LOBBranchModel(
                n_classes=3, backbone="tcn", pool="positional",
                cnn_channels=(32, 32, 64), interaction_channels=64,
                seq_len=SEQ_LEN,
            )

    def test_strict_state_dict_round_trip(self):
        torch.manual_seed(1)
        for pool, norm in (("positional", "revin_c"),
                           ("positional", "identity"),
                           ("multi_scale", "revin_c")):
            src = _build(pool, input_norm=norm)
            with torch.no_grad():
                for p in src.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            dst = _build(pool, input_norm=norm)
            dst.load_state_dict(src.state_dict(), strict=True)
            x = torch.randn(2, SEQ_LEN, 40)
            src.eval(), dst.eval()
            with torch.no_grad():
                torch.testing.assert_close(src(x), dst(x))

    def test_forward_backward_gradients_reach_pool(self):
        model = _build("positional")
        x = torch.randn(2, SEQ_LEN, 40)
        loss = model(x).sum()
        loss.backward()
        for branch_pool in model.pool.pools:
            self.assertIsNotNone(branch_pool.logits.grad)
            self.assertTrue(torch.isfinite(branch_pool.logits.grad).all())


if __name__ == "__main__":
    unittest.main()
