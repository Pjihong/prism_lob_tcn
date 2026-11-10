"""Contracts for the public branch-local positional readout.

Check its parameter layout, temporal weighting, gradients, parameter count,
and strict state-dict round trip using generated inputs and parameters.
"""
from __future__ import annotations

import unittest

import torch

from lob_xai.config import Config
from lob_xai.models import BranchLocalPool1D, LOBBranchModel, make_pool
from lob_xai.paper_protocol import branch_variant_parameter_count, ours_parameter_count

SEQ_LEN = 100
BRANCH_KEYS = tuple(f"pool.pools.{i}.logits" for i in range(4))


def _build(n_classes: int = 3, input_norm: str = "identity", **kwargs):
    return LOBBranchModel(
        n_classes=n_classes, backbone="tcn",
        cnn_channels=(32, 32, 64), cnn_kernel=3, cnn_dropout=0.1,
        input_norm=input_norm, seq_len=SEQ_LEN, **kwargs,
    )


class TestPositionalReadout(unittest.TestCase):
    def test_state_keys_and_shapes(self):
        model = _build()
        self.assertEqual(model.pool_kind, "positional")
        state = model.state_dict()
        for key in BRANCH_KEYS:
            self.assertIn(key, state)
            self.assertEqual(tuple(state[key].shape), (SEQ_LEN,))

    def test_unsupported_readouts_are_rejected(self):
        for kind in ("avg", "attention", "max", "avg_last", "avg+last", "multi_scale"):
            with self.subTest(kind=kind):
                for construct in (
                    lambda: _build(pool=kind),
                    lambda: Config(pool=kind),
                    lambda: make_pool(kind, 64),
                    lambda: BranchLocalPool1D(kind, (64,) * 4),
                    lambda: branch_variant_parameter_count(3, pool=kind),
                ):
                    with self.assertRaisesRegex(ValueError, "only positional"):
                        construct()

    def test_zero_initialization_weights_each_position_equally(self):
        torch.manual_seed(0)
        model = _build().eval()
        x = torch.randn(4, SEQ_LEN, 40)
        with torch.no_grad():
            x_bft = model.input_norm(x).transpose(1, 2)
            branches = [
                cnn(x_bft.index_select(1, getattr(model, f"_idx_{name}")))
                for name, cnn in zip(model.branch_names, model.cnns)
            ]
            expected = model.fc(torch.cat([branch.mean(-1) for branch in branches], dim=1))
            torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)

    def test_temporal_weights_are_convex_at_supported_lengths(self):
        model = _build()
        with torch.no_grad():
            for branch_pool in model.pool.pools:
                branch_pool.logits.normal_(std=0.5)
                for length in (37, SEQ_LEN):
                    w = branch_pool.temporal_weights(length)
                    self.assertGreaterEqual(float(w.min()), 0.0)
                    self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
                    self.assertEqual(w.shape, (length,))

    def test_parameter_counts_match_instantiated_model(self):
        for n_classes in (2, 3):
            model = _build(n_classes=n_classes)
            actual = sum(p.numel() for p in model.parameters())
            self.assertEqual(actual, ours_parameter_count(n_classes))
            self.assertEqual(actual, branch_variant_parameter_count(n_classes))
        normalized = _build(tcn_norm="layer")
        self.assertEqual(
            sum(p.numel() for p in normalized.parameters()),
            branch_variant_parameter_count(3, tcn_norm="layer"),
        )

    def test_branch_local_readout_forbids_interaction(self):
        with self.assertRaisesRegex(ValueError, "interaction_channels=0"):
            _build(interaction_channels=64)

    def test_strict_state_dict_round_trip(self):
        torch.manual_seed(1)
        for norm in ("revin_c", "identity"):
            src = _build(input_norm=norm)
            with torch.no_grad():
                for p in src.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            dst = _build(input_norm=norm)
            dst.load_state_dict(src.state_dict(), strict=True)
            x = torch.randn(2, SEQ_LEN, 40)
            src.eval(), dst.eval()
            with torch.no_grad():
                torch.testing.assert_close(src(x), dst(x))

    def test_forward_backward_gradients_reach_readout(self):
        model = _build()
        x = torch.randn(2, SEQ_LEN, 40)
        loss = model(x).sum()
        loss.backward()
        for branch_pool in model.pool.pools:
            self.assertIsNotNone(branch_pool.logits.grad)
            self.assertTrue(torch.isfinite(branch_pool.logits.grad).all())


if __name__ == "__main__":
    unittest.main()
