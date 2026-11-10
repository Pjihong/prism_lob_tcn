"""Validate explicit inputs for the standalone positional figure tools."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from lob_xai.counterfactual_test import build_ours
from lob_xai.models import LOBBranchModel
from lob_xai_viz.plot_positional_readout import _verify_config

ROOT = Path(__file__).resolve().parents[1]


class FigureInputTests(unittest.TestCase):
    def test_figure_commands_require_checkpoint_before_creating_output(self):
        for module in ("plot_alert_card", "plot_factorization"):
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "rendered"
                result = subprocess.run(
                    [sys.executable, "-m", f"lob_xai_viz.{module}",
                     "--out-dir", str(output)],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--ckpt is required", result.stderr)
                self.assertFalse(output.exists())

    def test_checkpoint_requires_explicit_positional_configuration(self):
        for pool in (None, "avg", "multi_scale"):
            with self.subTest(pool=pool), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                config = {"task": "toxic"}
                if pool is not None:
                    config["pool"] = pool
                # Empty state must never be interpreted as a supported model.
                torch.save({"config": config, "model_state": {}}, path)
                with self.assertRaisesRegex(ValueError, "explicitly declare"):
                    build_ours(path, torch.device("cpu"), "toxic")
                with self.assertRaisesRegex(ValueError, "expected 'positional'"):
                    _verify_config(config, "toxic", path)

    def test_positional_checkpoint_preserves_predictions(self):
        config = {
            "task": "toxic", "pool": "positional", "backbone": "tcn",
            "n_classes": 3, "cnn_channels": (2, 2, 2), "fc_hidden": (3,),
            "seq_len": 12,
        }
        model = LOBBranchModel(
            n_classes=3, pool="positional", cnn_channels=(2, 2, 2),
            fc_hidden=(3,), seq_len=12,
        ).eval()
        inputs = torch.randn(2, 12, 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": config, "model_state": model.state_dict()}, path)
            loaded, n_classes = build_ours(path, torch.device("cpu"), "toxic")
        self.assertEqual(n_classes, 3)
        with torch.no_grad():
            torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
