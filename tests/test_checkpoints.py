"""The shared loader must reproduce predictions and reject incomplete weights."""
from pathlib import Path
import tempfile
import unittest

import torch

from lob_xai.checkpoints import load_micro_windows_in_order, load_ours_model
from lob_xai.models import LOBBranchModel


class CheckpointTests(unittest.TestCase):
    def test_shared_loader_round_trip_and_strict_loading(self):
        torch.manual_seed(7)
        model = LOBBranchModel(
            n_classes=3, backbone="tcn", pool="positional",
            cnn_channels=(4, 4), fc_hidden=(8,), seq_len=12,
        ).eval()
        record = {
            "config": {"task": "toxic", "backbone": "tcn", "pool": "positional",
                       "cnn_channels": [4, 4], "fc_hidden": [8], "seq_len": 12},
            "model_state": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(record, path)
            loaded, cfg, _ = load_ours_model(str(path))
            x = torch.randn(2, 12, 40)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x), rtol=0, atol=0)
            self.assertEqual(cfg["pool"], "positional")
            self.assertFalse(loaded.training)

            # Task and data identity checks must fail before opening a CSV.
            with self.assertRaisesRegex(ValueError, "checkpoint task"):
                load_micro_windows_in_order(str(path), "absent.csv", task="vacuum")
            with self.assertRaisesRegex(ValueError, "data identity"):
                load_micro_windows_in_order(str(path), "absent.csv")

            record["model_state"].pop(next(iter(record["model_state"])))
            torch.save(record, path)
            with self.assertRaisesRegex(RuntimeError, "architecture mismatch"):
                load_ours_model(str(path))


if __name__ == "__main__":
    unittest.main()
