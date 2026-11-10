"""Keep training and saved-result plotting usable without the removed run logger."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lob_xai.training import train
from lob_xai_viz.plot_branch_importance import load_cam_time
from scripts import train_microstructure


class PublicCodeScopeTests(unittest.TestCase):
    def test_training_returns_history_without_writing_run_files(self):
        torch.manual_seed(3)
        model = torch.nn.Linear(4, 2)
        initial = model.weight.detach().clone()
        inputs = torch.randn(8, 4)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"LOB_XAI_ROOT": directory}), \
             redirect_stdout(io.StringIO()):
            history = train(
                model, loader, loader, epochs=1, n_classes=2,
                flat_class=0, use_class_weights=False,
            )
            self.assertEqual(history["epoch"], [1])
            self.assertTrue(np.isfinite(history["val_loss"][0]))
            self.assertFalse(torch.equal(initial, model.weight))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_refuses_to_reuse_an_existing_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "existing-result.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with patch.object(sys, "argv", [
                "train_microstructure.py", "--task", "toxic",
                "--output-dir", directory,
            ]), patch.object(train_microstructure, "file_identity", return_value={}), \
                 patch.object(train_microstructure, "build_microstructure_dataloaders") as build, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    train_microstructure.main()
                build.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_saved_cam_cache_is_loaded_without_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = np.arange(12, dtype=np.float32).reshape(4, 3)
            np.savez(
                Path(directory) / "cam_time_toxic_k10_seed1.npz",
                cam_time=expected,
                branch_names=np.array(["ask_price", "ask_volume", "bid_price", "bid_volume"]),
                n_cohort=7, checkpoint="private/model.pt", checkpoint_sha256="fixture",
            )
            args = SimpleNamespace(out_dir=directory, task="toxic", k=10)
            result = load_cam_time(args, seed=1)
            np.testing.assert_array_equal(result["cam_time"], expected)
            self.assertEqual(result["n_cohort"], 7)
            self.assertEqual(result["checkpoint_sha256"], "fixture")

    def test_missing_cam_cache_fails_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out_dir=directory, task="toxic", k=10)
            with self.assertRaisesRegex(SystemExit, "required saved CAM cache is missing"):
                load_cam_time(args, seed=1)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
