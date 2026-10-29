"""Protect metrics, paired interventions, CLI guards, and saved curve schemas."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lob_xai import counterfactual_test as cf
from lob_xai import metrics, training


class MetricsCompatibilityTests(unittest.TestCase):
    def test_old_metric_imports_remain_available(self):
        for name in (
            "_accuracy", "_f1_macro", "_mcc_multiclass",
            "_per_class_classification_metrics", "_correct_transaction_probability",
            "_pr_auc_binary", "_roc_auc_binary", "_lift_at_k", "_brier_binary",
            "_expected_cost", "_per_class_pr_auc", "_per_class_roc_auc",
            "find_threshold_high_precision", "find_threshold_youden", "find_threshold_cost",
        ):
            self.assertIs(getattr(training, name), getattr(metrics, name))

    def test_ties_and_missing_classes_keep_original_metric_conventions(self):
        scores = np.array([0.7, 0.7, 0.1, 0.1])
        labels = np.array([1, 0, 1, 0])
        self.assertEqual(metrics._roc_auc_binary(scores, labels), 0.5)
        self.assertAlmostEqual(metrics._pr_auc_binary(scores, labels), 5 / 6)
        for labels in (np.zeros(4), np.ones(4)):
            self.assertTrue(np.isnan(metrics._pr_auc_binary(scores, labels)))
            self.assertTrue(np.isnan(metrics._roc_auc_binary(scores, labels)))

    def test_empty_evaluation_still_reports_missing_results(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 4), torch.empty(0, dtype=torch.long)))
        result = training.evaluate(
            torch.nn.Linear(4, 2), loader, torch.device("cpu"),
            n_classes=2, flat_class=0, return_predictions=True,
        )
        self.assertTrue(np.isnan(result["mcc"]))
        self.assertEqual(result["n_actions"], 0)
        self.assertEqual(result["probs"].shape, (0, 2))


class CounterfactualWorkflowTests(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            model="ours", task="vacuum", method="hirescam", direction="both",
            pred_based=False, n_windows=2, mc=2, seed=17, neighbors_k=4,
            checkpoint_sha256="fixture", out=None,
        )

    def make_results(self):
        return {1: {
            "cam": np.array([[0.9, 0.4, 0.1], [0.8, 0.3, 0.1]]),
            "random": np.array([[0.9, 0.8, 0.7], [0.8, 0.7, 0.6]]),
            "anti": np.array([[0.9, 0.9, 0.9], [0.8, 0.8, 0.8]]),
            "_indices": np.array([7, 3]), "_confidence": np.array([0.9, 0.8]),
            "_flip": {m: np.array([True, False]) for m in ("cam", "random", "anti")},
            "_recovered": {m: np.array([False, True]) for m in ("cam", "random", "anti")},
        }}

    def test_deletion_and_insertion_use_opposite_auc_signs(self):
        args = self.make_args()
        results = self.make_results()
        q_grid = np.array([0.0, 0.25, 0.5])
        with redirect_stdout(io.StringIO()):
            deletion, del_windows = cf._summarize_deletion(
                args, cf.TASK_CLASSES["vacuum"], q_grid, results, 24,
            )
            insertion, ins_windows = cf._summarize_insertion(
                args, cf.TASK_CLASSES["vacuum"], q_grid, results, 24,
            )
        self.assertGreater(deletion[0]["dAUC_mean"], 0)
        self.assertEqual(deletion[0]["dAUC_mean"], -insertion[0]["ins_dAUC_mean"])
        self.assertEqual([r["window_index"] for r in del_windows], [7, 3])
        self.assertEqual([r["window_index"] for r in ins_windows], [7, 3])
        self.assertEqual(insertion[0]["baseline"], "unconditional_full_window_train_donor")
        self.assertEqual(deletion[0]["flip_cam"], 0.5)
        self.assertEqual(insertion[0]["recovery_cam"], 0.5)

    def test_modes_keep_sampling_order_and_share_each_insertion_baseline(self):
        args = self.make_args()
        sampler = Mock()
        baselines = [np.zeros((2, 100, 40)), np.ones((2, 100, 40))]
        sampler.sample.side_effect = baselines
        provider = Mock()
        provider.saliency.return_value = np.ones((1, 100))
        calls = []

        def deletion(*a, **kw):
            calls.append(("deletion", a[7], None))
            return np.array([0.9, 0.4]), True

        def insertion(*a, **kw):
            calls.append(("insertion", a[7], kw["base_variants"]))
            return np.array([0.1, 0.8]), True

        with patch.object(cf, "_make_provider", return_value=provider), \
             patch.object(cf, "deletion_curve", side_effect=deletion), \
             patch.object(cf, "insertion_curve", side_effect=insertion), \
             redirect_stdout(io.StringIO()):
            deleted, inserted = cf._collect_curves(
                args, Mock(), sampler, "cpu", np.array([0.0, 0.5]),
                np.zeros((2, 100, 40), dtype=np.float32), np.ones(2, dtype=int),
                np.array([[0.2, 0.8], [0.1, 0.9]]),
            )
        expected_order = [
            (direction, mode) for direction in ("deletion", "insertion")
            for mode in ("cam", "random", "anti")
        ] * 2
        self.assertEqual([(d, m) for d, m, _ in calls], expected_order)
        self.assertEqual(sampler.sample.call_count, 2)
        for i, baseline in enumerate(baselines):
            for _, _, actual in calls[i * 6 + 3:i * 6 + 6]:
                self.assertIs(actual, baseline)
        np.testing.assert_array_equal(deleted[1]["_indices"], [1, 0])
        np.testing.assert_array_equal(inserted[1]["_indices"], [1, 0])

    def test_cli_guards_keep_invalid_runs_out(self):
        invalid_options = [
            ["--model", "ours"],
            ["--model", "ours", "--demo", "--pool-size", "0"],
            ["--model", "ours", "--demo", "--q-max", "2"],
            ["--model", "ours", "--ckpt", "model.pt", "--train-csv", "same.csv",
             "--test-csv", "same.csv"],
        ]
        for options in invalid_options:
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    cf._parse_args(options)
                self.assertEqual(exc.exception.code, 2)

    def test_empirical_mode_requires_data_provenance(self):
        parser, args, _ = cf._parse_args(["--model", "ours", "--ckpt", "model.pt"])
        with patch.object(cf.torch, "load", return_value={}), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                cf._load_model_and_data(args, parser, torch.device("cpu"))
            self.assertEqual(exc.exception.code, 2)

    def test_saved_curves_retain_names_dtypes_and_cohort_order(self):
        args = self.make_args()
        results = self.make_results()
        q_grid = np.array([0.0, 0.25, 0.5])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args.out = str(Path(tmp) / "curves.png")
            rows, windows = cf._summarize_deletion(
                args, cf.TASK_CLASSES["vacuum"], q_grid, results, 24,
            )
            cf._write_curve_outputs(
                args, cf.TASK_CLASSES["vacuum"], q_grid, results, rows, windows, "deletion",
            )
            with np.load(Path(tmp) / "curves.curves.npz") as saved:
                self.assertEqual(saved["direction"].item(), "deletion")
                self.assertEqual(saved["vacuum__cam"].dtype, np.float32)
                self.assertEqual(saved["vacuum__indices"].dtype, np.int64)
                np.testing.assert_array_equal(saved["vacuum__indices"], [7, 3])
                np.testing.assert_array_equal(
                    saved["vacuum__cam"], results[1]["cam"].astype(np.float32),
                )
            with (Path(tmp) / "curves.windows.csv").open(newline="") as stream:
                saved_rows = list(csv.DictReader(stream))
            self.assertEqual([r["window_index"] for r in saved_rows], ["7", "3"])


if __name__ == "__main__":
    unittest.main()
