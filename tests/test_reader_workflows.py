"""Contracts for paper argument files and reader-facing documentation."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest

from scripts.check_docs import broken_links
from scripts.train_microstructure import build_parser
from scripts.collect_branch_dev_results import _signature

ROOT = Path(__file__).resolve().parents[1]


class ReaderWorkflowTests(unittest.TestCase):
    def test_paper_presets_select_architecture_and_task(self):
        for task, horizon in (("toxic", 10), ("vacuum", 30)):
            with self.subTest(task=task):
                args = build_parser().parse_args(
                    ["@" + str(ROOT / f"configs/paper/{task}.args"), "--seed", "3"]
                )
                self.assertEqual((args.task, args.k), (task, horizon))
                self.assertEqual((args.pool, args.input_norm), ("positional", "identity"))
                self.assertEqual(args.cnn_channels, [32, 32, 64])
                self.assertEqual((args.optimizer, args.scheduler), ("adamw", "cosine"))
                self.assertEqual((args.epochs, args.batch_size, args.early_stop_patience), (100, 256, 10))
                self.assertEqual(args.early_stop_metric, "val_f1")
                self.assertTrue(args.enforce_paper_split)
                self.assertTrue(args.dev_only)
                self.assertEqual(args.seed, 3)
        self.assertEqual(build_parser().parse_args(["--task", "toxic"]).pool, "positional")

    def test_cli_rejects_unsupported_model_settings(self):
        for args in (["--pool", "avg"], ["--interaction-channels", "64"],
                     ["--pool-windows", "10", "30", "100"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(["--task", "toxic", *args])
                self.assertEqual(error.exception.code, 2)

    def test_collector_requires_explicit_positional_configuration(self):
        for config in ({}, {"pool": "avg"}, {"pool": "positional", "interaction_channels": 64}):
            with self.assertRaisesRegex(ValueError, "positional"):
                _signature(config)
        self.assertIn("positional", _signature({"pool": "positional"}))

    def test_document_checker_catches_missing_targets_and_ignores_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "exists.md").write_text("ok", encoding="utf-8")
            doc = root / "README.md"
            doc.write_text(
                "[ok](exists.md#section)\n[bad](missing.md)\n"
                "[external](https://example.com)\n```md\n[example](not-real.md)\n```\n",
                encoding="utf-8",
            )
            errors = broken_links(doc)
            self.assertEqual(len(errors), 1)
            self.assertIn("missing.md", errors[0])


if __name__ == "__main__":
    unittest.main()
