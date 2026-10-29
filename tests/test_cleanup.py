"""Checks for the compatibility entry point and preservation verifier."""
import hashlib
import importlib.util
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preservation", ROOT / "scripts/verify_preservation.py")
preservation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preservation)


class PreservationTests(unittest.TestCase):
    def test_hash_missing_changed_and_escaping_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.txt"
            path.write_bytes(b"original")
            expected = hashlib.sha256(b"original").hexdigest()
            self.assertIsNone(preservation.check_file(root, "source.txt", expected))
            path.write_bytes(b"changed")
            self.assertIn("Changed", preservation.check_file(root, "source.txt", expected))
            self.assertIn("Missing", preservation.check_file(root, "missing.txt", expected))
            self.assertIn("escapes", preservation.check_file(root, "../source.txt", expected))

    def test_old_entrypoint_exports_the_same_module(self):
        import counterfactual_test as old
        from lob_xai import counterfactual_test as canonical
        self.assertIs(old, canonical)
        self.assertIs(old._interp_to_T, canonical._interp_to_T)

    def test_preservation_distinguishes_original_refactored_and_removed(self):
        row = {"disposition": "preserved", "destination": "training.py", "sha256": "original"}
        self.assertEqual(preservation.expected_files(row), {"training.py": "original"})
        files = {"training.py": "current", "metrics.py": "extracted"}
        row.update(disposition="refactored", refactored_files=files)
        self.assertEqual(preservation.expected_files(row), files)
        row["disposition"] = "removed_by_request"
        self.assertEqual(preservation.expected_files(row), {})
        row["disposition"] = "local_data"
        self.assertEqual(preservation.expected_files(row), {})
        self.assertEqual(
            preservation.expected_files(row, include_data=True), {"training.py": "original"},
        )
        row["disposition"] = "refactored"
        del row["refactored_files"]
        with self.assertRaises(KeyError):
            preservation.expected_files(row)

    def test_removed_manuscript_is_not_required_but_unmarked_missing_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs/PRESERVATION_MANIFEST.json").write_text("[]", encoding="utf-8")
            source = {"name": "draft.pdf", "sha256": "unused",
                      "disposition": "removed_by_request"}
            source_path = root / "docs/SOURCE_FILES.json"
            source_path.write_text(json.dumps([source]), encoding="utf-8")
            with patch.object(preservation, "ROOT", root), patch("sys.argv", ["verify"]):
                preservation.main()
                del source["disposition"]
                source_path.write_text(json.dumps([source]), encoding="utf-8")
                with self.assertRaises(SystemExit) as caught:
                    preservation.main()
                self.assertEqual(caught.exception.code, 1)

    def test_run_manifest_covers_current_figure_source(self):
        from scripts.make_run_manifest import source_tree_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "lob_xai_viz/plot_cf_mirror.py"
            script.parent.mkdir(parents=True)
            script.write_text("value = 1\n", encoding="utf-8")
            before = source_tree_sha256(root)
            script.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(before, source_tree_sha256(root))


if __name__ == "__main__":
    unittest.main()
