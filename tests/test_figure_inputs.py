"""Fail clearly before creating outputs when the optional CF inputs are absent."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FigureInputTests(unittest.TestCase):
    def test_cf_requires_original_inputs_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lob_xai").mkdir()
            output = root / "rendered"
            result = subprocess.run(
                [sys.executable, "-m", "lob_xai_viz.render",
                 "--root", str(root), "--only", "cf", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PNG-only figures folder", result.stderr)
            self.assertIn("reference/input_manifest.json", result.stderr)
            self.assertFalse(output.exists())

    def test_cards_require_checkpoint_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lob_xai").mkdir()
            output = root / "rendered"
            result = subprocess.run(
                [sys.executable, "-m", "lob_xai_viz.render",
                 "--root", str(root), "--only", "cards", "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing input:", result.stderr)
            self.assertIn("model.pt", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
