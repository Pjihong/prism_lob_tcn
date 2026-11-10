"""Checks for the retained compatibility entry point and new-run source identity."""
from pathlib import Path
import tempfile
import unittest


class PublicWorkflowTests(unittest.TestCase):
    def test_old_entrypoint_exports_the_same_module(self):
        import counterfactual_test as old
        from lob_xai import counterfactual_test as canonical
        self.assertIs(old, canonical)
        self.assertIs(old._interp_to_T, canonical._interp_to_T)

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
