import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BaselineSourceContractTests(unittest.TestCase):
    def test_builders_have_explicit_class_head(self):
        for filename, builder_name in (
            ("deeplob.py", "build_deeplob"),
            ("tlob.py", "build_tlob"),
            ("bin_ctabl.py", "build_bin_ctabl"),
        ):
            path = ROOT / "models" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"))
            builder = next(
                node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == builder_name
            )
            args = {arg.arg for arg in builder.args.args}
            self.assertIn("n_classes", args, filename)

    def test_models_return_logits_and_tlob_is_self_contained(self):
        sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in (ROOT / "models").glob("*.py")
        }
        for filename, source in sources.items():
            self.assertNotIn("nn.Softmax(", source, filename)
        tlob_tree = ast.parse(sources["tlob.py"])
        imports = {
            node.module for node in ast.walk(tlob_tree)
            if isinstance(node, ast.ImportFrom)
        }
        imports.update(
            alias.name for node in ast.walk(tlob_tree)
            if isinstance(node, ast.Import) for alias in node.names
        )
        self.assertNotIn("einops", imports)

    def test_no_parameter_data_mutation_in_forward(self):
        source = (ROOT / "models" / "bin_ctabl.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        model_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BiN_CTABL"
        )
        forward = next(
            node for node in model_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        attrs = [node.attr for node in ast.walk(forward) if isinstance(node, ast.Attribute)]
        self.assertNotIn("data", attrs)


if __name__ == "__main__":
    unittest.main()
