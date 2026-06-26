import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class HuggingFaceEntrypointTests(unittest.TestCase):
    def test_entrypoint_calls_ui_launch_with_supported_keywords(self):
        tree = ast.parse((REPO_ROOT / "app.py").read_text(encoding="utf-8"))
        launch_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "launch"
        ]

        self.assertEqual(len(launch_calls), 1)
        self.assertEqual(
            {keyword.arg for keyword in launch_calls[0].keywords},
            {"host", "port", "share"},
        )


if __name__ == "__main__":
    unittest.main()
