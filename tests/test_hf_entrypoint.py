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

    def test_space_metadata_pins_python_310(self):
        readme = (REPO_ROOT / "README_HF.md").read_text(encoding="utf-8")

        self.assertIn('python_version: "3.10"', readme)

    def test_space_requirements_keep_gradio_5_audio_separator_pins(self):
        requirements = (REPO_ROOT / "requirements_hf.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn("gradio==5.49.1", requirements)
        self.assertIn("fastapi>=0.115,<1", requirements)
        self.assertIn("pydantic>=2,<3", requirements)
        self.assertIn("jinja2>=3.1,<4", requirements)
        self.assertIn("pandas>=2,<3", requirements)
        self.assertIn("numpy>=2,<3", requirements)
        self.assertIn("audio-separator==0.44.1", requirements)
        self.assertIn("huggingface_hub>=0.19.0,<1.0", requirements)


if __name__ == "__main__":
    unittest.main()
