import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PlatformSupportContractTests(unittest.TestCase):
    def test_official_adapter_forwards_selected_device(self):
        source = (REPO_ROOT / "infer" / "official_adapter.py").read_text(
            encoding="utf-8-sig"
        )
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "convert_vocals_official_upstream"
        )
        self.assertIn("device", [argument.arg for argument in function.args.args])
        function_source = ast.get_source_segment(source, function) or ""
        self.assertIn('"--device"', function_source)
        self.assertIn("str(device)", function_source)

    def test_official_runner_overrides_vendored_auto_device(self):
        source = (REPO_ROOT / "infer" / "official_upstream_runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument("--device", default="auto")', source)
        self.assertIn("selected_device = get_device(args.device)", source)
        self.assertIn("config.device = selected_device", source)
        self.assertNotIn("config.is_half =", source)

    def test_cover_pipeline_forwards_device_on_both_official_routes(self):
        source = (REPO_ROOT / "infer" / "cover_pipeline.py").read_text(
            encoding="utf-8-sig"
        )
        tree = ast.parse(source)
        pipeline_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CoverPipeline"
        )
        process = next(
            node
            for node in pipeline_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        official_calls = [
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "convert_vocals_official_upstream"
        ]
        self.assertEqual(len(official_calls), 2)
        for call in official_calls:
            keyword_names = {keyword.arg for keyword in call.keywords}
            self.assertIn("device", keyword_names)

    def test_backend_requirement_files_have_one_runtime_extra_each(self):
        expected = {
            "requirements_cpu.txt": "audio-separator[cpu]==0.44.1",
            "requirements_cuda.txt": "audio-separator[gpu]==0.44.1",
            "requirements_dml.txt": "audio-separator[dml]==0.44.1",
        }
        for filename, selected in expected.items():
            with self.subTest(filename=filename):
                source = (REPO_ROOT / filename).read_text(encoding="utf-8")
                extras = [
                    line.strip()
                    for line in source.splitlines()
                    if line.strip().startswith("audio-separator[")
                ]
                self.assertEqual(extras, [selected])

    def test_platform_ci_covers_three_desktop_operating_systems(self):
        source = (
            REPO_ROOT / ".github" / "workflows" / "platform-contracts.yml"
        ).read_text(encoding="utf-8")
        for runner in ("windows-latest", "ubuntu-latest", "macos-14"):
            self.assertIn(runner, source)

    def test_colab_notebook_is_valid_json_and_checks_cuda_provider(self):
        notebook_path = REPO_ROOT / "AI_RVC_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        self.assertIn("python install.py --backend cuda --no-run", source)
        self.assertIn("CUDAExecutionProvider", source)


if __name__ == "__main__":
    unittest.main()
