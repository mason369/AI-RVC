import json
import unittest
from pathlib import Path


class ColabNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path("AI_RVC_Colab.ipynb")
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook.get("cells", [])
        )

    def test_notebook_json_is_valid_and_has_gpu_metadata(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["metadata"].get("accelerator"), "GPU")
        self.assertEqual(self.notebook["metadata"].get("colab", {}).get("gpuType"), "T4")

    def test_colab_uses_python_310_environment_for_fairseq(self):
        self.assertIn("uv python install 3.10", self.source)
        self.assertIn("uv venv --seed --python 3.10 venv310", self.source)
        self.assertIn("$PY -m pip --version", self.source)
        self.assertIn(
            "uv run --python 3.10 python install.py --backend cuda --no-run",
            self.source,
        )
        self.assertIn("assert sys.version_info[:2] == (3, 10)", self.source)

    def test_colab_preinstalls_gpu_torch_before_project_dependencies(self):
        self.assertIn("--index-url https://download.pytorch.org/whl/cu126", self.source)
        self.assertIn("pip install torch torchaudio", self.source)
        self.assertIn("raise RuntimeError('GPU PyTorch install completed but CUDA is not available.')", self.source)
        self.assertNotIn("https://download.pytorch.org/whl/cpu", self.source)

    def test_colab_checks_current_processing_model_defaults(self):
        self.assertIn(
            "ROFORMER_DEFAULT_MODEL == 'hybrid:leap_xe90_vocals+polarformer62_instrumental'",
            self.source,
        )
        self.assertIn("KARAOKE_DEFAULT_MODEL == 'ensemble:mvsep_9205_avg'", self.source)
        self.assertIn("check_required_default_separator_models", self.source)
        self.assertIn("get_missing_default_separator_model_files", self.source)
        self.assertIn(
            "ROFORMER_DEREVERB_DEFAULT_MODEL == 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt'",
            self.source,
        )
        self.assertIn("'onnxruntime-gpu': '>=1.17'", self.source)
        self.assertIn("SpecifierSet(constraint)", self.source)
        self.assertIn("'CUDAExecutionProvider' not in onnxruntime.get_available_providers()", self.source)
        self.assertNotIn("'onnxruntime':", self.source)

    def test_colab_no_longer_documents_legacy_single_roformer_as_default(self):
        self.assertNotIn("MVSEP Vocals SDR 11.01", self.source)
        self.assertNotIn("vocals_mel_band_roformer", self.source)

    def test_colab_launch_does_not_skip_checks(self):
        self.assertIn("$PY run.py --host 0.0.0.0 --port 7860 --share", self.source)
        self.assertNotIn("--skip-check", self.source)

    def test_colab_includes_full_processing_mode_matrix(self):
        self.assertIn("$PY tools/run_mode_matrix.py", self.source)
        self.assertIn("--output-dir /content/AI-RVC/outputs/mode_matrix_colab", self.source)
        self.assertIn("--require-cuda", self.source)
        self.assertIn("download_character_model('rin')", self.source)
        self.assertIn("Missing test audio", self.source)


if __name__ == "__main__":
    unittest.main()
