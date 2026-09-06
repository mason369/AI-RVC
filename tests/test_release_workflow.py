import unittest
from pathlib import Path


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = Path(".github/workflows/build-executables.yml").read_text(
            encoding="utf-8"
        )

    def test_torch_stack_is_complete_pinned_and_verified_after_audio_separator(self):
        self.assertEqual(
            self.workflow.count(
                "torch_stack: torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
            ),
            4,
        )
        self.assertEqual(
            self.workflow.count(
                "torch_stack: torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
            ),
            0,
        )
        self.assertEqual(self.workflow.count("https://download.pytorch.org/whl/cu128"), 2)
        self.assertIn(
            "pip install ${{ matrix.torch_stack }} --index-url ${{ matrix.pytorch_url }}",
            self.workflow,
        )
        self.assertIn("import torch, torchvision, torchaudio", self.workflow)
        self.assertIn("PyTorch stack changed after audio-separator install", self.workflow)
        self.assertIn("torch.__version__.split('+')[0]", self.workflow)
        self.assertNotIn('split(\\"+\\")', self.workflow)
        self.assertNotIn("pip install torch torchaudio", self.workflow)

    def test_release_upload_retry_preserves_the_real_exit_code(self):
        self.assertIn(
            "else\n              exit_code=$?\n            fi",
            self.workflow,
        )
        self.assertNotIn(
            "fi\n\n            exit_code=$?",
            self.workflow,
        )


    def test_build_includes_current_workers_and_model_dependencies(self):
        self.assertIn('_official_rvc_runtime${SEP}_official_rvc_runtime', self.workflow)
        self.assertIn('assets/hubert_base${SEP}assets/hubert_base', self.workflow)
        self.assertIn('rvc_mcp${SEP}rvc_mcp', self.workflow)
        self.assertNotIn('_official_rvc${SEP}_official_rvc"', self.workflow)
        for package in ('torchfcpe', 'transformers', 'mcp'):
            self.assertIn('--collect-all ' + package, self.workflow)
        self.assertIn('--internal-worker vc --help', self.workflow)
        self.assertIn('--internal-worker uvr5 --help', self.workflow)


if __name__ == "__main__":
    unittest.main()
