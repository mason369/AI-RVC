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
            2,
        )
        self.assertEqual(
            self.workflow.count(
                "torch_stack: torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
            ),
            2,
        )
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


if __name__ == "__main__":
    unittest.main()
