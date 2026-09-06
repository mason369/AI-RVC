"""Select a real PyTorch 2.11 wheel that includes the detected GPU kernels."""
import contextlib
import io
import unittest
from unittest import mock

import install


class CudaInstallArchitectureTests(unittest.TestCase):
    def test_driver_and_architecture_matrix(self):
        cases = (
            ("13.0", "12.0", "cu128"),
            ("12.8", "10.0", "cu128"),
            ("13.0", "8.9", "cu126"),
            ("13.0", "7.0", "cu126"),
            ("12.6", "6.1", "cu126"),
            ("12.4", "8.9", None),
            ("12.6", "12.0", None),
            ("13.0", "7.0\n12.0", None),
            ("13.0", "N/A", None),
            ("13.0", "", None),
        )
        for cuda, capability, index in cases:
            def run(command, **_kwargs):
                if command == ["nvidia-smi"]:
                    return mock.Mock(returncode=0, stdout=f"CUDA Version: {cuda}\n")
                if "--query-gpu=compute_cap" in command:
                    return mock.Mock(returncode=0, stdout=capability)
                if "--query-gpu=driver_version" in command:
                    return mock.Mock(returncode=0, stdout="580.82.07")
                raise AssertionError(command)

            with self.subTest(cuda=cuda, capability=capability), \
                    mock.patch("install.subprocess.run", side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                expected = f"https://download.pytorch.org/whl/{index}" if index else None
                self.assertEqual(install.detect_cuda_version(), expected)


if __name__ == "__main__":
    unittest.main()
