import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.package_runtime import validate_cuda_runtime


class PackageCudaRuntimeTests(unittest.TestCase):
    def test_linux_nvrtc_without_builtins_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
            library.parent.mkdir(parents=True)
            library.write_bytes(b'present-but-not-loadable')
            with patch('tools.package_runtime.sys.platform', 'linux'):
                with self.assertRaisesRegex(RuntimeError, 'libnvrtc-builtins.so.12.8'):
                    validate_cuda_runtime(root)

    def test_windows_nvrtc_without_builtins_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'torch/lib/nvrtc64_120_0.dll'
            library.parent.mkdir(parents=True)
            library.write_bytes(b'present-but-not-loadable')
            with patch('tools.package_runtime.sys.platform', 'win32'):
                with self.assertRaisesRegex(RuntimeError, 'nvrtc-builtins64_128.dll'):
                    validate_cuda_runtime(root)
