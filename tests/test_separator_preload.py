import unittest
from unittest import mock

from lib.separator_runtime import Separator, UpstreamSeparator


class SeparatorPreloadTests(unittest.TestCase):
    def test_cpu_initialization_never_requests_cuda_libraries(self):
        separator = Separator(info_only=True)
        with mock.patch("lib.separator_runtime.ort.get_available_providers", return_value=["CPUExecutionProvider"]), \
             mock.patch("lib.separator_runtime.ort.preload_dlls") as preload:
            separator.preload_onnxruntime_dependencies()
            preload.assert_called_once_with(cuda=False, cudnn=False)

    def test_cuda_keeps_upstream_preload(self):
        separator = Separator(info_only=True)
        with mock.patch("lib.separator_runtime.ort.get_available_providers", return_value=["CUDAExecutionProvider"]), \
             mock.patch.object(UpstreamSeparator, "preload_onnxruntime_dependencies") as preload:
            separator.preload_onnxruntime_dependencies()
            preload.assert_called_once()

    def test_cpu_preload_error_is_not_silenced(self):
        separator = Separator(info_only=True)
        with mock.patch("lib.separator_runtime.ort.get_available_providers", return_value=["CPUExecutionProvider"]), \
             mock.patch("lib.separator_runtime.ort.preload_dlls", side_effect=OSError("library error")):
            with self.assertRaisesRegex(OSError, "library error"):
                separator.preload_onnxruntime_dependencies()


if __name__ == "__main__":
    unittest.main()
