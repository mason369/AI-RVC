"""Keep CPU ONNX Runtime initialization independent of unavailable CUDA libraries."""
import onnxruntime as ort
from audio_separator.separator import Separator as UpstreamSeparator


class Separator(UpstreamSeparator):
    def preload_onnxruntime_dependencies(self):
        if "CUDAExecutionProvider" in ort.get_available_providers() or not hasattr(ort, "preload_dlls"):
            return super().preload_onnxruntime_dependencies()
        # ONNX Runtime's public API supports CPU/DirectML builds without CUDA/cuDNN.
        # Keep the rest of upstream device/model initialization and all errors intact.
        ort.preload_dlls(cuda=False, cudnn=False)
