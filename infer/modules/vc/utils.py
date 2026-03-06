import os
import logging
import functools
import torch

logger = logging.getLogger(__name__)

try:
    from fairseq import checkpoint_utils
    FAIRSEQ_AVAILABLE = True
except Exception:
    FAIRSEQ_AVAILABLE = False


def _patch_torch_load():
    """Patch torch.load to default weights_only=False for fairseq compatibility (PyTorch 2.6+)."""
    _original = torch.load

    @functools.wraps(_original)
    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original(*args, **kwargs)

    return _original, _patched


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    if FAIRSEQ_AVAILABLE:
        _original, _patched = _patch_torch_load()
        torch.load = _patched
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                ["assets/hubert/hubert_base.pt"],
                suffix="",
            )
        finally:
            torch.load = _original
        hubert_model = models[0]
        hubert_model = hubert_model.to(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        return hubert_model.eval()

    try:
        import torchaudio

        class HubertWrapper:
            def __init__(self, model):
                self.model = model
                self.final_proj = getattr(model, "final_proj", torch.nn.Identity())

            def extract_features(self, source, padding_mask=None, output_layer=None):
                feats, _ = self.model.extract_features(source)
                if output_layer is None:
                    idx = -1
                else:
                    idx = min(output_layer - 1, len(feats) - 1)
                return (feats[idx], None)

            def to(self, device):
                self.model = self.model.to(device)
                return self

            def half(self):
                self.model = self.model.half()
                return self

            def float(self):
                self.model = self.model.float()
                return self

            def eval(self):
                self.model.eval()
                return self

        model = torchaudio.pipelines.HUBERT_BASE.get_model()
        hubert_model = HubertWrapper(model).to(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        return hubert_model.eval()
    except Exception as e:
        raise RuntimeError(
            "HuBERT 模型加载失败，请检查 fairseq 和 torchaudio 是否已安装"
        ) from e
