from __future__ import annotations

import gc

import unittest
import tempfile
from pathlib import Path
import torch

from infer.lib.applio_algorithm.synthesizers import Synthesizer as ApplioSynthesizer
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.rvc_checkpoint import (
    RvcCheckpointCompatibilityError,
    build_rvc_synthesizer,
    checkpoint_weight_dtypes,
    checkpoint_storage_dtypes,
    prepare_rvc_checkpoint,
    select_rvc_runtime_precision,
)


CONFIG = [
    17,
    8,
    8,
    8,
    16,
    2,
    1,
    3,
    0,
    "1",
    [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [2, 2],
    16,
    [4, 4],
    2,
    4,
    32000,
]


def _official_checkpoint(version: str, f0: bool, *, include_encoder: bool = False):
    classes = {
        ("v1", True): SynthesizerTrnMs256NSFsid,
        ("v1", False): SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): SynthesizerTrnMs768NSFsid,
        ("v2", False): SynthesizerTrnMs768NSFsid_nono,
    }
    model = classes[(version, f0)](*CONFIG, **({"is_half": False} if f0 else {}))
    if not include_encoder:
        del model.enc_q
    checkpoint = {
        "weight": model.state_dict(),
        "config": list(CONFIG),
        "version": version,
        "f0": int(f0),
    }
    return model, checkpoint


class TelkNetRvcCheckpointCompatibilityTests(unittest.TestCase):
    def test_all_official_rvc_architectures_require_complete_exact_weights(self):
        for version, f0 in [('v1', True), ('v1', False), ('v2', True), ('v2', False)]:
            with self.subTest(version=version, f0=f0):
                model, checkpoint = _official_checkpoint(version, f0)

                prepared, report = prepare_rvc_checkpoint(
                    checkpoint,
                    model_label=f"{version}-{f0}",
                    validate_structure=True,
                )

                assert report.rvc_version == version
                assert report.f0_enabled is f0
                assert report.vocoder == "HiFi-GAN"
                assert report.compatibility_status == "verified_exact"
                assert report.inference_weight_coverage_percent == 100
                assert set(prepared["weight"]) == set(model.state_dict())
                del model
                gc.collect()

    def test_current_applio_hifigan_weight_norm_names_are_converted_losslessly(self):
        model = ApplioSynthesizer(
            *CONFIG,
            use_f0=True,
            text_enc_hidden_dim=768,
            vocoder="HiFi-GAN",
            randomized=False,
        )
        del model.enc_q
        source = model.state_dict()
        checkpoint = {
            "weight": source,
            "config": list(CONFIG),
            "version": "v2",
            "f0": 1,
            "vocoder": "HiFi-GAN",
        }

        prepared, report = prepare_rvc_checkpoint(
            checkpoint,
            model_label="applio-hifigan",
            validate_structure=True,
        )

        source_key = next(key for key in source if key.endswith(".parametrizations.weight.original0"))
        target_key = source_key.replace(".parametrizations.weight.original0", ".weight_g")
        assert report.source_weight_norm_format == "parametrized"
        assert report.weight_key_conversions > 0
        assert torch.equal(source[source_key], prepared["weight"][target_key])
        assert source[source_key].dtype == prepared["weight"][target_key].dtype
        assert not any("parametrizations.weight" in key for key in prepared["weight"])

    def test_applio_vocoder_architectures_load_their_declared_network(self):
        for vocoder in ['MRF HiFi-GAN', 'RefineGAN']:
            with self.subTest(vocoder=vocoder):
                model = ApplioSynthesizer(
                    *CONFIG,
                    use_f0=True,
                    text_enc_hidden_dim=768,
                    vocoder=vocoder,
                    randomized=False,
                )
                del model.enc_q
                checkpoint = {
                    "weight": model.state_dict(),
                    "config": list(CONFIG),
                    "version": "v2",
                    "f0": 1,
                    "vocoder": vocoder,
                }

                prepared, report = prepare_rvc_checkpoint(
                    checkpoint,
                    model_label=vocoder,
                    validate_structure=True,
                )
                loaded, runtime_checkpoint, runtime_report = build_rvc_synthesizer(
                    prepared,
                    model_label=vocoder,
                    is_half=False,
                )

                assert runtime_report.vocoder == vocoder
                assert runtime_report.runtime_architecture.startswith("applio_v2_f0_")
                assert set(runtime_checkpoint["weight"]) == set(loaded.state_dict())
                assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}

    def test_training_encoder_is_removed_but_every_inference_weight_remains_required(self):
        model, checkpoint = _official_checkpoint("v2", True, include_encoder=True)
        prepared, report = prepare_rvc_checkpoint(
            checkpoint,
            model_label="training-wrapper",
            validate_structure=True,
        )

        assert report.removed_training_tensor_count > 0
        assert report.source_parameter_count > report.parameter_count
        assert not any(key.startswith("enc_q.") for key in prepared["weight"])

        prepared["weight"].pop("enc_p.proj.bias")
        with self.assertRaisesRegex(RvcCheckpointCompatibilityError, "fully populate"):
            build_rvc_synthesizer(prepared, model_label="partial", is_half=False)
        del model

    def test_fp16_storage_stays_fp16_on_rtx_2080_ti_without_quantization(self):
        model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["weight"] = {
            name: tensor.half() for name, tensor in checkpoint["weight"].items()
        }

        precision = select_rvc_runtime_precision(
            checkpoint_weight_dtypes(checkpoint),
            "cuda:0",
            cuda_capability=(7, 5),
            cuda_device_name="NVIDIA GeForce RTX 2080 Ti",
        )
        loaded, _, report = build_rvc_synthesizer(
            checkpoint,
            model_label="fp16-storage",
            is_half=precision.is_half,
            runtime_dtype=precision.runtime_dtype,
        )
        loaded = loaded.to(dtype=precision.torch_dtype)

        assert report.weight_dtypes == ["float16"]
        assert report.runtime_dtype == "float16"
        assert report.runtime_precision_policy == "preserve_source_or_losslessly_widen"
        assert report.quantized is False
        assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float16}
        del model

    def test_fp32_storage_is_never_downcast_on_fp16_capable_gpu(self):
        model, checkpoint = _official_checkpoint("v2", True)

        precision = select_rvc_runtime_precision(
            checkpoint_weight_dtypes(checkpoint),
            "cuda:0",
            cuda_capability=(7, 5),
        )

        assert precision.runtime_dtype == "float32"
        assert precision.is_half is False
        assert "without downcast" in precision.reason
        del model

    def test_precision_plan_ignores_removed_training_encoder_tensors(self):
        _model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["weight"] = {
            name: tensor.half() for name, tensor in checkpoint["weight"].items()
        }
        checkpoint["weight"]["enc_q.training_only"] = torch.ones(1, dtype=torch.float32)

        assert checkpoint_weight_dtypes(checkpoint) == ["float16"]
        del _model

    def test_fp16_storage_is_losslessly_widened_when_native_fp16_is_unavailable(self):
        precision = select_rvc_runtime_precision(["float16"], "cpu")

        assert precision.runtime_dtype == "float32"
        assert precision.is_half is False
        assert "widened losslessly" in precision.reason

    def test_bfloat16_and_mixed_storage_use_exact_common_fp32_runtime(self):
        bf16 = select_rvc_runtime_precision(
            ["bfloat16"], "cuda:0", cuda_capability=(7, 5)
        )
        mixed = select_rvc_runtime_precision(
            ["float16", "float32"], "cuda:0", cuda_capability=(7, 5)
        )

        assert bf16.runtime_dtype == "float32"
        assert mixed.runtime_dtype == "float32"
        assert "mixed floating checkpoint" in mixed.reason

    def test_float64_checkpoint_and_mixed_float64_preserve_highest_source_precision(self):
        pure = select_rvc_runtime_precision(["float64"], "cuda:0", cuda_capability=(7, 5))
        mixed = select_rvc_runtime_precision(
            ["float16", "float32", "float64"],
            "cuda:0",
            cuda_capability=(7, 5),
        )

        assert pure.runtime_dtype == "float64"
        assert pure.torch_dtype == torch.float64
        assert "without downcast" in pure.reason
        assert mixed.runtime_dtype == "float64"
        assert mixed.torch_dtype == torch.float64
        assert "common FP64" in mixed.reason

    def test_fp64_weights_are_loaded_without_an_intermediate_fp32_round_trip(self):
        model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["weight"] = {
            name: tensor.double() for name, tensor in checkpoint["weight"].items()
        }
        key = next(
            name
            for name, tensor in checkpoint["weight"].items()
            if tensor.is_floating_point() and tensor.numel() > 0
        )
        flattened = checkpoint["weight"][key].reshape(-1)
        exact_value = torch.tensor(1.0000000000000002, dtype=torch.float64)
        flattened[0] = exact_value
        assert flattened[0] != flattened[0].float().double()

        loaded, _, report = build_rvc_synthesizer(
            checkpoint,
            model_label="fp64-exact",
            is_half=False,
            runtime_dtype="float64",
        )

        assert report.runtime_dtype == "float64"
        assert loaded.state_dict()[key].reshape(-1)[0] == exact_value
        assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float64}
        del model, loaded

    def test_fp64_checkpoint_completes_vocoder_inference_without_downcast(self):
        model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["weight"] = {
            name: tensor.double() for name, tensor in checkpoint["weight"].items()
        }
        loaded, _, report = build_rvc_synthesizer(
            checkpoint,
            model_label="fp64-inference",
            is_half=False,
            runtime_dtype="float64",
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loaded = loaded.to(device=device, dtype=torch.float64).eval()
        frame_count = 8
        phone = torch.randn(1, frame_count, 768, device=device, dtype=torch.float64)
        phone_lengths = torch.tensor([frame_count], device=device, dtype=torch.long)
        pitch = torch.full((1, frame_count), 128, device=device, dtype=torch.long)
        pitchf = torch.full(
            (1, frame_count),
            440.00000000000006,
            device=device,
            dtype=torch.float64,
        )
        sid = torch.zeros(1, device=device, dtype=torch.long)

        with torch.inference_mode():
            waveform = loaded.infer(phone, phone_lengths, pitch, pitchf, sid)[0]

        assert report.runtime_dtype == "float64"
        assert waveform.dtype == torch.float64
        assert waveform.device == device
        assert waveform.numel() > 0
        assert torch.isfinite(waveform).all()
        del model, loaded, waveform

    def test_checkpoint_storage_precision_is_read_per_component(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            mixed_path = tmp_path / "mixed.pt"
            half_path = tmp_path / "half.pt"
            torch.save(
                {
                    "full": torch.ones(2, dtype=torch.float32),
                    "half": torch.ones(2, dtype=torch.float16),
                },
                mixed_path,
            )
            torch.save({"half": torch.ones(2, dtype=torch.float16)}, half_path)

            assert checkpoint_storage_dtypes(mixed_path) == ["float16", "float32"]
            assert checkpoint_storage_dtypes(half_path) == ["float16"]

    def test_stale_version_and_speaker_metadata_are_derived_from_weights(self):
        model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["version"] = "v1"
        checkpoint["f0"] = 0
        checkpoint["config"][-3] = 999

        prepared, report = prepare_rvc_checkpoint(
            checkpoint,
            model_label="stale-metadata",
            validate_structure=True,
        )

        assert report.rvc_version == "v2"
        assert report.version_detection == "weight_shape"
        assert report.f0_enabled is True
        assert report.configured_f0_enabled is False
        assert report.f0_metadata_normalized is True
        assert report.configured_speaker_count == 999
        assert report.speaker_metadata_normalized is True
        assert prepared["config"][-3] == report.speaker_count
        del model

    def test_stale_vocoder_label_is_derived_from_exact_decoder_structure(self):
        model, checkpoint = _official_checkpoint("v2", True)
        checkpoint["vocoder"] = "UnknownGAN"

        prepared, report = prepare_rvc_checkpoint(
            checkpoint,
            model_label="stale-vocoder",
            validate_structure=True,
        )

        assert report.vocoder == "HiFi-GAN"
        assert report.configured_vocoder is None
        assert report.vocoder_metadata_normalized is True
        assert prepared["vocoder"] == "HiFi-GAN"
        del model
