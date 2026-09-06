"""Contract tests use generated fixtures; they are not real-weight quality scores."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import faiss
import numpy as np
import soundfile as sf
import torch

from configs.schema import validate_config
from infer.contracts import inspect_checkpoint, integer, number, read_index, validate_state_dict
from infer.pipeline import VoiceConversionPipeline
from lib.upstream_audio_output import preserve_float_vc_output

ROOT = Path(__file__).resolve().parents[1]


class EncodingFixture:
    def pipeline(self, audio_opt):
        audio_max = np.abs(audio_opt).max() / .99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        return audio_opt


class FeatureContractTests(unittest.TestCase):
    def test_removed_cover_parameters_are_not_in_public_signature(self):
        from infer.cover_pipeline import CoverPipeline
        from infer.official_adapter import convert_vocals_official_upstream
        from ui.app import process_cover
        removed = {"filter_radius", "hubert_layer", "singing_repair", "vc_preprocess_mode"}
        for function in (CoverPipeline.process, convert_vocals_official_upstream, process_cover):
            self.assertFalse(removed & set(inspect.signature(function).parameters))

    def test_unknown_and_obsolete_config_keys_are_rejected(self):
        for config in ({"typo": 1}, {"cover": {"filter_radius": 3}}, {"cover": {"f0_stabilize": True}},
                       {"cover": {"vc_preprocess_mode": "auto"}}, {"cover": {"protect": .55}},
                       {"cover": {"karaoke_separation": "false"}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_every_shipped_preset_matches_the_live_schema(self):
        config = json.loads((ROOT / "configs/config.json").read_text(encoding="utf-8"))
        validate_config(config)
        for path in (ROOT / "configs/presets").glob("*.json"):
            preset = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(preset=path.name):
                validate_config({**config, "cover": {**config["cover"], **preset["cover"]}})

    def test_nonfinite_booleans_strings_and_fractional_integers_are_rejected(self):
        for value in (True, "0.5", float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value, "index_rate", 0, 1)
        with self.assertRaises(ValueError):
            integer(1.5, "speaker_id", 0, 255)

    def test_float_encoding_preserves_values_below_one_pcm16_step(self):
        original = EncodingFixture()
        modified = EncodingFixture()
        preserve_float_vc_output(modified)
        audio = np.array([1e-7, .1234567, -.000004], dtype=np.float32)
        encoded = modified.pipeline(audio)
        self.assertEqual(encoded.dtype, np.float32)
        np.testing.assert_array_equal(encoded, audio)
        np.testing.assert_array_equal((encoded * 32768).astype(np.int16), original.pipeline(audio))

    def test_missing_reverb_dependency_fails_instead_of_skipping(self):
        from lib import mixer
        with mock.patch.object(mixer, "PEDALBOARD_AVAILABLE", False), self.assertRaises(RuntimeError):
            mixer.apply_reverb(np.ones((2, 200), dtype=np.float32), 44100)

    def test_index_load_failure_preserves_previous_valid_state(self):
        pipe = VoiceConversionPipeline("cpu")
        pipe.model_feature_dim = 256
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.index"
            invalid = Path(tmp) / "wrong.index"
            index = faiss.IndexFlatL2(256)
            index.add(np.zeros((2, 256), dtype=np.float32))
            faiss.write_index(index, str(valid))
            faiss.write_index(faiss.IndexFlatL2(768), str(invalid))
            pipe.load_index(str(valid))
            previous = pipe.index
            with self.assertRaises(ValueError):
                pipe.load_index(str(invalid))
            self.assertIs(pipe.index, previous)
            retrieved = pipe.search_index(np.zeros((3, 256), dtype=np.float32), k=8)
            self.assertTrue(np.isfinite(retrieved).all())
            self.assertEqual(retrieved.shape, (3, 256))
            with self.assertRaises(ValueError):
                read_index(valid, 256, min_vectors=8)

    def test_unknown_feature_dimensions_cannot_be_padded_or_truncated(self):
        for dim in (128, 512, 1024):
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                inspect_checkpoint({"weight": {"enc_p.emb_phone.weight": torch.zeros(8, dim)}})

    def test_incomplete_weights_are_not_loaded_with_random_inference_layers(self):
        layer = torch.nn.Linear(8, 4)
        with self.assertRaises(ValueError):
            validate_state_dict(layer, {"weight": layer.weight})

    def test_v1_uses_native_projection_and_v2_retains_all_features(self):
        pipe = VoiceConversionPipeline("cpu")
        encoder = mock.Mock()
        encoder.extract_features.return_value = (torch.ones(1, 3, 768),)
        encoder.final_proj.return_value = torch.ones(1, 3, 256)
        pipe.hubert_model, pipe.hubert_model_type = encoder, "fairseq"
        self.assertEqual(pipe.extract_features(np.ones(1600, dtype=np.float32), True).shape[-1], 256)
        self.assertEqual(encoder.extract_features.call_args.kwargs["output_layer"], 9)
        self.assertEqual(pipe.extract_features(np.ones(1600, dtype=np.float32), False).shape[-1], 768)
        self.assertEqual(encoder.extract_features.call_args.kwargs["output_layer"], 12)

    def test_all_native_version_f0_and_sample_rate_architectures(self):
        from infer.lib.infer_pack.models import (SynthesizerTrnMs256NSFsid, SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid_nono)
        classes = {("v1", 1): SynthesizerTrnMs256NSFsid, ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                   ("v2", 1): SynthesizerTrnMs768NSFsid, ("v2", 0): SynthesizerTrnMs768NSFsid_nono}
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.pth"
            for (version, f0), cls in classes.items():
                for sr, rates in [(32000, [10, 8, 2, 2]), (40000, [10, 10, 2, 2]), (48000, [10, 12, 2, 2])]:
                    with self.subTest(version=version, f0=f0, sample_rate=sr):
                        config = [17, 4, 8, 8, 16, 2, 1, 3, 0, "1", [3], [[1, 3, 5]], rates, 32,
                                  [2 * rate for rate in rates], 2, 8, sr]
                        model = cls(*config, is_half=False)
                        del model.enc_q
                        torch.save({"config": config, "weight": model.state_dict(), "f0": f0, "version": version}, path)
                        pipeline = VoiceConversionPipeline("cpu")
                        info = pipeline.load_voice_model(str(path))
                        dim = 256 if version == "v1" else 768
                        audio = pipeline._process_chunk(np.ones((3, dim), dtype=np.float32) * .01,
                                                       np.ones(6, dtype=np.float32) * 220, speaker_id=1)
                        self.assertEqual(len(audio), sr * 6 // 100)
                        self.assertTrue(np.isfinite(audio).all())
                        self.assertEqual(info["uses_f0"], bool(f0))
                        with self.assertRaises(ValueError):
                            pipeline._process_chunk(np.ones((3, dim), dtype=np.float32), np.ones(6), speaker_id=2)
                        pipeline.unload_all()
