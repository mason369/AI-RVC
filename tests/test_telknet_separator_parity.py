import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class TelkNetSeparatorParityTests(unittest.TestCase):



    def test_karaoke_rejects_unknown_stem_names_instead_of_guessing_order(self):
        from infer import separator

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "song.wav"
            first_path = tmp_path / "mystery_a.wav"
            second_path = tmp_path / "mystery_b.wav"
            audio = np.full((4000, 2), 0.1, dtype=np.float32)
            for path in (source_path, first_path, second_path):
                sf.write(path, audio, 44100)

            instance = object.__new__(separator.KaraokeSeparator)
            instance.model_filename = separator.KARAOKE_DEFAULT_MODEL
            instance.separator = types.SimpleNamespace(
                output_dir=str(tmp_path),
                separate=lambda _audio_path: [str(first_path), str(second_path)],
            )

            with (
                patch.object(instance, "load_model", return_value=None),
                patch.object(separator, "_get_karaoke_min_duration_seconds", return_value=0.0),
                self.assertRaisesRegex(RuntimeError, "无法唯一确定主唱和第二路"),
            ):
                instance.separate(str(source_path), str(tmp_path))

    def test_karaoke_rejects_apparently_reversed_stems_instead_of_swapping(self):
        from infer import separator

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "song.wav"
            lead_path = tmp_path / "song_(Vocals).wav"
            backing_path = tmp_path / "song_(Instrumental).wav"
            sf.write(source_path, np.full((4000, 2), 0.1, dtype=np.float32), 44100)
            sf.write(lead_path, np.zeros((4000, 2), dtype=np.float32), 44100)
            sf.write(backing_path, np.full((4000, 2), 0.1, dtype=np.float32), 44100)

            instance = object.__new__(separator.KaraokeSeparator)
            instance.model_filename = separator.KARAOKE_DEFAULT_MODEL
            instance.separator = types.SimpleNamespace(
                output_dir=str(tmp_path),
                separate=lambda _audio_path: [str(lead_path), str(backing_path)],
            )

            with (
                patch.object(instance, "load_model", return_value=None),
                patch.object(separator, "_get_karaoke_min_duration_seconds", return_value=0.0),
                self.assertRaisesRegex(RuntimeError, "输出疑似反转"),
            ):
                instance.separate(str(source_path), str(tmp_path))

    def test_backing_vocals_are_vocal_mix_minus_lead(self):
        sys.modules.setdefault("faiss", types.ModuleType("faiss"))
        from infer.cover_pipeline import CoverPipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            vocal_mix_path = tmp_path / "vocals_with_harmony.wav"
            lead_path = tmp_path / "lead_vocals.wav"
            output_path = tmp_path / "backing_vocals.wav"
            vocal_mix = np.asarray(
                [[0.30, 0.10], [0.50, -0.20], [-0.25, 0.40], [0.10, 0.20]],
                dtype=np.float32,
            )
            lead = np.asarray(
                [[0.20, 0.05], [0.35, -0.10], [-0.20, 0.25], [0.05, 0.10]],
                dtype=np.float32,
            )
            sf.write(vocal_mix_path, vocal_mix, 44100, subtype="FLOAT")
            sf.write(lead_path, lead, 44100, subtype="FLOAT")

            result = CoverPipeline(device="cpu")._derive_backing_vocals(
                vocals_with_harmony_path=str(vocal_mix_path),
                lead_vocals_path=str(lead_path),
                output_path=str(output_path),
            )

            actual, sample_rate = sf.read(result, always_2d=True, dtype="float32")
            self.assertEqual(sample_rate, 44100)
            np.testing.assert_allclose(actual, vocal_mix - lead, atol=2e-6)

    def test_cover_exports_telknet_harmony_and_accompaniment_contract(self):
        sys.modules.setdefault("faiss", types.ModuleType("faiss"))
        from infer import cover_pipeline as module
        from infer.cover_pipeline import CoverPipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pipeline = CoverPipeline(device="cpu")
            input_audio = tmp_path / "input.mp3"
            model_path = tmp_path / "voice.pth"
            index_path = tmp_path / "voice.index"
            vocals_path = tmp_path / "sota" / "leap_vocals.wav"
            pure_accompaniment_path = tmp_path / "sota" / "leap_accompaniment.wav"
            lead_path = tmp_path / "karaoke" / "lead_vocals.wav"
            accompaniment_with_harmony_path = (
                tmp_path / "karaoke" / "backing_plus_instrumental.wav"
            )
            for path, payload in (
                (input_audio, b"input"),
                (model_path, b"model"),
                (index_path, b"index"),
                (vocals_path, b"leap-vocals"),
                (pure_accompaniment_path, b"leap-accompaniment"),
                (lead_path, b"mvsep-lead"),
                (accompaniment_with_harmony_path, b"mvsep-backing-plus-instrumental"),
            ):
                _write(path, payload)

            calls = {}

            class FakeSeparator:
                def separate(self, audio_path: str, output_dir: str):
                    calls["separator_input"] = audio_path
                    return str(vocals_path), str(pure_accompaniment_path)

                def unload_model(self):
                    return None

            def fake_init_separator(*_args, **_kwargs):
                pipeline.separator = FakeSeparator()

            def fake_karaoke(*, source_audio_path: str, **_kwargs):
                calls["karaoke_input"] = source_audio_path
                return str(lead_path), str(accompaniment_with_harmony_path)

            def fake_prepare(vocals_path: str, *_args, **_kwargs):
                calls["prepare_input"] = vocals_path
                return vocals_path

            def fake_convert(*, vocals_path: str, output_path: str, **_kwargs):
                calls["vc_input"] = vocals_path
                _write(Path(output_path), b"converted-lead")

            def fake_mix(*, vocals_path: str, accompaniment_path: str, output_path: str, **_kwargs):
                calls.setdefault("mix_accompaniments", []).append(accompaniment_path)
                _write(Path(output_path), b"cover")

            def fake_derive(*, vocals_with_harmony_path: str, lead_vocals_path: str, output_path: str):
                calls["derive_vocal_mix"] = vocals_with_harmony_path
                calls["derive_lead"] = lead_vocals_path
                _write(Path(output_path), b"derived-harmony-only")
                return output_path

            with (
                patch.object(module, "_get_audio_duration", return_value=1.0),
                patch("infer.contracts.inspect_checkpoint", return_value=types.SimpleNamespace(speaker_count=1, uses_f0=True, feature_dim=768)),
                patch("infer.contracts.read_index"),
                patch.object(module.torch, "load", return_value={}),
                patch.object(pipeline, "_init_separator", side_effect=fake_init_separator),
                patch.object(pipeline, "_separate_karaoke", side_effect=fake_karaoke),
                patch.object(pipeline, "_prepare_vocals_for_vc", side_effect=fake_prepare),
                patch.object(pipeline, "_derive_backing_vocals", side_effect=fake_derive, create=True),
                patch.object(
                    pipeline,
                    "_prepare_mix_vocal_foreground",
                    side_effect=lambda **kwargs: {"vocals_path": kwargs["vocals_path"]},
                ),
                patch.object(pipeline, "_record_quality_debug", return_value=None),
                patch.object(pipeline, "_maybe_log_diagnostic_hint", return_value=None),
                patch.object(module, "convert_vocals_official_upstream", side_effect=fake_convert),
                patch.object(module, "mix_vocals_and_accompaniment", side_effect=fake_mix),
            ):
                result = pipeline.process(
                    input_audio=str(input_audio),
                    model_path=str(model_path),
                    index_path=str(index_path),
                    separator="roformer",
                    roformer_model="hybrid:leap_xe90_vocals+leap62_instrumental",
                    karaoke_separation=True,
                    karaoke_model="ensemble:mvsep_9205_avg",
                    karaoke_merge_backing_into_accompaniment=True,
                    source_constraint_mode="off",
                    vc_pipeline_mode="current",
                    output_dir=str(tmp_path / "out"),
                )
                result_without_harmony_mix = pipeline.process(
                    input_audio=str(input_audio),
                    model_path=str(model_path),
                    index_path=str(index_path),
                    separator="roformer",
                    roformer_model="hybrid:leap_xe90_vocals+leap62_instrumental",
                    karaoke_separation=True,
                    karaoke_model="ensemble:mvsep_9205_avg",
                    karaoke_merge_backing_into_accompaniment=False,
                    source_constraint_mode="off",
                    vc_pipeline_mode="current",
                    output_dir=str(tmp_path / "out_without_harmony_mix"),
                )

            self.assertEqual(calls["separator_input"], str(input_audio))
            self.assertEqual(calls["karaoke_input"], str(input_audio))
            self.assertEqual(calls["derive_vocal_mix"], str(vocals_path))
            self.assertEqual(calls["derive_lead"], str(lead_path))
            self.assertEqual(calls["prepare_input"], str(lead_path))
            self.assertEqual(calls["vc_input"], str(lead_path))
            self.assertEqual(
                calls["mix_accompaniments"],
                [
                    str(accompaniment_with_harmony_path),
                    str(pure_accompaniment_path),
                ],
            )
            self.assertEqual(
                Path(result["accompaniment"]).read_bytes(),
                b"mvsep-backing-plus-instrumental",
            )
            self.assertEqual(
                Path(result["accompaniment_without_harmony"]).read_bytes(),
                b"leap-accompaniment",
            )
            self.assertEqual(
                Path(result["backing_vocals"]).read_bytes(),
                b"derived-harmony-only",
            )
            self.assertEqual(
                Path(result_without_harmony_mix["accompaniment"]).read_bytes(),
                b"mvsep-backing-plus-instrumental",
            )
            self.assertEqual(
                Path(
                    result_without_harmony_mix["accompaniment_without_harmony"]
                ).read_bytes(),
                b"leap-accompaniment",
            )

    def test_cover_rejects_legacy_karaoke_model_for_public_output_contract(self):
        sys.modules.setdefault("faiss", types.ModuleType("faiss"))
        from infer.cover_pipeline import CoverPipeline

        with self.assertRaisesRegex(ValueError, "只支持 ensemble:mvsep_9205_avg"):
            CoverPipeline(device="cpu").process(
                input_audio="missing.wav",
                model_path="missing.pth",
                separator="roformer",
                karaoke_separation=True,
                karaoke_model="mel_band_roformer_karaoke_gabox.ckpt",
                vc_pipeline_mode="current",
            )


if __name__ == "__main__":
    unittest.main()
