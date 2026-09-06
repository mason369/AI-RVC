import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from infer import separator


class LeapInstrumentalTests(unittest.TestCase):
    def test_default_chain_names_the_actual_instrumental_model(self):
        labels = separator.get_separator_chain_labels(
            separator_name="roformer", roformer_model=separator.ROFORMER_DEFAULT_MODEL,
            karaoke_enabled=False, karaoke_model=separator.KARAOKE_DEFAULT_MODEL,
        )
        self.assertIn("纯伴奏: BS-RoFormer Leap Instrumental 62 bands (pcunwa)", labels)
        self.assertNotIn("PolarFormer", " ".join(labels))

    def test_new_models_have_pinned_revisions_and_distinct_config_directories(self):
        instrumental = separator._CUSTOM_AUDIO_SEPARATOR_MODELS["bs_roformer_leap_inst.ckpt"]
        dereverb = separator._CUSTOM_AUDIO_SEPARATOR_MODELS["dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt"]
        self.assertEqual(instrumental["revision"], "4e47d6662ae82eaa8b4ac4329fe66099a843b48e")
        self.assertEqual(dereverb["revision"], "bd5c6b55a429b4b74ce85fe5dcb690cfe36d91ec")
        self.assertNotEqual(instrumental["local_subdir"], dereverb["local_subdir"])
        for spec in (instrumental, dereverb):
            self.assertEqual(len(spec["model_sha256"]), 64)
            self.assertEqual(len(spec["config_sha256"]), 64)

    def test_shared_pcm_short_clip_roles_and_sequential_model_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcm = root / "source.wav"
            sf.write(pcm, np.ones((4000, 2), dtype=np.float32) * .1, 16000)
            seen, references = [], []

            class FakeSeparator:
                def __init__(self, output_dir):
                    self.output_dir = Path(output_dir)

                def separate(self, audio):
                    data, sr = sf.read(audio, always_2d=True)
                    self.assertion = len(data)
                    paths = [self.output_dir / "song_(vocals).wav", self.output_dir / "song_(other).wav"]
                    for path, value in zip(paths, (.2, .4)):
                        sf.write(path, np.ones_like(data) * value, sr)
                    return [str(path) for path in paths]

            def load(**kwargs):
                if references:
                    self.assertIsNone(references[-1](), "Previous model must be released before loading the next")
                seen.append(kwargs["model_spec"])
                result = FakeSeparator(kwargs["output_dir"])
                references.append(weakref.ref(result))
                return result

            runtime = separator._HybridLeapInstrumentalRuntime(str(root / "models"), str(root / "out"), "cpu")
            with (
                patch.object(separator, "_load_audio_separator_model", side_effect=load),
                patch.object(separator, "_ensure_separator_pcm_wav", return_value=str(pcm)) as normalize,
                patch.object(separator, "_get_custom_roformer_min_duration_seconds", return_value=.5),
                patch.object(separator, "_pad_audio_to_min_duration", wraps=separator._pad_audio_to_min_duration) as pad,
                patch.object(separator, "empty_device_cache"),
            ):
                outputs = runtime.separate("original.mp3")
            normalize.assert_called_once()
            self.assertEqual([call.args[0] for call in pad.call_args_list], [str(pcm), str(pcm)])
            self.assertEqual(seen, ["bs_roformer_leap_xe_voc.ckpt", "bs_roformer_leap_inst.ckpt"])
            for path, value in zip(outputs, (.2, .4)):
                data, sr = sf.read(path, always_2d=True)
                self.assertEqual((len(data), sr), (4000, 16000))
                np.testing.assert_allclose(data, value, atol=1 / 32768)
            self.assertTrue(all(ref() is None for ref in references))

    def test_first_model_failure_is_not_replaced_by_a_second_model(self):
        runtime = separator._HybridLeapInstrumentalRuntime("models", "unused", "cpu")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(separator, "_load_audio_separator_model", side_effect=RuntimeError("inference failed")) as load,
            patch.object(separator, "_ensure_separator_pcm_wav", return_value="input.wav"),
        ):
            runtime.output_dir = tmp
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                runtime.separate("input.wav")
            self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
