import unittest
from unittest.mock import patch


class RemovedSeparatorTests(unittest.TestCase):
    def test_retired_model_ids_stop_before_runtime_creation_or_download(self):
        from infer import separator

        retired = (
            "bs_polarformer_public_onnx_62bands",
            "bs_polarformer.onnx",
            "bs_polarformer_124bands_fp16",
            "hybrid:leap_xe90_vocals+polarformer62_instrumental",
            "ensemble:PolarFormer",
            [separator.LEAP_XE_VOCALS_MODEL, "bs_polarformer.onnx"],
            ("BS_POLARFORMER.ONNX",),
        )
        with patch.object(separator, "Separator") as runtime:
            for model_spec in retired:
                with self.subTest(model_spec=model_spec):
                    with self.assertRaisesRegex(ValueError, "PolarFormer 已移除"):
                        separator._load_audio_separator_model(
                            model_spec=model_spec,
                            output_dir="unused",
                            model_dir="unused",
                            device="cpu",
                        )
            runtime.assert_not_called()
