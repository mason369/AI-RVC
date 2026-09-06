import unittest
from pathlib import Path
from unittest import mock

from tools import download_models


class UpstreamPreparationTests(unittest.TestCase):
    def test_clean_model_preparation_includes_every_official_runtime(self):
        for prepare in (download_models.download_required_models, download_models.download_all_models):
            with (
                self.subTest(entrypoint=prepare.__name__),
                mock.patch.object(download_models, "check_model", return_value=True),
                mock.patch.object(download_models, "download_default_separator_models", return_value=True),
                mock.patch.object(download_models, "ensure_upstream_rvc_tree", return_value=Path("prepared")) as source,
            ):
                self.assertTrue(prepare())
                self.assertEqual(source.call_args_list, [mock.call(capability="vc"), mock.call(capability="uvr5")])

    def test_missing_uvr5_is_a_failed_preparation_even_when_vc_succeeds(self):
        for prepare in (download_models.download_required_models, download_models.download_all_models):
            with (
                self.subTest(entrypoint=prepare.__name__),
                mock.patch.object(download_models, "check_model", return_value=True),
                mock.patch.object(download_models, "download_default_separator_models", return_value=True),
                mock.patch.object(download_models, "ensure_upstream_rvc_tree", side_effect=[Path("vc"), RuntimeError("uvr5 unavailable")]),
            ):
                self.assertFalse(prepare())


if __name__ == "__main__":
    unittest.main()
