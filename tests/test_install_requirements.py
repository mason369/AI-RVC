import contextlib
import io
import unittest
from unittest import mock

import install


class InstallRequirementTests(unittest.TestCase):
    def test_audio_separator_below_required_version_is_marked_for_install(self):
        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            return_value="0.41.1",
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertIn(
            "audio-separator",
            {info["pip"] for info in missing},
        )

    def test_audio_separator_at_required_version_is_accepted(self):
        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            return_value="0.44.1",
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertNotIn(
            "audio-separator",
            {info["pip"] for info in missing},
        )

    def test_numpy_2_is_marked_for_downgrade(self):
        def version_for_package(_venv_py, distribution_name):
            if distribution_name == "numpy":
                return "2.2.6"
            if distribution_name == "audio-separator":
                return "0.44.1"
            return None

        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            side_effect=version_for_package,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertIn("numpy<2,>=1.23.0", {info["pip"] for info in missing})

    def test_audio_separator_install_restores_numpy_1_x(self):
        audio_separator_info = install.PACKAGES["audio_separator"]
        calls = []

        def fake_pip_install(_venv_py, package, **kwargs):
            calls.append((package, kwargs))
            return True

        with mock.patch("install.check_all", return_value=[audio_separator_info]), mock.patch(
            "install.detect_cuda_version",
            return_value=None,
        ), mock.patch("install.pip_install", side_effect=fake_pip_install), contextlib.redirect_stdout(
            io.StringIO()
        ):
            ok = install.install_all("python", gpu=False)

        self.assertTrue(ok)
        self.assertEqual(calls[0][0], "audio-separator")
        self.assertEqual(calls[0][1]["version_spec"], ">=0.44.1")
        self.assertIn(("numpy<2,>=1.23.0", {}), calls)


if __name__ == "__main__":
    unittest.main()
