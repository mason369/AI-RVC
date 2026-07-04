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

    def test_numpy_1_is_marked_for_upgrade(self):
        def version_for_package(_venv_py, distribution_name):
            if distribution_name == "numpy":
                return "1.26.4"
            if distribution_name == "audio-separator":
                return "0.44.1"
            if distribution_name == "gradio":
                return "5.49.1"
            if distribution_name == "huggingface_hub":
                return "0.36.0"
            if distribution_name == "fairseq":
                return "0.12.2"
            return None

        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            side_effect=version_for_package,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertIn("numpy>=2,<3", {info["pip"] for info in missing})

    def test_gradio_wrong_version_is_marked_for_upgrade(self):
        def version_for_package(_venv_py, distribution_name):
            versions = {
                "gradio": "3.50.2",
                "numpy": "2.2.6",
                "audio-separator": "0.44.1",
                "huggingface_hub": "0.36.0",
                "fairseq": "0.12.2",
            }
            return versions.get(distribution_name)

        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            side_effect=version_for_package,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertIn("gradio==5.49.1", {info["pip"] for info in missing})

    def test_huggingface_hub_1_is_marked_for_downgrade(self):
        def version_for_package(_venv_py, distribution_name):
            versions = {
                "gradio": "5.49.1",
                "numpy": "2.2.6",
                "audio-separator": "0.44.1",
                "huggingface_hub": "1.0.1",
                "fairseq": "0.12.2",
            }
            return versions.get(distribution_name)

        with mock.patch("install.check_package", return_value=True), mock.patch(
            "install.get_installed_version",
            side_effect=version_for_package,
            create=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            missing = install.check_all("python")

        self.assertIn(
            "huggingface_hub>=0.19.0,<1.0",
            {info["pip"] for info in missing},
        )

    def test_audio_separator_install_keeps_declared_numpy_2_stack(self):
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
        self.assertEqual(
            calls,
            [
                (
                    "audio-separator",
                    {"extra": "cpu", "version_spec": "==0.44.1"},
                )
            ],
        )

    def test_cuda_13_uses_latest_supported_pytorch_wheel(self):
        def fake_run(cmd, **_kwargs):
            if cmd == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]:
                return mock.Mock(returncode=0, stdout="575.51.03\n")
            if cmd == ["nvidia-smi"]:
                return mock.Mock(returncode=0, stdout="CUDA Version: 13.0\n")
            raise AssertionError(f"Unexpected command: {cmd}")

        with mock.patch("install.subprocess.run", side_effect=fake_run):
            self.assertEqual(
                install.detect_cuda_version(),
                "https://download.pytorch.org/whl/cu126",
            )

    def test_gpu_install_does_not_fall_back_to_cpu_torch(self):
        calls = []

        def fake_pip_install(_venv_py, package, **kwargs):
            calls.append((package, kwargs))
            return True

        with mock.patch(
            "install.check_all",
            return_value=[install.PACKAGES["torch"], install.PACKAGES["torchaudio"]],
        ), mock.patch("install.detect_cuda_version", return_value=None), mock.patch(
            "install.pip_install",
            side_effect=fake_pip_install,
        ), contextlib.redirect_stdout(io.StringIO()):
            ok = install.install_all("python", gpu=True)

        self.assertFalse(ok)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
