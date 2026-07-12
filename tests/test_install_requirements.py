import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import install


class InstallRequirementTests(unittest.TestCase):
    def test_pip_install_exposes_dependency_resolution_failure(self):
        failure = mock.Mock(
            returncode=1,
            stdout="",
            stderr="ERROR: ResolutionImpossible\n",
        )
        with mock.patch("install.subprocess.run", return_value=failure) as run_mock, \
                contextlib.redirect_stdout(io.StringIO()):
            ok = install.pip_install("python", "fairseq", version_spec="==0.12.2")

        self.assertFalse(ok)
        run_mock.assert_called_once()
        self.assertNotIn("--no-deps", run_mock.call_args.args[0])

    def test_runtime_check_does_not_claim_implicit_cpu_selection(self):
        source = Path("run.py").read_text(encoding="utf-8")
        self.assertNotIn("未检测到 GPU 加速，将使用 CPU", source)
        self.assertIn("若该设备不可用会停止并报错", source)

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

        def installed_version(_venv_py, distribution_name):
            if distribution_name == "onnxruntime-gpu":
                return None
            if distribution_name == "onnxruntime":
                return "1.23.2"
            return None

        with mock.patch("install.check_all", return_value=[audio_separator_info]), mock.patch(
            "install.detect_cuda_version",
            return_value=None,
        ), mock.patch("install.get_installed_version", side_effect=installed_version), mock.patch(
            "install.pip_install", side_effect=fake_pip_install
        ), mock.patch(
            "install.check_dependency_consistency", return_value=True
        ), mock.patch(
            "install.check_backend_available", return_value=True
        ), contextlib.redirect_stdout(io.StringIO()):
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

    def test_torch_stack_is_installed_together_from_one_index(self):
        missing = [install.PACKAGES["torchvision"]]
        with mock.patch("install.check_all", return_value=missing), mock.patch(
            "install.detect_cuda_version",
            return_value="https://download.pytorch.org/whl/cu126",
        ), mock.patch(
            "install.get_installed_version",
            side_effect=lambda _python, dist: "1.23.2" if dist == "onnxruntime-gpu" else None,
        ), mock.patch(
            "install.pip_install_packages",
            return_value=True,
        ) as install_stack, mock.patch(
            "install.check_dependency_consistency",
            return_value=True,
        ), mock.patch(
            "install.check_backend_available",
            return_value=True,
        ), contextlib.redirect_stdout(io.StringIO()):
            ok = install.install_all("python", gpu=True)

        self.assertTrue(ok)
        install_stack.assert_called_once_with(
            "python",
            ("torch", "torchvision", "torchaudio"),
            index_url="https://download.pytorch.org/whl/cu126",
        )

    def test_dependency_consistency_failure_is_exposed(self):
        failure = mock.Mock(
            returncode=1,
            stdout="torchvision requires torch==2.5.1, but torch 2.11.0 is installed.\n",
            stderr="",
        )
        with mock.patch("install.subprocess.run", return_value=failure), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            ok = install.check_dependency_consistency("python")

        self.assertFalse(ok)
        self.assertIn("torchvision requires torch", output.getvalue())

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

    def test_onnx_runtime_comes_only_from_audio_separator_extra(self):
        self.assertNotIn("onnxruntime", install.PACKAGES)
        for requirements_path in (Path("requirements.txt"), Path("requirements_hf.txt")):
            direct_runtime_lines = [
                line
                for line in requirements_path.read_text(encoding="utf-8").splitlines()
                if line.strip().lower().startswith("onnxruntime")
            ]
            self.assertEqual(direct_runtime_lines, [])

    def test_requirements_include_torchvision_for_audio_separator(self):
        for requirements_path in (Path("requirements.txt"), Path("requirements_hf.txt")):
            source = requirements_path.read_text(encoding="utf-8")
            self.assertIn("torchvision>=0.15.0", source)

    def test_cpu_install_writes_explicit_cpu_device(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "configs" / "config.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                json.dumps({"device": "cuda", "cover": {}}),
                encoding="utf-8",
            )

            with mock.patch.object(install, "ROOT_DIR", root):
                install.save_runtime_device("cpu")

            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["device"], "cpu")

    def test_runtime_device_supports_every_declared_backend(self):
        expected = {
            "cpu": "cpu",
            "cuda": "cuda",
            "rocm": "cuda",
            "xpu": "xpu",
            "directml": "directml",
            "mps": "mps",
        }
        for backend, device in expected.items():
            with self.subTest(backend=backend):
                self.assertEqual(
                    install.BACKEND_SETTINGS[backend]["runtime_device"],
                    device,
                )

    def test_backend_settings_select_matching_runtime_extras(self):
        expected = {
            "cpu": ("cpu", "onnxruntime"),
            "cuda": ("gpu", "onnxruntime-gpu"),
            "rocm": ("cpu", "onnxruntime"),
            "xpu": ("cpu", "onnxruntime"),
            "directml": ("dml", "onnxruntime-directml"),
            "mps": ("cpu", "onnxruntime"),
        }
        for backend, values in expected.items():
            with self.subTest(backend=backend):
                settings = install.BACKEND_SETTINGS[backend]
                self.assertEqual(
                    (settings["audio_extra"], settings["runtime_dist"]),
                    values,
                )

    def test_auto_backend_uses_mps_only_on_apple_silicon(self):
        with mock.patch.object(install.sys, "platform", "darwin"), mock.patch(
            "install.platform.machine",
            return_value="arm64",
        ):
            self.assertEqual(install.resolve_install_backend("auto"), "mps")

        with mock.patch.object(install.sys, "platform", "darwin"), mock.patch(
            "install.platform.machine",
            return_value="x86_64",
        ):
            self.assertEqual(install.resolve_install_backend("auto"), "cpu")

    def test_directml_install_uses_dml_extra_without_replacing_torch(self):
        audio_separator_info = install.PACKAGES["audio_separator"]
        with mock.patch("install.check_all", return_value=[audio_separator_info]), mock.patch(
            "install.get_installed_version",
            side_effect=lambda _python, dist: (
                "1.23.2" if dist == "onnxruntime-directml" else None
            ),
        ), mock.patch("install.pip_install", return_value=True) as pip_install, mock.patch(
            "install.pip_install_packages", return_value=True
        ) as install_torch, mock.patch(
            "install.check_backend_available", return_value=True
        ), mock.patch(
            "install.check_dependency_consistency", return_value=True
        ), contextlib.redirect_stdout(io.StringIO()):
            ok = install.install_all("python", backend="directml")

        self.assertTrue(ok)
        install_torch.assert_not_called()
        pip_install.assert_called_once_with(
            "python",
            "audio-separator",
            extra="dml",
            version_spec="==0.44.1",
        )

    def test_specialized_backend_does_not_replace_missing_torch_stack(self):
        for backend in ("rocm", "xpu", "directml"):
            with self.subTest(backend=backend), mock.patch(
                "install.check_all", return_value=[install.PACKAGES["torch"]]
            ), mock.patch(
                "install.get_installed_version",
                side_effect=lambda _python, dist: (
                    "1.23.2"
                    if dist == install.BACKEND_SETTINGS[backend]["runtime_dist"]
                    else None
                ),
            ), mock.patch(
                "install.pip_install_packages", return_value=True
            ) as install_torch, mock.patch(
                "install.check_backend_available", return_value=False
            ), mock.patch(
                "install.check_dependency_consistency", return_value=True
            ), contextlib.redirect_stdout(io.StringIO()):
                ok = install.install_all("python", backend=backend)

            self.assertFalse(ok)
            install_torch.assert_not_called()

    def test_runtime_variant_check_rejects_conflicting_distribution(self):
        versions = {
            "onnxruntime": "1.23.2",
            "onnxruntime-gpu": "1.23.2",
        }
        with mock.patch(
            "install.get_installed_version",
            side_effect=lambda _python, dist: versions.get(dist),
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.check_onnx_runtime_variant("python", gpu=True))

        versions["onnxruntime"] = None
        with mock.patch(
            "install.get_installed_version",
            side_effect=lambda _python, dist: versions.get(dist),
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.check_onnx_runtime_variant("python", gpu=True))

    def test_directml_runtime_variant_rejects_other_onnx_distributions(self):
        versions = {
            "onnxruntime": "1.23.2",
            "onnxruntime-directml": "1.23.2",
        }
        with mock.patch(
            "install.get_installed_version",
            side_effect=lambda _python, dist: versions.get(dist),
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(
                install.check_onnx_runtime_variant("python", backend="directml")
            )

    def test_backend_requirement_files_select_explicit_runtime_extra(self):
        expected = {
            "requirements_cpu.txt": "audio-separator[cpu]==0.44.1",
            "requirements_cuda.txt": "audio-separator[gpu]==0.44.1",
            "requirements_dml.txt": "audio-separator[dml]==0.44.1",
        }
        for filename, requirement in expected.items():
            with self.subTest(filename=filename):
                source = Path(filename).read_text(encoding="utf-8")
                self.assertIn("-r requirements.txt", source)
                self.assertIn(requirement, source)

    def test_portable_build_sets_device_and_has_no_cpu_fallback_claim(self):
        workflow = Path(".github/workflows/build-executables.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'config["device"] = "cpu" if sys.argv[1] == "CPU" else "cuda"',
            workflow,
        )
        self.assertNotIn("自动回退到 CPU 推理", workflow)
        self.assertNotIn('"onnxruntime>=1.18.0"', workflow)


if __name__ == "__main__":
    unittest.main()
