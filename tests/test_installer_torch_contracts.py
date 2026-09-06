"""Backend upgrades must satisfy upstream metadata without replacing the stack."""
import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

import install
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


class InstallerTorchContractTests(unittest.TestCase):
    def test_platform_minimum_matches_upstream_and_requirements(self):
        source = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        requirements = [Requirement(line) for line in source.splitlines() if line.startswith("torch>")]
        for system, machine, minimum in (("win32", "AMD64", "2.3.0"), ("linux", "x86_64", "2.3.0"),
                                         ("darwin", "arm64", "2.13.0"), ("darwin", "x86_64", "2.3.0")):
            with self.subTest(system=system, machine=machine), mock.patch.object(sys, "platform", system), \
                    mock.patch("platform.machine", return_value=machine):
                values = runpy.run_path(str(ROOT / "install.py"), run_name="installer_contract")
                self.assertEqual(values["PACKAGES"]["torch"]["min_version"], minimum)
                active = [r for r in requirements if r.marker.evaluate({"sys_platform": system, "platform_machine": machine})]
                self.assertEqual(len(active), 1)
                self.assertIn(minimum, active[0].specifier)
                self.assertNotIn("2.2.0", active[0].specifier)
                self.assertNotIn("3.0.0", active[0].specifier)
                if machine == "arm64":
                    self.assertNotIn("2.12.0", active[0].specifier)

    def test_old_torch_is_marked_before_installing_separator(self):
        with mock.patch.dict(install.PACKAGES, {"torch": install.PACKAGES["torch"]}, clear=True), \
                mock.patch("install.check_package", return_value=True), \
                mock.patch("install.get_installed_version", return_value="2.2.0"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(install.check_all("python"), [install.PACKAGES["torch"]])

    def test_specialized_missing_stack_stops_before_any_dependency_install(self):
        for backend in ("xpu", "rocm", "directml"):
            with self.subTest(backend=backend), mock.patch("install.check_all", return_value=[
                install.PACKAGES["torch"], install.PACKAGES["audio_separator"],
            ]), mock.patch("install.get_installed_version", return_value=None), \
                    mock.patch("install.pip_install") as package, \
                    mock.patch("install.pip_install_packages") as stack, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(install.install_all("python", backend=backend))
                package.assert_not_called()
                stack.assert_not_called()

    def test_failed_stack_install_stops_before_other_dependencies(self):
        with mock.patch("install.check_all", return_value=[install.PACKAGES["torch"], install.PACKAGES["audio_separator"]]), \
                mock.patch("install.get_installed_version", return_value=None), \
                mock.patch("install.pip_install_packages", return_value=False), \
                mock.patch("install.pip_install") as package, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.install_all("python", backend="cpu"))
            package.assert_not_called()

    def test_every_dependency_keeps_all_three_versions_and_build_suffixes(self):
        for backend, suffix in (("cpu", "+cpu"), ("cuda", "+cu128"), ("xpu", "+xpu"),
                                ("rocm", "+rocm6.4"), ("directml", ""), ("mps", "")):
            versions = {"torch": f"2.13.0{suffix}", "torchvision": f"0.28.0{suffix}", "torchaudio": f"2.13.0{suffix}",
                        install.BACKEND_SETTINGS[backend]["runtime_dist"]: "1.23.2"}
            with self.subTest(backend=backend), mock.patch("install.check_all", return_value=[
                install.PACKAGES["audio_separator"], install.PACKAGES["transformers"],
            ]), mock.patch("install.get_installed_version", side_effect=lambda _py, name: versions.get(name)), \
                    mock.patch("install.detect_cuda_version", return_value="https://download.pytorch.org/whl/cu128"), \
                    mock.patch("install.check_backend_available", return_value=True), \
                    mock.patch("install.check_dependency_consistency", return_value=True), \
                    mock.patch("install.install_project_dependencies", return_value=True) as package, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(install.install_all("python", backend=backend))
                self.assertEqual(package.call_count, 1)
                expected = tuple(f"{name}=={versions[name]}" for name in install.TORCH_STACK_NAMES)
                for call in package.call_args_list:
                    self.assertEqual(call.args[2], expected)

    def test_complete_manifest_is_resolved_with_the_selected_runtime(self):
        pins = ("torch==2.11.0+cu128", "torchvision==0.26.0+cu128", "torchaudio==2.11.0+cu128")
        with mock.patch("install.pip_install_packages", return_value=True) as pip:
            self.assertTrue(install.install_project_dependencies("python", "cuda", pins))
            self.assertEqual(pip.call_args.args[1], (
                "-r", str(ROOT / "requirements.txt"), "audio-separator[gpu]==0.47.0", *pins,
            ))

    def test_installed_but_inconsistent_environment_is_resolved_again(self):
        versions = {"torch": "2.11.0+cpu", "torchvision": "0.26.0+cpu", "torchaudio": "2.11.0+cpu", "onnxruntime": "1.23.2"}
        with mock.patch("install.check_all", return_value=[]), \
                mock.patch("install.get_installed_version", side_effect=lambda _py, name: versions.get(name)), \
                mock.patch("install.check_backend_available", return_value=True), \
                mock.patch("install.check_dependency_consistency", side_effect=[False, True]), \
                mock.patch("install.install_project_dependencies", return_value=True) as project, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.install_all("python", backend="cpu"))
            project.assert_called_once()

    def test_pip_receives_pins_in_the_same_resolver_transaction(self):
        pins = ("torch==2.11.0+xpu", "torchvision==0.26.0+xpu", "torchaudio==2.11.0+xpu")
        with mock.patch("install.subprocess.run", return_value=mock.Mock(returncode=1, stderr="ResolutionImpossible")) as run, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(install.pip_install("python", "audio-separator", extra="cpu", version_spec="==0.47.0", pinned_packages=pins))
            self.assertEqual(run.call_args.args[0], ["python", "-m", "pip", "install", "audio-separator[cpu]==0.47.0", *pins])
            self.assertIn("ResolutionImpossible", output.getvalue())

    def test_unknown_stack_version_stops_without_guessing_a_version(self):
        with mock.patch("install.check_all", return_value=[install.PACKAGES["audio_separator"]]), \
                mock.patch("install.get_installed_version", return_value=None), \
                mock.patch("install.pip_install") as package, contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.install_all("python", backend="cpu"))
            package.assert_not_called()


if __name__ == "__main__":
    unittest.main()
