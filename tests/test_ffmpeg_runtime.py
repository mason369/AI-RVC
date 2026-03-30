import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_ffmpeg_runtime_module():
    module_path = REPO_ROOT / "lib" / "ffmpeg_runtime.py"
    spec = importlib.util.spec_from_file_location("ffmpeg_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FfmpegRuntimeTests(unittest.TestCase):
    def test_bundled_ffmpeg_dir_is_preferred(self):
        module = _load_ffmpeg_runtime_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "tools" / "ffmpeg" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "ffmpeg.exe").write_bytes(b"exe")

            resolved = module.get_ffmpeg_bin_dir(root_dir=root)

        self.assertEqual(resolved, bin_dir)

    def test_runtime_setup_prepends_bundled_ffmpeg_to_path(self):
        module = _load_ffmpeg_runtime_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "tools" / "ffmpeg" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            ffmpeg = bin_dir / "ffmpeg.exe"
            ffmpeg.write_bytes(b"exe")

            env = {"PATH": r"C:\Windows\System32"}
            module.configure_ffmpeg_runtime(root_dir=root, env=env)

        self.assertTrue(env["PATH"].startswith(str(bin_dir)))
        self.assertEqual(env["FFMPEG_BINARY"], str(ffmpeg))

    def test_uvr5_module_does_not_shell_out_to_ffprobe(self):
        source = (REPO_ROOT / "infer" / "modules" / "uvr5" / "modules.py").read_text(encoding="utf-8")

        self.assertNotIn('cmd="ffprobe"', source)

    def test_run_configures_bundled_ffmpeg_runtime(self):
        source = (REPO_ROOT / "run.py").read_text(encoding="utf-8")

        self.assertIn("configure_ffmpeg_runtime()", source)

    def test_workflow_packages_bundled_ffmpeg_directory(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "build-executables.yml").read_text(encoding="utf-8")

        self.assertIn("tools/ffmpeg", workflow)


if __name__ == "__main__":
    unittest.main()
