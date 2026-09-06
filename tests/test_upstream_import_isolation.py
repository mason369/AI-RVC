"""Official packages must not hide the frozen app's dependency search path."""
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from lib.upstream_imports import activate_upstream_packages


class UpstreamImportIsolationTests(unittest.TestCase):
    def test_official_packages_replace_application_packages_and_keep_dependencies(self):
        for regular in (False, True):
            with self.subTest(regular=regular), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app, upstream = root / "application", root / "upstream"
                for name in ("configs", "i18n", "infer", "tools"):
                    (app / name).mkdir(parents=True)
                    (app / name / "__init__.py").write_text("ORIGIN = 'application'", encoding="utf-8")
                    (app / name / "probe.py").write_text("ORIGIN = 'application'", encoding="utf-8")
                    (upstream / name).mkdir(parents=True)
                    if regular:
                        (upstream / name / "__init__.py").write_text("ORIGIN = 'upstream'", encoding="utf-8")
                    (upstream / name / "probe.py").write_text("ORIGIN = 'upstream'", encoding="utf-8")
                archive = root / "bundled-dependencies.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr("audit_dependency.py", "ORIGIN = 'bundled'\n")
                script = f"""
import importlib, sys
from pathlib import Path
from lib.upstream_imports import activate_upstream_packages
sys.path[:0] = [{str(app)!r}, {str(archive)!r}]
import infer.probe
assert infer.probe.ORIGIN == 'application'
before = list(sys.path)
activate_upstream_packages(Path({str(upstream)!r}))
assert all(entry in sys.path for entry in before)
for name in ('configs', 'i18n', 'infer', 'tools'):
    module = importlib.import_module(name + '.probe')
    assert module.ORIGIN == 'upstream', module.__file__
import audit_dependency
assert audit_dependency.ORIGIN == 'bundled'
"""
                result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_upstream_package_does_not_activate_application_code(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(sys, "path", list(sys.path)):
            before = list(sys.path)
            with self.assertRaises(FileNotFoundError):
                activate_upstream_packages(Path(temporary))
            self.assertEqual(sys.path, before)


if __name__ == "__main__":
    unittest.main()
