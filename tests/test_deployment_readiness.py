import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import install

ROOT = Path(__file__).resolve().parents[1]


class DeploymentReadinessTests(unittest.TestCase):
    def test_bootstrap_uses_shared_requirements_and_exposes_failure(self):
        for code in (0, 1):
            with self.subTest(code=code), mock.patch(
                'install.subprocess.run',
                return_value=mock.Mock(returncode=code, stderr='resolver failed', stdout=''),
            ) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(install.prepare_install_tools('isolated-python'), code == 0)
                self.assertEqual(run.call_args.args[0], [
                    'isolated-python', '-m', 'pip', 'install', '-r', str(ROOT / 'pre-requirements.txt'),
                ])

    def test_existing_venv_propagates_bootstrap_failure(self):
        with mock.patch('install.os.path.isfile', return_value=True), mock.patch(
            'install.subprocess.run', return_value=mock.Mock(returncode=0, stdout='Python 3.10.12'),
        ), mock.patch('install.prepare_install_tools', return_value=False), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(install.create_venv())

    def test_space_includes_every_local_feature_dependency(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        def names(filename):
            return {
                canonicalize_name(Requirement(line.split('#')[0].strip()).name)
                for line in (ROOT / filename).read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.lstrip().startswith(('#', '-'))
            }
        self.assertEqual(names('requirements.txt') - names('requirements_hf.txt'), set())

    def test_release_build_does_not_skip_dependency_resolution(self):
        source = (ROOT / '.github/workflows/build-executables.yml').read_text(encoding='utf-8')
        self.assertNotIn('--no-deps', source)
        self.assertIn('python -m pip check', source)
        self.assertIn('pip install -r requirements.txt', source)

    def test_native_xpu_does_not_require_ipex(self):
        from lib import device
        with mock.patch.object(device.torch, 'xpu', create=True) as xpu:
            xpu.is_available.return_value = True
            self.assertTrue(device._has_xpu())
            xpu.is_available.return_value = False
            self.assertFalse(device._has_xpu())
        with mock.patch('install.subprocess.run', return_value=mock.Mock(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(install.check_backend_available('python', 'xpu'))
            self.assertNotIn('intel_extension_for_pytorch', run.call_args.args[0][-1])


if __name__ == '__main__':
    unittest.main()
