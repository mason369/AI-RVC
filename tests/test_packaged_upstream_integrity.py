"""Portable runtimes verify immutable source contents without system Git."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import upstream_runtime as runtime


def write_fixture(project):
    root = runtime.runtime_root(project)
    files = {}
    for name in runtime.REQUIRED_FILES['vc']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = '# fixture\n'
        if name == 'infer/vc/modules.py':
            text = 'class VC:\n    def vc_single(' + ', '.join(runtime.VC_SINGLE_PARAMETERS) + '):\n        pass\n'
        path.write_bytes(text.encode())
        files[name] = {'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': True}
    manifest = project / 'upstream_source_manifest.json'
    manifest.write_text(json.dumps({'vc': {'revision': runtime.REVISIONS['vc'], 'files': files}}))
    return root, manifest


class PackagedUpstreamIntegrityTests(unittest.TestCase):
    def test_pinned_portable_tree_accepts_crlf_without_invoking_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root, _ = write_fixture(project)
            for path in root.rglob('*.py'):
                path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
            with patch.object(runtime, '__file__', str(project / 'upstream_runtime.py')), \
                    patch.object(runtime.sys, 'frozen', True, create=True), \
                    patch.object(runtime, '_git', side_effect=AssertionError('Git must not run')):
                self.assertEqual(runtime.ensure_source(project), root)

    def test_modified_or_extra_python_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root, _ = write_fixture(project)
            path = root / 'infer/hubert.py'
            path.write_bytes(b'# changed\n')
            (root / 'extra.py').write_bytes(b'# unexpected\n')
            with patch.object(runtime, '__file__', str(project / 'upstream_runtime.py')), \
                    patch.object(runtime.sys, 'frozen', True, create=True):
                problems = runtime.source_problems(root)
            self.assertTrue(any('SHA-256' in problem for problem in problems))
            self.assertTrue(any('extra.py' in problem for problem in problems))

    def test_missing_or_wrong_manifest_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            root, manifest = write_fixture(project)
            payload = json.loads(manifest.read_text())
            payload['vc']['revision'] = '0' * 40
            manifest.write_text(json.dumps(payload))
            with patch.object(runtime, '__file__', str(project / 'upstream_runtime.py')), \
                    patch.object(runtime.sys, 'frozen', True, create=True):
                self.assertTrue(runtime.source_problems(root))
                manifest.unlink()
                self.assertTrue(runtime.source_problems(root))

    def test_missing_portable_source_never_downloads_or_uses_git(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runtime.sys, 'frozen', True, create=True), \
                patch.object(runtime, '_git') as git:
            with self.assertRaises(RuntimeError):
                runtime.ensure_source(Path(tmp))
            git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
