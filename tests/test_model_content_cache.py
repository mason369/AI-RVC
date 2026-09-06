"""Model integrity must survive same-size writes with unchanged timestamps."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import download_models
from tools.character_assets import inspect_character_directory
from tests.character_fixtures import write_character_fixture


class ModelContentCacheTests(unittest.TestCase):
    def test_base_asset_overwrite_invalidates_verification_with_identical_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'model.pth'
            valid = b'valid model content'
            spec = {'path': 'model.pth', 'size_bytes': len(valid),
                    'sha256': hashlib.sha256(valid).hexdigest()}
            with patch.object(download_models, 'MODELS', {'model': spec}), \
                    patch.object(download_models, 'get_project_root', return_value=root):
                path.write_bytes(valid)
                self.assertTrue(download_models.check_model('model'))
                stamp = path.stat()
                path.write_bytes(b'x' * len(valid))
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns)
                self.assertFalse(download_models.check_model('model'))
                path.write_bytes(valid)
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertTrue(download_models.check_model('model'))

    def test_character_weight_and_index_overwrites_cannot_reuse_cached_validation(self):
        for changed in ('voice.pth', 'voice.index'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_character_fixture(root / 'voice.pth', root / 'voice.index')
                self.assertEqual(inspect_character_directory(root)['feature_dim'], 256)
                path = root / changed
                stamp = path.stat()
                path.write_bytes(bytes(stamp.st_size))
                os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                self.assertEqual(path.stat().st_mtime_ns, stamp.st_mtime_ns)
                with self.assertRaises(Exception):
                    inspect_character_directory(root)


if __name__ == '__main__':
    unittest.main()
