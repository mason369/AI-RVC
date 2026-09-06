import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from tools import character_models, mega_download
from tests.character_fixtures import write_character_fixture


class CharacterDownloadContractsTests(unittest.TestCase):
    def test_general_model_list_exposes_ambiguity_without_hiding_entries(self):
        from infer.pipeline import list_voice_models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            character = root / 'characters/ambiguous'
            character.mkdir(parents=True)
            for filename in ('a.pth', 'b.pth'):
                (character / filename).write_bytes(b'weight')
            rows = list_voice_models(str(root))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all('多个 .pth' in row['index_error'] for row in rows))
            self.assertTrue(all(row['index_path'] is None for row in rows))

    def test_general_model_list_keeps_character_archive_index_pairing(self):
        from infer.pipeline import list_voice_models
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            character = root / 'characters/yu'
            character.mkdir(parents=True)
            (character / 'Yu.pth').write_bytes(b'weight')
            index = character / 'added_IVF_Yu.index'
            index.write_bytes(b'index')
            self.assertEqual(list_voice_models(str(root))[0]['index_path'], str(index))

    def test_official_and_diagnostic_paths_never_guess_or_ignore_explicit_missing_index(self):
        from infer.official_adapter import _resolve_index_path
        from tools.diagnose_vc_session import _resolve_index_for_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'Voice.pth'
            model.write_bytes(b'weight')
            (root / 'added_Voice_OTHER.index').write_bytes(b'wrong')
            for resolve in (_resolve_index_path, _resolve_index_for_model):
                self.assertIsNone(resolve(model, None))
                with self.assertRaises(FileNotFoundError):
                    resolve(model, str(root / 'missing.index'))

    def test_drive_uses_supported_confirmation_api_and_verifies_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'download.zip'
            def download(**kwargs):
                path.write_bytes(b'actual-download-test')
                return str(path)
            with patch('gdown.download', side_effect=download) as call:
                self.assertTrue(character_models._download_gdrive_file('public-id', path))
            call.assert_called_once_with(id='public-id', output=str(path), quiet=True,
                                         use_cookies=False, resume=True)
            with patch('gdown.download', return_value=None), self.assertRaisesRegex(RuntimeError, '有效文件'):
                character_models._download_gdrive_file('public-id', path)

    def test_mega_error_is_not_success_and_does_not_disclose_link(self):
        url = 'https://mega.nz/file/public#key'
        result = subprocess.CompletedProcess([], 1, '', f'ENOENT {url}')
        with tempfile.TemporaryDirectory() as tmp, patch.object(mega_download, 'prepare_megatools', return_value=Path('megatools')), \
             patch.object(mega_download.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'ENOENT') as caught:
                mega_download.download_public_file(url, Path(tmp) / 'download', Path(tmp))
            self.assertNotIn(url, str(caught.exception))

    def test_mega_requires_one_nonempty_file(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(mega_download, 'prepare_megatools', return_value=Path('megatools')):
            directory = Path(tmp) / 'download'
            def run(*args, **kwargs):
                (directory / 'model.zip').write_bytes(b'test-archive')
                return subprocess.CompletedProcess([], 0, '', '')
            with patch.object(mega_download.subprocess, 'run', side_effect=run):
                self.assertEqual(mega_download.download_public_file('public-link', directory, Path(tmp)), directory / 'model.zip')
            with patch.object(mega_download.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')):
                with self.assertRaisesRegex(RuntimeError, '非空模型文件'):
                    mega_download.download_public_file('public-link', Path(tmp) / 'empty', Path(tmp))

    def test_unsupported_platform_needs_a_real_installed_client(self):
        with patch.object(mega_download, 'build_key', return_value=None), \
             patch.object(mega_download.shutil, 'which', return_value=None):
            self.assertFalse(mega_download.download_supported())
            with self.assertRaisesRegex(RuntimeError, 'Megatools'):
                mega_download.prepare_megatools(Path('unused'))

    def test_rate_limit_stops_batch_and_never_reports_complete_progress(self):
        error = requests.HTTPError('rate limited')
        error.response = requests.Response()
        error.response.status_code = 429
        progress = []
        with patch.object(character_models, 'list_available_characters', return_value=[{'name': 'a'}, {'name': 'b'}]), \
             patch.object(character_models, 'download_character_model', side_effect=error) as download:
            result = character_models.download_all_character_models(progress_callback=lambda msg, value: progress.append(value))
        self.assertEqual(download.call_count, 1)
        self.assertEqual(result['success'], [])
        self.assertEqual(result['failed'], ['a'])
        self.assertEqual(result['not_attempted'], ['b'])
        self.assertNotIn(1, progress)

    def test_ambiguous_directory_remains_visible_but_has_no_selected_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            directory = Path(tmp) / 'assets/weights/characters/ambiguous'
            directory.mkdir(parents=True)
            (directory / 'a.pth').write_bytes(b'a')
            (directory / 'b.pth').write_bytes(b'b')
            records = character_models.list_downloaded_characters()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['compatibility_status'], 'invalid')
            self.assertIsNone(records[0]['model_path'])
            with self.assertRaisesRegex(ValueError, '多个 .pth'):
                character_models.get_character_model_path('ambiguous')

    def test_direct_weight_gets_the_same_actual_architecture_validation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            directory = Path(tmp) / 'assets/weights/characters'
            directory.mkdir(parents=True)
            write_character_fixture(directory / 'direct.pth', directory / 'direct.index')
            record = character_models.list_downloaded_characters()[0]
            self.assertEqual(record['compatibility_status'], 'validated')
            self.assertEqual(record['model_contract']['feature_dim'], 256)

    def test_similarity_is_not_an_index_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            weight = directory / 'voice.pth'
            weight.write_bytes(b'x')
            (directory / 'added_voice_v1.index').write_bytes(b'x')
            (directory / 'added_voice_v2.index').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, '配对不唯一'):
                character_models._find_index_file(weight)

    def test_known_unavailable_source_is_not_hidden_or_mislabeled_validated(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            record = character_models._build_character_record('shizuku_osaka', character_models.CHARACTER_MODELS['shizuku_osaka'])
            self.assertEqual(record['source_check']['reason'], 'MEGA ENOENT')
            self.assertEqual(record['compatibility_status'], 'unverified')
            self.assertEqual(len(character_models.CHARACTER_MODELS), 181)


if __name__ == '__main__':
    unittest.main()
