import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tools import character_models
from tools.character_assets import inspect_character_directory, model_files
from tests.character_fixtures import write_character_fixture


class CharacterAssetValidationTests(unittest.TestCase):
    def test_architecture_comes_from_weights_not_distribution_name(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            directory = Path(tmp) / 'assets/weights/characters/fake_v2'
            directory.mkdir(parents=True)
            write_character_fixture(directory / 'fake_v2.pth', directory / 'fake_v2.index')
            record = character_models._build_character_record('fake_v2', {'zh_name': 'fixture', 'variant': 'v2'})
            self.assertEqual(record['compatibility_status'], 'validated')
            self.assertIn('RVC v1', record['version_label'])
            self.assertIn('256D', record['display'])

    def test_metadata_only_directory_is_explicitly_invalid(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            directory = Path(tmp) / 'assets/weights/characters/missing'
            directory.mkdir(parents=True)
            (directory / 'metadata.json').write_text('{}')
            record = character_models._build_character_record('missing', {'zh_name': 'fixture'})
            self.assertEqual(record['compatibility_status'], 'invalid')
            self.assertIn('.pth', record['compatibility_error'])

    def test_invalid_uploaded_checkpoint_is_rejected_not_listed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            weight = Path(tmp) / 'invalid.pth'
            weight.write_bytes(b'not-a-checkpoint')
            with self.assertRaises(Exception):
                character_models.import_custom_character_model(str(weight), display_name='invalid')
            self.assertEqual(character_models.list_downloaded_characters(), [])

    def test_multiple_weights_and_indices_are_not_arbitrarily_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'a.pth').write_bytes(b'a')
            (directory / 'b.pth').write_bytes(b'b')
            with self.assertRaisesRegex(ValueError, '多个 .pth'):
                model_files(directory)
            (directory / 'b.pth').unlink()
            (directory / 'a.index').write_bytes(b'a')
            (directory / 'b.index').write_bytes(b'b')
            with self.assertRaisesRegex(ValueError, '多个索引'):
                model_files(directory)

    def test_downloaded_archive_without_weights_never_reports_success(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            archive = Path(tmp) / 'invalid.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('metadata.json', '{}')
            progress = []
            with patch.dict(character_models.CHARACTER_MODELS, {'fixture': {'file': 'invalid.zip'}}), \
                 patch.object(character_models, 'hf_hub_download', return_value=str(archive)):
                with self.assertRaisesRegex(ValueError, '没有 .pth'):
                    character_models.download_character_model('fixture', lambda msg, value: progress.append(value))
            self.assertNotIn(1.0, progress)
            self.assertFalse((Path(tmp) / 'assets/weights/characters/fixture').exists())

    def test_complete_download_is_validated_and_previous_files_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(character_models, 'get_project_root', return_value=Path(tmp)):
            root = Path(tmp)
            weight, index = root / 'voice.pth', root / 'voice.index'
            write_character_fixture(weight, index)
            archive = root / 'voice.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.write(weight, 'nested/voice.pth')
                stream.write(index, 'nested/voice.index')
            target = root / 'assets/weights/characters/fixture'
            target.mkdir(parents=True)
            (target / 'user-note.txt').write_text('preserve me')
            with patch.dict(character_models.CHARACTER_MODELS, {'fixture': {'file': 'voice.zip'}}), \
                 patch.object(character_models, 'hf_hub_download', return_value=str(archive)):
                self.assertTrue(character_models.download_character_model('fixture'))
            self.assertEqual(inspect_character_directory(target)['feature_dim'], 256)
            notes = list((root / 'temp/downloads/characters').rglob('user-note.txt'))
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].read_text(), 'preserve me')
            self.assertEqual(json.loads((target / 'ai_rvc_model.json').read_text(encoding='utf8'))['compatibility_status'], 'validated')

    def test_partial_multifile_download_cannot_be_treated_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_character_fixture(directory / 'voice.pth')
            with self.assertRaisesRegex(FileNotFoundError, '缺少 .index'):
                inspect_character_directory(directory, require_index=True)


if __name__ == '__main__':
    unittest.main()
