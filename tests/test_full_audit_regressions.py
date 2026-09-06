import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fastapi
from fastapi.testclient import TestClient
from starlette.responses import FileResponse

from lib import device
from tools import apply_preset
from ui import app


class FullAuditRegressions(unittest.TestCase):
    def test_nonexistent_cuda_index_is_rejected(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            self.assertEqual(str(device.get_device('cuda:0')), 'cuda:0')
            with self.assertRaises(ValueError):
                device.get_device('cuda:1')

    def test_filter_clears_previous_character_selection(self):
        with patch.object(app, 'get_available_character_choices', return_value=[]), patch.object(app, 'get_downloaded_character_choices', return_value=[]):
            self.assertIsNone(app.update_download_choices('all', 'missing')['value'])
            self.assertIsNone(app.update_downloaded_choices('all', 'missing')['value'])

    def test_presets_apply_and_restore_without_changing_other_settings(self):
        original = {'device':'cpu', 'language':'en_US', 'cover':{'index_rate':0.3,'protect':0.3,'rms_mix_rate':0.8}}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.json'
            backup = Path(directory) / 'backup.json'
            with patch.object(apply_preset, 'CONFIG_FILE', config), patch.object(apply_preset, 'BACKUP_FILE', backup):
                for name in apply_preset.PRESETS.values():
                    config.write_text(json.dumps(original))
                    apply_preset.apply_preset(name)
                    self.assertEqual(json.loads(config.read_text())['language'], 'en_US')
                    apply_preset.restore_config()
                    self.assertEqual(json.loads(config.read_text()), original)

    def test_gradio_prefixed_route_preserves_range_and_errors(self):
        server = fastapi.FastAPI()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'audio.wav'
            source.write_bytes(b'0123456789')

            @server.get('/gradio_api/file={path_or_url:path}')
            async def download(path_or_url: str):
                if path_or_url != 'audio.wav':
                    raise fastapi.HTTPException(403)
                return FileResponse(source)

            class Blocks:
                server_app = server

            app._patch_gradio_file_download(Blocks())
            app._patch_gradio_file_download(Blocks())
            client = TestClient(server)
            response = client.get('/gradio_api/file=audio.wav', headers={'Range':'bytes=2-5'})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b'2345')
            self.assertIn('audio.wav', response.headers['content-disposition'])
            self.assertEqual(client.get('/gradio_api/file=forbidden').status_code, 403)


if __name__ == '__main__':
    unittest.main()
