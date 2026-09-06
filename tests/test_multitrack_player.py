import html
import asyncio
import json
import logging
import re
import sys
import unittest
from pathlib import Path

from lib.logger import ColoredFormatter
from ui.gradio_assets import AFTER, BEFORE, patched_upload_asset
from ui.multitrack_player import player_html
from ui.server_runtime import selector_loop_factory

ROOT = Path(__file__).resolve().parents[1]


class MultitrackPlayerTests(unittest.TestCase):
    def test_uvicorn_custom_factory_returns_a_usable_event_loop(self):
        import uvicorn
        factory = uvicorn.Config(lambda scope, receive, send: None, loop='ui.server_runtime:selector_loop_factory').get_loop_factory()
        loop = factory()
        try:
            self.assertIsInstance(loop, asyncio.AbstractEventLoop)
            self.assertEqual(loop.run_until_complete(asyncio.sleep(0, result=42)), 42)
        finally:
            loop.close()

    def test_localized_player_embeds_offline_bundle(self):
        for language, tag in (("zh_CN", "zh"), ("en_US", "en")):
            markup = player_html(language)
            srcdoc = html.unescape(re.search(r'srcdoc="(.*)"></iframe>', markup, re.S).group(1))
            self.assertIn(f'<html lang="{tag}">', srcdoc)
            self.assertNotRegex(srcdoc, r'<script[^>]+src=')
            self.assertIn('rvc-tracks', srcdoc)

    def test_locales_cover_all_player_translation_keys(self):
        source = (ROOT / 'ui/multitrack/src/MultiTrackAudioMixer.tsx').read_text(encoding='utf-8')
        keys = set(re.findall(r'(?<![\w.])t\("(\w+)"', source))
        keys.update(re.findall(r': "(audioStem\w+)"', source))
        for language in ('zh_CN', 'en_US'):
            catalog = json.loads((ROOT / f'i18n/{language}.json').read_text(encoding='utf-8'))['multitrack']
            self.assertFalse(keys - catalog.keys())

    def test_gradio_upload_assigns_reactive_id_before_render_and_request(self):
        source = patched_upload_asset()
        self.assertNotIn(BEFORE, source)
        self.assertEqual(source.count(AFTER), 1)

    def test_exception_traceback_survives_colored_logging(self):
        try:
            raise ValueError('diagnostic marker')
        except ValueError:
            record = logging.LogRecord('test', logging.ERROR, __file__, 1, 'failure', (), sys.exc_info())
        formatted = ColoredFormatter().format(record)
        self.assertIn('Traceback (most recent call last)', formatted)
        self.assertIn('ValueError: diagnostic marker', formatted)


if __name__ == '__main__':
    unittest.main()
