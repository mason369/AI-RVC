"""A startup-pinned device must never appear to be changed by an ignored save."""
import os
import unittest
from unittest import mock

import gradio as gr

from ui import app


class ManagedDeviceSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_device_is_readonly_and_api_cannot_persist_ignored_value(self):
        with mock.patch.dict(os.environ, {'AI_RVC_DEVICE': 'cpu'}), \
             mock.patch.dict(app.config, {'device': 'cpu'}), \
             mock.patch.object(app, 'update_config') as update:
            blocks = app.create_ui()
            radio = next(b for b in blocks.blocks.values() if isinstance(b, gr.Radio) and b.value == 'cpu')
            self.assertFalse(radio.interactive)
            save = next(b.fn for b in blocks.fns.values() if b.fn and b.fn.__name__ == 'save_settings')
            with self.assertRaises(gr.Error):
                save('cpu')
            update.assert_not_called()

    async def test_native_device_setting_remains_editable_and_saves(self):
        with mock.patch.dict(os.environ, {'AI_RVC_DEVICE': ''}), \
             mock.patch.dict(app.config, {'device': 'cpu'}), \
             mock.patch.object(app, 'update_config') as update:
            blocks = app.create_ui()
            radio = next(b for b in blocks.blocks.values() if isinstance(b, gr.Radio) and b.value == 'cpu')
            self.assertTrue(radio.interactive)
            save = next(b.fn for b in blocks.fns.values() if b.fn and b.fn.__name__ == 'save_settings')
            self.assertEqual(save('cpu'), app.t('settings_saved_restart', 'settings'))
            update.assert_called_once_with(app.CONFIG_PATH, {'device': 'cpu'})


if __name__ == '__main__':
    unittest.main()
