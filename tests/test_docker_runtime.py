"""Persisted user data and headless/authenticated startup are release contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from configs.persistence import update_config
from docker.entrypoint import authentication, initialize_data


class DockerRuntimeTests(unittest.TestCase):
    def test_empty_volume_preserves_every_quality_parameter(self):
        default = Path("configs/config.json")
        expected = json.loads(default.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            saved = initialize_data(data, default, "cpu")
            actual = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(actual["cover"], expected["cover"])
            self.assertEqual(actual["device"], "cpu")
            actual["language"] = "en_US"
            actual["cover"]["index_rate"] = 0.71
            saved.write_text(json.dumps(actual), encoding="utf-8")
            original = saved.read_bytes()
            recording = data / "outputs" / "keep.wav"
            recording.write_bytes(b"user recording")
            initialize_data(data, default, "cuda")
            self.assertEqual(saved.read_bytes(), original)
            self.assertEqual(recording.read_bytes(), b"user recording")

    def test_invalid_existing_config_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / "config.json"
            saved.write_text('{"cover":{"unknown_parameter":1}}', encoding="utf-8")
            original = saved.read_bytes()
            with self.assertRaises(ValueError):
                initialize_data(Path(directory), Path("configs/config.json"), "cpu")
            self.assertEqual(saved.read_bytes(), original)

    def test_atomic_updates_keep_symlink_and_user_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "persisted.json"
            link = Path(directory) / "config.json"
            original = {"device": "cpu", "language": "zh_CN", "cover": {"index_rate": 0.71}}
            data.write_text(json.dumps(original), encoding="utf-8")
            link.symlink_to(data)
            update_config(link, {"language": "en_US"})
            self.assertTrue(link.is_symlink())
            self.assertEqual(json.loads(data.read_text())["language"], "en_US")
            self.assertEqual(json.loads(data.read_text())["cover"], original["cover"])
            before = data.read_bytes()
            with self.assertRaises(ValueError):
                update_config(link, {"cover": {"index_rate": 2}})
            self.assertEqual(data.read_bytes(), before)
            self.assertTrue(link.is_symlink())

    def test_password_is_read_from_secret_and_partial_auth_fails(self):
        self.assertIsNone(authentication({}))
        for env in ({"AI_RVC_AUTH_USER": "rvc"}, {"AI_RVC_AUTH_PASSWORD_FILE": "/missing"}):
            with self.assertRaises(ValueError):
                authentication(env)
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "password"
            env = {"AI_RVC_AUTH_USER": "rvc", "AI_RVC_AUTH_PASSWORD_FILE": str(secret)}
            secret.write_text("test-password\n", encoding="utf-8")
            self.assertEqual(authentication(env), ("rvc", "test-password"))
            secret.write_text("\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                authentication(env)

    def test_headless_launch_passes_auth_and_proxy_path_without_opening_browser(self):
        from ui import app
        blocks = mock.Mock(local_url="http://127.0.0.1:7860")
        with mock.patch.object(app, "create_ui", return_value=blocks), \
             mock.patch.object(app, "config", {"output_dir": str(Path(tempfile.gettempdir()) / "rvc-volume-outputs")}), \
             mock.patch("ui.server_runtime.configure_browser_server"), \
             mock.patch.object(app, "_patch_gradio_file_download"), \
             mock.patch("ui.gradio_assets.patch_upload_route"), \
             mock.patch("webbrowser.open") as browser:
            app.launch(open_browser=False, auth=("rvc", "test-password"), root_path="/ai-rvc")
            browser.assert_not_called()
            self.assertEqual(blocks.launch.call_args.kwargs["auth"], ("rvc", "test-password"))
            self.assertEqual(blocks.launch.call_args.kwargs["root_path"], "/ai-rvc")
            self.assertFalse(blocks.launch.call_args.kwargs["share"])
            self.assertEqual(blocks.launch.call_args.kwargs["allowed_paths"],
                             [str((Path(tempfile.gettempdir()) / "rvc-volume-outputs").resolve())])
            blocks.block_thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
