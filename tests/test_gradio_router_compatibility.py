"""Exercise real router inclusion, file permissions and patched browser assets."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import fastapi
import gradio
from fastapi.testclient import TestClient
from gradio.routes import App
from starlette.responses import FileResponse

from ui.app import _patch_gradio_file_download
from ui.gradio_assets import ASSET_NAME, AFTER, patch_upload_route


class GradioRouterCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_router_preserves_auth_range_and_filename(self):
        server = fastapi.FastAPI()

        async def require_login(request: fastapi.Request):
            if request.headers.get("Authorization") != "Bearer audit":
                raise fastapi.HTTPException(401)

        leaf = fastapi.APIRouter(dependencies=[fastapi.Depends(require_login)])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "audio.wav"
            source.write_bytes(b"0123456789")

            @leaf.get("/file={path_or_url:path}")
            async def download(path_or_url: str):
                if path_or_url != "audio.wav":
                    raise fastapi.HTTPException(403)
                return FileResponse(source)

            parent = fastapi.APIRouter()
            parent.include_router(leaf)
            server.include_router(parent, prefix="/gradio_api")
            blocks = SimpleNamespace(server_app=server)
            _patch_gradio_file_download(blocks)
            _patch_gradio_file_download(blocks)
            client = TestClient(server)
            url = "/gradio_api/file=audio.wav"
            self.assertEqual(client.get(url).status_code, 401)
            headers = {"Authorization": "Bearer audit", "Range": "bytes=2-5"}
            response = client.get(url, headers=headers)
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"2345")
            self.assertEqual(response.headers["content-range"], "bytes 2-5/10")
            self.assertIn("audio.wav", response.headers["content-disposition"])
            self.assertEqual(client.get("/gradio_api/file=denied", headers=headers).status_code, 403)

    async def test_real_gradio_file_route_preserves_allowed_path_check(self):
        with gradio.Blocks() as blocks:
            gradio.Textbox()
        with tempfile.TemporaryDirectory() as directory:
            permitted = Path(directory) / "allowed"
            permitted.mkdir()
            source = permitted / "audio.wav"
            source.write_bytes(b"0123456789")
            secret = Path(directory) / "private.txt"
            secret.write_text("private", encoding="utf-8")
            blocks.allowed_paths = [str(permitted)]
            blocks.server_app = App.create_app(blocks)
            _patch_gradio_file_download(blocks)
            client = TestClient(blocks.server_app)
            response = client.get("/gradio_api/file=" + quote(str(source)), headers={"Range": "bytes=0-3"})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, b"0123")
            self.assertIn("audio.wav", response.headers["content-disposition"])
            self.assertEqual(client.get("/gradio_api/file=" + quote(str(secret))).status_code, 403)

    async def test_real_gradio_serves_fixed_upload_and_stream_assets(self):
        with gradio.Blocks() as blocks:
            gradio.Textbox()
        blocks.server_app = App.create_app(blocks)
        patch_upload_route(blocks)
        client = TestClient(blocks.server_app)
        response = client.get("/assets/" + ASSET_NAME)
        self.assertEqual(response.status_code, 200)
        self.assertIn(AFTER, response.text)
        response = client.get("/assets/index-BvFm9VBJ.js")
        self.assertEqual(response.status_code, 200)
        self.assertIn("new AbortController", response.text)
        self.assertNotIn('close:()=>{console.warn("Method not implemented.")}', response.text)


if __name__ == "__main__":
    unittest.main()
