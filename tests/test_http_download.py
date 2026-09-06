import contextlib
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from tools.http_download import download_http


class HttpDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / 'model.bin'
        self.body = b'complete model bytes' * 100
        self.mode = 'full'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                mode = owner.mode
                if mode in ('403', '416', '429', '404'):
                    self.send_response(int(mode)); self.end_headers(); return
                start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
                ranged = mode in ('range', 'bad_range')
                body = owner.body[start:] if ranged else owner.body
                self.send_response(206 if ranged else 200)
                self.send_header('ETag', '"v1"')
                if ranged:
                    self.send_header('Content-Range', f'bytes {start + (mode == "bad_range")}-{len(owner.body)-1}/{len(owner.body)}')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body[:9] if mode == 'truncated' else body)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}/asset'
        self.delay = patch('tools.http_download.time.sleep')
        self.delay.start()
        self.addCleanup(self.delay.stop)

    def partial(self):
        self.dest.with_name('model.bin.part').write_bytes(self.body[:11])
        self.dest.with_name('model.bin.part.json').write_text(json.dumps({
            'url_hash': hashlib.sha256(self.url.encode()).hexdigest(), 'etag': '"v1"'}))

    def test_range_resume_matches_original_bytes(self):
        self.partial(); self.mode = 'range'
        download_http(self.url, self.dest)
        self.assertEqual(self.dest.read_bytes(), self.body)

    def test_ignored_range_replaces_partial_instead_of_appending(self):
        self.partial()
        download_http(self.url, self.dest)
        self.assertEqual(self.dest.read_bytes(), self.body)

    def test_http_failures_never_publish_or_overwrite(self):
        self.dest.write_bytes(b'existing file')
        self.partial()
        for mode in ['403', '404', '416', '429']:
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaisesRegex(RuntimeError, f'HTTP {mode}'):
                    download_http(self.url, self.dest)
                self.assertEqual(self.dest.read_bytes(), b'existing file')

    def test_invalid_content_range_is_rejected(self):
        self.partial(); self.mode = 'bad_range'
        with self.assertRaisesRegex(RuntimeError, 'range'):
            download_http(self.url, self.dest)
        self.assertFalse(self.dest.exists())

    def test_truncated_body_is_not_published(self):
        self.mode = 'truncated'
        with self.assertRaises(Exception):
            download_http(self.url, self.dest)
        self.assertFalse(self.dest.exists())

    def test_checksum_mismatch_preserves_original(self):
        self.dest.write_bytes(b'existing file')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            download_http(self.url, self.dest, expected_sha256='0' * 64)
        self.assertEqual(self.dest.read_bytes(), b'existing file')

    def test_verified_file_is_published_and_partial_removed(self):
        download_http(self.url, self.dest, expected_sha256=hashlib.sha256(self.body).hexdigest())
        self.assertEqual(self.dest.read_bytes(), self.body)
        self.assertEqual(sorted(p.name for p in self.dest.parent.iterdir()), ['model.bin'])
