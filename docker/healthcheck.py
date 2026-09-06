"""Readiness is HTTP availability after the startup model checks, with or without login."""
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=4) as response:
    if response.status != 200:
        raise RuntimeError(f"Web UI 未就绪: HTTP {response.status}")
