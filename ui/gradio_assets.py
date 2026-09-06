"""Backport Gradio's upload ID initialization fix without modifying site-packages.

Upstream: https://github.com/gradio-app/gradio/pull/12637
The pinned 5.49.1 bundle passes a local ID to upload(), but never assigns the
reactive ID consumed by UploadProgress. Assign it before tick()/uploading.
"""
from pathlib import Path

import gradio
from starlette.responses import Response

ASSET_NAME = "ModifyUpload-DeA0LoIt.js"
BEFORE = 'async function qe(u,b){await ht(),b||(b=Math.random().toString(36).substring(2,15)),t(2,N=!0);try{const v=await Z(u,A,b,Y??1/0);'
AFTER = 'async function qe(u,b){t(19,Se=b||Math.random().toString(36).substring(2,15));await ht();t(2,N=!0);try{const v=await Z(u,A,Se,Y??1/0);'


def patched_upload_asset() -> str:
    asset = Path(gradio.__file__).parent / "templates/frontend/assets" / ASSET_NAME
    source = asset.read_text(encoding="utf-8")
    if gradio.__version__ != "5.49.1" or source.count(BEFORE) != 1:
        raise RuntimeError("Gradio upload asset differs from the verified 5.49.1 source")
    return source.replace(BEFORE, AFTER)


def patch_upload_route(blocks) -> None:
    from ui.gradio_routes import iter_gradio_routes
    assets = {ASSET_NAME: patched_upload_asset(), "index-BvFm9VBJ.js": patched_stream_asset()}
    route = next(route for route in iter_gradio_routes(blocks.server_app.routes)
                 if getattr(route, "path", None) == "/assets/{path:path}")
    original = route.app

    async def serve(scope, receive, send):
        name = scope.get("path_params", {}).get("path")
        if name in assets:
            await Response(assets[name], media_type="text/javascript", headers={"cache-control": "no-cache"})(scope, receive, send)
        else:
            await original(scope, receive, send)

    route.app = serve


def patched_stream_asset() -> str:
    asset = Path(gradio.__file__).parent / "templates/frontend/assets/index-BvFm9VBJ.js"
    source = asset.read_text(encoding="utf-8")
    start = source.index("function es(t,s={}){")
    end = source.index("function ts(", start)
    original = source[start:end]
    if 'close:()=>{console.warn("Method not implemented.")}' not in original:
        raise RuntimeError("Gradio stream asset differs from the verified 5.49.1 source")
    replacement = original.replace(
        'function es(t,s={}){const e={close:()=>{console.warn("Method not implemented.")}',
        'function es(t,s={}){const c=new AbortController,a=()=>c.abort(s.signal?.reason);if(s.signal){if(s.signal.aborted)a();else s.signal.addEventListener("abort",a,{once:true})}const e={close:()=>{c.abort();e.readyState=e.CLOSED}',
    ).replace('Kt(t,s)', 'Kt(t,{...s,signal:c.signal})')
    replacement = replacement.replace('e.readyState=e.OPEN;try{', 'if(c.signal.aborted)return;e.readyState=e.OPEN;try{')
    replacement = replacement.replace('e.onmessage&&e.onmessage(o)', '!c.signal.aborted&&e.onmessage&&e.onmessage(o)')
    replacement = replacement.replace('e.onerror&&e.onerror(o)', '!c.signal.aborted&&e.onerror&&e.onerror(o)')
    replacement = replacement.replace('console.error(n),e.onerror&&e.onerror(n)', '(!c.signal.aborted&&(console.error(n),e.onerror&&e.onerror(n)))')
    replacement = replacement.replace('}),e}', '}).finally(()=>s.signal?.removeEventListener("abort",a)),e}')
    return source[:start] + replacement + source[end:]
