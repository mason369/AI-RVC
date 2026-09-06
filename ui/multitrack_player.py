"""Offline result player: Gradio owns file authorization and audio postprocessing."""
import html
import json
from pathlib import Path


def player_html(language: str) -> str:
    document = (Path(__file__).parent / "multitrack/dist/player.html").read_text(encoding="utf-8")
    document = document.replace('<html>', '<html lang="en">' if language == "en_US" else '<html lang="zh">')
    catalog = json.loads((Path(__file__).parent.parent / 'i18n' / f'{language}.json').read_text(encoding='utf-8'))
    title = html.escape(catalog['multitrack']['audioMixerTitle'], quote=True)
    return '<iframe id="rvc-multitrack-frame" title="' + title + '" style="width:100%;border:0;height:112px;display:block" allow="autoplay" srcdoc="' + html.escape(document, quote=True) + '"></iframe>'


PLAYER_BRIDGE_JS = """() => {
    const frame = () => document.getElementById('rvc-multitrack-frame');
    window.rvcPlayerPayload = null;
    window.rvcPlayerLanguage = '__RVC_LANGUAGE__';
    window.rvcUpdatePlayer = (files, status) => {
        const payload = {files, status, language: window.rvcPlayerLanguage};
        window.rvcPlayerPayload = payload;
        frame()?.contentWindow?.postMessage({type:'rvc-tracks', payload}, location.origin);
    };
    if (window.rvcPlayerListener) window.removeEventListener('message', window.rvcPlayerListener);
    window.rvcPlayerListener = (event) => {
        const target = frame();
        if (!target || event.source !== target.contentWindow || event.origin !== location.origin) return;
        if (event.data?.type === 'rvc-player-height' && Number.isFinite(event.data.height)) {
            target.style.height = Math.max(100, event.data.height + 6) + 'px';
        }
        if (event.data?.type === 'rvc-player-ready' && window.rvcPlayerPayload) {
            target.contentWindow.postMessage({type:'rvc-tracks', payload:window.rvcPlayerPayload}, location.origin);
        }
    };
    window.addEventListener('message', window.rvcPlayerListener);
    if (!window.rvcWheelGuard) {
        window.rvcWheelGuard = (event) => {
            if (event.target instanceof HTMLInputElement && ['range', 'number'].includes(event.target.type)) event.preventDefault();
        };
        document.addEventListener('wheel', window.rvcWheelGuard, {passive:false, capture:true});
    }
}"""

PLAYER_UPDATE_JS = "(...files) => { window.rvcUpdatePlayer(files, 'complete'); }"
PLAYER_CLEAR_JS = "(...args) => { window.rvcUpdatePlayer([], 'processing'); return args; }"
