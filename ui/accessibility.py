"""Localize hardcoded Gradio 5.49.1 accessible names on the host page."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TEXT = {
    code: json.loads((_ROOT / 'i18n' / f'{code}.json').read_text(encoding='utf-8'))['accessibility']
    for code in ('zh_CN', 'en_US')
}

ACCESSIBILITY_JS = r"""
    const accessibleText = __ACCESSIBLE_TEXT__;
    const accessibleNames = {
        'Upload file': 'upload',
        'Record audio': 'record',
        'Click to upload or drop files': 'drop',
        'Reset to default value': 'reset',
        'Adjust volume': 'adjustVolume',
        'High volume': 'highVolume',
        'Low volume': 'lowVolume',
        'Muted volume': 'mutedVolume',
        'Reset audio': 'resetAudio',
        'Trim audio to selection': 'trimAudio',
        'undo': 'undo',
        'Empty value': 'empty',
        'Uploading': 'uploading',
    };
    if (!window.rvcAccessibleNames) {
        const originals = new WeakMap();
        const update = () => {
            const chinese = document.documentElement.lang.startsWith('zh');
            const text = accessibleText[chinese ? 'zh_CN' : 'en_US'];
            document.querySelectorAll('main [aria-label], main [title], main svg title').forEach(element => {
                const attributes = element.localName === 'title' ? ['textContent'] : ['aria-label', 'title'];
                for (const attribute of attributes) {
                    const current = attribute === 'textContent' ? element.textContent : element.getAttribute(attribute);
                    if (!current) continue;
                    let state = originals.get(element);
                    if (!state) {state = {}; originals.set(element, state);}
                    const previous = state[attribute];
                    const source = previous && current === previous.rendered ? previous.source : current;
                    let translated = text[accessibleNames[source]];
                    const input = source.match(/^(number input for|range slider for) (.+)$/);
                    if (input) translated = text[input[1] === 'number input for' ? 'number' : 'range'].replace('{label}', input[2]);
                    const speed = source.match(/^Adjust playback speed to ([0-9.]+)x$/);
                    if (speed) translated = text.speed.replace('{value}', speed[1]);
                    const skip = source.match(/^Skip (backwards|forward) by ([0-9.]+) seconds$/);
                    if (skip) translated = text[skip[1] === 'backwards' ? 'backward' : 'forward'].replace('{value}', skip[2]);
                    if (!translated) continue;
                    state[attribute] = {source, rendered: translated};
                    if (translated !== current) {
                        if (attribute === 'textContent') element.textContent = translated;
                        else element.setAttribute(attribute, translated);
                    }
                }
            });
            // Gradio keeps separate text nodes for the upload count; preserve them.
            const uploadWords = {'Uploading ': 'uploadingPrefix', 'file': 'fileSingular', 'files': 'filePlural'};
            document.querySelectorAll('main span.uploading').forEach(element => {
                element.childNodes.forEach(node => {
                    if (node.nodeType !== Node.TEXT_NODE) return;
                    const previous = originals.get(node);
                    const source = previous && node.data === previous.rendered ? previous.source : node.data;
                    const translated = text[uploadWords[source]];
                    if (!translated) return;
                    originals.set(node, {source, rendered: translated});
                    if (node.data !== translated) node.data = translated;
                });
            });
        };
        const observer = new MutationObserver(update);
        observer.observe(document.documentElement, {subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['aria-label','title','lang']});
        window.rvcAccessibleNames = update;
    }
    window.rvcAccessibleNames();
""".replace('__ACCESSIBLE_TEXT__', json.dumps(_TEXT, ensure_ascii=False))
