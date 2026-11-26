"""Offline result player & theme bridge: Gradio owns file authorization and audio postprocessing."""
import html
import json
from pathlib import Path

from ui.live_i18n import THEME_LABELS_JSON


def player_html(language: str) -> str:
    document = (Path(__file__).parent / "multitrack/dist/player.html").read_text(encoding="utf-8")
    document = document.replace('<html>', '<html lang="en">' if language == "en_US" else '<html lang="zh">')
    catalog = json.loads((Path(__file__).parent.parent / 'i18n' / f'{language}.json').read_text(encoding='utf-8'))
    title = html.escape(catalog['multitrack']['audioMixerTitle'], quote=True)
    return (
        '<iframe id="rvc-multitrack-frame" class="rvc-multitrack-frame" title="'
        + title
        + '" style="width:100%;border:0;height:132px;display:block" allow="autoplay" srcdoc="'
        + html.escape(document, quote=True)
        + '"></iframe>'
    )


THEME_INIT_SCRIPT = """<script>
(() => {
    const STORAGE_KEY = 'ai-rvc-theme';
    let saved = null;
    try { saved = localStorage.getItem(STORAGE_KEY); } catch {}
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = (saved === 'light' || saved === 'dark') ? saved : (prefersDark ? 'dark' : 'light');
    const root = document.documentElement;
    root.dataset.rvcTheme = theme;
    root.style.colorScheme = theme;
    root.classList.toggle('dark', theme === 'dark');
    window.rvcCurrentTheme = theme;
    const syncBody = () => {
        if (document.body) document.body.classList.toggle('dark', root.dataset.rvcTheme === 'dark');
    };
    if (document.body) syncBody();
    else document.addEventListener('DOMContentLoaded', syncBody, {once: true});
})();
</script>"""


PLAYER_BRIDGE_JS = r"""() => {
    const STORAGE_KEY = 'ai-rvc-theme';
    const THEME_LABELS = __RVC_THEME_LABELS__;
    const SUN_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>';
    const MOON_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>';

    const frame = () => document.getElementById('rvc-multitrack-frame');
    window.rvcPlayerPayload = null;
    window.rvcPlayerLanguage = '__RVC_LANGUAGE__';

    const readTheme = () => {
        const current = document.documentElement.dataset.rvcTheme;
        if (current === 'light' || current === 'dark') return current;
        let saved = null;
        try { saved = localStorage.getItem(STORAGE_KEY); } catch {}
        if (saved === 'light' || saved === 'dark') return saved;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    };

    const syncThemeToggle = (lang) => {
        const code = (lang === 'en_US' || lang === 'zh_CN') ? lang : (window.rvcPlayerLanguage || 'zh_CN');
        const labels = THEME_LABELS[code] || THEME_LABELS.zh_CN;
        const theme = readTheme();
        const isDark = theme === 'dark';
        const button = document.getElementById('rvc-theme-toggle');
        if (!button) return;
        const nextLabel = isDark ? labels.toLight : labels.toDark;
        const currentText = isDark ? labels.dark : labels.light;
        button.setAttribute('aria-label', nextLabel);
        button.setAttribute('title', nextLabel);
        button.setAttribute('aria-pressed', String(isDark));
        button.dataset.theme = theme;
        const iconSlot = button.querySelector('.rvc-theme-icon');
        const textSlot = button.querySelector('.rvc-theme-text');
        if (iconSlot) iconSlot.innerHTML = isDark ? MOON_ICON : SUN_ICON;
        if (textSlot) textSlot.textContent = currentText;
    };

    const postThemeToPlayer = (theme) => {
        frame()?.contentWindow?.postMessage({type: 'rvc-theme', theme}, location.origin);
    };

    const applyTheme = (theme, persist = false) => {
        const normalized = theme === 'light' ? 'light' : 'dark';
        const root = document.documentElement;
        root.dataset.rvcTheme = normalized;
        root.style.colorScheme = normalized;
        root.classList.toggle('dark', normalized === 'dark');
        if (document.body) document.body.classList.toggle('dark', normalized === 'dark');
        const container = document.querySelector('.gradio-container');
        if (container) container.classList.toggle('dark', normalized === 'dark');
        window.rvcCurrentTheme = normalized;
        if (persist) {
            try { localStorage.setItem(STORAGE_KEY, normalized); } catch {}
        }
        syncThemeToggle(window.rvcPlayerLanguage);
        postThemeToPlayer(normalized);
    };

    window.rvcSyncThemeToggle = syncThemeToggle;
    window.rvcApplyTheme = applyTheme;
    window.rvcToggleTheme = () => {
        const next = readTheme() === 'dark' ? 'light' : 'dark';
        applyTheme(next, true);
    };

    if (!window.rvcThemeClickBound) {
        document.addEventListener('click', (event) => {
            const target = event.target instanceof Element ? event.target.closest('#rvc-theme-toggle') : null;
            if (!target) return;
            event.preventDefault();
            window.rvcToggleTheme();
        });
        window.rvcThemeClickBound = true;
    }

    if (!window.rvcThemeMediaBound && window.matchMedia) {
        const media = window.matchMedia('(prefers-color-scheme: dark)');
        const onSchemeChange = (event) => {
            let saved = null;
            try { saved = localStorage.getItem(STORAGE_KEY); } catch {}
            if (saved === 'light' || saved === 'dark') return;
            applyTheme(event.matches ? 'dark' : 'light', false);
        };
        if (media.addEventListener) media.addEventListener('change', onSchemeChange);
        else if (media.addListener) media.addListener(onSchemeChange);
        window.rvcThemeMediaBound = true;
    }

    applyTheme(readTheme(), false);

    window.rvcUpdatePlayer = (files, status) => {
        const payload = {files, status, language: window.rvcPlayerLanguage, theme: readTheme()};
        window.rvcPlayerPayload = payload;
        frame()?.contentWindow?.postMessage({type:'rvc-tracks', payload}, location.origin);
    };
    if (window.rvcPlayerListener) window.removeEventListener('message', window.rvcPlayerListener);
    window.rvcPlayerListener = (event) => {
        const target = frame();
        if (!target || event.source !== target.contentWindow || event.origin !== location.origin) return;
        if (event.data?.type === 'rvc-player-height' && Number.isFinite(event.data.height)) {
            target.style.height = Math.max(116, event.data.height + 8) + 'px';
        }
        if (event.data?.type === 'rvc-player-ready') {
            postThemeToPlayer(readTheme());
            if (window.rvcPlayerPayload) {
                target.contentWindow.postMessage(
                    {type:'rvc-tracks', payload: {...window.rvcPlayerPayload, theme: readTheme()}},
                    location.origin
                );
            }
        }
    };
    window.addEventListener('message', window.rvcPlayerListener);
    if (!window.rvcWheelGuard) {
        window.rvcWheelGuard = (event) => {
            if (event.target instanceof HTMLInputElement && ['range', 'number'].includes(event.target.type)) event.preventDefault();
        };
        document.addEventListener('wheel', window.rvcWheelGuard, {passive:false, capture:true});
    }
}""".replace('__RVC_THEME_LABELS__', THEME_LABELS_JSON)

PLAYER_UPDATE_JS = "(...files) => { window.rvcUpdatePlayer(files, 'complete'); }"
PLAYER_CLEAR_JS = "(...args) => { window.rvcUpdatePlayer([], 'processing'); return args; }"
