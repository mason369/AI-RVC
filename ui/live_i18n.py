"""Use Gradio's native translation metadata for labels, preserving form state."""
import json
from pathlib import Path

import gradio as gr
from gradio.i18n import I18nData
from ui.accessibility import ACCESSIBILITY_JS

ROOT = Path(__file__).resolve().parents[1]
LOCALES = {'zh_CN': 'zh-CN', 'en_US': 'en'}


def _read_version() -> str:
    for candidate in (ROOT / 'VERSION', ROOT.parent / 'VERSION'):
        if candidate.is_file():
            return candidate.read_text(encoding='utf-8').strip()
    return ''


_VERSION = _read_version()


def _theme_labels() -> dict[str, dict[str, str]]:
    labels = {}
    for code in LOCALES:
        settings = json.loads((ROOT / 'i18n' / f'{code}.json').read_text(encoding='utf-8'))['settings']
        labels[code] = {
            'light': settings['theme_light'],
            'dark': settings['theme_dark'],
            'toLight': settings['theme_switch_to_light'],
            'toDark': settings['theme_switch_to_dark'],
        }
    return labels


THEME_LABELS_JSON = json.dumps(_theme_labels(), ensure_ascii=False)


def catalog() -> gr.I18n:
    def flatten(data, prefix=''):
        result = {}
        for key, value in data.items():
            key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                result.update(flatten(value, key))
            else:
                result[key] = value
        return result

    version_badge = f'<span class="top-brand-ver">v{_VERSION}</span>' if _VERSION else ''
    translations = {}
    for code, locale in LOCALES.items():
        data = flatten(json.loads((ROOT / 'i18n' / f'{code}.json').read_text(encoding='utf-8')))
        data['brand'] = (
            '<div class="top-brand-wrap">'
            + '<div class="top-brand-row">'
            + '<span class="top-brand-title">' + data['app_title'] + '</span>'
            + version_badge
            + '</div>'
            + '<div class="top-brand-subtitle">' + data['app_description'] + '</div>'
            + '</div>'
        )
        translations[locale] = data
    return gr.I18n(**translations)


def ui_text(key: str, section: str | None = None):
    return I18nData(f'{section}.{key}' if section else key)


def bind_static_translations(blocks):
    """Some Gradio 5 values translate only on prop updates, so refresh those too."""
    specifications = []
    for component in blocks.blocks.values():
        properties = {}
        for name in ('label', 'info', 'placeholder', 'value'):
            value = getattr(component, name, None)
            if isinstance(value, I18nData) or isinstance(value, str) and '__i18n__' in value:
                properties[name] = value
        if properties:
            specifications.append((component, properties))

    def refresh(choice):
        code = {'中文': 'zh_CN', 'English': 'en_US', 'zh_CN': 'zh_CN', 'en_US': 'en_US'}[choice]
        translations = catalog().translations_dict[LOCALES[code]]

        def resolve(value):
            if isinstance(value, I18nData):
                return translations[value.key]
            marker = '__i18n__'
            if marker not in value:
                return value
            before, encoded = value.split(marker, 1)
            metadata, consumed = json.JSONDecoder().raw_decode(encoded)
            return before + translations[metadata['key']] + resolve(encoded[consumed:])

        return [gr.update(**{name: resolve(value) for name, value in properties.items()})
                for _, properties in specifications]

    return [component for component, _ in specifications], refresh


LANGUAGE_CHANGE_JS = r"""async (choice) => {
    const code = {'中文':'zh_CN','English':'en_US',zh_CN:'zh_CN',en_US:'en_US'}[choice];
    if (!code) throw new Error('Unsupported interface language');
    const base = new URL((window.gradio_config.root || location.origin).replace(/\/$/, '') + '/');
    const {changeLocale} = await import(new URL('assets/index-CCsXLF_U.js', base).href);
    changeLocale(code === 'en_US' ? 'en' : 'zh-CN');
    document.title = __RVC_TITLES__[code];
    document.documentElement.lang = code === 'en_US' ? 'en' : 'zh-CN';
    __RVC_ACCESSIBILITY__
    window.rvcPlayerLanguage = code;
    if (window.rvcSyncThemeToggle) window.rvcSyncThemeToggle(code);
    const theme = window.rvcCurrentTheme || document.documentElement.dataset.rvcTheme || 'dark';
    const payload = {...(window.rvcPlayerPayload || {files:[]}), language:code, theme};
    window.rvcPlayerPayload = payload;
    const playerFrame = document.getElementById('rvc-multitrack-frame');
    if (playerFrame) {
        playerFrame.title = __RVC_PLAYER_TITLES__[code];
        playerFrame.contentWindow?.postMessage({type:'rvc-language',payload}, location.origin);
    }
    return [];
}""".replace(
    '__RVC_TITLES__',
    json.dumps({code: catalog().translations_dict[locale]['app_title'] for code, locale in LOCALES.items()}, ensure_ascii=False),
).replace(
    '__RVC_PLAYER_TITLES__',
    json.dumps({code: catalog().translations_dict[locale]['multitrack.audioMixerTitle'] for code, locale in LOCALES.items()}, ensure_ascii=False),
).replace('__RVC_ACCESSIBILITY__', ACCESSIBILITY_JS)
