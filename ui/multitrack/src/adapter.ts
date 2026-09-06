import zh from '../../../i18n/zh_CN.json';
import en from '../../../i18n/en_US.json';
export const cn = (...args: unknown[]) => args.filter(Boolean).join(' ');
function translator(catalog: Record<string,string>) {
  return (key: string, values: Record<string, unknown> = {}) => {
    if (!(key in catalog)) throw new Error(`Missing player translation: ${key}`);
    return (catalog as Record<string,string>)[key].replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? `{${name}}`));
  };
}
const translators = {zh: translator(zh.multitrack), en: translator(en.multitrack)};
export function useTranslations(_namespace?: string) {
  return document.documentElement.lang === 'en' ? translators.en : translators.zh;
}
export function reportMixerError(error: unknown) {
  console.error('AI-RVC multitrack:', error);
  window.dispatchEvent(new CustomEvent('rvc-player-error', {detail: error instanceof Error ? error.message : String(error)}));
}
