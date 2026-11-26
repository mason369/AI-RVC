"""Check an original Windows CPU archive in Chrome, preserving visual evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.release_manifest import inventory


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def unpack(archives: Path, output: Path) -> Path:
    name = 'AI-RVC-Windows-CPU-Portable'
    checksum = archives / (name + '.SHA256SUMS')
    expected = {}
    for line in checksum.read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})\s+\*?(?:\./)?(' + re.escape(name) + r'\.7z\.[0-9]{3})', line)
        if not match:
            raise RuntimeError('Invalid original archive checksum entry')
        value, filename = match.groups()
        if filename in expected or digest(archives / filename) != value:
            raise RuntimeError('Original archive checksum mismatch: ' + filename)
        expected[filename] = value
    if not expected or {path.name for path in archives.iterdir()} != set(expected) | {checksum.name}:
        raise RuntimeError('Missing or unexpected original archive files')
    destination = output.parent / 'portable-browser-中文'
    destination.mkdir(exist_ok=False)
    with (output / 'extraction.log').open('w', encoding='utf-8') as log:
        subprocess.run(['7z', 'x', str(archives / (name + '.7z.001')), '-o' + str(destination)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    package = destination / name
    actual = inventory(package)
    if actual != json.loads((package / 'PACKAGE-MANIFEST.json').read_text(encoding='utf-8')):
        raise RuntimeError('Extracted original package does not match its complete inventory')
    (output / 'integrity.json').write_text(json.dumps({
        'archives': expected, 'files': actual['count'], 'bytes': actual['bytes'],
        'manifest': 'verified before first launch',
    }, indent=2), encoding='utf-8')
    return package


BROWSER_CODE = r'''async (page) => {
  const URL = __URL__, OUTPUT = __OUTPUT__, CATALOGS = __CATALOGS__;
  const errors = [], failedRequests = [], consoleErrors = [], closedEventStreams = [];
  const eventStreams = new WeakSet();
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('response', response => {
    if (response.url().startsWith(URL + '/gradio_api/queue/data?') && response.status() === 200 &&
        response.headers()['content-type']?.startsWith('text/event-stream')) eventStreams.add(response.request());
    if (response.url().startsWith(URL) && response.status() >= 400)
      failedRequests.push({url: response.url(), status: response.status()});
  });
  page.on('requestfailed', request => {
    if (!request.url().startsWith(URL)) return;
    const failure = {url: request.url(), error: request.failure()};
    // Chrome reports closing a successful SSE stream as ERR_ABORTED, including
    // after queue completion and reload. Keep that evidence separate from faults.
    if (eventStreams.has(request) && failure.error?.errorText === 'net::ERR_ABORTED')
      closedEventStreams.push(failure);
    else failedRequests.push(failure);
  });
  await page.goto(URL, {waitUntil: 'domcontentloaded', timeout: 60000});
  const observations = [];
  async function inspect(code, phase = '') {
    const catalog = CATALOGS[code];
    await page.getByRole('button', {name: catalog.settings.save_language, exact: true}).waitFor({timeout: 60000});
    for (const label of Object.values(catalog.tabs)) {
      if (!await page.getByRole('tab', {name: label, exact: true}).isVisible())
        throw new Error('Missing visible localized tab: ' + label);
    }
    await page.getByRole('tab', {name: catalog.tabs.models, exact: true}).click();
    await page.waitForFunction(text => Array.from(document.querySelectorAll('textarea')).some(el => el.value.includes(text)), catalog.models.status_scope);
    const snapshot = await page.locator('body').ariaSnapshot();
    if (!snapshot.includes('listbox') || !snapshot.includes(catalog.settings.save_language))
      throw new Error('Language controls are missing from the current rendered snapshot');
    for (const key of ['model_name', 'model_path', 'index_path']) {
      await page.getByRole('columnheader').getByText(catalog.ui[key], {exact: true})
        .first().waitFor({state: 'visible'});
    }
    await page.screenshot({path: OUTPUT + '/' + code + phase + '-models.png', fullPage: true});
    await page.getByRole('tab', {name: catalog.tabs.cover, exact: true}).click();
    const coverSnapshot = await page.locator('body').ariaSnapshot();
    if (!coverSnapshot.includes(catalog.tabs.cover)) throw new Error('Cover tab did not render');
    const frame = page.frameLocator('iframe#rvc-multitrack-frame');
    await frame.getByText(catalog.multitrack.empty, {exact: true}).waitFor({timeout: 30000});
    const batch = page.getByRole('tab', {name: catalog.cover.batch, exact: true});
    await batch.click();
    await page.getByRole('button', {name: catalog.cover.batch_start, exact: true}).waitFor();
    await page.getByRole('tab', {name: catalog.cover.single_mode, exact: true}).click();
    const theme = await page.locator('html').getAttribute('data-rvc-theme');
    if (!['light', 'dark'].includes(theme)) throw new Error('The workspace theme did not initialize');
    const switchedTheme = theme === 'light' ? 'dark' : 'light';
    async function setTheme(value) {
      await page.getByRole('button', {name: catalog.settings['theme_switch_to_' + value], exact: true}).click();
      await page.waitForFunction(value => document.documentElement.dataset.rvcTheme === value, value);
      await frame.locator('html[data-theme="' + value + '"]').waitFor();
    }
    await setTheme(switchedTheme);
    await setTheme(theme);
    const playerText = await frame.locator('body').innerText();
    if (!playerText.trim()) throw new Error('The bundled player rendered an empty document');
    await page.screenshot({path: OUTPUT + '/' + code + phase + '-cover.png', fullPage: true});
    observations.push({language: code, phase, snapshot, coverSnapshot, playerText});
  }
  await inspect('zh_CN');
  // Read each changed UI before interacting; option references are never hard-coded.
  await page.locator('body').ariaSnapshot();
  await page.getByRole('listbox', {name: CATALOGS.zh_CN.settings.language, exact: true}).click();
  const options = await page.locator('body').ariaSnapshot();
  if (!options.includes('English')) throw new Error('English is missing from the language choices');
  await page.getByRole('option', {name: 'English', exact: true}).click();
  await page.getByRole('button', {name: CATALOGS.zh_CN.settings.save_language, exact: true}).click();
  await inspect('en_US');
  await page.reload({waitUntil: 'domcontentloaded'});
  await inspect('en_US', '-reloaded');
  await page.locator('body').ariaSnapshot();
  await page.getByRole('listbox', {name: CATALOGS.en_US.settings.language, exact: true}).click();
  await page.locator('body').ariaSnapshot();
  await page.getByRole('option', {name: '中文', exact: true}).click();
  await page.getByRole('button', {name: CATALOGS.en_US.settings.save_language, exact: true}).click();
  await inspect('zh_CN', '-restored');
  if (errors.length || failedRequests.length || consoleErrors.length)
    throw new Error(JSON.stringify({errors, failedRequests, consoleErrors}));
  return {
    status: 'passed', observations, languagePersistsAfterReload: true,
    pageErrors: errors, failedLocalRequests: failedRequests, consoleErrors, closedEventStreams
  };
}'''


def verify(package: Path, output: Path) -> None:
    executable = package / 'AI-RVC-Windows-CPU.exe'
    catalogs = {code: json.loads((package / '_internal/i18n' / (code + '.json')).read_text(encoding='utf-8'))
                for code in ('zh_CN', 'en_US')}
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    working = output / 'launch-中文'
    working.mkdir()
    env = dict(os.environ, PYTHONPATH='', PYTHONHOME='', AI_RVC_DEVICE='cpu',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GRADIO_ANALYTICS_ENABLED='False')
    npx = shutil.which('npx.cmd')
    if not npx:
        raise RuntimeError('The browser verifier requires the runner Node/npm runtime')
    command = [npx, '--yes', '--package', '@playwright/cli@0.1.22', 'playwright-cli', '-s=portable-rebuild']

    def cli(*args: str, strict: bool = True) -> str:
        result = subprocess.run([*command, *args], cwd=output, text=True, encoding='utf-8',
                                capture_output=True, timeout=240)
        with (output / 'browser-cli.log').open('a', encoding='utf-8') as log:
            log.write(result.stdout + result.stderr)
        if strict and (result.returncode or '### Error' in result.stdout):
            raise RuntimeError('Browser CLI failed; see browser-cli.log: ' + result.stdout[-2000:])
        return result.stdout

    with (output / 'application.log').open('w', encoding='utf-8') as log:
        app = subprocess.Popen([str(executable), '--port', str(port)], cwd=working, env=env,
                               stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 180
            while True:
                if app.poll() is not None:
                    raise RuntimeError('The original executable exited before HTTP startup')
                try:
                    with urllib.request.urlopen(url + '/config', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Original executable startup timed out')
                time.sleep(1)
            script = output / 'browser-check.js'
            script.write_text(BROWSER_CODE.replace('__URL__', json.dumps(url))
                              .replace('__OUTPUT__', json.dumps(output.as_posix()))
                              .replace('__CATALOGS__', json.dumps(catalogs, ensure_ascii=False)), encoding='utf-8')
            cli('open', 'about:blank', '--browser=chrome')
            cli('resize', '1440', '1080')
            result = cli('run-code', '--filename=' + str(script), '--raw')
            report = json.loads(result)
            if report.get('status') != 'passed':
                raise RuntimeError('The real browser did not return a completed verification report')
        finally:
            try:
                state = cli('run-code', 'async (page) => ({url: page.url(), snapshot: await page.locator("body").ariaSnapshot()})', '--raw', strict=False)
                (output / 'last-page.json').write_text(state, encoding='utf-8')
                cli('screenshot', '--filename=' + str(output / 'last-page.png'), strict=False)
                cli('console', strict=False)
                cli('requests', strict=False)
                cli('close', strict=False)
            finally:
                app.terminate()
                app.wait(timeout=30)
    if 'Traceback (most recent call last)' in (output / 'application.log').read_text(encoding='utf-8'):
        raise RuntimeError('The original app logged a server exception during browser verification')
    (output / 'browser-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Original Windows CPU executable passed Chrome checks in zh_CN and en_US', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    verify(unpack(args.archives.resolve(), output), output)


if __name__ == '__main__':
    main()
