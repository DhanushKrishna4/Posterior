// Development-time SVG rasterization only. The website serves a static PNG.
import {chromium} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import {readFile, mkdir, readdir, writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import os from 'node:os';
import path from 'node:path';

const root = fileURLToPath(new URL('../..', import.meta.url));
process.chdir(root);
const python = process.env.PYTHON || (existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python3');
execFileSync(python, ['tools/og/compose.py'], {stdio:'inherit'});
await mkdir('tools/og/proofs', {recursive:true});
let executablePath = process.env.PLAYWRIGHT_EXECUTABLE;
const bundledPath = chromium.executablePath();
const bundledDirectory = bundledPath.match(/^(.*[/\\]chromium-\d+)[/\\]/)?.[1];
if (!executablePath && bundledDirectory && existsSync(path.join(bundledDirectory, 'INSTALLATION_COMPLETE')) && existsSync(bundledPath)) executablePath = bundledPath;
if (!executablePath) {
  // Reuse a complete local Playwright installation, as the existing browser suites do.
  const cache = process.env.PLAYWRIGHT_BROWSERS_PATH || (process.platform === 'darwin'
    ? path.join(os.homedir(), 'Library/Caches/ms-playwright') : path.join(os.homedir(), '.cache/ms-playwright'));
  const dirs = (await readdir(cache).catch(() => [])).filter(name => /^chromium-\d+$/.test(name)
    && existsSync(path.join(cache, name, 'INSTALLATION_COMPLETE'))).sort((a,b) => Number(b.split('-')[1])-Number(a.split('-')[1]));
  for (const dir of dirs) {
    const candidates = process.platform === 'darwin'
      ? ['chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing','chrome-mac/Chromium.app/Contents/MacOS/Chromium']
      : ['chrome-linux64/chrome','chrome-linux/chrome'];
    executablePath = candidates.map(name => path.join(cache,dir,name)).find(name => existsSync(name));
    if (executablePath) break;
  }
}
const browser = await chromium.launch({headless:true, executablePath, args:['--disable-gpu', '--force-color-profile=srgb']});
try {
  const page = await browser.newPage({viewport:{width:1200,height:630}, deviceScaleFactor:2, reducedMotion:'reduce'});
  await page.route('**/*', route => route.abort());
  for (const [source, output] of [
    ['tools/og/studies/landscape.svg', 'tools/og/studies/landscape.png'],
    ['tools/og/studies/folded.svg', 'tools/og/studies/folded.png'],
    ['tools/og/card.svg', 'tools/og/proofs/render-2x.png'],
  ]) {
    const svg = await readFile(source, 'utf8');
    await page.setContent(`<html><head><style>*{box-sizing:border-box}body{margin:0;width:1200px;height:630px;overflow:hidden}svg{display:block}</style></head><body>${svg}</body></html>`);
    await page.evaluate(async () => { await document.fonts.ready; });
    // Every glyph is already outlined. Rasterization cannot silently substitute fonts.
    if (await page.locator('text, image, foreignObject').count()) throw new Error('Export contains an unresolved font or image dependency');
    await page.screenshot({path:output, animations:'disabled', type:'png'});
  }
  console.log(`Rasterized with Chromium ${browser.version()}`);
  await writeFile('tools/og/proofs/renderer.json', JSON.stringify({chromium:browser.version(), rasterization:'CPU (--disable-gpu)', colorProfile:'sRGB', deviceScaleFactor:2, outputScale:1, animation:'none', externalRequests:'blocked'},null,2)+'\n');
} finally { await browser.close(); }
execFileSync(python, ['tools/og/finish.py'], {stdio:'inherit'});
