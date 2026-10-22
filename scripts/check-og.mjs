// Inspect the static build; optionally verify delivery at /Posterior/ over HTTP.
// Usage: node scripts/check-og.mjs [--url http://127.0.0.1:4173/Posterior/]
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const root = new URL('../', import.meta.url);
const canonical = 'https://dhanushkrishna4.github.io/Posterior/';
const filename = 'posterior-og-v1.png';
const imageURL = new URL(filename, canonical).href;
const title = 'Posterior — Bayesian inference, made visible.';
const description = 'A local Bayesian inference workbench. Compare real inference with exact references, inspect diagnostics, and see how evidence changes a belief.';
const alt = 'Large Posterior wordmark above a sage Beta(2,2) prior and an indigo Beta(9,5) posterior: the exact update after 7 successes in 10 trials.';
const expected = new Map(Object.entries({
  'og:type': 'website',
  'og:site_name': 'Posterior',
  'og:title': title,
  'og:description': description,
  'og:url': canonical,
  'og:image': imageURL,
  'og:image:secure_url': imageURL,
  'og:image:type': 'image/png',
  'og:image:width': '1200',
  'og:image:height': '630',
  'og:image:alt': alt,
  'twitter:card': 'summary_large_image',
  'twitter:title': title,
  'twitter:description': description,
  'twitter:image': imageURL,
  'twitter:image:alt': alt,
}));

const args = process.argv.slice(2);
assert(args.length === 0 || (args.length === 2 && args[0] === '--url'),
  'Usage: node scripts/check-og.mjs [--url URL]');
const base = args.length ? new URL(args[1]) : null;
if (base) {
  assert.equal(base.pathname, '/Posterior/', 'Serve the production site at its case-sensitive Pages subpath.');
  assert(!base.search && !base.hash && !base.username && !base.password, 'Use a plain public page URL.');
  assert(base.protocol === 'https:' || (base.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname)),
    'Delivery must use HTTPS except for a local preview.');
}

function decode(value) {
  return value.replace(/&(?:amp|quot|apos|lt|gt|#\d+|#x[0-9a-f]+);/gi, entity => {
    const name = entity.slice(1, -1).toLowerCase();
    if (name.startsWith('#x')) return String.fromCodePoint(parseInt(name.slice(2), 16));
    if (name.startsWith('#')) return String.fromCodePoint(parseInt(name.slice(1), 10));
    return {amp: '&', quot: '"', apos: "'", lt: '<', gt: '>'}[name];
  });
}

function attributes(tag) {
  const pairs = [...tag.matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g)];
  return new Map(pairs.map(([, key, double, single, bare]) => [key.toLowerCase(), decode(double ?? single ?? bare)]));
}

function inspectHTML(html, label) {
  const head = html.match(/<head\b[^>]*>([\s\S]*?)<\/head>/i)?.[1];
  assert(head, `${label}: static HTML head missing.`);
  const cleanHead = head.replace(/<!--[\s\S]*?-->/g, '').replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, '');
  const found = new Map();
  for (const [tag] of cleanHead.matchAll(/<meta\b[^>]*>/gi)) {
    const attrs = attributes(tag);
    const key = attrs.get('property') ?? attrs.get('name');
    if (!key?.startsWith('og:') && !key?.startsWith('twitter:')) continue;
    assert(!found.has(key), `${label}: duplicate ${key}.`);
    assert(expected.has(key), `${label}: unexpected sharing tag ${key}.`);
    assert.equal(attrs.get(key.startsWith('og:') ? 'property' : 'name'), key, `${label}: incorrect attribute for ${key}.`);
    found.set(key, attrs.get('content'));
  }
  for (const [key, value] of expected) assert.equal(found.get(key), value, `${label}: ${key} must match the card.`);
  const canonicalLinks = [...cleanHead.matchAll(/<link\b[^>]*>/gi)]
    .map(([tag]) => attributes(tag)).filter(attrs => attrs.get('rel')?.split(/\s+/).includes('canonical'));
  assert.equal(canonicalLinks.length, 1, `${label}: require exactly one canonical link.`);
  assert.equal(canonicalLinks[0].get('href'), canonical, `${label}: canonical mismatch.`);
  for (const key of ['og:url', 'og:image', 'og:image:secure_url', 'twitter:image']) {
    const url = new URL(found.get(key));
    assert.equal(url.protocol, 'https:', `${label}: ${key} must use absolute HTTPS.`);
    assert(url.pathname.startsWith('/Posterior/'), `${label}: ${key} must retain the Pages subpath.`);
  }
  return {sharingTags: found.size, canonical};
}

function inspectPNG(bytes, label) {
  assert(bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), `${label}: not PNG bytes.`);
  assert(bytes.length < 1_000_000, `${label}: image must be under 1 MB.`);
  let offset = 8;
  const chunks = [];
  while (offset < bytes.length) {
    assert(offset + 12 <= bytes.length, `${label}: truncated PNG chunk.`);
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    assert(offset + 12 + length <= bytes.length, `${label}: invalid ${type} length.`);
    chunks.push({type, data: bytes.subarray(offset + 8, offset + 8 + length)});
    offset += 12 + length;
  }
  assert.equal(chunks[0]?.type, 'IHDR', `${label}: missing initial IHDR.`);
  assert.equal(chunks.filter(c => c.type === 'IHDR').length, 1);
  const header = chunks[0].data;
  assert.equal(header.length, 13);
  assert.equal(header.readUInt32BE(0), 1200, `${label}: width must be 1200.`);
  assert.equal(header.readUInt32BE(4), 630, `${label}: height must be 630.`);
  assert.equal(header[8], 8, `${label}: bit depth must be 8.`);
  assert.equal(header[9], 2, `${label}: PNG must be truecolor RGB, color type 2.`);
  assert.equal(header[10], 0, `${label}: invalid compression method.`);
  assert.equal(header[11], 0, `${label}: invalid filter method.`);
  assert(!chunks.some(c => c.type === 'tRNS'), `${label}: transparency is not permitted.`);
  const srgb = chunks.filter(c => c.type === 'sRGB');
  assert.equal(srgb.length, 1, `${label}: require an explicit sRGB chunk.`);
  assert.equal(srgb[0].data.length, 1);
  assert(srgb[0].data[0] <= 3, `${label}: invalid sRGB rendering intent.`);
  assert(chunks.some(c => c.type === 'IDAT'), `${label}: missing image data.`);
  assert.equal(chunks.at(-1)?.type, 'IEND', `${label}: missing terminal IEND.`);
  assert.equal(chunks.at(-1).data.length, 0);
  return {width: 1200, height: 630, mode: 'RGB', colorSpace: 'sRGB', bytes: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex')};
}

async function fetchResponse(url, mime) {
  const response = await fetch(url, {signal: AbortSignal.timeout(30_000), redirect: 'follow'});
  assert.equal(response.status, 200, `${url}: HTTP ${response.status}.`);
  assert.equal(response.headers.get('content-type')?.split(';')[0].trim().toLowerCase(), mime, `${url}: incorrect Content-Type.`);
  return response;
}

const [html, image, sourceImage] = await Promise.all([
  readFile(new URL('dist/index.html', root), 'utf8'),
  readFile(new URL(`dist/${filename}`, root)),
  readFile(new URL(`public/${filename}`, root)),
]);
const metadata = inspectHTML(html, 'Production HTML');
const png = inspectPNG(image, 'Production PNG');
assert(image.equals(sourceImage), 'Production image differs from the editable-source export in public/.');
const report = {status: 'passed', local: {metadata, image: png}};
if (base) {
  const pageResponse = await fetchResponse(base, 'text/html');
  inspectHTML(await pageResponse.text(), 'Served HTML');
  const servedImageURL = new URL(filename, base);
  const imageResponse = await fetchResponse(servedImageURL, 'image/png');
  const servedImage = Buffer.from(await imageResponse.arrayBuffer());
  const served = inspectPNG(servedImage, 'Served PNG');
  assert.equal(served.sha256, png.sha256, 'Served PNG differs from the tested local production image.');
  report.delivery = {page: base.href, image: servedImageURL.href, metadata: 'matched', mime: 'image/png', sha256: served.sha256};
}
console.log(JSON.stringify(report, null, 2));
