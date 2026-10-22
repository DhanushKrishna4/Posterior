"""Opaque RGB/sRGB conversion, deterministic optimization and sharing-size proofs."""
from pathlib import Path
from PIL import Image, PngImagePlugin
import hashlib
import json

ROOT = Path(__file__).resolve().parent
metadata = PngImagePlugin.PngInfo()
metadata.add(b'sRGB', b'\x00')


def save(image, path):
    image.convert('RGB').save(path, format='PNG', optimize=True, compress_level=9, pnginfo=metadata)


for name in ['landscape', 'folded']:
    p = ROOT / f'studies/{name}.png'
    with Image.open(p) as image:
        save(image.resize((1200, 630), Image.Resampling.LANCZOS), p)

with Image.open(ROOT / 'proofs/render-2x.png') as source:
    image = source.convert('RGB').resize((1200, 630), Image.Resampling.LANCZOS)
target = ROOT.parent.parent / 'public/posterior-og-v1.png'
save(image, target)
for width, height in [(600,315), (300,158)]:
    save(image.resize((width, height), Image.Resampling.LANCZOS), ROOT / f'proofs/{width}x{height}.png')
save(image.crop((40, 21, 1160, 609)), ROOT / 'proofs/edge-trim.png')
save(image.crop((40, 0, 1160, 630)).resize((600,338), Image.Resampling.LANCZOS), ROOT / 'proofs/16x9.png')
save(image.crop((285,0,915,630)).resize((315,315), Image.Resampling.LANCZOS), ROOT / 'proofs/square-stress.png')
(ROOT / 'proofs/render-2x.png').unlink()
report = {'image':'public/posterior-og-v1.png', 'width':1200, 'height':630, 'mode':'RGB', 'color_space':'sRGB', 'bytes':target.stat().st_size, 'sha256':hashlib.sha256(target.read_bytes()).hexdigest(), 'source':'tools/og/compose.py', 'vector':'tools/og/card.svg', 'reference':{'prior':[2,2], 'successes':7, 'trials':10, 'posterior':[9,5], 'kind':'exact conjugate update; not sampler output'}}
(ROOT / 'proofs/export.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
