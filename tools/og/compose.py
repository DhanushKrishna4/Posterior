"""Posterior's editable OG artwork. Geometry is analytic, type is outlined at export.

System fonts stay on the rendering machine; no font binary is redistributed.
See README.md for licensed-font overrides and the one-command export.
"""
from pathlib import Path
import hashlib
import html
import json
import math
import os

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parent
PAPER, GROUND, INK = '#fcfaf6', '#f4f1e9', '#252d35'
INDIGO, SAGE, MUTED, LINE = '#414c99', '#93a28f', '#626d73', '#d9d6cd'


class Typeface:
    def __init__(self, variable, fallback, index=0):
        self.path = Path(os.environ.get(variable, fallback))
        if not self.path.is_file():
            raise SystemExit(f'Missing licensed font: set {variable}. See tools/og/README.md.')
        self.font = TTFont(self.path, fontNumber=index)
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.units = self.font['head'].unitsPerEm
        self.kern = {}
        if 'kern' in self.font:
            for table in self.font['kern'].kernTables:
                if table.version == 0:
                    self.kern.update(table.kernTable)

    def text(self, value, x, y, size, fill=INK, tracking=0, anchor='start', accents=None):
        cursor, previous, parts = 0, None, []
        scale = size / self.units
        for char in value:
            name = self.cmap.get(ord(char))
            if name is None:
                raise ValueError(f'Missing glyph {char!r} in {self.path.name}')
            cursor += self.kern.get((previous, name), 0) * scale
            pen = SVGPathPen(self.glyphs)
            self.glyphs[name].draw(pen)
            color = (accents or {}).get(char, fill)
            parts.append(f'<path fill="{color}" transform="translate({cursor:.5f} 0) scale({scale:.8f} {-scale:.8f})" d="{pen.getCommands()}"/>')
            cursor += self.font['hmtx'][name][0] * scale + tracking
            previous = name
        width = cursor - tracking
        if anchor == 'end':
            x -= width
        elif anchor == 'middle':
            x -= width / 2
        return f'<g aria-label="{html.escape(value, quote=True)}" fill="{fill}" transform="translate({x:.3f} {y:.3f})">' + ''.join(parts) + '</g>'


serif = Typeface('POSTERIOR_SERIF_FONT', '/System/Library/Fonts/Supplemental/Georgia.ttf')
italic = Typeface('POSTERIOR_ITALIC_FONT', '/System/Library/Fonts/Supplemental/Georgia Italic.ttf')
sans = Typeface('POSTERIOR_SANS_FONT', '/System/Library/Fonts/HelveticaNeue.ttc')


def line(x1, y1, x2, y2, color=LINE, width=1, extra=''):
    return f'<path d="M{x1} {y1}H{x2}" stroke="{color}" stroke-width="{width}" {extra}/>' if y1 == y2 else f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" stroke-width="{width}" {extra}/>'


def circle(x, y, r, fill=INDIGO, stroke='none', width=1):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def density(x, a, b):
    # Exact integer-parameter Beta PDF; the default experiment is conjugate.
    constant = math.factorial(a + b - 1) / (math.factorial(a - 1) * math.factorial(b - 1))
    return constant * x ** (a - 1) * (1 - x) ** (b - 1)


def curve(a, b, x0, y0, width, scale, color, fill='none', opacity=1, dashed=False):
    samples = [(x0 + i / 1000 * width, y0 - density(i / 1000, a, b) * scale) for i in range(1001)]
    d = 'M' + 'L'.join(f'{x:.3f},{y:.3f}' for x, y in samples)
    area = f'<path d="{d}Z" fill="{fill}" opacity="{opacity}"/>' if fill != 'none' else ''
    dash = ' stroke-dasharray="9 8"' if dashed else ''
    return area + f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3.5" stroke-linejoin="round"{dash}/>'


def svg(body, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">A real analytic Beta–Binomial update: Beta(2,2) prior, seven successes in ten trials, Beta(9,5) posterior. Curves share a probability-density scale; no sampler output is depicted.</desc>
<rect width="1200" height="630" fill="{GROUND}"/>
{''.join(body)}
</svg>'''


def landscape():
    """Study A: one broad probability landscape below an oversized name."""
    b = [sans.text('A LABORATORY FOR BAYESIAN INFERENCE', 66, 62, 15, INDIGO, 2.0)]
    b += [serif.text('Posterior.', 58, 210, 164, INK, -5.5)]
    b += [sans.text('Belief, revised.', 68, 258, 27, MUTED)]
    b += [curve(2, 2, 68, 532, 1064, 75, SAGE, SAGE, .13, True)]
    b += [curve(9, 5, 68, 532, 1064, 75, INDIGO, INDIGO, .20)]
    b += [line(68, 532, 1132, 532)]
    b += [sans.text('PRIOR', 230, 433, 15, MUTED, 1.5)]
    b += [sans.text('POSTERIOR', 846, 302, 15, INDIGO, 1.5)]
    b += [sans.text('7 successes / 10 trials', 1132, 70, 18, MUTED, anchor='end')]
    for i in range(10):
        b += [circle(926 + 21 * i, 101, 5, INDIGO if i < 7 else GROUND, INDIGO, 1.5)]
    b += [sans.text('Exact update · Beta(2, 2) to Beta(9, 5)', 68, 576, 18, MUTED)]
    b += [sans.text('dhanushkrishna4.github.io/Posterior', 1132, 576, 16, MUTED, anchor='end')]
    return svg(b, 'Posterior — study A, probability landscape')


def folded():
    """Study B: two sequential plates on an indigo field beside the name."""
    b = [f'<rect x="598" width="602" height="630" fill="{INDIGO}"/>']
    b += [sans.text('BAYESIAN INFERENCE', 62, 72, 16, INDIGO, 2.4)]
    b += [serif.text('Posterior.', 56, 268, 112, INK, -4)]
    b += [italic.text('Belief,', 64, 348, 48, INDIGO), italic.text('revised.', 64, 404, 48, INDIGO)]
    b += [sans.text('An inference workbench.', 64, 557, 23, MUTED)]
    b += [sans.text('01 / BEFORE', 650, 69, 15, PAPER, 1.6)]
    b += [curve(2, 2, 650, 245, 494, 48, '#ced4b9', '#ced4b9', .15)]
    b += [line(650, 245, 1144, 245, '#8d94c0')]
    b += [sans.text('7 successes in 10 trials', 650, 302, 20, PAPER)]
    b += [line(650, 330, 1144, 330, '#8d94c0')]
    b += [sans.text('02 / AFTER', 650, 375, 15, PAPER, 1.6)]
    b += [curve(9, 5, 650, 557, 494, 48, PAPER, PAPER, .22)]
    b += [line(650, 557, 1144, 557, '#8d94c0')]
    return svg(b, 'Posterior — study B, two evidence plates')


def final_card():
    """A large typographic masthead above a single, exact probability landscape."""
    b = [sans.text('BELIEF, REVISED.', 66, 62, 17, INDIGO, 2.2)]
    b += [serif.text('Posterior.', 58, 210, 164, INK, -5.5, accents={'.':INDIGO})]
    b += [sans.text('Bayesian inference, made visible.', 68, 260, 28, INK)]
    # Same x and density scales for both normalized PDFs. No independent scaling.
    b += [curve(2, 2, 68, 532, 1064, 87, SAGE, SAGE, .22, True)]
    b += [curve(9, 5, 68, 532, 1064, 87, INDIGO, INDIGO, .94)]
    b += [curve(2, 2, 68, 532, 1064, 87, SAGE, dashed=True)]
    b += [line(68, 532, 1132, 532, '#a5aba3')]
    b += [sans.text('PRIOR', 202, 423, 16, MUTED, 1.5)]
    b += [sans.text('POSTERIOR', 865, 273, 16, INDIGO, 1.5)]
    b += [sans.text('7 successes / 10 trials', 1132, 70, 18, MUTED, anchor='end')]
    for i in range(10):
        b += [circle(926 + 21 * i, 101, 5, INDIGO if i < 7 else GROUND, INDIGO, 1.5)]
    b += [sans.text('EXACT UPDATE', 68, 578, 14, INDIGO, 1.8)]
    b += [sans.text('Beta(2, 2)', 242, 578, 19, MUTED)]
    b += [line(343, 572, 379, 572, MUTED, 1.5), f'<path d="M373 567L379 572L373 577" fill="none" stroke="{MUTED}" stroke-width="1.5"/>']
    b += [sans.text('Beta(9, 5)', 394, 578, 19, MUTED)]
    return svg(b, 'Posterior — Bayesian inference, made visible.')


def main():
    (ROOT / 'studies').mkdir(exist_ok=True)
    (ROOT / 'studies/landscape.svg').write_text(landscape())
    (ROOT / 'studies/folded.svg').write_text(folded())
    (ROOT / 'card.svg').write_text(final_card())
    fonts = {name: {'filename': face.path.name, 'sha256': hashlib.sha256(face.path.read_bytes()).hexdigest()} for name, face in [('serif', serif), ('italic', italic), ('sans', sans)]}
    (ROOT / 'fonts.json').write_text(json.dumps(fonts, indent=2) + '\n')


if __name__ == '__main__':
    main()
