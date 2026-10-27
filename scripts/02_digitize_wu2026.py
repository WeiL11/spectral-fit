"""Digitize the group-mean white-matter amplitude spectrum of Wu et al. (2026), Fig. 1B.

Usage:  python scripts/02_digitize_wu2026.py path/to/nihpp-rs10770821v1.pdf [--check]

Source: Wu, S. et al. A distinct second spectral peak in white matter BOLD signals.
Research Square preprint rs.3.rs-10770821/v1 (2026), CC BY 4.0. Requires poppler's pdftoppm.

Page 11 is rendered at 500 dpi and panel B cropped. Axis calibration from the grid lines
(verifiable with --check): f = 0.02 + (x - 246.5)/9025 Hz, amplitude = 1.0 + (691 - y)/354.7.
The red "observed spectrum" curve is located column by column (legend masked).
Output: data/wu2026_fig1b_wm_spectrum_raw.csv (f_Hz, amplitude).
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf.data import DATA

DPI, PAGE = 500, 11
CROP = (2153, 326, 4057, 1173)          # = (560, 85, 1055, 305) at 130 dpi, scaled by 500/130
f_of = lambda x: 0.02 + (x - 246.5) / 9025.0
v_of = lambda y: 1.0 + (691 - y) / 354.7


def render(pdf):
    d = tempfile.mkdtemp()
    subprocess.run(['pdftoppm', '-png', '-r', str(DPI), '-f', str(PAGE), '-l', str(PAGE), pdf, os.path.join(d, 'p')], check=True)
    png = [f for f in os.listdir(d) if f.endswith('.png')][0]
    return np.asarray(Image.open(os.path.join(d, png)).convert('RGB').crop(CROP)).astype(int)


def check(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    H, W = R.shape
    grid = (abs(R - G) < 8) & (abs(G - B) < 8) & (R > 200) & (R < 245)
    def groups(v):
        out, cur = [], [v[0]]
        for x in v[1:]:
            if x - cur[-1] <= 2: cur.append(x)
            else: out.append(round(float(np.mean(cur)), 1)); cur = [x]
        return out + [round(float(np.mean(cur)), 1)]
    print('vertical grid', groups([i for i in range(W) if grid[:, i].sum() > 0.5 * H]), '(0.02..0.20 Hz at ~246.5 ... 1871)')
    print('horizontal grid', groups([i for i in range(H) if grid[i].sum() > 0.5 * W]), '(2.5, 2, 1.5, 1 at ~159, 336, 513.5, 691)')


def main():
    pdf = sys.argv[1]
    a = render(pdf)
    if '--check' in sys.argv:
        check(a)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 180) & (G < 110) & (B < 110)
    red[:260, 1240:] = False                 # legend
    rows = []
    for x in range(160, 1860):
        ys = np.where(red[:, x])[0]
        if len(ys):
            rows.append((f_of(x), v_of(ys.mean())))
    rows = np.array(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, 'wu2026_fig1b_wm_spectrum_raw.csv')
    np.savetxt(out, rows, delimiter=',', header='f_Hz,amplitude', comments='')
    print(f'wrote {out}: {len(rows)} columns; amplitude {rows[0,1]:.3f} at {rows[0,0]:.4f} Hz to {rows[-1,1]:.3f} at {rows[-1,0]:.4f} Hz')


if __name__ == '__main__':
    main()
