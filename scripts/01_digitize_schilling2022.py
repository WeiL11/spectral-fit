"""Digitize the population-averaged white-matter HRF of Schilling et al. (2022), Fig. 3 (top panel).

Usage:  python scripts/01_digitize_schilling2022.py path/to/tgac035.pdf [--check]

Source: Schilling, K. G. et al. Anomalous and heterogeneous characteristics of the BOLD
hemodynamic response function in white matter. Cereb. Cortex Commun. 3, tgac035 (2022),
open access (CC BY 4.0). Requires poppler's pdftoppm on the PATH.

Page 4 is rendered at 400 dpi and the HRF panel cropped. Axis calibration (from the plot frame
and tick marks, verifiable with --check): t = 5 + (x - 263.5)/30.08 s, value = (484.75 - y)/291.5.
For every pixel column the blue (WM) band is located; its mean row gives the mean HRF and its
extreme rows the band edges. Output: data/schilling2022_fig3_wm_hrf_raw.csv
(t, mean, band edge at lowest pixel, band edge at highest pixel, pixel count).
Beyond ~12.7 s the WM band overlaps the GM band; that tail is read manually (wmhrf/data.py, HRF_TAIL).
"""
import os, sys, subprocess, tempfile
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf.data import DATA

DPI, PAGE = 400, 4
CROP = (1709, 254, 3018, 1000)          # = (470, 70, 830, 275) at 110 dpi, scaled by 400/110
t_of = lambda x: 5 + (x - 263.5) / 30.08
v_of = lambda y: (484.75 - y) / 291.5


def render(pdf):
    d = tempfile.mkdtemp()
    subprocess.run(['pdftoppm', '-png', '-r', str(DPI), '-f', str(PAGE), '-l', str(PAGE), pdf, os.path.join(d, 'p')], check=True)
    png = [f for f in os.listdir(d) if f.endswith('.png')][0]
    return np.asarray(Image.open(os.path.join(d, png)).convert('RGB').crop(CROP)).astype(int)


def check(a):
    """Print the detected frame lines and x tick positions used for the calibration."""
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    gray = (abs(R - G) < 20) & (abs(G - B) < 20) & (R < 200)
    cs, rs = gray.sum(0), gray.sum(1)
    print('frame columns', sorted(int(i) for i in np.argsort(cs)[-4:]), '(expect ~134.5 and ~871.5)')
    print('frame rows   ', sorted(int(i) for i in np.argsort(rs)[-4:]), '(expect ~47.5 = 1.5 and ~630.5 = -0.5)')
    g2 = (abs(R - G) < 25) & (abs(G - B) < 25) & (R < 200)
    band = g2[618:628, 140:866].sum(0)
    print('x ticks', [i + 140 for i in np.where(band >= 6)[0]], '(expect 5,10,20,25 s at ~263.5, 413.5, 714.5, 865)')


def main():
    pdf = sys.argv[1]
    a = render(pdf)
    if '--check' in sys.argv:
        check(a)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    blue = (B > R + 50) & (B > G + 50)
    rows = []
    for x in range(137, 870):
        ys = np.where(blue[50:628, x])[0] + 50
        if len(ys) >= 2:
            rows.append((t_of(x), v_of(ys.mean()), v_of(ys.max()), v_of(ys.min()), len(ys)))
    rows = np.array(rows)
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, 'schilling2022_fig3_wm_hrf_raw.csv')
    np.savetxt(out, rows, delimiter=',', header='t_s,mean,band_edge_low_pixel,band_edge_high_pixel,n_pixels', comments='')
    i = np.argmin(rows[rows[:, 0] < 6, 1]); j = np.argmax(rows[:, 1])
    print(f'wrote {out}: {len(rows)} columns; dip {rows[i,1]:.3f} at {rows[i,0]:.2f} s, peak {rows[j,1]:.3f} at {rows[j,0]:.2f} s')


if __name__ == '__main__':
    main()
