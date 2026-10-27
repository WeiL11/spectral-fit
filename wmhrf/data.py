"""Target data: white-matter spectra (Wu et al., 2026) and HRF (Schilling et al., 2022).

The digitized curves are produced by scripts/01_digitize_schilling2022.py and
scripts/02_digitize_wu2026.py and stored in data/. See data/README.md.
"""
import os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
TR = 0.72                                   # HCP Young Adult repetition time (s)
F = np.linspace(0.01, 0.20, 400)           # frequency grid of all spectral fits (Hz)

# Wu et al. (2026), Table 1: per-subject parameters of equation (1), (c, a, b, A, f0, gamma)
TABLE1_MEDIAN = (0.9039, 2.6496, 47.197, 0.0561, 0.0602, 0.0150)
TABLE1_MEAN = (0.8927, 2.9398, 51.366, 0.0644, 0.0597, 0.0153)
TABLE1_SD = (0.1133, 1.1427, 25.581, 0.0296, 0.0123, 0.0044)

# Tail of the WM HRF beyond 12.7 s, where the WM and GM bands overlap in Schilling et al.
# Fig. 3; read manually from the figure (time s, value).
HRF_TAIL = [(13.3, 0.11), (14.0, 0.04), (15.0, -0.025), (16.3, -0.047), (17.0, -0.059),
            (18.0, -0.063), (20.0, -0.051), (22.0, -0.035), (24.0, -0.015), (25.2, -0.005)]


def _load(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found: run the digitization scripts first (see data/README.md)')
    return np.loadtxt(path, delimiter=',', skiprows=1)


def hrf_raw():
    """Digitized WM HRF columns: t, mean, band edge 1, band edge 2, pixel count."""
    return _load('schilling2022_fig3_wm_hrf_raw.csv')


def hrf_target(tmax=None):
    """WM HRF sampled at the TR from 0 to 25.2 s (PCHIP through the digitized points)."""
    raw = hrf_raw()
    m = raw[:, 0] <= 12.7
    tt = np.r_[0.0, raw[m, 0], [t for t, _ in HRF_TAIL]]
    vv = np.r_[0.0, raw[m, 1], [v for _, v in HRF_TAIL]]
    o = np.argsort(tt)
    tg = np.arange(0, 25.2, TR)
    hs = PchipInterpolator(tt[o], vv[o])(tg)
    if tmax is not None:
        k = tg <= tmax
        tg, hs = tg[k], hs[k]
    return tg, hs


def hrf_band_halfwidth(tg):
    """Half-width of the +-1 SD band at tg (digitized to 12.7 s, 0.01 assumed beyond)."""
    raw = hrf_raw()
    m = raw[:, 0] <= 12.7
    bw = np.abs(raw[m, 3] - raw[m, 2]) / 2
    bt = np.r_[0, raw[m, 0], 13.3, 40]
    return np.interp(tg, bt, np.r_[0.005, uniform_filter1d(bw, 5), 0.01, 0.01])


def spectrum_groupmean():
    """Group-mean WM amplitude spectrum on F (9-point smoothing of the digitized curve)."""
    raw = _load('wu2026_fig1b_wm_spectrum_raw.csv')
    o = np.argsort(raw[:, 0])
    return np.interp(F, raw[o, 0], uniform_filter1d(raw[o, 1], 9))


def spectrum_individual():
    """Representative individual spectrum: equation (1) with the Table 1 mean parameters."""
    from .model import eq1
    return eq1(TABLE1_MEAN, F)


def save_csv(name, header, cols):
    os.makedirs(RESULTS, exist_ok=True)
    np.savetxt(os.path.join(RESULTS, name), np.column_stack(cols), delimiter=',', header=header, comments='')
