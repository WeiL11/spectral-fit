"""Amplitude spectrum of the measured WM HRF, and a simulated superposition of randomly timed HRFs.

Manuscript: Results, "The hemodynamic filter" (band-pass WM HRF; Fig. 2).
"""
import os, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import data, save_result, report, starts

tg, hs = data.hrf_target()
dt = 0.05
tt = np.arange(0, 25.2, dt)
h = PchipInterpolator(tg, hs)(tt)
n = int(2000 / dt)                                    # zero-padded to 2000 s
fr = np.fft.rfftfreq(n, dt)
spec = lambda x: np.abs(np.fft.rfft(x, n)) * dt


def peak(x):
    A = spec(x); m = fr < 0.3; k = np.argmax(A[m])
    return A[m][k] / A[0], fr[m][k]


nodip = h.copy(); nodip[(tt < 5) & (h < 0)] = 0
nound = h.copy(); nound[(tt > 12) & (h < 0)] = 0
both = h.copy(); both[h < 0] = 0
print('Band-pass property of the measured WM HRF (max |H| / |H(0)| and its frequency)')
for name, x, e in (('measured', h, '1.35 at 0.048 Hz'), ('initial dip removed', nodip, '1.09 at 0.032 Hz'),
                   ('undershoot removed', nound, '1.08 at 0.046 Hz'), ('both removed', both, '1.00 (low-pass)')):
    g, fp = peak(x)
    report(name, g, e, '{:.3f}'); report('   at frequency (Hz)', fp, None, '{:.4f}')

# simulated superposition: Poisson events convolved with the measured HRF, HCP-like sampling
rng = np.random.default_rng(0)
TR, N, dtf, up, rate = 0.72, 1200, 0.08, 9, 0.1
hk = PchipInterpolator(tg, hs)(np.arange(0, 25.2, dtf))
specs = []
for s in range(starts(120)):
    L = N * up + len(hk)
    ev = rng.poisson(rate * dtf, L) * rng.normal(1, 0.5, L)
    x = np.convolve(ev, hk)[len(hk):len(hk) + N * up][::up]
    x = x - x.mean()
    specs.append(np.abs(np.fft.rfft(x * np.hanning(N))))
fs = np.fft.rfftfreq(N, TR)
S = np.mean(specs, 0)
k = (fs > 0.005) & (fs < 0.25)
Sm = uniform_filter1d(S[k], 7)
j = np.argmax(np.where(fs[k] > 0.02, Sm, 0))
report('simulated superposition: frequency of maximum (Hz)', fs[k][j], 0.050, '{:.4f}')

data.save_csv('hrf_spectra.csv', 'f_Hz,measured,dip_removed,undershoot_removed',
              [fr[fr <= 0.2], spec(h)[fr <= 0.2], spec(nodip)[fr <= 0.2], spec(nound)[fr <= 0.2]])
data.save_csv('superposition_simulation.csv', 'f_Hz,mean_amplitude,smoothed', [fs[k], S[k], Sm])
save_result('hrf_bandpass.json', dict(measured=peak(h), dip_removed=peak(nodip), undershoot_removed=peak(nound),
                                      both_removed=peak(both), simulation_fmax=float(fs[k][j])))
