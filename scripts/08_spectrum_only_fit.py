"""Underdamped filter fitted to the individual spectrum alone (all input filtered; tau_m, tau_v, alpha fixed).

Gives a near-perfect spectral fit whose implied HRF disagrees strongly with the measured HRF.
Used as a seed for the trade-off analysis (09). Standalone.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, report, starts

f = data.F
Ti = data.spectrum_individual()
tg, hs = data.hrf_target()
TM = model.TAU_M
rng = np.random.default_rng(5)
lo = [.3, .01, .01, 0, 0, 0, 0, 0, 1, 0]
hi = [15, 5, 5, 8, 50, 50, 20, 50, 1000, 5]
res = fitting.multistart(lambda v: model.model_spectrum(np.r_[v[:6], TM], v[6:], f) - Ti, lo, hi, starts(150), rng,
                         start_ranges=([1, .05, .05, 0, 0, 0, 0, 0, 5, 0], [10, 1, .5, 5, 5, 5, 1, 3, 200, 1]))
v = res[0].x
q = np.r_[v[:6], TM]
f0, Q, zeta = model.pole(q)
print('Spectrum-only fit to the individual spectrum')
report('R2 spectrum', fitting.r2(model.model_spectrum(q, v[6:], f), Ti), 1.000)
report('pole Q', Q, 2.37); report('pole frequency (Hz)', f0, 0.060)
report('R2 of implied HRF vs measured', fitting.r2(model.hrf_at(q, tg), hs), -1.97)
save_result('spectrum_only_fit.json', dict(params=v.tolist()))
