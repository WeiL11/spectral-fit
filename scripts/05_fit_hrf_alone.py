"""Fit equation (13) to the measured WM HRF alone (tau_m, tau_v, alpha fixed at gray-matter values).

Used as the starting point for the joint fit (06) and reported in a manuscript comment
(tau0 ~ 4.4 s, Delta ~ 3.1 s, G_m/G_f ~ 1.95, R2 ~ 0.999).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, report, starts

tg, hs = data.hrf_target()
rng = np.random.default_rng(0)
TM = model.TAU_M
lo = [.3, .01, .01, 0, 0, 0, TM - 1e-6]
hi = [15, 5, 5, 8, 50, 50, TM + 1e-6]
res = fitting.multistart(lambda p: model.hrf_at(p, tg) - hs, lo, hi, starts(300), rng,
                         start_ranges=([0.8, .1, .05, 0, 0, 0, TM - 1e-6], [6, 1.5, .6, 4, 3, 3, TM + 1e-6]))
p = res[0].x
y = model.hrf_at(p, tg)
f0, Q, zeta = model.pole(p)
t, h = model.impulse_response(p)
ft = model.hrf_features(t, h, dip_window=6)
print('Equation (13) fitted to the measured WM HRF')
for nm, v, e in zip(model.PARAM_NAMES[:4], p[:4], (4.47, 0.628, 0.270, 3.11)):
    report(nm, v, e)
report('G_m/G_f', p[4] / p[5], 1.95)
report('R2', fitting.r2(y, hs), 0.9985)
report('damping ratio', zeta, 0.60)
report('dip', ft['dip'], '-0.32 at 3.3 s'); report('peak', ft['peak'], '0.71 at 8.1 s')
save_result('fit_hrf_alone.json', dict(params=p.tolist(), r2=fitting.r2(y, hs)))
