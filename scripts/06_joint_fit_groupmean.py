"""Joint fit of the WM HRF and the group-mean WM spectrum (equation (14), Fig. 3).

All neural input is filtered by the HRF; instrumental noise is added in power (equation (5)).
Residuals of each data set are divided by its SD and sqrt(number of points) and weighted equally
(w = 0.5). Free: tau0, kappa, gamma_a, Delta, G_m, G_f, input (c, a, b) and N0; tau_m, tau_v,
alpha fixed. Requires results/fit_hrf_alone.json (script 05).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, load_result, report, starts

f = data.F
T = data.spectrum_groupmean()
tg, hs = data.hrf_target()
pH = np.array(load_result('fit_hrf_alone.json')['params'])
TM = model.TAU_M
rng = np.random.default_rng(11)

# (a) HRF fixed at the HRF-alone fit, input and noise free
res = fitting.multistart(lambda z: model.model_spectrum(pH, z, f) - T, [0, 0, 1, 0], [20, 50, 1000, 5], starts(60), rng,
                         start_ranges=([0, 0, 5, 0], [1, 3, 150, 1]))
yfix = model.model_spectrum(pH, res[0].x, f)
print('Spectrum predicted by the HRF-alone filter (input free)')
report('R2 spectrum', fitting.r2(yfix, T), 0.973)

# (b) joint fit
sh, sT, nh, nT, w = hs.std(), T.std(), len(hs), len(T), 0.5
def resid(v):
    q = np.r_[v[:6], TM]
    return np.r_[np.sqrt(w) * (model.hrf_at(q, tg) - hs) / sh / np.sqrt(nh),
                 np.sqrt(1 - w) * (model.model_spectrum(q, v[6:], f) - T) / sT / np.sqrt(nT)]
lo = [.3, .01, .01, 0, 0, 0, 0, 0, 1, 0]
hi = [15, 5, 5, 8, 50, 50, 20, 50, 1000, 5]
seeds = [np.r_[pH[:6] * rng.uniform(0.6, 1.5, 6), rng.uniform(0, 1), rng.uniform(0, 3), rng.uniform(5, 150), rng.uniform(0, 1)]
         for _ in range(40)]
res = fitting.multistart(resid, lo, hi, starts(40), rng, seeds=seeds)
v = res[0].x
q, z = np.r_[v[:6], TM], v[6:]
y = model.model_spectrum(q, z, f)
f0, Q, zeta = model.pole(q)
gain, fmax = model.bandpass(q)
band = (f >= 0.03) & (f <= 0.09)
r = y - T
print('Joint fit, equation (14)')
for nm, val, e in zip(model.PARAM_NAMES[:4], q[:4], (4.15, 0.540, 0.218, 2.91)):
    report(nm, val, e)
report('G_m/G_f', q[4] / q[5], 2.35)
report('R2 HRF', fitting.r2(model.hrf_at(q, tg), hs), 0.993)
report('R2 spectrum', fitting.r2(y, T), 0.987)
report('damping ratio kappa/2sqrt(gamma_a)', zeta, 0.58)
report('max filter gain / gain at 0 Hz', gain, 1.48); report('   at frequency (Hz)', fmax, 0.047)
report('input b (s)', z[2], 111); report('sqrt(N0)', np.sqrt(z[3]), 0.88)
report('largest residual', abs(r).max(), '0.15 (9%) at 0.06 Hz')
report('excess of data over model, 0.03-0.09 Hz', np.trapezoid(np.clip(T - y, 0, None)[band], f[band]) / np.trapezoid(T[band], f[band]), '~2%')
save_result('joint_fit_groupmean.json', dict(hrf_params=q.tolist(), input_noise=z.tolist(),
            r2_hrf=fitting.r2(model.hrf_at(q, tg), hs), r2_spectrum=fitting.r2(y, T), r2_spectrum_hrf_alone_filter=fitting.r2(yfix, T)))
