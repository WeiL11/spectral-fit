"""Sensitivity of the individual-spectrum peak to each HRF parameter (Fig. 6; aging predictions).

Starting point: the jointly fitted HRF (script 06) and the rhythmic-input description of the
per-subject mean spectrum (script 07). Each parameter is varied alone. Elasticity = % change in
peak prominence per % change in the parameter (central difference, +-10%). Also finds the change
needed to halve or double the peak. Requires scripts 06 and 07.
"""
import os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, load_result, report, starts

f = data.F
q = np.array(load_result('joint_fit_groupmean.json')['hrf_params'])
zA = np.array(load_result('individual_spectrum.json')['input_with_line'])
NAMES = ['tau0', 'kappa', 'gamma_a', 'Delta', 'ratio', 'tau_m', 'tau_v', 'alpha']
base = dict(tau0=q[0], kappa=q[1], gamma_a=q[2], Delta=q[3], ratio=q[4] / q[5], G_f=q[5], tau_m=q[6],
            tau_v=model.TAU_V, alpha=model.ALPHA)


def vec(P):
    return [P['tau0'], P['kappa'], P['gamma_a'], P['Delta'], P['ratio'] * P['G_f'], P['G_f'], P['tau_m'], P['tau_v'], P['alpha']]


def feats(P):
    p = vec(P)
    y = model.model_spectrum(p, zA[:4], f, line=zA[4:])
    prom, fpk = fitting.peak_in_band(f, y)
    t, h = model.impulse_response(p)
    ft = model.hrf_features(t, h)
    g, fm = model.bandpass(p, 0.2, 2001)
    return dict(prom=prom, fpk=fpk, gain=g, fmax=fm, dip=-ft['dip'] / ft['peak'], ttp=ft['t_peak'])


b = feats(base)
expected = dict(kappa=-2.27, tau0=-1.62, gamma_a=1.19, ratio=0.92, Delta=0.90, tau_m=-0.84, alpha=0.20, tau_v=0.18)
print('Elasticity of the 0.06 Hz peak prominence (manuscript value in brackets)')
el = {}
for nm in NAMES:
    up, dn = dict(base), dict(base)
    up[nm] *= 1.1; dn[nm] *= 0.9
    fu, fd = feats(up), feats(dn)
    el[nm] = {k: (fu[k] - fd[k]) / (0.2 * b[k]) if b[k] else float('nan') for k in b}
    report(nm, el[nm]['prom'], expected[nm], '{:+.2f}')

print('Change in one parameter needed to halve the peak (dip/peak and time to peak at that point)')
xs = np.linspace(0.3, 3.0, 55)
halve = {}
EXP = dict(kappa=('x1.34', 0.56, 8.4), tau0=('x1.50', 0.45, 8.8), gamma_a=('x0.73', 0.33, 9.0), tau_m=('x2.35', 0.08, 7.9),
           ratio=('not reached', None, None), Delta=('not reached', None, None))
for nm, (e, edip, ettp) in EXP.items():
    vals = np.array([feats({**base, nm: base[nm] * x})['prom'] for x in xs])
    tgt = 0.5 * b['prom']
    idx = np.where(np.diff(np.sign(vals - tgt)) != 0)[0]
    if len(idx) == 0:
        report(nm, float('nan'), e); halve[nm] = None; continue
    i = idx[np.argmin(abs(xs[idx] - 1))]
    x = brentq(lambda x: feats({**base, nm: base[nm] * x})['prom'] - tgt, xs[i], xs[i + 1])
    fe = feats({**base, nm: base[nm] * x})
    report(nm, x, e, 'x{:.2f}'); report('   dip/peak at that point', fe['dip'], edip, '{:.2f}'); report('   time to peak (s)', fe['ttp'], ettp, '{:.1f}')
    halve[nm] = x
    data.save_csv(f'sensitivity_curve_{nm}.csv', 'scale,prominence', [xs, vals])
for nm in ('ratio', 'Delta', 'alpha', 'tau_v'):
    vals = np.array([feats({**base, nm: base[nm] * x})['prom'] for x in xs])
    data.save_csv(f'sensitivity_curve_{nm}.csv', 'scale,prominence', [xs, vals])
save_result('sensitivity.json', dict(baseline=b, elasticities=el, halve_factor=halve))
