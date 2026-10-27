"""How the dip governs the band-pass filter: sweeps of the metabolic weight G_m/G_f and the delay Delta (Fig. 5).

All other parameters at the joint fit (equation (14)). Requires script 06.
"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, data, save_result, load_result, report

q = np.array(load_result('joint_fit_groupmean.json')['hrf_params'])


def feats(p):
    t, h = model.impulse_response(p)
    ft = model.hrf_features(t, h)
    g, fm = model.bandpass(p)
    return -ft['dip'] / ft['peak'], g, fm, ft['t_peak']


rows_r = []
for r in np.linspace(1.0, 3.25, 10):
    p = q.copy(); p[4] = r * p[5]; rows_r.append((r,) + feats(p))
rows_d = []
for D in np.linspace(0.5, 5, 10):
    p = q.copy(); p[3] = D; rows_d.append((D,) + feats(p))
R, Dd = np.array(rows_r), np.array(rows_d)
print('Metabolic weight sweep: G_m/G_f, dip/peak, max gain, f_max (Hz), time to peak (s)')
print(np.round(R, 3))
report('dip/peak from ratio 1.0 to 3.25', R[0, 1], '13% to 79%'); report('   end', R[-1, 1])
report('max gain at ratio 3.25', R[-1, 2], 9.5); report('f_max at ratio 3.25', R[-1, 3], 0.058)
report('Spearman(dip, f_max)', spearmanr(R[:, 1], R[:, 3]).correlation, 1.0)
for r in (3.0, 3.25, 3.5, 3.75):
    p = q.copy(); p[4] = r * p[5]
    report(f'net HRF area (DC gain) at ratio {r}', model.transfer(p, np.array([1e-7]))[0].real, 'crosses zero ~3.4')
print('Delay sweep: Delta (s), dip/peak, max gain, f_max (Hz), time to peak (s)')
print(np.round(Dd, 3))
report('dip/peak from Delta 0.5 to 5 s', Dd[0, 1], '3% to 58%'); report('   end', Dd[-1, 1])
report('f_max from Delta 0.5 to 5 s', Dd[0, 3], '0.035 to 0.051'); report('   end', Dd[-1, 3])
report('max gain: start, minimum, end', Dd[0, 2], '1.63, 1.45, 1.86'); report('   minimum', Dd[:, 2].min()); report('   end', Dd[-1, 2])
save_result('dip_link_sweeps.json', dict(ratio_sweep=R.tolist(), delay_sweep=Dd.tolist(),
            columns=['value', 'dip_over_peak', 'max_gain', 'f_max_Hz', 't_peak_s']))
