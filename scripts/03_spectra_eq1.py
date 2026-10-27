"""Equation (1) fit to the group-mean spectrum, and the effect of averaging subject spectra.

Manuscript: Introduction (group-mean Eq. 1 parameters) and Results, "Group-mean and individual
spectra" (Fig. 4A).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, report, starts

f = data.F
Tobs = data.spectrum_groupmean()
rng = np.random.default_rng(0)

# group-level Eq. (1) fit, 200 random starts
lo, hi = [0, 0, 1, 0, 0.03, 0.002], [3, 20, 500, 1, 0.1, 0.08]
res = fitting.multistart(lambda p: model.eq1(p, f) - Tobs, lo, hi, starts(200), rng,
                         start_ranges=([.7, 1, 20, .01, .04, .005], [1.1, 4, 120, .1, .07, .04]))
p = res[0].x
print('Equation (1) fitted to the group-mean spectrum')
for name, v, e in zip(('c', 'a', 'b (s)', 'A', 'f0 (Hz)', 'gamma (Hz)'), p, (0.892, 2.53, 43.2, 0.0769, 0.0576, 0.0241)):
    report(name, v, e)
report('R2', fitting.r2(model.eq1(p, f), Tobs), 0.9996)
report('Q = f0/2gamma', p[4] / 2 / p[5], 1.2)
report('rise of the maximum above the trough', fitting.peak_in_band(f, Tobs, 0.03, 0.1)[0], 0.10)

# averaging spectra of simulated subjects drawn from Table 1 means and SDs
m, s = data.TABLE1_MEAN, data.TABLE1_SD
S = []
for i in range(starts(4000)):
    q = [m[0] + s[0] * rng.standard_normal(), max(0.2, m[1] + s[1] * rng.standard_normal()),
         max(10, m[2] + s[2] * rng.standard_normal()), max(0.005, m[3] + s[3] * rng.standard_normal()),
         np.clip(m[4] + s[4] * rng.standard_normal(), 0.03, 0.15), np.clip(m[5] + s[5] * rng.standard_normal(), 0.004, 0.06)]
    S.append(model.eq1(q, f))
Sg = np.mean(S, 0)
Tind = data.spectrum_individual()
print('Group-mean vs individual spectra')
report('peak rise, Table 1 mean parameters', fitting.peak_in_band(f, Tind)[0], 0.62)
report('peak rise, average of 4000 simulated subjects', fitting.peak_in_band(f, Sg)[0], 0.19)
report('R2 of the average vs observed group mean', fitting.r2(Sg, Tobs), 0.978)
report('R2 of Table 1 median curve vs observed', fitting.r2(model.eq1(data.TABLE1_MEDIAN, f), Tobs), '0.95 (not reported)')

save_result('eq1_groupmean.json', dict(params=p.tolist(), r2=fitting.r2(model.eq1(p, f), Tobs)))
data.save_csv('spectra.csv', 'f_Hz,observed_groupmean,eq1_table1_median,eq1_table1_mean,eq1_groupfit,average_simulated_subjects',
              [f, Tobs, model.eq1(data.TABLE1_MEDIAN, f), Tind, model.eq1(p, f), Sg])
