"""Individual spectrum (Table 1 mean parameters) with the jointly fitted HRF; rhythmic-input description.

Manuscript: Results, "Group-mean and individual spectra" (Fig. 4B). Requires script 06.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, load_result, report, starts

f = data.F
Tg, Ti = data.spectrum_groupmean(), data.spectrum_individual()
J = load_result('joint_fit_groupmean.json')
q = np.array(J['hrf_params'])
rng = np.random.default_rng(2)
lo, hi = [0, 0, 1, 0], [20, 50, 1000, 5]
loL, hiL = lo + [0, 0.03, 0.003], hi + [1, 0.1, 0.06]

# broadband input only
zi = fitting.multistart(lambda z: model.model_spectrum(q, z, f) - Ti, lo, hi, starts(60), rng)[0].x
yi = model.model_spectrum(q, zi, f)
print('Individual spectrum, jointly fitted HRF, broadband input')
report('R2', fitting.r2(yi, Ti), 0.925)
report('frequency of broad maximum (Hz)', fitting.peak_in_band(f, yi, 0.03, 0.1, 0.005)[1], 0.046)

# rhythmic input: Lorentzian line added to the input, filtered by the same HRF
def withline(z, T):
    return model.model_spectrum(q, z[:4], f, line=z[4:]) - T
zA = fitting.multistart(lambda z: withline(z, Ti), loL, hiL, starts(80), rng)[0].x
zAg = fitting.multistart(lambda z: withline(z, Tg), loL, hiL, starts(80), rng)[0].x
band = (f >= 0.03) & (f <= 0.09)
def share(z):
    ln = model.lorentz(*z[4:], f); inp = z[0] + z[1] * np.exp(-z[2] * f)
    return np.trapezoid(ln[band], f[band]) / np.trapezoid((inp + ln)[band], f[band])
print('Rhythmic input (filtered Lorentzian line)')
report('individual: R2', fitting.r2(withline(zA, Ti) + Ti, Ti), 0.999)
report('individual: line f0 (Hz)', zA[5], 0.061); report('individual: line Q', zA[5] / 2 / zA[6], 2.2)
report('individual: share of input amplitude 0.03-0.09 Hz', share(zA), '38%')
report('group mean: R2', fitting.r2(withline(zAg, Tg) + Tg, Tg), 0.9998)
report('group mean: line Q', zAg[5] / 2 / zAg[6], 1.5); report('group mean: share', share(zAg), '24%')
save_result('individual_spectrum.json', dict(input_broadband=zi.tolist(), r2_broadband=fitting.r2(yi, Ti),
            input_with_line=zA.tolist(), groupmean_input_with_line=zAg.tolist()))
