"""Trade-off between HRF fit and individual-spectrum fit (Fig. 4D) and the spectrum-prioritized fit (Fig. 4B, C).

All nine hemodynamic parameters free (including tau_m, tau_v, alpha), plus input (c, a, b) and N0.
Residuals normalized by SD and sqrt(n) of each data set, weighted sqrt(w) (HRF) and sqrt(1 - w) (spectrum).

  --tradeoff     weights 0.1 ... 0.9, HRF over 0-25 s, N starts per weight (manuscript: 80 per weight, 720 fits)
  --prioritized  w = 0.1, HRF over 0-20 s only (manuscript: 80 starts)  [default]
  --starts N     number of starts per weight (default 80)

Run time: a few seconds per start. Requires scripts 06, 07 and 08 (starting seeds).
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, fitting, data, save_result, load_result, report, starts

ap = argparse.ArgumentParser()
ap.add_argument('--tradeoff', action='store_true')
ap.add_argument('--prioritized', action='store_true')
ap.add_argument('--starts', type=int, default=starts(80))
args = ap.parse_args()
if not args.tradeoff and not args.prioritized:
    args.prioritized = True

f = data.F
Ti = data.spectrum_individual()
DT, TW = 0.05, 200.0                       # impulse-response grid for these fits
LO = [0.3, 0.01, 0.01, 0, 0, 0, 0.5, 0, 0.2, 0, 0, 1, 0]
HI = [15, 5, 5, 8, 60, 60, 8, 30, 0.6, 20, 50, 1000, 5]
q = load_result('joint_fit_groupmean.json')['hrf_params']
zi = load_result('individual_spectrum.json')['input_broadband']
vs = load_result('spectrum_only_fit.json')['params']
SEEDS = [np.r_[q[:6], model.TAU_M, model.TAU_V, model.ALPHA, zi], np.r_[vs[:6], model.TAU_M, model.TAU_V, model.ALPHA, vs[6:]]]


def run(w, tmax, n, seed):
    tg, hs = data.hrf_target(tmax)
    sh, sT, nh, nT = hs.std(), Ti.std(), len(hs), len(Ti)
    def resid(v):
        p = v[:9]
        return np.r_[np.sqrt(w) * (model.hrf_at(p, tg, DT, TW) - hs) / sh / np.sqrt(nh),
                     np.sqrt(1 - w) * (model.model_spectrum(p, v[9:], f) - Ti) / sT / np.sqrt(nT)]
    rng = np.random.default_rng(seed)
    lo, hi = np.array(LO), np.array(HI)
    res = fitting.multistart(resid, lo, hi, n, rng, seeds=SEEDS,
                             start_ranges=(lo, np.minimum(hi, lo + (hi - lo) * 0.6)), max_nfev=3000)
    tg25, hs25 = data.hrf_target()
    k20 = tg25 <= 20
    out = []
    for r in res:
        p, z = r.x[:9], r.x[9:]
        y = model.model_spectrum(p, z, f)
        yh = model.hrf_at(p, tg25, DT, TW)
        out.append(dict(w=w, cost=r.cost, r2_hrf_0_25=fitting.r2(yh, hs25), r2_hrf_0_20=fitting.r2(yh[k20], hs25[k20]),
                        r2_spectrum=fitting.r2(y, Ti), prominence=fitting.peak_in_band(f, y)[0],
                        Q=model.pole(p)[1], params=r.x.tolist()))
    return out


if args.prioritized:
    out = run(0.1, 20.0, args.starts, 1)
    b = out[0]; p = np.array(b['params'][:9])
    t, h = model.impulse_response(p, DT, TW)
    ft = model.hrf_features(t, h, tmax=45)
    late = t > 18
    jl = np.argmax(np.where(late & (t < 45), h, -np.inf))
    print('Spectrum-prioritized joint fit (w = 0.1, HRF 0-20 s)')
    report('R2 spectrum', b['r2_spectrum'], 0.999); report('line prominence', b['prominence'], '0.58 (target 0.62)')
    report('R2 HRF over 0-20 s', b['r2_hrf_0_20'], 0.95); report('pole Q', b['Q'], 2.2)
    report('dip / time (s)', ft['dip'], '-0.30 at 2.2 s'); report('peak', ft['peak'], '0.65 at 8.6 s')
    report('undershoot', ft['under'], -0.12)
    report('second lobe / peak', h[jl] / ft['peak'], '0.26 at 24.5 s'); report('   at (s)', t[jl], 24.5)
    for nm, v in zip(model.PARAM_NAMES, p):
        report(nm, v)
    save_result('spectrum_prioritized_fit.json', b)

if args.tradeoff:
    allr = []
    for i, w in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        allr += run(w, None, args.starts, 100 + i)
        print(f'  weight {w}: done')
    pts = sorted(allr, key=lambda r: -r['r2_hrf_0_25'])
    front, best = [], -9
    for r in pts:
        if r['r2_spectrum'] > best + 1e-4:
            front.append(r); best = r['r2_spectrum']
    print('Pareto front (R2 HRF 0-25 s, R2 spectrum, prominence, Q)')
    for r in front:
        print(f"  {r['r2_hrf_0_25']:.3f}  {r['r2_spectrum']:.4f}  {r['prominence']:.3f}  {r['Q']:.2f}  (w={r['w']})")
    c = [r for r in allr if r['r2_hrf_0_25'] >= 0.99]
    if c:
        report('best spectrum R2 with HRF R2 >= 0.99', max(r['r2_spectrum'] for r in c), 0.935)
    save_result('tradeoff_joint_fits.json', dict(all=allr, front=front))
