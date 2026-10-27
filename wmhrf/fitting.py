"""Multi-start nonlinear least squares and fit statistics."""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def r2(y, target):
    target = np.asarray(target)
    return 1 - np.sum((y - target) ** 2) / np.sum((target - target.mean()) ** 2)


def multistart(resid, lo, hi, n_starts, rng, seeds=(), start_ranges=None, max_nfev=None):
    """Run trust-region reflective least squares from several starts; return all results sorted by cost.

    seeds: explicit starting vectors tried first (clipped into the bounds).
    start_ranges: (lo, hi) for drawing random starts; defaults to the bounds.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    slo, shi = (lo, hi) if start_ranges is None else map(np.asarray, start_ranges)
    out = []
    for i in range(n_starts):
        if i < len(seeds):
            x0 = np.clip(np.asarray(seeds[i], float), lo + 1e-6, hi - 1e-6)
        else:
            x0 = rng.uniform(slo, shi)
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=max_nfev)
        except Exception:
            continue
        out.append(r)
    out.sort(key=lambda r: r.cost)
    return out


def peak_in_band(f, y, fmin=0.03, fmax=0.1, min_prom=0.0):
    """Prominence and frequency of the first local maximum between fmin and fmax (0, nan if none)."""
    pk, pr = find_peaks(y, prominence=min_prom)
    for i, p in zip(pk, pr['prominences']):
        if fmin < f[i] < fmax:
            return float(p), float(f[i])
    return 0.0, float('nan')
