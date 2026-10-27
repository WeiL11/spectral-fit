"""Hemodynamic filter and spectral models.

The white-matter HRF is the linearized revised balloon model of Gore et al. (2026),
equation (13) of the manuscript:

    H_R(s) = -G_m / ((1 + tau_m s)^2 (1 + tau_0 s))
             + G_f exp(-Delta s) [rho (1 + tau_0 s) + (1 - alpha + alpha tau_v s)]
               / ((s^2 + kappa s + gamma_a)(1 + tau_c s)(1 + tau_0 s))

with tau_c = alpha (tau_0 + tau_v) and rho = alpha (1 - 1/beta_D).

Parameter vectors use the order of PARAM_NAMES. Seven-element vectors hold
tau_v and alpha at their gray-matter defaults.
"""
import numpy as np

BETA_D = 3.900 / 3.105          # Davis exponent from the 3 T signal constants (Gore et al., 2026)
TAU_M, TAU_V, ALPHA = 2.99, 10.03, 0.326   # canonical-HRF fit of Gore et al. (2026), Table 4
PARAM_NAMES = ('tau0', 'kappa', 'gamma_a', 'Delta', 'G_m', 'G_f', 'tau_m', 'tau_v', 'alpha')


def full_params(p):
    """Return a 9-element parameter vector (fills tau_m, tau_v, alpha with defaults)."""
    p = list(p)
    if len(p) == 6:
        p += [TAU_M]
    if len(p) == 7:
        p += [TAU_V, ALPHA]
    if len(p) != 9:
        raise ValueError('expected 6, 7 or 9 parameters')
    return np.asarray(p, float)


def transfer(p, f):
    """Complex transfer function H_R(i 2 pi f) of equation (13)."""
    t0, k, ga, D, Gm, Gf, tm, tv, al = full_params(p)
    s = 2j * np.pi * np.asarray(f, float)
    rho = al * (1 - 1 / BETA_D)
    tc = al * (t0 + tv)
    met = -Gm / ((1 + tm * s) ** 2 * (1 + t0 * s))
    flo = Gf * np.exp(-D * s) * (rho * (1 + t0 * s) + (1 - al + al * tv * s)) / (
        (s * s + k * s + ga) * (1 + tc * s) * (1 + t0 * s))
    return met + flo


def impulse_response(p, dt=0.02, T=400.0):
    """HRF h(t) by numerical inverse Fourier transform of equation (13)."""
    n = int(round(T / dt))
    fr = np.fft.rfftfreq(n, dt)
    h = np.fft.irfft(transfer(p, fr), n) / dt
    return np.arange(n) * dt, h


def hrf_at(p, tq, dt=0.02, T=400.0):
    t, h = impulse_response(p, dt, T)
    return np.interp(tq, t, h)


def eq1(p, f):
    """Exponential-Lorentzian description of the amplitude spectrum (Wu et al., 2026), equation (1)."""
    c, a, b, A, f0, g = p
    return c + a * np.exp(-b * f) + A / np.pi * g / ((f - f0) ** 2 + g ** 2)


def lorentz(A, f0, g, f):
    return A / np.pi * g / ((f - f0) ** 2 + g ** 2)


def model_spectrum(p, z, f, line=None):
    """Equation (5): all neural input filtered by the HRF, instrumental noise added in power.

    z = (c, a, b, N0): input amplitude c + a exp(-b f) and noise power N0.
    line = (A, f0, gamma): optional Lorentzian line added to the input (rhythmic input).
    """
    c, a, b, N0 = z
    inp = c + a * np.exp(-b * f)
    if line is not None:
        inp = inp + lorentz(*line, f)
    return np.sqrt(inp ** 2 * np.abs(transfer(p, f)) ** 2 + N0)


def hrf_features(t, h, tmax=40.0, dip_window=8.0):
    """Dip, peak and undershoot of an impulse response (values and times)."""
    m = t < tmax
    tt, hh = t[m], h[m]
    i = np.argmin(hh[tt < dip_window])
    j = np.argmax(hh)
    u = j + np.argmin(hh[j:])
    return dict(dip=hh[i], t_dip=tt[i], peak=hh[j], t_peak=tt[j], under=hh[u], t_under=tt[u])


def bandpass(p, fmax=0.3, n=3001):
    """Maximum filter gain relative to zero frequency, and its frequency."""
    fh = np.linspace(0, fmax, n)
    H = np.abs(transfer(p, fh))
    k = np.argmax(H)
    return H[k] / H[0], fh[k]


def pole(p):
    """Damped frequency (Hz), quality factor omega_d/kappa and damping ratio of the flow stage."""
    t0, k, ga = full_params(p)[:3]
    wd = np.sqrt(max(ga - k * k / 4, 0.0))
    return wd / (2 * np.pi), (wd / k if k > 0 else np.nan), k / (2 * np.sqrt(ga))
