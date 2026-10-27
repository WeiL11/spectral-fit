"""Manuscript figures 1-6 (PDF and 600 dpi PNG) from the saved results of scripts 03-11."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmhrf import model, data, load_result

BL, OR, PU, RD, GR = '#1f6fb4', '#e08a00', '#9b3fb5', '#d62728', '#6b6b6b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.8, 'lines.linewidth': 2})
os.makedirs(data.FIGURES, exist_ok=True)
def save(fig, n):
    fig.tight_layout()
    fig.savefig(os.path.join(data.FIGURES, f'Figure_{n}.pdf'))
    fig.savefig(os.path.join(data.FIGURES, f'Figure_{n}.png'), dpi=600)
    plt.close(fig)
ld = lambda name: np.loadtxt(os.path.join(data.RESULTS, name), delimiter=',', skiprows=1)
f = data.F
J = load_result('joint_fit_groupmean.json'); q, z = np.array(J['hrf_params']), np.array(J['input_noise'])
I = load_result('individual_spectrum.json')
tg, hs = data.hrf_target()

# Figure 1: neural input before filtering (Table 1 median parameters)
fx = np.linspace(0.005, 0.2, 500); c, a, b = data.TABLE1_MEDIAN[:3]
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(fx, c + a * np.exp(-b * fx), color=BL, label='$H_c(f)+H_a(f)$, equation (10)')
ax.axhline(c, color=GR, lw=1, ls='--', label='impulse component $H_c = c$')
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Amplitude (a.u.)'); ax.legend(frameon=False, fontsize=9)
save(fig, 1)

# Figure 2: band-pass WM HRF
dt = 0.05; tt = np.arange(0, 25.2, dt); h = PchipInterpolator(tg, hs)(tt)
nodip = h.copy(); nodip[(tt < 5) & (h < 0)] = 0
Hs = ld('hrf_spectra.csv'); Sim = ld('superposition_simulation.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].axhline(0, color='#bbbbbb', lw=0.8)
ax[0].plot(tt, h, color=BL, label='measured WM HRF'); ax[0].plot(tt, nodip, color=OR, ls='--', lw=1.6, label='initial dip removed')
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('BOLD response (a.u.)'); ax[0].set_title('A  White-matter HRF', loc='left', fontsize=11); ax[0].legend(frameon=False, fontsize=8.5)
ax[1].plot(Hs[:, 0], Hs[:, 1] / Hs[0, 1], color=BL, label='measured')
ax[1].plot(Hs[:, 0], Hs[:, 2] / Hs[0, 2], color=OR, ls='--', lw=1.6, label='dip removed')
ax[1].plot(Sim[:, 0], Sim[:, 2] / np.interp(0.012, Sim[:, 0], Sim[:, 2]) * np.interp(0.012, Hs[:, 0], Hs[:, 1]) / Hs[0, 1],
           color=RD, lw=1, alpha=0.8, label='simulated superposition of HRFs')
ax[1].axhline(1, color='#bbbbbb', lw=0.8)
ax[1].set_xlabel('Frequency (Hz)'); ax[1].set_ylabel('|H(f)| / |H(0)|'); ax[1].set_title('B  Amplitude spectra', loc='left', fontsize=11); ax[1].legend(frameon=False, fontsize=8.5, loc='lower left')
save(fig, 2)

# Figure 3: joint fit
Tg = data.spectrum_groupmean()
H = np.abs(model.transfer(q, f)); inp = z[0] + z[1] * np.exp(-z[2] * f)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(f, Tg, color=RD, label='WM group-mean spectrum (Wu et al., 2026)')
ax[0].plot(f, model.model_spectrum(q, z, f), color=BL, label=f"model, equation (5), R² = {J['r2_spectrum']:.3f}")
ax[0].plot(f, inp * H, color=BL, lw=1.2, ls=':', label='neural input × HRF')
ax[0].plot(f, inp * H[0], color=GR, lw=1.2, ls=':', label='neural input before filtering (scaled)')
ax[0].axhline(np.sqrt(z[3]), color=GR, lw=1, ls='--', label='instrumental noise floor')
ax[0].set_xlabel('Frequency (Hz)'); ax[0].set_ylabel('Amplitude (a.u.)'); ax[0].set_title('A  Resting-state spectrum', loc='left', fontsize=11); ax[0].legend(frameon=False, fontsize=8.5)
t, hq = model.impulse_response(q); m = t < 25.5
ax[1].axhline(0, color='#bbbbbb', lw=0.8)
ax[1].plot(tg, hs, 'o', color=RD, ms=4.5, mec='white', mew=0.6, label='WM HRF (Schilling et al., 2022)')
ax[1].plot(t[m], hq[m], color=BL, label=f"model, equation (13), R² = {J['r2_hrf']:.3f}")
ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('BOLD response (a.u.)'); ax[1].set_title('B  Hemodynamic response', loc='left', fontsize=11); ax[1].legend(frameon=False, fontsize=8.5)
save(fig, 3)

# Figure 4: group-mean vs individual spectra and the trade-off
Sp = ld('spectra.csv'); Ti = data.spectrum_individual()
fig, ax = plt.subplots(2, 2, figsize=(11, 8)); A, B, C, D = ax.flat
A.plot(Sp[:, 0], Sp[:, 3], color=GR, ls='--', label='Eq. (1), Table 1 mean parameters')
A.plot(Sp[:, 0], Sp[:, 5], color=OR, label='average of simulated subject spectra'); A.plot(Sp[:, 0], Sp[:, 1], color=RD, label='observed group mean')
A.set_xlabel('Frequency (Hz)'); A.set_ylabel('Amplitude (a.u.)'); A.set_title('A  Averaging across subjects flattens the peak', loc='left', fontsize=11); A.legend(frameon=False, fontsize=8.5)
B.plot(f, Ti, color=GR, ls='--', lw=2.4, label='individual spectrum (Table 1 mean)')
B.plot(f, model.model_spectrum(q, I['input_broadband'], f), color=BL, lw=1.6, label='jointly fitted HRF, broadband input')
try:
    P = load_result('spectrum_prioritized_fit.json'); vp = np.array(P['params'])
    B.plot(f, model.model_spectrum(vp[:9], vp[9:], f), color=PU, lw=1.8, label='underdamped HRF, joint fit')
except FileNotFoundError:
    P = None
zA = np.array(I['input_with_line'])
B.plot(f, model.model_spectrum(q, zA[:4], f, line=zA[4:]), color=OR, lw=1.6, ls=':', label='jointly fitted HRF + rhythmic input')
B.set_xlabel('Frequency (Hz)'); B.set_ylabel('Amplitude (a.u.)'); B.set_title('B  Individual spectrum: three descriptions', loc='left', fontsize=11); B.legend(frameon=False, fontsize=8.5)
C.axvspan(20, 40, color='#bbbbbb', alpha=0.18, lw=0); C.text(20.6, 0.66, 'HRF not well\ndetermined here', fontsize=8.5, color=GR)
C.axhline(0, color='#bbbbbb', lw=0.8)
half = data.hrf_band_halfwidth(tg)
C.fill_between(tg, hs - half, hs + half, color=RD, alpha=0.15, lw=0)
C.plot(tg, hs, 'o', color=RD, ms=4, mec='white', mew=0.6, label='measured WM HRF ± 1 SD (to 25 s)')
if P is not None:
    t9, h9 = model.impulse_response(vp[:9], 0.05, 200.0); m9 = t9 < 40
    C.plot(t9[m9], h9[m9], color=PU, lw=1.8, label='underdamped HRF')
m = t < 40; C.plot(t[m], hq[m], color=BL, lw=1.6, label='jointly fitted HRF')
C.set_xlabel('Time (s)'); C.set_ylabel('BOLD response (a.u.)'); C.set_title('C  HRFs: agreement up to 20 s', loc='left', fontsize=11); C.legend(frameon=False, fontsize=8.5, loc='lower right')
try:
    TF = load_result('tradeoff_joint_fits.json')['front']
    D.plot([r['r2_spectrum'] for r in TF], [r['r2_hrf_0_25'] for r in TF], '-o', color=BL, ms=4, label='joint fits, HRF weight 0.9 to 0.1')
    D.set_xlim(0.9, 1.003); D.set_ylim(0.9, 1.0); D.legend(frameon=False, fontsize=8.5, loc='lower left')
except FileNotFoundError:
    D.text(0.5, 0.5, 'run 09_tradeoff_joint_fits.py --tradeoff', ha='center', transform=D.transAxes, color=GR)
D.set_xlabel('R² of individual spectrum'); D.set_ylabel('R² of HRF (0–25 s)'); D.set_title('D  Trade-off between HRF and spectral fits', loc='left', fontsize=11)
save(fig, 4)

# Figure 5: metabolic weight sweep
ratios, cols = [1.25, 1.75, round(q[4] / q[5], 2), 2.75, 3.0], ['#b9d3ea', '#86b4dc', BL, '#155289', '#0b2f52']
fh = np.linspace(0, 0.2, 2001)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.9)); ax[0].axhline(0, color='#bbbbbb', lw=0.8)
for k, (r, cl) in enumerate(zip(ratios, cols)):
    p = q.copy(); p[4] = r * p[5]; t5, h5 = model.impulse_response(p); mk = t5 < 25.5
    ax[0].plot(t5[mk], h5[mk] / h5[mk].max(), color=cl, lw=2.4 if k == 2 else 1.6, label=f'G_m/G_f = {r}' + (' (fitted)' if k == 2 else ''))
    Hh = np.abs(model.transfer(p, fh)); ax[1].plot(fh, Hh / Hh[0], color=cl, lw=2.4 if k == 2 else 1.6)
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('HRF (normalized to peak)'); ax[0].set_title('A  HRF', loc='left', fontsize=11); ax[0].legend(frameon=False, fontsize=8.5)
ax[1].axhline(1, color='#bbbbbb', lw=0.8); ax[1].set_ylim(0, 4); ax[1].set_xlabel('Frequency (Hz)'); ax[1].set_ylabel('|H(f)| / |H(0)|'); ax[1].set_title('B  Filter gain', loc='left', fontsize=11)
R = np.array(load_result('dip_link_sweeps.json')['ratio_sweep'])
ax[2].plot(R[:, 1], R[:, 3], '-o', color=BL, ms=6, mec='white', mew=1)
ax[2].set_xlabel('Dip depth (dip / peak)'); ax[2].set_ylabel('Frequency of maximum gain (Hz)'); ax[2].set_title('C  Dip depth sets the band-pass frequency', loc='left', fontsize=11)
save(fig, 5)

# Figure 6: sensitivity
lab = {'kappa': 'κ  vasoactive decay rate', 'tau0': 'τ₀  resting transit time', 'gamma_a': 'γa  autoregulatory feedback',
       'ratio': 'Gm/Gf  metabolic weight', 'Delta': 'Δ  neurovascular delay', 'tau_m': 'τm  metabolic time constant'}
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a_, group, title in ((ax[0], ['kappa', 'tau0', 'gamma_a'], 'A  Vascular factors'), (ax[1], ['ratio', 'Delta', 'tau_m'], 'B  Metabolic and timing factors')):
    a_.axhline(1, color='#bbbbbb', lw=0.8); a_.axvline(1, color='#bbbbbb', lw=0.8)
    for nm, cl in zip(group, [BL, OR, PU]):
        S = ld(f'sensitivity_curve_{nm}.csv'); mm = (S[:, 0] >= 0.5) & (S[:, 0] <= 2.0)
        a_.plot(S[mm, 0], S[mm, 1] / S[np.argmin(abs(S[:, 0] - 1)), 1], color=cl, label=lab[nm])
    a_.set_xscale('log'); a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a_.set_xticks([0.5, 0.7, 1, 1.4, 2]); a_.set_xticklabels(['0.5×', '0.7×', '1×', '1.4×', '2×'])
    a_.set_xlabel('Parameter relative to fitted value'); a_.set_title(title, loc='left', fontsize=11); a_.legend(frameon=False, fontsize=8.5, loc='upper center')
ax[0].set_ylabel('0.06 Hz peak prominence (relative)'); ax[0].set_ylim(0, 3.6)
save(fig, 6)
print('figures written to', data.FIGURES)
