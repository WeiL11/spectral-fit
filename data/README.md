# Data

All target data are digitized from published figures. No new data were acquired.

| File | Source | Created by |
| --- | --- | --- |
| `schilling2022_fig3_wm_hrf_raw.csv` | Population-averaged white-matter HRF (mean and ±1 SD band), Fig. 3 (top), Schilling et al. (2022). Resting-state HRFs of 199 HCP subjects estimated by blind deconvolution (rsHRF toolbox) with a double-gamma basis. | `scripts/01_digitize_schilling2022.py` |
| `wu2026_fig1b_wm_spectrum_raw.csv` | Group-mean white-matter BOLD amplitude spectrum ("observed spectrum"), Fig. 1B, Wu et al. (2026). Welch amplitude spectra, 0.01-0.20 Hz, 120 HCP subjects, TR = 0.72 s. | `scripts/02_digitize_wu2026.py` |

Processing applied in `wmhrf/data.py`:

- HRF: digitized means up to 12.7 s, the manually read tail (`HRF_TAIL`) beyond it, and zero at t = 0, joined by
  a monotone cubic (PCHIP) interpolant and sampled at TR = 0.72 s from 0 to 25.2 s.
- Spectrum: 9-point moving average of the digitized curve, interpolated onto 400 frequencies over 0.01-0.20 Hz.
- Individual spectrum: equation (1) with the per-subject mean parameters of Wu et al. (2026, Table 1)
  (`TABLE1_MEAN`; medians and SDs are also listed).

Sources (both open access under CC BY 4.0; the digitized curves are adapted from these figures):

- Schilling, K. G. et al. Anomalous and heterogeneous characteristics of the BOLD hemodynamic response function in
  white matter. *Cereb. Cortex Commun.* **3**, tgac035 (2022). https://doi.org/10.1093/texcom/tgac035
- Wu, S. et al. A distinct second spectral peak in white matter BOLD signals: mathematical modeling, spatial
  distribution and age-related evolution. Research Square preprint (2026). https://doi.org/10.21203/rs.3.rs-10770821/v1
