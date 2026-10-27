# White-matter HRF and the resting-state BOLD spectrum

Code for the analyses and fits in:

> Gore, J. C., Li, T.-W., Li, M., Boschen, W. W., Schilling, K., Anderson, A. W., Ding, Z. & Gao, Y.
> *The white-matter hemodynamic response acts as a band-pass filter that shapes the resting-state BOLD spectrum.* (submitted)

The resting-state signal is modeled as shot noise: randomly timed neural events filtered by the hemodynamic
response function (HRF), plus instrumental noise. The HRF is the closed-form, linearized balloon model of
Gore et al. (*The balloon model revisited*), equation (13) of the paper. The code fits this model jointly to a
measured white-matter HRF (Schilling et al., 2022) and to white-matter resting-state spectra (Wu et al., 2026).

## Layout

```
wmhrf/            model (HRF transfer function, spectra), multi-start fitting, data loading
scripts/          numbered analysis scripts, run in order
data/             digitized source curves (created by scripts 01-02; see data/README.md)
results/          fitted parameters and derived curves (JSON, CSV)
figures/          manuscript figures 1-6 (PDF, PNG)
```

## Installation

Python 3.9 or later.

```
pip install -r requirements.txt
```

Digitizing the source figures (scripts 01-02) also needs `pdftoppm` from Poppler
(`brew install poppler`, `apt install poppler-utils` or `conda install -c conda-forge poppler`).

## Data

No new data were acquired. Two published curves are digitized from the source PDFs (both CC BY 4.0):

```
python scripts/01_digitize_schilling2022.py path/to/tgac035.pdf --check
python scripts/02_digitize_wu2026.py path/to/nihpp-rs10770821v1.pdf --check
```

`--check` prints the detected plot frame, ticks and grid lines used for the axis calibration.
Details and sources are in [data/README.md](data/README.md). The per-subject spectrum uses the
equation (1) parameters of Wu et al. (2026, Table 1), which are coded in `wmhrf/data.py`.

## Running the analyses

```
bash run_all.sh                 # scripts 03-12; about 1 h, most of it in the trade-off fits
WMHRF_QUICK=1 bash run_all.sh   # fast smoke test with ~1/20 of the random starts
```

Each script prints its results beside the values reported in the manuscript.

| Script | What it computes | Manuscript |
| --- | --- | --- |
| `03_spectra_eq1.py` | Equation (1) fit to the group-mean spectrum; averaging of simulated subject spectra | Introduction; Fig. 4A |
| `04_hrf_bandpass.py` | Amplitude spectrum of the measured WM HRF, with the dip or undershoot removed; simulated superposition of HRFs | Fig. 2 |
| `05_fit_hrf_alone.py` | Equation (13) fitted to the measured WM HRF alone | Starting point for 06 |
| `06_joint_fit_groupmean.py` | Joint fit of the HRF and the group-mean spectrum, all input filtered (equation (5)) | Equation (14); Fig. 3 |
| `07_individual_spectrum.py` | Individual spectrum with the jointly fitted HRF; rhythmic input as a filtered Lorentzian line | Fig. 4B |
| `08_spectrum_only_fit.py` | Underdamped filter fitted to the individual spectrum alone | Seed for 09 |
| `09_tradeoff_joint_fits.py` | Spectrum-prioritized joint fit (weight 0.1, HRF 0-20 s) and the HRF-versus-spectrum trade-off (weights 0.1-0.9) | Fig. 4B-D |
| `10_dip_link_sweeps.py` | Metabolic-weight and delay sweeps: dip depth, filter gain and its frequency | Fig. 5 |
| `11_sensitivity.py` | Elasticity of the 0.06 Hz peak to each HRF parameter; changes that halve the peak | Fig. 6 |
| `12_figures.py` | Figures 1-6 | Figs. 1-6 |

## Notes on reproducibility

- Fits use trust-region reflective least squares (`scipy.optimize.least_squares`) from many random starts;
  the best solution is kept. Random seeds are fixed, but results can differ slightly between NumPy and SciPy
  versions and with small differences in the digitized curves.
- Several parameters are weakly determined (notably Delta and the gain ratio G_m/G_f in some fits), and the
  spectrum-prioritized fit places Delta and alpha at the limits of their allowed ranges; see the paper.
- The HRF beyond 12.7 s, where the white- and gray-matter curves overlap in the source figure, was read by
  eye; those values are listed in `wmhrf/data.py` (`HRF_TAIL`).

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the sources of the data:

- Schilling, K. G. et al. Anomalous and heterogeneous characteristics of the BOLD hemodynamic response function
  in white matter. *Cereb. Cortex Commun.* **3**, tgac035 (2022).
- Wu, S. et al. A distinct second spectral peak in white matter BOLD signals: mathematical modeling, spatial
  distribution and age-related evolution. Preprint at https://doi.org/10.21203/rs.3.rs-10770821/v1 (2026).

## License

MIT (see [LICENSE](LICENSE)) for the code. Digitized data are derived from CC BY 4.0 figures; see data/README.md.
