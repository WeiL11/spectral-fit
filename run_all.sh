#!/usr/bin/env bash
# Run all analyses after the source curves have been digitized (scripts 01-02; see README.md).
set -euo pipefail
cd "$(dirname "$0")"
for s in 03_spectra_eq1 04_hrf_bandpass 05_fit_hrf_alone 06_joint_fit_groupmean 07_individual_spectrum 08_spectrum_only_fit; do
  echo "== $s"; python scripts/$s.py
done
echo "== 09_tradeoff_joint_fits"; python scripts/09_tradeoff_joint_fits.py --prioritized --tradeoff
for s in 10_dip_link_sweeps 11_sensitivity 12_figures; do
  echo "== $s"; python scripts/$s.py
done
