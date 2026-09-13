#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the whole study on simulated data.
#
#   bash scripts/run_all.sh [source] [n_boot]
#
# The default source is the simulator, so this runs with no provider data and
# no network access. Nothing it produces is a claim about real football; it
# demonstrates that the pipeline is complete and that the calibration machinery
# behaves as specified.
#
# To run on real data: complete the relevant adapter in
# src/pcc/data/loaders.py, place the data under data/raw/<key>/, then pass the
# source key as the first argument.
# ---------------------------------------------------------------------------
set -euo pipefail

SOURCE="${1:-simulated}"
N_BOOT="${2:-500}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

step() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

step "0/8  environment"
python3 00_environment_report.py

step "1/8  data availability (presence and reachability only)"
python3 01_check_data_availability.py || true

step "2/8  instrument validation - must pass before anything else"
python3 09_validate_instrument.py

step "3/8  build dataset"
python3 02_build_dataset.py --source "$SOURCE"

step "4/8  main analysis (RQ1, RQ2, RQ5)"
python3 03_main_analysis.py --source "$SOURCE" --n-boot "$N_BOOT"

step "5/8  subgroup calibration (RQ3, RQ4)"
python3 04_subgroup_analysis.py --source "$SOURCE"

step "6/8  selection analysis"
python3 05_selection_analysis.py --source "$SOURCE"

step "7/8  downstream impact"
python3 06_downstream_analysis.py --source "$SOURCE"

step "8/8  ablations"
python3 07_ablations.py --source "$SOURCE"

printf '\n\033[1mDone.\033[0m Results under results/. Each directory carries a manifest.json\n'
printf 'recording the git revision, package versions and the config hash used.\n'
if [ "$SOURCE" = "simulated" ]; then
  printf '\n\033[1mREMINDER:\033[0m the source was the simulator. These outputs describe the\n'
  printf 'simulator, not football, and must not appear in a results section.\n'
fi
