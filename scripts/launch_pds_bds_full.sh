#!/bin/bash
# Full PDS + BDS analysis pipeline:
#   1) Run PDS test (all datasets, 14 default metrics, 1000 permutations)
#   2) Run BDS test (all datasets, 14 default metrics, 999 permutations, max 50 perts)
#
# Usage:
#   bash analyses/perturbation_discrimination/launch_pds_bds_full.sh 2>&1 | tee pds_bds_full.log
set -euo pipefail

DATASETS="adamson16 frangieh21 nadig25hepg2 norman19 replogle22k562 wessels23 replogle20"

echo "============================================"
echo " STEP 1: PDS test (all datasets)"
echo "============================================"
echo ""

uv run python analyses/perturbation_discrimination/run_pds_test.py \
    --datasets $DATASETS \
    --n-permutations 1000 \
    --n-trials 1 \
    --max-perturbations 500 \
    --workers 64 \
    --no-progress \
    --overwrite

echo ""
echo "============================================"
echo " STEP 2: BDS test (all datasets)"
echo "============================================"
echo ""

uv run python analyses/perturbation_discrimination/run_bds_test.py \
    --datasets $DATASETS \
    --n-permutations 999 \
    --n-trials 1 \
    --max-perturbations 500 \
    --max-control-cells 500 \
    --workers 64

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
