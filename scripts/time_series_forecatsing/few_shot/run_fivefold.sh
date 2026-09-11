#!/usr/bin/env bash
set -euo pipefail

mkdir -p results
cv_dir="$(mktemp -d "./results/cv_XXXXXX")"

for fold in 0 1 2 3 4; do
  bash scripts/time_series_forecatsing/few_shot/sempo_icecore_crossvar_chem_accum.sh \
    --fold_id "$fold" \
    --cv_dir "$cv_dir" \
    --checkpoints "$cv_dir/checkpoints" \
    --itr 1 \
    2>&1 | tee "$cv_dir/fold_${fold}.log"
done

python summarize_fivefold.py "$cv_dir"