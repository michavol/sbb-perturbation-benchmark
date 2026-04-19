#!/usr/bin/env bash
# Delete model checkpoints and related files to free up space, only if cache exists.
set -euo pipefail

# Set to 1 for dry-run (only print), 0 to actually delete
DRY_RUN=1

# Root of the models directory (relative to repo root)
MODELS_ROOT="models"

# Model families (prefixes) to clean.
# Each maps to directories like <prefix>_<dataset>, e.g. pdae_wessels23, gears_replogle20, ...
ALLOWED_MODEL_PREFIXES=(
  "pdae"
  "presage"
  "gears"
  "scgpt"
  "sclambda"
  "fmlp_esm2"
  "fmlp_geneformer"
  "fmlp_genept"
  "fmlp_scgpt"
)

# Large training artifacts we want to delete when a cache exists
DELETE_DIR_NAMES=("checkpoints" "processed_data" "presage_data")
DELETE_FILE_SUFFIXES=(".pth" ".pt" ".ckpt" ".npz")

# Files we always keep
SAFE_KEEP_FILES=(
  "training_checkpoint.json"
  "inference_metadata.json"
  "metadata.json"
  "model_config.json"
  "presage_training_hparams.json"
  "args.json"
  "vocab.json"
  "control_mean.csv"
)

echo "Scanning ${MODELS_ROOT} for cache-backed runs to clean..."
echo "DRY_RUN=${DRY_RUN}"
echo "Model families included:"
for p in "${ALLOWED_MODEL_PREFIXES[@]}"; do echo "  - $p"; done
echo

in_allowed_family() {
  local bname="$1"
  for p in "${ALLOWED_MODEL_PREFIXES[@]}"; do
    case "$bname" in
      "${p}_"*) return 0 ;;
    esac
  done
  return 1
}

in_array() {
  local needle="$1"; shift
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

# Accumulate total bytes that would be (or were) deleted
TOTAL_BYTES=0

for model_dir in $(find "${MODELS_ROOT}" -maxdepth 1 -mindepth 1 -type d | sort); do
  model_base="$(basename "$model_dir")"
  if ! in_allowed_family "$model_base"; then
    continue
  fi

  echo "[$model_base]"
  # Iterate over run dirs (hashed IDs)
  while IFS= read -r run_dir; do
    run_id="$(basename "$run_dir")"
    ck="${run_dir}/training_checkpoint.json"
    cache_root="${run_dir}/predictions_cache"

    if [[ ! -f "$ck" || ! -d "$cache_root" ]]; then
      echo "  ${run_id}: SKIP (no cache or no training_checkpoint.json)"
      continue
    fi

    # Require at least one predictions.h5ad in cache
    if ! find "$cache_root" -type f -name 'predictions.h5ad' -print -quit | grep -q .; then
      echo "  ${run_id}: SKIP (predictions_cache has no predictions.h5ad)"
      continue
    fi

    echo "  ${run_id}: HAS CACHE -> deleting training artifacts:"

    # Delete dirs
    for dname in "${DELETE_DIR_NAMES[@]}"; do
      dpath="${run_dir}/${dname}"
      if [[ -d "$dpath" ]]; then
        # Size in bytes of this directory
        size=$(du -sb "$dpath" 2>/dev/null | cut -f1 || echo 0)
        TOTAL_BYTES=$((TOTAL_BYTES + size))
        echo "    DIR  $dpath  (~$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "${size}B"))"
        if [[ "$DRY_RUN" -eq 0 ]]; then
          rm -rf "$dpath"
        fi
      fi
    done

    # Delete files with heavy suffixes
    while IFS= read -r f; do
      base="$(basename "$f")"
      [[ -d "$f" ]] && continue
      if in_array "$base" "${SAFE_KEEP_FILES[@]}"; then
        continue
      fi
      suffix=".${base##*.}"
      delete_this=0
      for s in "${DELETE_FILE_SUFFIXES[@]}"; do
        if [[ "$suffix" == "$s" ]]; then
          delete_this=1
          break
        fi
      done
      if [[ "$delete_this" -eq 1 ]]; then
        size=$(stat -c%s "$f" 2>/dev/null || echo 0)
        TOTAL_BYTES=$((TOTAL_BYTES + size))
        echo "    FILE $f  (~$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "${size}B"))"
        if [[ "$DRY_RUN" -eq 0 ]]; then
          rm -f "$f"
        fi
      fi
    done < <(find "$run_dir" -maxdepth 1 -type f | sort)

  done < <(find "$model_dir" -maxdepth 1 -mindepth 1 -type d | sort)
  echo
done

# Summary
echo "Done."
human_total=$(numfmt --to=iec --suffix=B "$TOTAL_BYTES" 2>/dev/null || echo "${TOTAL_BYTES}B")
if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Total space freed: ${human_total}"
else
  echo "Total space that would be freed: ${human_total}"
  echo "This was a dry run. To actually delete files, set DRY_RUN=0 in the script and run again."
fi
