#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CONDA_SH="${CONDA_SH:-/home/ubuntu/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-matbench-discovery}"

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  if [[ -f "$CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
  else
    echo "[WARN] Conda setup file not found: $CONDA_SH"
    echo "[WARN] Continuing with the current Python environment."
  fi
fi

if [[ $# -gt 0 ]]; then
  models=("$@")
else
  models=(
    chgnet
    m3gnet
    mace-mpa-0
    mattersim-v1-5m
    orb-v3
    sevennet-l3i5
  )
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
failed=()
skipped=()
converted=()

echo "Working directory: $SCRIPT_DIR"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-current-shell}"
echo "Models: ${models[*]}"
echo

for model in "${models[@]}"; do
  input="outputs/${model}/raw_predictions/${model}_wbm_computed_full_raw.csv"
  output="outputs/${model}/eform/${model}_wbm_computed_full.csv"
  log_dir="outputs/${model}/logs"
  log="${log_dir}/convert_energy_to_eform_${timestamp}.log"

  mkdir -p "outputs/${model}/eform" "$log_dir"

  if [[ ! -s "$input" ]]; then
    echo "[SKIP] $model"
    echo "       Missing raw prediction file: $input"
    skipped+=("$model")
    continue
  fi

  echo "[RUN]  $model"
  echo "       Input : $input"
  echo "       Output: $output"
  echo "       Log   : $log"

  if python convert_energy_to_eform.py \
    --baseline-model "$model" \
    --input "$input" \
    --output "$output" \
    2>&1 | tee "$log"; then
    echo "[DONE] $model"
    converted+=("$model")
  else
    echo "[FAIL] $model"
    echo "       See log: $log"
    failed+=("$model")
  fi
  echo
done

echo "Summary"
echo "  Converted: ${converted[*]:-(none)}"
echo "  Skipped  : ${skipped[*]:-(none)}"
echo "  Failed   : ${failed[*]:-(none)}"

if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi
