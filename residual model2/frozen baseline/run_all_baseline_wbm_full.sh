#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CHUNK_SIZE="${CHUNK_SIZE:-2000}"

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
completed=()
failed=()

echo "Working directory: $SCRIPT_DIR"
echo "Models: ${models[*]}"
echo "Chunk size: $CHUNK_SIZE"
echo

for model in "${models[@]}"; do
  log_dir="outputs/${model}/logs"
  log="${log_dir}/run_baseline_wbm_full_${timestamp}.log"
  mkdir -p "$log_dir"

  echo "[RUN]  $model"
  echo "       Log: $log"

  if python run_baseline_wbm_full.py \
    --baseline-model "$model" \
    --chunk-size "$CHUNK_SIZE" \
    2>&1 | tee "$log"; then
    echo "[DONE] $model"
    completed+=("$model")
  else
    echo "[FAIL] $model"
    echo "       See log: $log"
    failed+=("$model")
  fi
  echo
done

echo "Summary"
echo "  Completed: ${completed[*]:-(none)}"
echo "  Failed   : ${failed[*]:-(none)}"

if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi

