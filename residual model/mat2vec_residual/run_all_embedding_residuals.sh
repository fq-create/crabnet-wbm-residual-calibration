#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate matbench-discovery

OUTPUT_ROOT="mat2vec_residual/outputs_all_embeddings_residual"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

run_model() {
  local model="$1"
  local predictions="$2"
  local output_dir="${OUTPUT_ROOT}/${model}/crabnet_frontend"
  local log_path="${LOG_DIR}/${model}_all_embedding_residual.log"

  if [[ ! -s "${predictions}" ]]; then
    echo "[$(date --iso-8601=seconds)] MISSING ${model}: ${predictions}" >&2
    return 1
  fi

  echo "[$(date --iso-8601=seconds)] START ${model}"
  python "${SCRIPT_DIR}/residual_calibration_full_all_embeddings.py" \
    --baseline-model "${model}" \
    --predictions "${predictions}" \
    --output-dir "${output_dir}" \
    > "${log_path}" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE  ${model}"
}

run_model "chgnet" "frozen baseline/outputs/chgnet/eform/chgnet_wbm_computed_full.csv"
run_model "m3gnet" "frozen baseline/outputs/m3gnet/eform/m3gnet_wbm_computed_full.csv"
run_model "mace-mpa-0" "frozen baseline/outputs/mace-mpa-0/eform/mace-mpa-0_wbm_computed_full.csv"
run_model "mattersim-v1-5m" "frozen baseline/outputs/mattersim-v1-5m/eform/mattersim-v1-5m_wbm_computed_full.csv"
run_model "orb-v3" "frozen baseline/outputs/orb-v3/eform/orb-v3_wbm_computed_full.csv"
run_model "sevennet-l3i5" "frozen baseline/outputs/sevennet-l3i5/eform/sevennet-l3i5_wbm_computed_full.csv"

python "${SCRIPT_DIR}/aggregate_all_embedding_residual_outputs.py" \
  --input-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/combined"

python "${SCRIPT_DIR}/compute_all_embedding_oof_metrics.py" \
  --input-roots "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/oof_metrics"

echo "[$(date --iso-8601=seconds)] ALL DONE"
