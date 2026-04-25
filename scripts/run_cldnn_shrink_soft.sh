#!/usr/bin/env bash
set -euo pipefail

# Keep CPU thread pools predictable and use the first visible GPU by default.
unset OMP_NUM_THREADS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/amc-pytorch-baseline}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/processed_v2_stratified_64_16_20}"
CONFIG_PATH="${CONFIG_PATH:-configs/cldnn_shrink_soft.yaml}"
MODE="${MODE:-mmap}"
RUN_NAME="${RUN_NAME:-cldnn_shrink_soft_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-runs/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/train.log}"

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}"

nohup python -u scripts/train_fast.py \
  --config "${CONFIG_PATH}" \
  --data-root "${DATA_ROOT}" \
  --mode "${MODE}" \
  --run-name "${RUN_NAME}" \
  > "${LOG_PATH}" 2>&1 &

echo "RUN_NAME=${RUN_NAME}"
echo "RUN_DIR=${PROJECT_ROOT}/${RUN_DIR}"
echo "LOG_PATH=${PROJECT_ROOT}/${LOG_PATH}"
echo "PID=$!"
