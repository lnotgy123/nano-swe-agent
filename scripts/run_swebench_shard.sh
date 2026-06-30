#!/usr/bin/env bash
set -euo pipefail

START_INDEX="${1:?usage: scripts/run_swebench_shard.sh START_INDEX END_INDEX}"
END_INDEX="${2:?usage: scripts/run_swebench_shard.sh START_INDEX END_INDEX}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/envs/swe-agent-lf/bin/python}"
ENV_PYTHON="${ENV_PYTHON:-/environment/miniconda3/bin/python}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY_FILE="${PROJECT_ROOT}/data/runs/swebench_lite_batch_${START_INDEX}_${END_INDEX}_${STAMP}.jsonl"
LOCK_FILE="${PROJECT_ROOT}/data/runs/swebench_shard.lock"
RECREATE_ENV_ARGS=()
CONTINUE_ON_ENV_FAILED_ARGS=()
SKIP_EXISTING_ARGS=()

mkdir -p "${PROJECT_ROOT}/data/runs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec 9>"${LOCK_FILE}"
if [ "${WAIT_FOR_LOCK:-0}" = "1" ]; then
  flock 9
elif ! flock -n 9; then
  echo "Another SWE-bench shard is already running. Lock: ${LOCK_FILE}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
echo "summary_file: ${SUMMARY_FILE}"

if [ "${RECREATE_ENV:-0}" = "1" ]; then
  RECREATE_ENV_ARGS+=(--recreate-env)
fi
if [ "${CONTINUE_ON_ENV_FAILED:-0}" = "1" ]; then
  CONTINUE_ON_ENV_FAILED_ARGS+=(--continue-on-env-failed)
fi
if [ "${SKIP_EXISTING:-0}" = "1" ]; then
  SKIP_EXISTING_ARGS+=(--skip-existing)
fi

"${PYTHON_BIN}" scripts/run_swebench_batch.py \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --max-steps "${MAX_STEPS:-150}" \
  --agent-mode sweagent_xml \
  --dataset-offline \
  --repo-cache-root data/repo_cache \
  --install-env \
  "${RECREATE_ENV_ARGS[@]}" \
  "${CONTINUE_ON_ENV_FAILED_ARGS[@]}" \
  --reuse-existing-env \
  "${SKIP_EXISTING_ARGS[@]}" \
  --reuse-existing-workspace \
  --reset-existing-workspace \
  --env-python "${ENV_PYTHON}" \
  --install-timeout 1800 \
  --eval-timeout 60 \
  --command-timeout 300 \
  --summary-file "${SUMMARY_FILE}"
