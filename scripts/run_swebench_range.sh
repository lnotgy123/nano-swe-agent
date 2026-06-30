#!/usr/bin/env bash
set -euo pipefail

START_INDEX="${1:?usage: scripts/run_swebench_range.sh START_INDEX END_INDEX [SHARD_SIZE]}"
END_INDEX="${2:?usage: scripts/run_swebench_range.sh START_INDEX END_INDEX [SHARD_SIZE]}"
SHARD_SIZE="${3:-30}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/envs/swe-agent-lf/bin/python}"
LOG_DIR="${PROJECT_ROOT}/data/runs/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RANGE_LOG="${LOG_DIR}/swebench_range_${START_INDEX}_${END_INDEX}_${STAMP}.log"

mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
echo "range_log: ${RANGE_LOG}"

current="${START_INDEX}"
while [ "${current}" -lt "${END_INDEX}" ]; do
  next=$((current + SHARD_SIZE))
  if [ "${next}" -gt "${END_INDEX}" ]; then
    next="${END_INDEX}"
  fi

  echo "== shard ${current}-${next} started at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==" | tee -a "${RANGE_LOG}"
  if WAIT_FOR_LOCK=1 ./scripts/run_swebench_shard.sh "${current}" "${next}" 2>&1 | tee -a "${RANGE_LOG}"; then
    latest_summary="$(ls -t "data/runs/swebench_lite_batch_${current}_${next}_"*.jsonl | head -n 1)"
    echo "== analyze ${latest_summary} ==" | tee -a "${RANGE_LOG}"
    "${PYTHON_BIN}" scripts/analyze_swebench_runs.py "${latest_summary}" 2>&1 | tee -a "${RANGE_LOG}"
  else
    echo "== shard ${current}-${next} failed; continuing to next shard ==" | tee -a "${RANGE_LOG}"
  fi

  current="${next}"
done

echo "== range ${START_INDEX}-${END_INDEX} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==" | tee -a "${RANGE_LOG}"
