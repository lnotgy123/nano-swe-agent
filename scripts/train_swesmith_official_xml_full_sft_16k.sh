#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/environment/miniconda3/envs/swe-agent-lf/bin/python}"
VENV_BIN="$(dirname "${PYTHON_BIN}")"
export CUDA_HOME="${CUDA_HOME:-${VENV_BIN}/../lib/python3.11/site-packages/nvidia/cu13}"
LLAMA_FACTORY_DIR="${PROJECT_ROOT}/third_party/LLaMA-Factory"
CONFIG="${PROJECT_ROOT}/configs/llamafactory_qwen25_swesmith_official_xml_full_sft_16k.yaml"

export PATH="${CUDA_HOME}/bin:${VENV_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-warning}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/data/cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/data/tmp}"

mkdir -p "${TMPDIR}" "${PROJECT_ROOT}/data/runs/logs"

cd "${LLAMA_FACTORY_DIR}"
"${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG}"
