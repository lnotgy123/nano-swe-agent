#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UNIT="${UNIT:-swebench-prepare-0-100}"
PYTHON="${PYTHON:-python}"
HOME_DIR="${HOME_DIR:-${HOME}}"
HF_HOME="${HF_HOME:-${HOME_DIR}/.cache/huggingface}"
DOCKER_PROXY_URL="${DOCKER_PROXY_URL:-}"

if systemctl is-active --quiet "$UNIT.service"; then
  echo "$UNIT.service is already running."
  exit 0
fi

if [[ -n "$DOCKER_PROXY_URL" ]]; then
  if ! curl --proxy "$DOCKER_PROXY_URL" --head --location --silent --show-error \
    --connect-timeout 8 --max-time 20 https://registry-1.docker.io/v2/ >/dev/null; then
    echo "Proxy ${DOCKER_PROXY_URL} cannot reach Docker Hub." >&2
    exit 1
  fi
fi

sudo systemctl stop "$UNIT.service" 2>/dev/null || true
sudo systemctl reset-failed "$UNIT.service" 2>/dev/null || true
sudo systemd-run \
  --unit="$UNIT" \
  --working-directory="$ROOT" \
  --setenv=HOME="$HOME_DIR" \
  --setenv=HF_HOME="$HF_HOME" \
  --setenv=HF_HUB_OFFLINE=1 \
  --setenv=HF_DATASETS_OFFLINE=1 \
  /bin/bash -lc "until $PYTHON -u scripts/cache_swebench_images.py --start-index 0 --end-index 100 --pull-timeout 300 --retries 6; do echo 'Cache pass failed; retrying in 30 seconds.'; sleep 30; done"

sleep 2
systemctl --no-pager --full status "$UNIT.service" | sed -n '1,24p'
