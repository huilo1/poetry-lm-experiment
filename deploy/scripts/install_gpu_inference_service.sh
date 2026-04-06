#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

sudo mkdir -p /etc/poetry-lm
if [[ ! -f /etc/poetry-lm/gpu-inference-api.env ]]; then
  echo "Copy deploy/env/gpu-inference-api.env.example to /etc/poetry-lm/gpu-inference-api.env and edit it first." >&2
  exit 1
fi

sudo cp "$ROOT_DIR/deploy/systemd/poetry-inference-api.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now poetry-inference-api.service
sudo systemctl status --no-pager poetry-inference-api.service
