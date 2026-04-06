#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for env_file in /etc/poetry-lm/web-app.env /etc/poetry-lm/inference-tunnel.env; do
  if [[ ! -f "$env_file" ]]; then
    echo "Missing $env_file. Create it from deploy/env/*.example first." >&2
    exit 1
  fi
done

sudo cp "$ROOT_DIR/deploy/systemd/poetry-inference-tunnel.service" /etc/systemd/system/
sudo cp "$ROOT_DIR/deploy/systemd/poetry-web-app.service" /etc/systemd/system/
sudo cp "$ROOT_DIR/deploy/nginx/ebekkuev.runningdog.org.conf" /etc/nginx/sites-available/ebekkuev.runningdog.org.conf
sudo ln -sf /etc/nginx/sites-available/ebekkuev.runningdog.org.conf /etc/nginx/sites-enabled/ebekkuev.runningdog.org.conf
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now poetry-inference-tunnel.service
sudo systemctl enable --now poetry-web-app.service
sudo systemctl reload nginx
sudo systemctl status --no-pager poetry-inference-tunnel.service
sudo systemctl status --no-pager poetry-web-app.service
