#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $EUID -ne 0 ]]; then
  echo "deploy/deploy-prod.sh must run as root" >&2
  exit 1
fi

if [[ ! -f deploy/model-release.env ]]; then
  echo "deploy/model-release.env not found" >&2
  exit 1
fi

set -a
source deploy/model-release.env
set +a

: "${APP_DIR:?APP_DIR must be set}"
: "${APP_HOST:?APP_HOST must be set}"
: "${APP_PORT:?APP_PORT must be set}"
: "${DEPLOY_PUBLIC_HOST:?DEPLOY_PUBLIC_HOST must be set}"
: "${EDGE_PROXY:?EDGE_PROXY must be set}"
: "${HEALTHCHECK_URL:?HEALTHCHECK_URL must be set}"
: "${UV_PYTHON_VERSION:?UV_PYTHON_VERSION must be set}"
: "${REMOTE_SSH_HOST:?REMOTE_SSH_HOST must be set}"
: "${REMOTE_SSH_PORT:?REMOTE_SSH_PORT must be set}"
: "${REMOTE_SSH_USER:?REMOTE_SSH_USER must be set}"
: "${REMOTE_PROJECT_ROOT:?REMOTE_PROJECT_ROOT must be set}"
: "${REMOTE_INFERENCE_HOST:?REMOTE_INFERENCE_HOST must be set}"
: "${REMOTE_INFERENCE_PORT:?REMOTE_INFERENCE_PORT must be set}"
: "${REMOTE_INFERENCE_DEVICE:?REMOTE_INFERENCE_DEVICE must be set}"
: "${REMOTE_ENV_DIR:?REMOTE_ENV_DIR must be set}"
: "${REMOTE_SYSTEMD_DIR:?REMOTE_SYSTEMD_DIR must be set}"
: "${RUNTIME_SSH_KEY_PATH:?RUNTIME_SSH_KEY_PATH must be set}"
: "${TUNNEL_LOCAL_PORT:?TUNNEL_LOCAL_PORT must be set}"
: "${GPU_API_PORT:?GPU_API_PORT must be set}"

traefik_config_dir="${TRAEFIK_CONFIG_DIR:-}"
if [[ "$EDGE_PROXY" == "traefik" && -z "$traefik_config_dir" ]]; then
  echo "TRAEFIK_CONFIG_DIR must be set when EDGE_PROXY=traefik" >&2
  exit 1
fi

required_local_files=(
  "pyproject.toml"
  "scripts/web_app.py"
  "scripts/inference_api.py"
  "src/poetry_lm/model_registry.py"
  "deploy/systemd/poetry-web-app.service"
  "deploy/systemd/poetry-inference-tunnel.service"
)

if [[ "$EDGE_PROXY" == "nginx" ]]; then
  required_local_files+=("deploy/nginx/ebekkuev.runningdog.org.conf")
fi

for path in "${required_local_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Required file is missing: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$RUNTIME_SSH_KEY_PATH" ]]; then
  echo "SSH key is missing: $RUNTIME_SSH_KEY_PATH" >&2
  exit 1
fi

wait_for_url() {
  local url="$1"
  local label="$2"
  local timeout="${3:-60}"
  local deadline=$((SECONDS + timeout))

  while ! curl --fail --silent --show-error "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${label}: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x /root/.local/bin/uv ]]; then
  uv_bin="/root/.local/bin/uv"
else
  echo "uv is required on the web host" >&2
  exit 1
fi

remote_target="${REMOTE_SSH_USER}@${REMOTE_SSH_HOST}"
ssh_opts=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -i "$RUNTIME_SSH_KEY_PATH"
  -p "$REMOTE_SSH_PORT"
)
rsync_ssh_cmd="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${RUNTIME_SSH_KEY_PATH} -p ${REMOTE_SSH_PORT}"
sync_excludes=(
  --exclude ".git/"
  --exclude ".github/"
  --exclude ".venv/"
  --exclude ".ruff_cache/"
  --exclude ".pytest_cache/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "*.egg-info/"
  --exclude "artifacts/"
  --exclude "data/"
)

echo "Syncing current checkout to GPU host ${remote_target}:${REMOTE_PROJECT_ROOT}"
ssh "${ssh_opts[@]}" "$remote_target" "mkdir -p '$REMOTE_PROJECT_ROOT' '$REMOTE_ENV_DIR' '$REMOTE_SYSTEMD_DIR'"
rsync -az "${sync_excludes[@]}" -e "$rsync_ssh_cmd" "$repo_root/" "${remote_target}:${REMOTE_PROJECT_ROOT}/"

echo "Preparing GPU inference runtime"
ssh "${ssh_opts[@]}" "$remote_target" bash -s -- \
  "$REMOTE_PROJECT_ROOT" \
  "$REMOTE_ENV_DIR" \
  "$REMOTE_SYSTEMD_DIR" \
  "$REMOTE_INFERENCE_HOST" \
  "$REMOTE_INFERENCE_PORT" \
  "$REMOTE_INFERENCE_DEVICE" <<'EOF'
set -euo pipefail

project_root="$1"
env_dir="$2"
systemd_dir="$3"
inference_host="$4"
inference_port="$5"
inference_device="$6"

mkdir -p "$env_dir" "$systemd_dir"

cat > "${env_dir}/gpu-inference-api.env" <<ENVFILE
PROJECT_ROOT=${project_root}
INFERENCE_API_HOST=${inference_host}
INFERENCE_API_PORT=${inference_port}
INFERENCE_DEVICE=${inference_device}
ENVFILE

cat > "${systemd_dir}/poetry-inference-api.service" <<'UNIT'
[Unit]
Description=Poetry LM inference API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=%h/.config/poetry-lm/gpu-inference-api.env
WorkingDirectory=/
ExecStart=/usr/bin/bash -lc 'cd "$PROJECT_ROOT" && . .venv/bin/activate && export PYTHONPATH=src && exec python scripts/inference_api.py --host "$INFERENCE_API_HOST" --port "$INFERENCE_API_PORT" --device "$INFERENCE_DEVICE"'
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
UNIT

cd "$project_root"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

systemctl --user daemon-reload
systemctl --user enable --now poetry-inference-api.service
systemctl --user restart poetry-inference-api.service

python - "$inference_port" <<'PY'
import sys
import time
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/health"
deadline = time.time() + 30
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(1)

raise SystemExit(f"remote healthcheck failed: {last_error}")
PY
EOF

echo "Syncing current checkout to web host app dir ${APP_DIR}"
install -d "$APP_DIR"
rsync -az --delete "${sync_excludes[@]}" "$repo_root/" "$APP_DIR/"

echo "Preparing web runtime"
"$uv_bin" python install "$UV_PYTHON_VERSION"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  "$uv_bin" venv --python "$UV_PYTHON_VERSION" "$APP_DIR/.venv"
fi
"$uv_bin" pip install \
  --python "$APP_DIR/.venv/bin/python" \
  "fastapi>=0.115.2" \
  "httpx>=0.28.1" \
  "uvicorn>=0.30.0"

install -d /etc/poetry-lm

cat > /etc/poetry-lm/web-app.env <<EOF
PROJECT_ROOT=${APP_DIR}
WEBAPP_HOST=${APP_HOST}
WEBAPP_PORT=${APP_PORT}
INFERENCE_API_BASE_URL=http://127.0.0.1:${TUNNEL_LOCAL_PORT}
EOF

cat > /etc/poetry-lm/inference-tunnel.env <<EOF
SSH_USER=${REMOTE_SSH_USER}
SSH_HOST=${REMOTE_SSH_HOST}
SSH_PORT=${REMOTE_SSH_PORT}
SSH_KEY_PATH=${RUNTIME_SSH_KEY_PATH}
TUNNEL_LOCAL_PORT=${TUNNEL_LOCAL_PORT}
GPU_API_PORT=${GPU_API_PORT}
EOF

echo "Installing systemd units and public route"
install -m 0644 "$APP_DIR/deploy/systemd/poetry-inference-tunnel.service" /etc/systemd/system/poetry-inference-tunnel.service
install -m 0644 "$APP_DIR/deploy/systemd/poetry-web-app.service" /etc/systemd/system/poetry-web-app.service
systemctl daemon-reload
systemctl enable --now poetry-inference-tunnel.service
systemctl enable --now poetry-web-app.service
systemctl restart poetry-inference-tunnel.service
systemctl restart poetry-web-app.service

if [[ "$EDGE_PROXY" == "nginx" ]]; then
  install -m 0644 "$APP_DIR/deploy/nginx/ebekkuev.runningdog.org.conf" /etc/nginx/sites-available/ebekkuev.runningdog.org.conf
  ln -sf /etc/nginx/sites-available/ebekkuev.runningdog.org.conf /etc/nginx/sites-enabled/ebekkuev.runningdog.org.conf
  nginx -t
  systemctl reload nginx
elif [[ "$EDGE_PROXY" == "traefik" ]]; then
  install -d "$traefik_config_dir"
  cat > "${traefik_config_dir}/${DEPLOY_PUBLIC_HOST//./-}.yml" <<EOF
http:
  routers:
    poetry-web-http:
      rule: "Host(\`${DEPLOY_PUBLIC_HOST}\`)"
      entryPoints:
        - web
      service: poetry-web
    poetry-web-secure:
      rule: "Host(\`${DEPLOY_PUBLIC_HOST}\`)"
      entryPoints:
        - websecure
      tls:
        certResolver: letsencrypt
      service: poetry-web
  services:
    poetry-web:
      loadBalancer:
        servers:
          - url: "http://host.docker.internal:${APP_PORT}"
EOF
else
  echo "Unsupported EDGE_PROXY=${EDGE_PROXY}" >&2
  exit 1
fi

echo "Running healthchecks"
wait_for_url "http://127.0.0.1:${TUNNEL_LOCAL_PORT}/health" "local SSH tunnel"
wait_for_url "http://127.0.0.1:${APP_PORT}/api/health" "local web app"
wait_for_url "${HEALTHCHECK_URL}" "public endpoint"

echo "Deploy completed successfully."
