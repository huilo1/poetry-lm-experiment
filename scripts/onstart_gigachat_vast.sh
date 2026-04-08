#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/workspace/Poetry"
LOG_DIR="${BASE_DIR}/artifacts"
mkdir -p "${BASE_DIR}" "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "onstart_gigachat_vast.sh booting"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv git rsync ca-certificates

log "waiting for code sync marker ${BASE_DIR}/.ready"
while [ ! -f "${BASE_DIR}/.ready" ]; do
  sleep 5
done

cd "${BASE_DIR}"

if [ ! -d .venv ]; then
  log "creating venv"
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[qwen]'

log "environment prepared"
nvidia-smi || true

log "starting benchmark run"
./scripts/run_gigachat_vast_pipeline.sh \
  configs/benchmark_gigachat3_10b_a1_8b_aabb_lora_bf16.json \
  data/gigachat_aabb_qf2_sft/test.jsonl.gz \
  ai-sage/GigaChat3-10B-A1.8B-base
touch "${LOG_DIR}/benchmark_gigachat.done"
log "benchmark run finished"

log "waiting for full-run marker ${BASE_DIR}/.run_full"
while [ ! -f "${BASE_DIR}/.run_full" ]; do
  sleep 5
done

log "starting full run"
./scripts/run_gigachat_vast_pipeline.sh \
  configs/vast_gigachat3_10b_a1_8b_aabb_qf2_lora_bf16.json \
  data/gigachat_aabb_qf2_sft/test.jsonl.gz \
  ai-sage/GigaChat3-10B-A1.8B-base
touch "${LOG_DIR}/gigachat_full.done"
log "full run finished"
