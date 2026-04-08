#!/usr/bin/env bash
set -euo pipefail

cd /home/angel/projects/Poetry || exit 1

INSTANCE_ID="${INSTANCE_ID:-34242647}"
SSH_HOST="${SSH_HOST:-ssh9.vast.ai}"
SSH_PORT="${SSH_PORT:-12646}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/Poetry/artifacts/checkpoints/vast_qwen3_8b_aabb_qf2_lora_bf16}"
LOCAL_DIR="${LOCAL_DIR:-artifacts/downloaded/vast_qwen3_8b_aabb_qf2_lora_bf16}"
LOG_PATH="${LOG_PATH:-artifacts/logs/watch_qwen_vast_run.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$(dirname "$LOG_PATH")" "$LOCAL_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_PATH"
}

remote_status() {
  ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@${SSH_HOST}" "
    if test -f '${REMOTE_DIR}/eval8.json'; then
      echo done
    elif pgrep -af 'evaluate_qwen_8line.py --base-model Qwen/Qwen3-8B-Base|train_qwen_sft.py --config configs/vast_qwen3_8b_aabb_qf2_lora_bf16.json' >/dev/null; then
      echo running
    else
      echo idle
    fi
  " 2>>"$LOG_PATH"
}

destroy_instance() {
  if vastai show instance "$INSTANCE_ID" --raw >/dev/null 2>&1; then
    log "Destroying Vast instance ${INSTANCE_ID}"
    vastai destroy instance "$INSTANCE_ID" >>"$LOG_PATH" 2>&1 || log "Destroy failed for ${INSTANCE_ID}"
  else
    log "Instance ${INSTANCE_ID} is already unavailable"
  fi
}

log "Watcher started for instance ${INSTANCE_ID} (${SSH_HOST}:${SSH_PORT})"

while true; do
  state="$(remote_status || echo ssh_error)"
  case "$state" in
    done)
      log "Remote eval finished, starting artifact sync"
      break
      ;;
    running)
      log "Remote run still active"
      sleep "$POLL_SECONDS"
      ;;
    idle)
      log "Remote run is no longer active; syncing partial/final artifacts"
      break
      ;;
    ssh_error)
      log "SSH status check failed, will retry"
      sleep "$POLL_SECONDS"
      ;;
    *)
      log "Unexpected state: $state"
      sleep "$POLL_SECONDS"
      ;;
  esac
done

log "Syncing ${REMOTE_DIR} -> ${LOCAL_DIR}"
rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" "root@${SSH_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/" >>"$LOG_PATH" 2>&1

destroy_instance

log "Watcher finished"
