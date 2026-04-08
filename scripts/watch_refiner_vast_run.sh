#!/usr/bin/env bash
set -euo pipefail

cd /home/angel/projects/Poetry || exit 1

INSTANCE_ID="${INSTANCE_ID:?INSTANCE_ID is required}"
SSH_HOST="${SSH_HOST:?SSH_HOST is required}"
SSH_PORT="${SSH_PORT:?SSH_PORT is required}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/Poetry/artifacts/checkpoints/vast_refiner_aabb_20m}"
LOCAL_DIR="${LOCAL_DIR:-artifacts/downloaded/vast_refiner_aabb_20m}"
LOG_PATH="${LOG_PATH:-artifacts/logs/watch_refiner_vast_run.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAIN_PATTERN="${TRAIN_PATTERN:-train_refiner.py --config configs/vast_refiner_aabb_20m.json}"
EVAL_PATTERN="${EVAL_PATTERN:-evaluate_refiner_8line.py --baseline-checkpoint artifacts/checkpoints/host_5060_8line_20m/best.pt}"

mkdir -p "$(dirname "$LOG_PATH")" "$LOCAL_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_PATH"
}

remote_status() {
  ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "root@${SSH_HOST}" "
    if test -f '${REMOTE_DIR}/best.refined.eval8.json'; then
      echo done
    elif pgrep -af '${EVAL_PATTERN}|${TRAIN_PATTERN}' >/dev/null; then
      echo running
    else
      echo idle
    fi
  " 2>>"$LOG_PATH"
}

sync_artifacts() {
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p ${SSH_PORT}" "root@${SSH_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/" >>"$LOG_PATH" 2>&1
}

validate_sync() {
  test -f "${LOCAL_DIR}/best.pt" &&
  test -f "${LOCAL_DIR}/final.pt" &&
  test -f "${LOCAL_DIR}/summary.json" &&
  test -f "${LOCAL_DIR}/log.jsonl" &&
  test -f "${LOCAL_DIR}/best.refined.eval8.json"
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
      log "Remote eval finished"
      break
      ;;
    running)
      log "Remote run still active"
      sleep "$POLL_SECONDS"
      ;;
    idle)
      log "Remote run is idle"
      break
      ;;
    ssh_error)
      log "SSH status check failed, retrying"
      sleep "$POLL_SECONDS"
      ;;
    *)
      log "Unexpected state: $state"
      sleep "$POLL_SECONDS"
      ;;
  esac
done

attempt=1
while true; do
  log "Artifact sync attempt ${attempt}"
  sync_artifacts
  if validate_sync; then
    log "All required artifacts are present locally"
    destroy_instance
    log "Watcher finished successfully"
    exit 0
  fi
  if (( attempt >= 10 )); then
    log "Required artifacts are still missing after ${attempt} attempts; instance left running for manual inspection"
    exit 1
  fi
  log "Required artifacts missing after sync; sleeping before retry"
  attempt=$((attempt + 1))
  sleep "$POLL_SECONDS"
done
