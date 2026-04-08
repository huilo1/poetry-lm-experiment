#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Poetry

CONFIG_PATH="${1:?usage: run_gigachat_vast_pipeline.sh <config-path> [test-file] [base-model]}"
TEST_FILE="${2:-data/gigachat_aabb_qf2_sft/test.jsonl.gz}"
BASE_MODEL="${3:-ai-sage/GigaChat3-10B-A1.8B-base}"

OUT_DIR="$(python - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(cfg['out_dir'])
PY
)"

mkdir -p "${OUT_DIR}"

python -m pip install --upgrade pip
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[qwen]'

python scripts/train_gigachat_sft.py --config "$CONFIG_PATH" | tee "${OUT_DIR}/train.log"
python scripts/evaluate_gigachat_8line.py \
  --base-model "$BASE_MODEL" \
  --adapter-dir "${OUT_DIR}/best_adapter" \
  --test-file "$TEST_FILE" \
  --device cuda \
  --limit 300 \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --top-k 40 \
  --bf16 | tee "${OUT_DIR}/eval.log"
