#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate
export PYTHONPATH=src

python scripts/prepare_tokens.py \
  --input-dir data/processed \
  --output-dir data/processed_fullpoem_stage1tok \
  --tokenizer-model artifacts/tokenizer_aabb8/poetry.model

python scripts/prepare_tokens.py \
  --input-dir data/processed_aabb8_qf2 \
  --output-dir data/processed_aabb8_qf2_stage1tok \
  --tokenizer-model artifacts/tokenizer_aabb8/poetry.model

python scripts/train.py \
  --config configs/host_5060_fullpoem_20m_stage1.json

python scripts/train.py \
  --config configs/host_5060_aabb_qf2_stage2_from_fullpoem_20m.json \
  --init-from artifacts/checkpoints/host_5060_fullpoem_20m_stage1/best.pt

python scripts/evaluate_8line.py \
  --checkpoint artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m/best.pt \
  --tokenizer-model artifacts/tokenizer_aabb8/poetry.model \
  --test-file data/processed_aabb8_qf2/test.jsonl.gz \
  --device cuda \
  --limit 300 \
  > artifacts/checkpoints/host_5060_aabb_qf2_stage2_from_fullpoem_20m/best.eval8.json
