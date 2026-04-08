#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Poetry

python -m pip install --upgrade pip
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .

PYTHONPATH=src python scripts/train_refiner.py --config configs/vast_refiner_aabb_20m.json | tee artifacts/checkpoints/vast_refiner_aabb_20m/train.log

PYTHONPATH=src python scripts/evaluate_refiner_8line.py \
  --baseline-checkpoint artifacts/checkpoints/host_5060_8line_20m/best.pt \
  --baseline-tokenizer artifacts/tokenizer_aabb8/poetry.model \
  --refiner-checkpoint artifacts/checkpoints/vast_refiner_aabb_20m/best.pt \
  --refiner-tokenizer artifacts/tokenizer_refiner/poetry.model \
  --test-file data/processed_aabb8_qf2/test.jsonl.gz \
  --device cuda \
  --limit 300 \
  --max-new-tokens 160 \
  --temperature 0.9 \
  --top-k 50 \
  --refiner-steps 8 \
  --refiner-temperature 0.8 \
  --refiner-top-k 32 | tee artifacts/checkpoints/vast_refiner_aabb_20m/eval.log
