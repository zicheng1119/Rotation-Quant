#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_a_weight_only.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_a \
  --bits 4 3 2 \
  --methods direct_absmax hadamard_absmax hadamard_lm \
  --block-size 128
