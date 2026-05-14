#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_accuracy.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --benchmark piqa
