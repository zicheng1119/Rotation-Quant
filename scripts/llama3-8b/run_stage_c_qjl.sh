#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_qjl.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --qjl-specs hadamard_lm_k3 hadamard_lm_k2 hadamard_lm_k2_qjl hadamard_lm_k3_qjl \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 64 \
  --layer-limit 4
