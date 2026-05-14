#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_kv_local.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --kv-specs fp16 absmax_k4v4 absmax_k3v4 absmax_k4v3 hadamard_lm_k4v4 hadamard_lm_k3v4 hadamard_lm_k4v3 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 64 \
  --layer-limit 4
