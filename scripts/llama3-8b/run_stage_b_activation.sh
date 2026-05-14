#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_b_activation.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_b \
  --methods direct_absmax rot_absmax rot_lm rot_mxfp4 randhadamard_lm randortho_lm \
  --w-bits 4 3 2 \
  --a-bits 4 3 2 \
  --block-size 128 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 256 \
  --layer-limit 4
