#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_a_ppl.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_a \
  --methods fp16 direct_absmax hadamard_absmax hadamard_lm \
  --bits 4 3 2 \
  --block-size 128 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 512
