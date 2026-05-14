#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_b_ppl.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_b \
  --methods fp16 ffn_direct_absmax_w4a4 ffn_rot_absmax_w4a4 ffn_rot_lm_w4a4 ffn_rot_lm_w3a4 ffn_rot_lm_w4a3 \
  --block-size 128 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 512 \
  --sequence-length 2048 \
  --stride 2048
