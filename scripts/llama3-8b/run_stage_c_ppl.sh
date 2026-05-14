#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_ppl.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --methods fp16 attn_identity_fp16 attn_kv_hlm_k4v4 attn_kv_hlm_k3v4 attn_rot_lm_w4a4_hlm_k4v4 attn_rot_lm_w3a4_hlm_k3v4 attn_rot_lm_w4a3_hlm_k4v3 \
  --block-size 128 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 512 \
  --sequence-length 2048 \
  --stride 2048
