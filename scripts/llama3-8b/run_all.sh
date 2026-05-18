#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
export MODEL_DIR

echo "========================================"
echo "Llama3-8B 全实验流水线"
echo "模型路径: $MODEL_DIR"
echo "开始时间: $(date)"
echo "========================================"

# ── Stage A ──────────────────────────────
echo ""
echo ">>> Stage A: 权重量化"
bash scripts/llama3-8b/run_stage_a_tensor_sweep.sh
bash scripts/llama3-8b/run_stage_a_ppl.sh

# ── Stage B ──────────────────────────────
echo ""
echo ">>> Stage B: 权重 + 激活伪量化"
bash scripts/llama3-8b/run_stage_b_activation.sh
bash scripts/llama3-8b/run_stage_b_local.sh
bash scripts/llama3-8b/run_stage_b_ppl.sh

# ── Stage C ──────────────────────────────
echo ""
echo ">>> Stage C: Attention / KV Cache"
bash scripts/llama3-8b/run_stage_c_invariance.sh
bash scripts/llama3-8b/run_stage_c_kv_local.sh
bash scripts/llama3-8b/run_stage_c_qjl.sh
bash scripts/llama3-8b/run_stage_c_attention_layer.sh
bash scripts/llama3-8b/run_stage_c_ppl.sh
bash scripts/llama3-8b/run_stage_c_accuracy.sh

echo ""
echo "========================================"
echo "全部实验完成: $(date)"
echo "========================================"
