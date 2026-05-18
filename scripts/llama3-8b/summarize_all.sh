#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "汇总所有 Stage 结果"
echo "========================================"

# Stage A: tensor sweep (tensor_metrics.csv)
LATEST_A_TENSOR=$(ls -dt outputs/stage_a/*_stage_a_tensor_sweep 2>/dev/null | head -1)
if [ -n "$LATEST_A_TENSOR" ]; then
    echo ""
    echo ">>> Stage A tensor sweep: $(basename $LATEST_A_TENSOR)"
    PYTHONPATH=src python experiments/summarize_stage_a_tensor.py "$LATEST_A_TENSOR"
else
    echo "No stage_a tensor sweep output found."
fi

# Stage A: PPL (ppl.csv — read directly, no summarizer needed)
LATEST_A_PPL=$(ls -dt outputs/stage_a/*_stage_a_ppl 2>/dev/null | head -1)
if [ -n "$LATEST_A_PPL" ]; then
    echo ""
    echo ">>> Stage A PPL: $(basename $LATEST_A_PPL)"
    PYTHONPATH=src python experiments/summarize_stage_b.py "$LATEST_A_PPL"
fi

# Stage B: activation (activation_metrics.csv)
LATEST_B_ACT=$(ls -dt outputs/stage_b/*_stage_b_activation 2>/dev/null | head -1)
if [ -n "$LATEST_B_ACT" ]; then
    echo ""
    echo ">>> Stage B activation: $(basename $LATEST_B_ACT)"
    PYTHONPATH=src python experiments/summarize_stage_b.py "$LATEST_B_ACT"
fi

# Stage B: local (linear_metrics.csv + ffn_metrics.csv)
LATEST_B_LOCAL=$(ls -dt outputs/stage_b/*_stage_b_local 2>/dev/null | head -1)
if [ -n "$LATEST_B_LOCAL" ]; then
    echo ""
    echo ">>> Stage B local: $(basename $LATEST_B_LOCAL)"
    PYTHONPATH=src python experiments/summarize_stage_b.py "$LATEST_B_LOCAL"
fi

# Stage B: PPL (ppl.csv)
LATEST_B_PPL=$(ls -dt outputs/stage_b/*_stage_b_ppl 2>/dev/null | head -1)
if [ -n "$LATEST_B_PPL" ]; then
    echo ""
    echo ">>> Stage B PPL: $(basename $LATEST_B_PPL)"
    PYTHONPATH=src python experiments/summarize_stage_b.py "$LATEST_B_PPL"
fi

# Stage C: invariance (invariance_metrics.csv — self-summarized)
LATEST_C_INV=$(ls -dt outputs/stage_c/*_stage_c_invariance 2>/dev/null | head -1)
if [ -n "$LATEST_C_INV" ]; then
    echo ""
    echo ">>> Stage C invariance: $(basename $LATEST_C_INV)"
    # self-contained summary in its own directory
    [ -f "$LATEST_C_INV/summary.md" ] && cat "$LATEST_C_INV/summary.md"
fi

# Stage C: KV local (kv_metrics.csv — self-summarized)
LATEST_C_KV=$(ls -dt outputs/stage_c/*_stage_c_kv_local 2>/dev/null | head -1)
if [ -n "$LATEST_C_KV" ]; then
    echo ""
    echo ">>> Stage C KV local: $(basename $LATEST_C_KV)"
    [ -f "$LATEST_C_KV/summary.md" ] && cat "$LATEST_C_KV/summary.md"
fi

# Stage C: QJL (qjl_metrics.csv — self-summarized)
LATEST_C_QJL=$(ls -dt outputs/stage_c/*_stage_c_qjl 2>/dev/null | head -1)
if [ -n "$LATEST_C_QJL" ]; then
    echo ""
    echo ">>> Stage C QJL: $(basename $LATEST_C_QJL)"
    [ -f "$LATEST_C_QJL/summary.md" ] && cat "$LATEST_C_QJL/summary.md"
fi

# Stage C: attention layer
LATEST_C_ATTN=$(ls -dt outputs/stage_c/*_stage_c_attention_layer 2>/dev/null | head -1)
if [ -n "$LATEST_C_ATTN" ]; then
    echo ""
    echo ">>> Stage C attention layer: $(basename $LATEST_C_ATTN)"
    PYTHONPATH=src python experiments/summarize_stage_c.py "$LATEST_C_ATTN"
fi

# Stage C: PPL
LATEST_C_PPL=$(ls -dt outputs/stage_c/*_stage_c_ppl 2>/dev/null | head -1)
if [ -n "$LATEST_C_PPL" ]; then
    echo ""
    echo ">>> Stage C PPL: $(basename $LATEST_C_PPL)"
    PYTHONPATH=src python experiments/summarize_stage_c.py "$LATEST_C_PPL"
fi

# Stage C: accuracy
LATEST_C_ACC=$(ls -dt outputs/stage_c/*_stage_c_accuracy 2>/dev/null | head -1)
if [ -n "$LATEST_C_ACC" ]; then
    echo ""
    echo ">>> Stage C accuracy: $(basename $LATEST_C_ACC)"
    [ -f "$LATEST_C_ACC/summary.md" ] && cat "$LATEST_C_ACC/summary.md"
fi

echo ""
echo "========================================"
echo "汇总完成: $(date)"
echo "以上 summary.md 位于各自的 outputs/stage_X/<run_id>/ 目录"
echo "========================================"
