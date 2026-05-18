#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "汇总所有 Stage 结果"
echo "========================================"

# Stage A tensor sweep — find latest run
LATEST_A=$(ls -dt outputs/stage_a/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_A" ]; then
    RUN_ID_A=$(basename "$LATEST_A")
    echo ""
    echo ">>> Stage A tensor sweep: $RUN_ID_A"
    PYTHONPATH=src python experiments/summarize_stage_a_tensor.py "$LATEST_A"
else
    echo "No stage_a output found."
fi

# Stage B — find latest run
LATEST_B=$(ls -dt outputs/stage_b/b*_stage_b_* 2>/dev/null | head -1)
if [ -z "$LATEST_B" ]; then
    LATEST_B=$(ls -dt outputs/stage_b/*/ 2>/dev/null | head -1)
fi
if [ -n "$LATEST_B" ]; then
    RUN_ID_B=$(basename "$LATEST_B")
    echo ""
    echo ">>> Stage B: $RUN_ID_B"
    PYTHONPATH=src python experiments/summarize_stage_b.py "$LATEST_B"
else
    echo "No stage_b output found."
fi

# Stage C — find latest run
LATEST_C=$(ls -dt outputs/stage_c/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_C" ]; then
    RUN_ID_C=$(basename "$LATEST_C")
    echo ""
    echo ">>> Stage C: $RUN_ID_C"
    PYTHONPATH=src python experiments/summarize_stage_c.py "$LATEST_C"
else
    echo "No stage_c output found."
fi

echo ""
echo "========================================"
echo "汇总完成: $(date)"
echo "========================================"
