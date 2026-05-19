# Llama3-8B 量化实验迁移实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将 rotation-based quantization 实验代码从 TinyLlama-1.1B/MPS 迁移到 Llama3-8B/CUDA，改动最小化，保持向后兼容。

**Architecture:** 参数化模型路径和设备类型，泛型 capture 类改名去 TinyLlama 前缀但保留兼容别名，所有实验脚本已有 `--model-dir` 参数只需改传入值。新增 Llama3-8B 专用 shell 脚本与 TinyLlama 脚本并存。

**Tech Stack:** Python 3.11+, PyTorch 2.11.0, transformers 5.8.0, CUDA 12.x, 4090 24G

**预估改动文件:** 5 个 Python 源文件修改 + 2 个 Python 实验文件 import 更新 + 12 个新 shell 脚本

---

### Task 1: modeling.py — 模型路径参数化 + device_map 默认值

**Files:**

- Modify: `src/rotationquant/modeling.py`

- [x] **Step 1: 修改 load_causal_lm 签名和 TINYLLAMA_BASE_DIR**

`load_causal_lm` 第一个参数改为 `model_dir`（不再是可选），`device_map` 默认值改为 `"auto"`。移除 `TINYLLAMA_BASE_DIR` 常量。

```python
# modeling.py — 修改后

from __future__ import annotations

from collections.abc import Iterator

import torch


LLAMA_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def iter_llama_target_linears(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Linear]]:
    """Yield only the q/k/v/o and FFN linears targeted by Stage A."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(LLAMA_LINEAR_SUFFIXES):
            yield name, module


def iter_llama_decoder_layers(model: torch.nn.Module) -> Iterator[tuple[int, torch.nn.Module]]:
    """Yield decoder layers from Hugging Face LLaMA-like causal LM models."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Expected a Hugging Face LLaMA-like model with model.layers.")
    for index, layer in enumerate(layers):
        yield index, layer


def iter_llama_ffn_modules(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield FFN modules; Hugging Face calls them `mlp`, but Stage B uses FFN."""
    for index, layer in iter_llama_decoder_layers(model):
        ffn = getattr(layer, "mlp", None)
        if ffn is None:
            raise ValueError(f"Decoder layer {index} has no mlp/FFN module.")
        yield f"model.layers.{index}.mlp", ffn


def load_causal_lm(model_dir: str, dtype: str = "float16", device_map: str | None = "auto"):
    """Load a local Hugging Face causal LM without tying the code to one checkpoint."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load a causal LM. Install project requirements first."
        ) from exc

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    load_kwargs = {
        "device_map": device_map,
        "local_files_only": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch_dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, **load_kwargs)
    model.eval()
    return model, tokenizer
```

- [x] **Step 2: 验证 TinyLlama 旧路径仍能加载**

```bash
PYTHONPATH=src python -c "
from rotationquant.modeling import load_causal_lm
# device_map=None 避免在 Mac 上尝试 auto
m, t = load_causal_lm('models/TinyLlama-1.1B-intermediate-step-1431k-3T', device_map=None)
print('Model loaded:', type(m).__name__)
print('Tokenizer:', type(t).__name__)
"
```

Expected: Model loaded with no errors.

- [x] **Step 3: Commit**

```bash
git add src/rotationquant/modeling.py
git commit -m "refactor: parameterize model_dir in load_causal_lm, default device_map=auto"
```

---

### Task 2: run_metadata.py — 添加 CUDA 检测字段

**Files:**

- Modify: `src/rotationquant/run_metadata.py:30-42`

- [x] **Step 1: 修改 torch_runtime_status 函数**

在 MPS 字段之后新增 CUDA 字段：

```python
def torch_runtime_status() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch_importable": False}
    status: dict[str, object] = {
        "torch_importable": True,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_built": bool(torch.backends.cuda.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if hasattr(torch, "mps"):
        status["mps_device_count"] = int(torch.mps.device_count())
    if torch.cuda.is_available():
        status["cuda_device_count"] = int(torch.cuda.device_count())
        status["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return status
```

- [x] **Step 2: 验证 MPS 机器上 CUDA 字段正常为 false**

```bash
PYTHONPATH=src python -c "
from rotationquant.run_metadata import torch_runtime_status
status = torch_runtime_status()
print('cuda_available:', status['cuda_available'])
print('cuda_built:', status['cuda_built'])
print('mps_available:', status['mps_available'])
"
```

Expected (Mac): cuda_available=False, cuda_built=False (or True if PyTorch was built with CUDA), mps_available=True.

- [x] **Step 3: Commit**

```bash
git add src/rotationquant/run_metadata.py
git commit -m "feat: add CUDA runtime detection to run_metadata"
```

---

### Task 3: activation_capture.py — 类重命名 + 兼容别名

**Files:**

- Modify: `src/rotationquant/activation_capture.py`

- [x] **Step 1: 重命名类并添加兼容别名**

将 `TinyLlamaActivationCapture` → `LlamaActivationCapture`，`TinyLlamaLocalIOCapture` → `LlamaLocalIOCapture`。文件末尾加旧名别名。

```python
# activation_capture.py — 修改后 (仅显示变更部分)

# class TinyLlamaActivationCapture → class LlamaActivationCapture
# class TinyLlamaLocalIOCapture → class LlamaLocalIOCapture

# 文件末尾添加:
TinyLlamaActivationCapture = LlamaActivationCapture
TinyLlamaLocalIOCapture = LlamaLocalIOCapture
```

- [x] **Step 2: 验证旧名和新名等价**

```bash
PYTHONPATH=src python -c "
from rotationquant.activation_capture import (
    LlamaActivationCapture,
    LlamaLocalIOCapture,
    TinyLlamaActivationCapture,
    TinyLlamaLocalIOCapture,
)
assert LlamaActivationCapture is TinyLlamaActivationCapture, 'alias mismatch'
assert LlamaLocalIOCapture is TinyLlamaLocalIOCapture, 'alias mismatch'
print('All aliases correct')
"
```

Expected: All aliases correct.

- [x] **Step 3: Commit**

```bash
git add src/rotationquant/activation_capture.py
git commit -m "refactor: rename capture classes to Llama* with backward-compat aliases"
```

---

### Task 4: attention_capture.py — 类重命名 + 兼容别名

**Files:**

- Modify: `src/rotationquant/attention_capture.py`

- [x] **Step 1: 重命名类并添加兼容别名**

```python
# attention_capture.py — 变更部分

# class TinyLlamaAttentionCapture → class LlamaAttentionCapture

# 文件末尾添加:
TinyLlamaAttentionCapture = LlamaAttentionCapture
```

- [x] **Step 2: 验证别名**

```bash
PYTHONPATH=src python -c "
from rotationquant.attention_capture import (
    LlamaAttentionCapture,
    TinyLlamaAttentionCapture,
)
assert LlamaAttentionCapture is TinyLlamaAttentionCapture
print('Alias correct')
"
```

Expected: Alias correct.

- [x] **Step 3: Commit**

```bash
git add src/rotationquant/attention_capture.py
git commit -m "refactor: rename TinyLlamaAttentionCapture to LlamaAttentionCapture with alias"
```

---

### Task 5: 更新实验脚本 import（使用新类名）

**Files:**

- Modify: `experiments/stage_b_activation.py` (line 12, 141)
- Modify: `experiments/stage_b_local.py` (line 12, 159)
- Modify: `experiments/stage_c_invariance.py` (line 11, 92)
- Modify: `experiments/stage_c_kv_local.py` (line 12, 122)
- Modify: `experiments/stage_c_qjl.py` (line 12, 107)
- Modify: `experiments/stage_c_attention_layer.py` (line 12, 218)

- [x] **Step 1: 更新 6 个实验脚本的 import 和引用**

每处改动相同模式：`from rotationquant.activation_capture import TinyLlamaActivationCapture` → `from rotationquant.activation_capture import LlamaActivationCapture`，引用处同步更新。

以 `experiments/stage_b_activation.py` 为例：

```python
# Line 12: import
from rotationquant.activation_capture import LlamaActivationCapture

# Line 141: usage
with LlamaActivationCapture(model, layer_limit=args.layer_limit) as capture:
```

其他 5 个文件同理。

- [x] **Step 2: 验证 import 正确**

```bash
PYTHONPATH=src python -c "
from experiments.stage_b_activation import main
from experiments.stage_b_local import main
from experiments.stage_c_invariance import main
from experiments.stage_c_kv_local import main
from experiments.stage_c_qjl import main
from experiments.stage_c_attention_layer import main
print('All experiment imports OK')
"
```

Expected: All experiment imports OK.

- [x] **Step 3: Commit**

```bash
git add experiments/stage_b_activation.py experiments/stage_b_local.py \
        experiments/stage_c_invariance.py experiments/stage_c_kv_local.py \
        experiments/stage_c_qjl.py experiments/stage_c_attention_layer.py
git commit -m "refactor: update experiment imports to use new Llama* capture class names"
```

---

### Task 6: 创建 Llama3-8B 专用 shell 脚本

**Files:**

- Create: `scripts/llama3-8b/run_stage_a_tensor_sweep.sh`
- Create: `scripts/llama3-8b/run_stage_a_ppl.sh`
- Create: `scripts/llama3-8b/run_stage_b_activation.sh`
- Create: `scripts/llama3-8b/run_stage_b_local.sh`
- Create: `scripts/llama3-8b/run_stage_b_ppl.sh`
- Create: `scripts/llama3-8b/run_stage_c_invariance.sh`
- Create: `scripts/llama3-8b/run_stage_c_kv_local.sh`
- Create: `scripts/llama3-8b/run_stage_c_qjl.sh`
- Create: `scripts/llama3-8b/run_stage_c_attention_layer.sh`
- Create: `scripts/llama3-8b/run_stage_c_ppl.sh`
- Create: `scripts/llama3-8b/run_stage_c_accuracy.sh`

- [x] **Step 1: 创建目录 + Stage A 脚本**

```bash
mkdir -p scripts/llama3-8b
```

**`scripts/llama3-8b/run_stage_a_tensor_sweep.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_a_weight_only.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_a \
  --bits 4 3 2 \
  --methods direct_absmax hadamard_absmax hadamard_lm \
  --block-size 128
```

**`scripts/llama3-8b/run_stage_a_ppl.sh`:**

```bash
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
```

关键差异 vs TinyLlama 脚本：`--device-map auto` 替代 `--device mps`，`MODEL_DIR` 环境变量支持覆盖。

- [x] **Step 2: Stage B 脚本**

**`scripts/llama3-8b/run_stage_b_activation.sh`:**

```bash
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
```

**`scripts/llama3-8b/run_stage_b_local.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_b_local.py \
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
  --max-samples 128 \
  --layer-limit 4
```

**`scripts/llama3-8b/run_stage_b_ppl.sh`:**

```bash
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
```

- [x] **Step 3: Stage C 脚本**

**`scripts/llama3-8b/run_stage_c_invariance.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_invariance.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 64 \
  --layer-limit 4
```

**`scripts/llama3-8b/run_stage_c_kv_local.sh`:**

```bash
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
```

**`scripts/llama3-8b/run_stage_c_qjl.sh`:**

```bash
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
```

**`scripts/llama3-8b/run_stage_c_attention_layer.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_attention_layer.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --methods fp16 attn_identity_fp16 attn_kv_hlm_k4v4 attn_kv_hlm_k3v4 attn_rot_lm_w4a4_hlm_k4v4 attn_rot_lm_w3a4_hlm_k3v4 attn_rot_lm_w4a3_hlm_k4v3 \
  --dtype float16 \
  --device-map auto \
  --dataset wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --split test \
  --max-samples 64 \
  --layer-limit 4
```

**`scripts/llama3-8b/run_stage_c_ppl.sh`:**

```bash
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
```

**`scripts/llama3-8b/run_stage_c_accuracy.sh`:**

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-models/Meta-Llama-3-8B}"
PYTHONPATH=src python experiments/stage_c_accuracy.py \
  --model-dir "$MODEL_DIR" \
  --output-dir outputs/stage_c \
  --benchmark piqa
```

- [x] **Step 4: 赋予执行权限并验证脚本语法**

```bash
chmod +x scripts/llama3-8b/*.sh
for f in scripts/llama3-8b/*.sh; do
  bash -n "$f" && echo "OK: $f"
done
```

Expected: OK for all scripts.

- [x] **Step 5: Commit**

```bash
git add scripts/llama3-8b/
git commit -m "feat: add Llama3-8B shell scripts with device-map auto for CUDA"
```

---

### Task 7: 服务器部署流程（文档化）

**Files:**

- Create: `docs/server_setup_guide.md`

- [x] **Step 1: 编写服务器部署指南**

```markdown
# 优云智算服务器部署指南

## 1. 创建实例
- 平台: 优云智算 (youyunzhisuan.com)
- 配置: RTX 4090 24G × 1, CUDA 12.x 镜像
- Python >= 3.11

## 2. 克隆代码
```bash
git clone https://github.com/CarreyLiu-code/Rotation-Quant.git -b stage1-report
cd Rotation-Quant
```

## 3. 安装依赖

```bash
python -m venv rotationquant
source rotationquant/bin/activate
pip install -r requirements.txt
# 如果 torch 无 CUDA wheel:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121
```

## 4. 下载模型

```bash
huggingface-cli login
# 需要先去 huggingface.co/meta-llama/Meta-Llama-3-8B 申请访问权限
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B
```

## 5. 验证环境

```bash
PYTHONPATH=src python -c "
from rotationquant.modeling import load_causal_lm
from rotationquant.run_metadata import torch_runtime_status
print(torch_runtime_status())
m, t = load_causal_lm('models/Meta-Llama-3-8B')
print('Layers:', len(list(m.named_modules())))
"
```

## 6. 运行实验

```bash
# Stage A
bash scripts/llama3-8b/run_stage_a_tensor_sweep.sh
bash scripts/llama3-8b/run_stage_a_ppl.sh

# Stage B
bash scripts/llama3-8b/run_stage_b_activation.sh
bash scripts/llama3-8b/run_stage_b_local.sh
bash scripts/llama3-8b/run_stage_b_ppl.sh

# Stage C
bash scripts/llama3-8b/run_stage_c_invariance.sh
bash scripts/llama3-8b/run_stage_c_kv_local.sh
bash scripts/llama3-8b/run_stage_c_qjl.sh
bash scripts/llama3-8b/run_stage_c_attention_layer.sh
bash scripts/llama3-8b/run_stage_c_ppl.sh
bash scripts/llama3-8b/run_stage_c_accuracy.sh  # 可选
```

## 7. 汇总结果

```bash
PYTHONPATH=src python experiments/summarize_stage_a_tensor.py outputs/stage_a/<run_id>
PYTHONPATH=src python experiments/summarize_stage_b.py outputs/stage_b/<run_id>
PYTHONPATH=src python experiments/summarize_stage_c.py outputs/stage_c/<run_id>
```

```

- [x] **Step 2: Commit**

```bash
git add docs/server_setup_guide.md
git commit -m "docs: add server setup guide for Llama3-8B experiments"
```

---

### Task 8: inspect 脚本适配 + 最终验证

**Files:**

- Modify: `experiments/inspect_tinyllama_arch.py`

- [x] **Step 1: 更新 inspect 脚本名和 import**

脚本已经接受 `--model-dir` 参数，只需更新注释中的模型名引用：

```python
# experiments/inspect_tinyllama_arch.py
# Line 8: 移除 TINYLLAMA_BASE_DIR 常量, 改用 argparse default
# Line 12: 更新 description
parser = argparse.ArgumentParser(description="Inspect LLaMA architecture for quantization planning.")
parser.add_argument("--model-dir", default="models/TinyLlama-1.1B-intermediate-step-1431k-3T")
```

- [x] **Step 2: 本地验证所有 import 路径正确**

```bash
PYTHONPATH=src python -c "
# 验证所有核心模块可导入
from rotationquant import modeling, rotations, quantizers, metrics, ppl, run_metadata
from rotationquant import stage_a, stage_a_model, stage_b, stage_c, stage_c_model
from rotationquant import activation_capture, attention_capture
print('All core modules imported successfully')
"
```

Expected: All core modules imported successfully.

- [x] **Step 3: Commit**

```bash
git add experiments/inspect_tinyllama_arch.py
git commit -m "refactor: update inspect script for generic LLaMA support"
```

---

## 改动总结

| 文件                                        | 操作      | 说明                                          |
| ------------------------------------------- | --------- | --------------------------------------------- |
| `src/rotationquant/modeling.py`           | 修改      | 移除 TINYLLAMA_BASE_DIR，device_map 默认 auto |
| `src/rotationquant/run_metadata.py`       | 修改      | 加 CUDA 检测字段                              |
| `src/rotationquant/activation_capture.py` | 修改      | 类重命名 + 别名                               |
| `src/rotationquant/attention_capture.py`  | 修改      | 类重命名 + 别名                               |
| `experiments/stage_b_activation.py`       | 修改      | import 更新                                   |
| `experiments/stage_b_local.py`            | 修改      | import 更新                                   |
| `experiments/stage_c_invariance.py`       | 修改      | import 更新                                   |
| `experiments/stage_c_kv_local.py`         | 修改      | import 更新                                   |
| `experiments/stage_c_qjl.py`              | 修改      | import 更新                                   |
| `experiments/stage_c_attention_layer.py`  | 修改      | import 更新                                   |
| `experiments/inspect_tinyllama_arch.py`   | 修改      | description 更新                              |
| `scripts/llama3-8b/*.sh`                  | 新建 (11) | Llama3-8B 专用运行脚本                        |
| `docs/server_setup_guide.md`              | 新建      | 服务器部署文档                                |

**不改动的文件：** `rotations.py`, `quantizers.py`, `metrics.py`, `ppl.py`, `stage_a.py`, `stage_b.py`, `stage_c.py`, `stage_a_model.py`, `stage_c_model.py`
