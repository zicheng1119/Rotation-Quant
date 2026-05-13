# Llama3-8B 量化实验迁移设计

**Date**: 2026-05-13
**Status**: draft

## 目标

将 rotation-based low-bit quantization 实验从 TinyLlama-1.1B 迁移到 Llama3-8B，验证量化方法在大参数量模型上的泛化能力。同时完成 MPS → CUDA 的设备适配。

## 范围

- Stage A（权重量化）：张量级扫描 + 模型级 PPL
- Stage B（W/A 伪量化）：激活扫描 + 局部误差 + FFN-only PPL
- Stage C（Attention/KV cache）：旋转不变性验证 + KV 局部扫描 + QJL 残差校正 + 结构化 attention 层量化 + attention-only PPL + 零样本准确率

## 模型架构差异

| 参数 | TinyLlama-1.1B | Llama3-8B | 适配情况 |
|---|---|---|---|
| hidden_size | 2048 (2^11) | 4096 (2^12) | FWHT 直接兼容，均为 2 的幂 |
| intermediate_size | 5632 | 14336 | 均非 2 的幂，block padding 机制已覆盖 |
| head_dim | 64 | 128 | 均为 2 的幂 |
| num_attention_heads | 32 | 32 | 无硬编码假设 |
| num_kv_heads | 4 | 8 | 泛型遍历，无硬编码 |
| num_layers | 22 | 32 | `iter_llama_decoder_layers` 泛型遍历 |
| LayerNorm | RMSNorm | RMSNorm | Hook 名称一致 (`input_layernorm`, `post_attention_layernorm`) |

### Stage C 专项分析

Stage C 是三个 Stage 中对架构差异最敏感的部分：

#### head_dim 自适配

`headwise_rotation`、`make_head_rotation_matrix`、`make_head_signs` 均通过 `x.shape[-1]` 或传入的 `head_dim` 参数动态推断维度，无需硬编码。Llama3-8B 的 head_dim=128 是 2 的幂，FWHT 直接可用。

#### kv_block_size 默认值 64 的影响

| 指标 | TinyLlama-1.1B (head_dim=64) | Llama3-8B (head_dim=128) |
|---|---|---|
| head_dim | 64 | 128 |
| kv_block_size=64 下每 head 块数 | 1 block (64/64) | 2 blocks (128/64) |
| kv_block_size=32 下每 head 块数 | 2 blocks (64/32) | 4 blocks (128/32) |

block_size=64 在 Llama3-8B 上相当于 TinyLlama 上 block_size=32 的分块粒度。这对量化精度有利（更细的 scale granularity），但不会破坏任何逻辑。

#### QJL projection_dim 语义变化

QJL 使用随机高斯投影将 head_dim 维的 residual error 压缩到 `projection_dim` 维，再用量化码本近似。现有 `QJLSpec` 硬编码 `projection_dim=64`：

| 场景 | head_dim | projection_dim | 意义 |
|---|---|---|---|
| TinyLlama | 64 | 64 | 无压缩（identity mapping），QJL 直接拟合残差 |
| Llama3-8B | 128 | 64 | 2:1 压缩，QJL 在低维子空间拟合残差 |

这对 Llama3-8B 反而更有意义——在真实压缩条件下测试 QJL，而非 TinyLlama 上的退化为 identity。不需要改代码，但结果解读时需要注意。

#### value rotation + o_proj absorb

`make_head_rotation_matrix(..., head_dim, block_size)` 动态构建块对角旋转矩阵。head_dim=128, block_size=64 时生成 2 个 H64 块组成的 128×128 矩阵，`absorb_o_proj_weight_headwise` 将转置吸收到 `o_proj.weight`。逻辑泛型，无需改动。

#### 注意力包装器兼容性

`_CaptureAttentionWrapper` 通过 `original_attention.head_dim`、`original_attention.num_key_value_groups`、`original_attention.scaling` 读取注意力配置，均为 Hugging Face LLaMA attention 标准属性，Llama3-8B 完全兼容。RoPE 函数 `apply_rotary_pos_emb` 来自同一个 `transformers.models.llama.modeling_llama` 路径。

## 代码改动

### 1. modeling.py — 模型路径配置化

- 移除 `TINYLLAMA_BASE_DIR` 硬编码常量
- `load_causal_lm` 改为接受 `model_dir` 参数（必传）
- 加 `device_map="auto"` 让 Hugging Face 自动处理 GPU 放置

### 2. run_metadata.py — CUDA 设备检测

- 现有 MPS 检测字段保留不变
- 新增 `cuda_built`、`cuda_available`、`cuda_device_count`、`cuda_device_names` 字段

### 3. activation_capture.py & attention_capture.py — 类重命名

- `TinyLlamaActivationCapture` → `LlamaActivationCapture`
- `TinyLlamaLocalIOCapture` → `LlamaLocalIOCapture`
- `TinyLlamaAttentionCapture` → `LlamaAttentionCapture`
- 保留旧名作为兼容别名，避免破坏现有脚本

### 4. experiments/*.py — 加 --model_dir 参数

- 所有实验脚本新增 `--model_dir` 参数，默认值为 `models/TinyLlama-1.1B-intermediate-step-1431k-3T`（保持向后兼容）
- `inspect_tinyllama_arch.py` 加 `--model_dir` 参数

### 5. 不改动的文件

- `rotations.py` — 块旋转逻辑泛型，无需改动
- `quantizers.py` — 量化器与模型架构无关，无需改动
- `stage_a.py`、`stage_b.py`、`stage_c.py` — 通过 `iter_*` 泛型接口访问模型，无需改动
- `ppl.py` — 已接受通用 model，无需改动
- `metrics.py` — 张量级指标，与架构无关

## CUDA 适配策略

- 不引入显式的 `.to("cuda")` 或 `.cuda()` 调用
- 利用 Hugging Face `device_map="auto"` 自动分布模型层到 GPU
- 量化代码已通过 `x.device` 惯用法跟随张量所在设备（见 quantizers.py 中 `device=blocks.device` 等用法）
- 唯一需要确认的是 FP16 推理：4090 24G 运行 Llama3-8B FP16 约需 16GB 显存，余量充足

## 环境搭建（优云智算 4090 24G）

### 步骤

1. 创建实例：4090 24G × 1，预装 CUDA 12.x + Python 3.11+ 系统镜像
2. `git clone https://github.com/CarreyLiu-code/Rotation-Quant.git -b stage1-report`
3. `pip install -r requirements.txt`（锁定版本）
4. 如果 torch 2.11.0 在 PyPI 无 CUDA 12.x wheel：`pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121`
5. `huggingface-cli login` + 下载 `meta-llama/Meta-Llama-3-8B` 到 `models/`
6. 验证：`PYTHONPATH=src python -c "from rotationquant.modeling import load_causal_lm; m, t = load_causal_lm('models/Meta-Llama-3-8B')"`

### 依赖兼容性风险

- `torch 2.11.0`：需确认 PyPI 有 CUDA 12.x 对应版本
- `transformers 5.8.0`：可能已更新到 5.9+，需要测试兼容性
- `sentencepiece 0.2.1`：Llama3-8B tokenizer 使用 tiktoken/BPE，不依赖 sentencepiece
- 其他依赖（numpy、pandas、pyyaml、tqdm）风险低

## 实验执行

### Stage A（权重量化）

1. **arch inspect**：`PYTHONPATH=src python experiments/inspect_tinyllama_arch.py --model_dir models/Meta-Llama-3-8B`
2. **tensor sweep**：`run_stage_a_tensor_sweep.sh`（改 MODEL_DIR）
3. **PPL**：`run_stage_a_ppl.sh`（改 MODEL_DIR）
4. **汇总**：`summarize_stage_a_tensor.py`

### Stage B（W/A 伪量化）

1. **activation capture**：`run_stage_b_activation.sh`
2. **local error**：`run_stage_b_local.sh`
3. **PPL**：`run_stage_b_ppl.sh`
4. **汇总**：`summarize_stage_b.py`

### Stage C（Attention/KV cache 量化）

1. **旋转不变性**：`run_stage_c_invariance.sh` — 验证 post-RoPE H128 旋转保持 attention scores
2. **KV 局部扫描**：`run_stage_c_kv_local.sh` — KV cache 量化方法 × 位宽扫描
3. **QJL 残差校正**：`run_stage_c_qjl.sh` — QJL residual 校正效果
4. **结构化 attention 层**：`run_stage_c_attention_layer.sh` — 完整的 Q/K/V/O W/A + KV cache 量化
5. **PPL**：`run_stage_c_ppl.sh` — attention-only 模型级 PPL
6. **零样本准确率**：`run_stage_c_accuracy.sh` — 可选
7. **汇总**：`summarize_stage_c.py`

### 预估时间（4090 24G）

| Stage | 实验 | 预估时间 |
|---|---|---|
| A | tensor sweep | ~15-20 分钟 |
| A | PPL | ~10 分钟 |
| B | activation capture | ~15 分钟 |
| B | local error | ~10 分钟 |
| B | PPL | ~20 分钟 |
| C | invariance | ~5 分钟 |
| C | KV local | ~15 分钟 |
| C | QJL | ~10 分钟 |
| C | attention layer | ~20 分钟 |
| C | PPL | ~25 分钟 |
| C | accuracy (可选) | ~30 分钟 |
| **合计** | | **约 2.5-3 小时**（不含 accuracy）

## 输出

所有结果写入 `outputs/stage_a/<run_id>/`、`outputs/stage_b/<run_id>/` 和 `outputs/stage_c/<run_id>/`，与 TinyLlama 输出目录平行，便于横向对比。
