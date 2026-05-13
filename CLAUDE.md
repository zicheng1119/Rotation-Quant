# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

基于旋转的大语言模型低位量化研究代码库（第一阶段）。在 TinyLlama-1.1B 上评估 Hadamard 旋转 + Lloyd-Max 非均匀量化，与均匀 absmax 和 MXFP4 E2M1 基线进行对比，横跨三个实验阶段。

模型：`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`（22 层 LLaMA，hidden 2048，32 个 Q heads / 4 个 KV heads，head dim 64）。

## 环境配置

```
conda 环境: rotationquant
PYTHONPATH=src（所有脚本均需设置）
设备: mps（Apple Silicon GPU）
```

安装依赖：`pip install -r requirements.txt`（PyTorch 2.11、transformers 5.8、datasets 4.8 等）

运行实验前，须将模型下载至 `models/TinyLlama-1.1B-intermediate-step-1431k-3T/`。

## 运行实验

所有实验遵循以下模式：
```
PYTHONPATH=src conda run -n rotationquant python experiments/<script>.py [args]
```

Stage A（仅权重量化）：
- `scripts/run_stage_a_tensor_sweep.sh` — 张量级权重量化扫描
- `scripts/run_stage_a_ppl.sh` — 模型级 PPL，仅权重伪量化

Stage B（权重/激活伪量化，QuaRot 风格）：
- `scripts/run_stage_b_activation.sh` — 激活张量级扫描
- `scripts/run_stage_b_local.sh` — 局部 Linear/FFN 权重/激活输出误差
- `scripts/run_stage_b_ppl.sh` — 仅 FFN 的模型级 PPL

Stage C（Attention/KV cache）：
- `scripts/run_stage_c_invariance.sh` — post-RoPE 旋转不变性验证
- `scripts/run_stage_c_kv_local.sh` — KV 局部 attention 量化扫描
- `scripts/run_stage_c_qjl.sh` — QJL 残差校正扫描
- `scripts/run_stage_c_attention_layer.sh` — 结构化 attention 层量化
- `scripts/run_stage_c_ppl.sh` — 仅 attention 的模型级 PPL
- `scripts/run_stage_c_accuracy.sh` — 可选零样本准确率

Stage 1 补充脚本（`scripts/run_stage1_supplement_*.sh`）增加 MXFP4 和旋转消融实验。

汇总任意运行结果：`PYTHONPATH=src conda run -n rotationquant python experiments/summarize_stage_<X>.py outputs/<stage>/<run_id>`

## 架构

### 核心库（`src/rotationquant/`）

| 模块 | 职责 |
|---|---|
| `rotations.py` | 快速 Walsh-Hadamard 变换（FWHT）、带 padding 的分块旋转、随机符号/正交矩阵、PolarQuant 风格的分块归一化 + 旋转 |
| `quantizers.py` | 对称 absmax（均匀）、分块 absmax、MXFP4 E2M1（每 32 个元素共享 2 的幂次缩放的 microscaling FP4）、Gaussian Lloyd-Max（在 N(0,1) 上拟合的非均匀质心码本） |
| `metrics.py` | 张量指标（相对 MSE、余弦相似度、SQNR）、分布指标（峰度、离群比例）、attention 专属指标（score 偏置/方差、softmax KL、top-k 重叠） |
| `modeling.py` | TinyLlama 模型加载、LLaMA Linear 层识别 |
| `ppl.py` | 在 WikiText2 上的因果语言模型滑动窗口 PPL |
| `run_metadata.py` | Run ID 生成、git 状态、包版本、PyTorch 运行时元数据 |
| `stage_a.py` | Stage A 方法注册（direct_absmax、mxfp4、hadamard_absmax、hadamard_lm、hadamard_mxfp4、randhadamard_lm、randortho_lm）及权重量化流水线 |
| `stage_a_model.py` | 模型级 A16Wb 权重替换包装器 |
| `stage_b.py` | Stage B 方法注册、权重/激活位宽规格、分块 Hadamard 旋转、Linear/FFN 权重/激活伪量化、仅 FFN 的模型包装器 |
| `activation_capture.py` | 基于 hook 从 TinyLlama attention 和 FFN 位置捕获激活值 |
| `attention_capture.py` | Attention 包装器/hook，捕获 q/k/v 投影、post-RoPE Q/K、attention scores/probs/outputs |
| `stage_c.py` | Stage C 方法注册、逐 head Hadamard 旋转、post-RoPE KV 量化、QJL 残差估计器、score/attention 输出指标 |
| `stage_c_model.py` | 仅 attention 的模型包装器，含 q/k/v/o 权重/激活伪量化、KV cache 量化、value 旋转 + o_proj 吸收 |

### 实验脚本（`experiments/`）

每个脚本按 方法 × 位宽 × 层 进行扫描，将 JSONL/CSV + 汇总写入 `outputs/<stage>/<run_id>/`。输出始终包含 `run_metadata.json` 和 `summary.md`。

### 关键设计决策

- **仅伪量化（Fake Quant）**：所有量化均输出反量化后的浮点张量，仅用于数值质量评估；不涉及实际的 INT GEMM 或推理加速。
- **分块大小**：权重和激活旋转默认使用 128；KV cache 使用 head-dim 64（或 32）的分块。分块大小是一个超参数——在较低位宽下，32/64 通常优于 128。
- **Value 旋转 + o_proj 吸收**：在 Stage C 中，value 旋转通过块对角 H64 块在结构上吸收到 `o_proj.weight` 中，避免了旧版重建路径不匹配的问题。
- **Key 量化在 post-RoPE 之后**：Hadamard 旋转在 RoPE 之后作用于每个 head 的 64 维向量。Post-RoPE Q/K 逐 head H64 旋转保持 attention scores 不变（C1 不变性已验证）。
- **PolarQuant 风格归一化**：权重在旋转前按块进行 L2 归一化，再乘以 `sqrt(block_size)` 以近似标准正态分布——即 Lloyd-Max 的拟合域。
- **MXFP4 是与硬件相关的基线**：E2M1 格式，group_size=32，共享 2 的幂次缩放。Hadamard-LM 在数值上持续优于它，但 MXFP4 拥有更清晰的硬件实现路径。
