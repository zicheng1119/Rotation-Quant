# 量化全流程走查：一个 Token 在 Llama3-8B 中的完整旅程

本文以 Llama3-8B 为例，追踪一个 token 从输入到输出的完整处理流程，并标注 Stage A/B/C 量化在每个环节的介入位置。

## 模型结构速查

```
Llama3-8B:
  hidden_size      = 4096
  intermediate_size = 14336
  num_attention_heads = 32
  num_kv_heads     = 8
  head_dim         = 128
  num_layers       = 32
  vocab_size       = 128256
```

## 零、启动阶段：量化权重的"预处理"（Stage A 核心）

在任何 forward 之前，Stage A 实验会**一次性**对所有 7 类 Linear 层的权重做伪量化：

```
对每一层 (layer 0..31):
  对 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj 的 .weight:
    ┌─────────────────────────────────────────────────────────┐
    │ Stage A 量化管线 (以 hadamard_lm 为例)                      │
    │                                                         │
    │  weight [4096×4096]  →  reshape 1D  →  block_view(128) │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │ 对每个 block [128]:                                 │   │
    │  │   1. L2 normalize → block 内单位范数               │   │
    │  │   2. FWHT (快速 Walsh-Hadamard 变换)              │   │
    │  │   3. × √128 → 近似 N(0,1) 高斯分布                │   │
    │  │   4. Lloyd-Max 非均匀量化 (在标准正态上拟合的码本)  │   │
    │  │   5. 反量化得到 dequantized block                  │   │
    │  │   6. FWHT 逆变换 → /√128 → × norms                 │   │
    │  └──────────────────────────────────────────────────┘   │
    │  重组 → 拷贝回 layer.weight                              │
    └─────────────────────────────────────────────────────────┘
```

**结果：** 所有权重被替换为"伪量化"后的 float 值。后续 forward 使用的是量化过的权重，但计算仍是 FP16 GEMM。

---

## 一、Token 的旅程：逐层 Forward

### 输入：一个 Token

```
input_ids: [1]  (token id = 1059, 例如 "The")
    │
    ▼
┌──────────────────────────────────────┐
│  Embedding                           │
│  [1] × vocab[128256×4096] → [4096]   │
│  ⚠️ Embedding 不被量化 (Stage A/B/C  │
│    只量化指定的 7 类 Linear 层)        │
└──────────────────────────────────────┘
    │
    ▼  hidden_states: [1, 4096] (FP16)
    │
    │  ╔══════════════════════════════╗
    ╔══╣  32 个 Decoder Layer (重复)   ╠══════════════════╗
    ║  ╚══════════════════════════════╝                  ║
    ║                                                     ║
    ║  ┌─────────────────────────────────────────────┐    ║
    ║  │ RMSNorm (input_layernorm)                   │    ║
    ║  │ [1, 4096] → [1, 4096]                       │    ║
    ║  │ ⚠️ Norm 层不量化                              │    ║
    ║  └─────────────────────────────────────────────┘    ║
    ║         │                                           ║
    ║         ▼                                           ║
    ║  ╔══════════════════════════════════════════════╗   ║
    ║  ║         Self-Attention 模块                   ║   ║
    ║  ╚══════════════════════════════════════════════╝   ║
    ║         │                                           ║
    ║         │  (详见第二节：Attention 量化流程)            ║
    ║         │                                           ║
    ║         ▼  attn_output: [1, 4096]                   ║
    ║  ┌─────────────────────────────────────────────┐    ║
    ║  │ Residual Add: x = x + attn_output           │    ║
    ║  │ [1, 4096] + [1, 4096] → [1, 4096]          │    ║
    ║  └─────────────────────────────────────────────┘    ║
    ║         │                                           ║
    ║         ▼                                           ║
    ║  ┌─────────────────────────────────────────────┐    ║
    ║  │ RMSNorm (post_attention_layernorm)           │    ║
    ║  │ [1, 4096] → [1, 4096]                       │    ║
    ║  └─────────────────────────────────────────────┘    ║
    ║         │                                           ║
    ║         ▼                                           ║
    ║  ╔══════════════════════════════════════════════╗   ║
    ║  ║              FFN 模块 (SwiGLU)                ║   ║
    ║  ╚══════════════════════════════════════════════╝   ║
    ║         │                                           ║
    ║         │  (详见第三节：FFN 量化流程)                  ║
    ║         │                                           ║
    ║         ▼  ffn_output: [1, 4096]                    ║
    ║  ┌─────────────────────────────────────────────┐    ║
    ║  │ Residual Add: x = x + ffn_output            │    ║
    ║  │ [1, 4096] + [1, 4096] → [1, 4096]          │    ║
    ║  └─────────────────────────────────────────────┘    ║
    ║         │                                           ║
    ╚═════════╧═══════════════════════════════════════════╝
              │
              ▼ (经过 32 层后)
    ┌──────────────────────────────────────┐
    │  RMSNorm (final_layernorm)            │
    │  [1, 4096] → [1, 4096]               │
    ╚══════════════════════════════════════╝
              │
              ▼
    ┌──────────────────────────────────────┐
    │  LM Head (lm_head)                    │
    │  [1, 4096] × [4096, 128256]          │
    │  → [1, 128256] (logits)              │
    │  ⚠️ lm_head 不被量化                   │
    ╚══════════════════════════════════════╝
              │
              ▼
         cross-entropy loss → PPL
```

---

## 二、Attention 模块内部：Stage C 量化介入点

```
输入: hidden_states [1, 4096]
│
├─ Q/K/V 投影 ─────────────────────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ Stage C quantize_qkv=True 时:                      │
│  │                                                    │
│  │   1. 激活量化 (Stage B 风格的 activation domain):   │
│  │      hidden [1,4096] → block_rotation_last_dim     │
│  │      → 分块 Hadamard 旋转 (block_size=128)          │
│  │      → Lloyd-Max / absmax 量化 → dequant           │
│  │                                                    │
│  │   2. 权重 × 量化激活 (权重已在 Stage A 预处理过):    │
│  │      q_proj: [1,4096] × [4096,4096] → [1,4096]    │
│  │      k_proj: [1,4096] × [4096,1024] → [1,1024]    │
│  │      v_proj: [1,4096] × [4096,1024] → [1,1024]    │
│  └────────────────────────────────────────────────────┘
│
├─ Reshape 为多头 ───────────────────────────────────────
│
│  Q: [1,4096] → [1, 32, 128]     (32 heads × 128)
│  K: [1,1024] → [1, 8, 128]      (8 KV heads × 128)
│  V: [1,1024] → [1, 8, 128]
│
├─ RoPE 位置编码 ────────────────────────────────────────
│
│  Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
│  → Q_rope [1, 32, 128], K_rope [1, 8, 128]
│
├─ Head-wise Hadamard 旋转 ────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ 对每个 head 独立施加 H128 旋转:                      │
│  │                                                    │
│  │   query_rotated  = FWHT(Q_head, dim=-1) / √128     │
│  │   key_rotated    = FWHT(K_head, dim=-1) / √128     │
│  │   value_rotated  = FWHT(V_head, dim=-1) / √128     │
│  │                                                    │
│  │   (注: 如果用 randomized_hadamard, 则先乘 sign)     │
│  └────────────────────────────────────────────────────┘
│
├─ KV Cache 量化 (Stage C 核心) ────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ 对旋转后的 K, V 做逐 token 逐 head 量化:             │
│  │                                                    │
│  │   key  [1, 8, 128]                                 │
│  │     → block_view with kv_block_size=64             │
│  │     → 每 head 分 2 个 block [64] (128/64=2)         │
│  │     → 对每个 block: RMS scale → Lloyd-Max 量化      │
│  │     → dequant → key_hat [1, 8, 128]                │
│  │                                                    │
│  │   value [1, 8, 128] — 同上述流程                     │
│  │     → value_hat [1, 8, 128]                        │
│  │                                                    │
│  │   量化方法由 KVQuantSpec 决定:                       │
│  │     hadamard_lm:  RMS scale + Lloyd-Max            │
│  │     absmax:       absmax scale + uniform quant     │
│  │     fp16:         不量化                            │
│  └────────────────────────────────────────────────────┘
│
├─ GQA 扩展 ──────────────────────────────────────────
│
│  K_hat: [1, 8, 128] → repeat_interleave(4) → [1, 32, 128]
│  V_hat: 同上
│
├─ Attention Score 计算 ────────────────────────────────
│
│  scores = Q_rotated @ K_hat^T / √128
│         = [1, 32, 1, 128] @ [1, 32, 128, 1] → [1, 32, 1, 1]
│
│  → causal mask → softmax → probs [1, 32, 1, 1]
│
├─ Attention Output ────────────────────────────────────
│
│  output_heads = probs @ V_hat
│               = [1, 32, 1, 1] @ [1, 32, 1, 128] → [1, 32, 1, 128]
│
├─ Value 路径分叉 ──────────────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ 路径 A: "reconstruct"                               │
│  │   output_heads (仍在旋转域) → 逐 head 逆旋转 H128    │
│  │   → 回到原始域 → concat → o_proj                     │
│  │                                                    │
│  │ 路径 B: "o_proj_absorb" (推荐)                      │
│  │   output_heads 保持在旋转域 (不做逆旋转)             │
│  │   o_proj.weight 已在初始化时预先左乘 R^T:            │
│  │     W_o_absorbed = W_o @ R_blockdiag^T             │
│  │   其中 R_blockdiag = 32 个 H128 块组成的对角阵       │
│  │   数学上等价于先逆旋转再乘 W_o，但省去运行时旋转      │
│  └────────────────────────────────────────────────────┘
│
├─ Concat heads ────────────────────────────────────────
│
│  [1, 32, 1, 128] → transpose → [1, 1, 4096]
│
├─ O 投影 (带量化) ──────────────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ quantize_o=True 时:                                │
│  │   激活量化 → F.linear(o_input, o_weight, o_bias)    │
│  │   → attn_output [1, 4096]                          │
│  │                                                    │
│  │   (o_proj_absorb 路径下: o_weight 已包含旋转吸收)    │
│  └────────────────────────────────────────────────────┘
│
└─ 输出: attn_output [1, 4096]
```

---

## 三、FFN 模块内部：Stage B 量化介入点

```
输入: hidden_states [1, 4096] (来自 post_attention_layernorm)
│
├─ 激活量化 (输入侧) ────────────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ 输入 activation 先量化:                              │
│  │                                                    │
│  │   [1, 4096]                                        │
│  │     → block_rotation_last_dim (block_size=128)     │
│  │     → 分块 Hadamard 旋转 (32 blocks × 128)          │
│  │     → quantize (Lloyd-Max / absmax / MXFP4)        │
│  │     → dequant → x_quant [1, 4096]                  │
│  │                                                    │
│  │ 注: gate_proj 和 up_proj 共享同一个量化激活          │
│  └────────────────────────────────────────────────────┘
│
├─ Gate 路径 ───────────────────────────────────────────
│
│  gate = x_quant @ gate_weight^T + gate_bias
│       = [1, 4096] @ [4096, 14336] → [1, 14336]
│
│  gate_weight 已在 Stage A / Stage B prep 中量化
│
│  → SiLU(gate) [1, 14336]
│
├─ Up 路径 ─────────────────────────────────────────────
│
│  up = x_quant @ up_weight^T + up_bias
│      = [1, 4096] @ [4096, 14336] → [1, 14336]
│
│  up_weight 同样已量化
│
├─ 门控融合 ────────────────────────────────────────────
│
│  intermediate = SiLU(gate) ⊙ up
│               = [1, 14336] ⊙ [1, 14336] → [1, 14336]
│
├─ 激活量化 (中间侧) ────────────────────────────────────
│
│  ┌────────────────────────────────────────────────────┐
│  │ 中间激活 quantization:                               │
│  │                                                    │
│  │   [1, 14336]                                       │
│  │     → block_rotation_last_dim (block_size=128)     │
│  │     → pad 到 14464 (113 blocks × 128 + pad=96)     │
│  │     → 每 block Hadamard 旋转                        │
│  │     → quantize → dequant                           │
│  │     → unpad → down_input [1, 14336]                │
│  └────────────────────────────────────────────────────┘
│
│  注: 14336 不是 2 的幂，也不是 128 的整数倍。
│      block_rotation_last_dim 自动填充到 14464 (113×128)
│      量化后切除填充部分，恢复到 14336。
│
├─ Down 投影 ────────────────────────────────────────────
│
│  down = down_input @ down_weight^T + down_bias
│        = [1, 14336] @ [14336, 4096] → [1, 4096]
│
│  down_weight 已量化
│
└─ 输出: ffn_output [1, 4096]
```

---

## 四、三个 Stage 的量化范围总结

```
                    Stage A        Stage B        Stage C
                    (weight-only)  (W/A FFN)      (W/A Attention + KV)
                    ────────────   ───────────    ────────────────────
q_proj.weight         ✓              —               ✓ (W)
k_proj.weight         ✓              —               ✓ (W)
v_proj.weight         ✓              —               ✓ (W)
o_proj.weight         ✓              —               ✓ (W)
gate_proj.weight      ✓              ✓ (W)            —
up_proj.weight        ✓              ✓ (W)            —
down_proj.weight      ✓              ✓ (W)            —
FFN activation in     —              ✓ (A)            —
FFN activation mid    —              ✓ (A)            —
Q/K/V activation in   —              —               ✓ (A)
O activation in       —              —               ✓ (A)
Key cache             —              —               ✓ (KV)
Value cache           —              —               ✓ (KV)
embedding             ✗              ✗               ✗
lm_head               ✗              ✗               ✗
RMSNorm               ✗              ✗               ✗

✓ = 量化   — = 不适用   ✗ = 永不量化   W = Weight   A = Activation
```

## 五、一个完整的量化方法链示例

以 `hadamard_lm` 方法为例，一个 token 经过第 0 层 FFN 时的完整变换链：

```
输入 activation x [1, 4096]
  │
  ├─ 1. 分块 (block_size=128)
  │     [1, 4096] → [32, 128]   (32 blocks)
  │
  ├─ 2. L2 normalize 每 block
  │     x_norm[i] = x[i] / ||x[i]||₂
  │
  ├─ 3. Hadamard 旋转 (FWHT)
  │     x_rot[i] = FWHT(x_norm[i]) / √128
  │     此时每 block 近似 N(0,1) 高斯分布
  │
  ├─ 4. Lloyd-Max 非均匀量化 (在标准正态码本上)
  │     对 x_rot 的每个值:
  │       → 找到最近的 Lloyd-Max centroid (如 4-bit 有 16 个质心)
  │       → 用 centroid 值替代原值
  │     x_dequant = codebook[argmin |x_rot - centroid|]
  │
  ├─ 5. 逆 Hadamard + 恢复 norm
  │     x_hat[i] = FWHT(x_dequant[i]) / √128 × ||x[i]||₂
  │
  ├─ 6. gate = SiLU(x_hat @ gate_weight^T)    — gate_weight 也是 hadamard_lm 处理的
  │    up   = x_hat @ up_weight^T              — up_weight 同上
  │    intermediate = SiLU(gate) ⊙ up
  │
  ├─ 7. 对 intermediate 重复步骤 1-5
  │
  └─ 8. down_output = intermediate_hat @ down_weight^T

每一层的量化误差通过 residual connection 累积，最终体现在 PPL 退化上。
```

## 六、Llama3-8B vs TinyLlama 关键差异一览

| 环节 | TinyLlama-1.1B | Llama3-8B | 影响 |
|---|---|---|---|
| hidden_size | 2048 | 4096 | 块数翻倍，FWHT 开销增大但仍在 O(n log n) |
| head_dim | 64 | 128 | H64 → H128，旋转矩阵维度翻倍 |
| kv_block_size per head | 64 → 1 block/head | 64 → 2 blocks/head | 更细粒度的 scale，有利量化精度 |
| QJL projection_dim | 64 (identity) | 64 (2:1 压缩) | 真正的降维压缩，不再是退化的 identity |
| intermediate_size | 5632 (pad→5760) | 14336 (pad→14464) | 填充率相似 (~2.3%) |
| 单层参数量 | ~28M | ~224M | 8 倍 |
| 4090 24G FP16 | 模型 ~2.2GB | 模型 ~16GB | 刚好跑通 |
