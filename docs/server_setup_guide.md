# Llama3-8B 量化实验 — 完整操作手册

> 本文档覆盖从零开始到拿到全部实验结果的完整流程。预计总耗时：**下载模型 30min + 运行实验 2-4h**。

---

## 前置准备（本地完成，仅一次）

### 1. 申请 Llama3-8B 模型访问权限

浏览器打开 <https://huggingface.co/meta-llama/Meta-Llama-3-8B> ，登录 HuggingFace 账号后点击 "Request Access"。Meta 通常几分钟内自动审批。

### 2. 创建 HuggingFace Access Token

访问 <https://huggingface.co/settings/tokens> → New token → 选 Read 权限 → 复制保存。后续下载模型时需要。

### 3. 创建优云智算实例

打开 <https://youyunzhisuan.com> → 创建实例：

| 配置项 | 选择 |
| ------ | ---- |
| GPU    | RTX 4090 24G × 1 |
| 镜像   | CUDA 12.x（官方推荐） |
| 磁盘   | 100G 以上（模型 16G + 依赖 ~5G） |

---

## 第一步：SSH 登录服务器

创建实例后，优云智算会显示 SSH 连接信息。在你的本地终端执行：

```bash
ssh root@<服务器IP> -p <端口>
# 示例: ssh root@123.456.789.0 -p 22222
```

> **提示：** 优云智算也会提供 JupyterLab 链接，可以用它自带的终端操作，省去 SSH。

---

## 第二步：克隆代码

```bash
git clone https://github.com/zicheng1119/Rotation-Quant.git
cd Rotation-Quant
```

确认代码版本（应该看到最新的 commit）：

```bash
git log --oneline -3
```

期望输出类似：

```text
5697083 feat: add one-click run_all and summarize_all scripts, update server guide
0c68e16 refactor: remove TINYLLAMA_BASE_DIR, each script now owns its default model path
81feabf docs: add server setup guide for Llama3-8B experiments
```

---

## 第三步：安装 Python 环境

```bash
# 创建虚拟环境
python3 -m venv rotationquant

# 激活（之后每次进服务器都要先执行这一行）
source rotationquant/bin/activate

# 安装依赖
pip install -r requirements.txt
```

如果上面的 `pip install` 提示 torch 没有 CUDA，手动指定 CUDA 版本：

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121
```

验证 GPU 可见：

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

期望输出：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

---

## 第四步：下载模型

```bash
# 登录 HuggingFace（粘贴刚才申请的 Access Token）
huggingface-cli login

# 下载 Llama3-8B（约 16GB，需 15-30 分钟）
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B
```

下载完成后确认文件完整：

```bash
ls models/Meta-Llama-3-8B/
# 应该看到: config.json  tokenizer.model  model-00001-of-00004.safetensors  ...
```

---

## 第五步：验证环境

```bash
PYTHONPATH=src python -c "
from rotationquant.modeling import load_causal_lm
from rotationquant.run_metadata import torch_runtime_status

# 打印运行时信息
status = torch_runtime_status()
for k, v in status.items():
    print(f'  {k}: {v}')

# 加载模型
m, t = load_causal_lm('models/Meta-Llama-3-8B')
print(f'Model loaded: {type(m).__name__}')
print(f'Layers: {len(list(m.named_modules()))}')
print('Environment OK')
"
```

期望看到 `cuda_available: True`、`Model loaded: LlamaForCausalLM`、`Environment OK`。

---

## 第六步：一键运行全部实验

```bash
# 开始跑全部 11 个实验（预计 2-4 小时）
bash scripts/llama3-8b/run_all.sh
```

这个脚本会按顺序执行：

| 阶段     | 实验                              | 说明                       | 预计耗时 |
| -------- | --------------------------------- | -------------------------- | -------- |
| Stage A  | `run_stage_a_tensor_sweep`        | 权重张量级量化扫描         | 5min     |
| Stage A  | `run_stage_a_ppl`                 | 仅权重的模型级 PPL         | 20min    |
| Stage B  | `run_stage_b_activation`          | 激活张量级扫描             | 10min    |
| Stage B  | `run_stage_b_local`               | 局部 Linear/FFN 误差       | 10min    |
| Stage B  | `run_stage_b_ppl`                 | 仅 FFN 的模型级 PPL        | 30min    |
| Stage C  | `run_stage_c_invariance`          | post-RoPE 旋转不变性       | 5min     |
| Stage C  | `run_stage_c_kv_local`            | KV cache 局部量化          | 10min    |
| Stage C  | `run_stage_c_qjl`                 | QJL 残差校正               | 10min    |
| Stage C  | `run_stage_c_attention_layer`     | 结构化 attention 量化      | 15min    |
| Stage C  | `run_stage_c_ppl`                 | 仅 attention 模型级 PPL    | 30min    |
| Stage C  | `run_stage_c_accuracy`            | 零样本准确率 (PIQA)        | 10min    |

> **中断恢复：** 如果中途失败，修复问题后只需重新运行 `run_all.sh`。已完成的实验会覆盖写入，不会冲突。

---

## 第七步：汇总结果

```bash
bash scripts/llama3-8b/summarize_all.sh
```

这会自动找到最新的 run_id 并生成每个 stage 的 `summary.md`，存放在对应的 `outputs/stage_X/<run_id>/` 目录下。

---

## 第八步：下载结果到本地

```bash
# 在你的本地 Mac 上执行（不是在服务器上）
scp -r -P <端口> root@<服务器IP>:/root/Rotation-Quant/outputs/ ./outputs/
```

或者用优云智算的文件管理直接在浏览器下载。

关键文件：

- `outputs/stage_a/<run_id>/summary.md` — Stage A 权重结果
- `outputs/stage_b/<run_id>/summary.md` — Stage B 权重+激活结果
- `outputs/stage_c/<run_id>/summary.md` — Stage C attention/KV 结果

---

## 故障排查

### PyTorch 没有 CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
# False
```

解决：重装 CUDA 版 torch

```bash
pip uninstall torch -y
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu121
```

### 显存不足 (OOM)

单个实验的显存峰值约 20G，4090 24G 足够。如果出现 OOM：

```bash
# 检查显存使用
nvidia-smi

# 如果跑多个进程，kill 掉旧的
pkill -f python
```

### HuggingFace 下载失败

```text
ConnectionError: Failed to download...
```

解决：设置镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B
```

### 想在后台跑（断开 SSH 不中断）

```bash
nohup bash scripts/llama3-8b/run_all.sh > run.log 2>&1 &
# 查看进度
tail -f run.log
```

---

## 命令速查

```bash
# === 首次部署 ===
git clone https://github.com/zicheng1119/Rotation-Quant.git && cd Rotation-Quant
python3 -m venv rotationquant && source rotationquant/bin/activate
pip install -r requirements.txt
huggingface-cli login
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B

# === 每次进服务器 ===
cd Rotation-Quant && source rotationquant/bin/activate

# === 运行实验 ===
bash scripts/llama3-8b/run_all.sh

# === 查看结果 ===
bash scripts/llama3-8b/summarize_all.sh

# === 如果更新了代码 ===
git pull origin main
bash scripts/llama3-8b/run_all.sh
```
