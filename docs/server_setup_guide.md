# 优云智算服务器部署指南

## 1. 创建实例
- 平台: 优云智算 (youyunzhisuan.com)
- 配置: RTX 4090 24G × 1, CUDA 12.x 镜像
- Python >= 3.11

## 2. 克隆代码
```bash
git clone https://github.com/zicheng1119/Rotation-Quant.git
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
# 一键运行全部 11 个实验
bash scripts/llama3-8b/run_all.sh
```

## 7. 汇总结果

```bash
# 自动找到最新的 run_id 并汇总
bash scripts/llama3-8b/summarize_all.sh
```
