# 环境说明

本项目后续统一使用下面这个 conda 环境：

```bash
conda activate swe-agent-lf
```

环境路径：

```text
/root/miniconda3/envs/swe-agent-lf
```

## 已验证依赖

当前已验证的主要依赖如下：

```text
Python: 3.11.15
PyTorch: 2.6.0+cu124
Transformers: 4.57.6
LLaMA-Factory: 0.9.6.dev0
Datasets: 4.0.0
Accelerate: 1.11.0
PEFT: 0.18.1
TRL: 0.24.0
BitsAndBytes: 0.49.2
GPU: NVIDIA A100-PCIE-40GB
```

## 验证命令

检查 LLaMA-Factory 环境：

```bash
llamafactory-cli env
```

检查本地 Qwen 推理：

```bash
python scripts/smoke_test_qwen.py \
  --prompt "用一句话说明 Python 中 list 和 tuple 的区别。"
```

把 agent 轨迹转换成 LLaMA-Factory SFT 数据：

```bash
python scripts/convert_trajectories_to_sft.py \
  --input-glob 'data/trajectories/**/*.jsonl' \
  --output-file data/llamafactory/swe_agent_sft.jsonl \
  --dataset-name swe_agent_sft
```

## 环境复现

当前环境创建过程如下：

```bash
conda create -y -n swe-agent-lf python=3.11 pip
conda activate swe-agent-lf
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
cd third_party/LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
pip install bitsandbytes
```

我们把 LLaMA-Factory 放在独立 conda 环境中，是为了让训练依赖和系统默认环境解耦，避免后续升级训练依赖时影响已经跑通的 agent 推理链路。
