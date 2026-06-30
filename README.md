# SWE Agent

这是一个面向 SWE-bench 的软件工程智能体实验项目。项目目标是让本地代码模型在受控 workflow 下完成真实仓库的问题定位、代码编辑、patch 生成和评测。

当前实现重点是核心 agent 链路，而不是提交大模型权重、训练产物或评测缓存。

```text
issue -> agent workflow -> tool actions -> code patch -> SWE-bench prediction
```

## 核心技术栈

- 模型：Qwen2.5-Coder-7B-Instruct 或兼容的本地代码模型
- 推理：Transformers / PyTorch
- 微调：LLaMA-Factory，LoRA SFT
- 数据：SWE-smith trajectories
- 任务：SWE-bench / SWE-bench Lite
- 隔离执行：Docker
- Agent 协议：XML action，主要包含 `bash`、`str_replace_editor`、`submit`

## 提交内容

建议开源仓库只包含核心代码、配置模板和复现实验脚本：

```text
agent/
  llm_client.py                  # 本地模型推理封装
  sweagent_xml_workflow.py        # 核心 XML workflow
  trajectory.py                   # 轨迹记录
  types.py                        # agent 数据结构

bench/
  swebench_lite.py                # SWE-bench Lite 数据读取
  predictions.py                  # trajectory -> predictions
  evaluator.py                    # 本地评估辅助
  test_commands.py                # 测试命令辅助逻辑

tools/
  executor.py                     # 工具执行抽象
  docker_executor.py              # Docker 执行器

scripts/
  run_swebench_shared_docker_batch.py
  run_official_eval_cached_specs.py
  export_swebench_predictions.py
  cache_swebench_test_specs.py
  cache_swebench_images.py
  build_swesmith_official_sft.py
  build_swesmith_stage2_key_actions.py
  train_*.sh

configs/
  qwen_local.yaml                 # 推理配置模板
  llamafactory_*.yaml             # 训练配置模板

docs/
  ENVIRONMENT.md
  SWEBENCH.md
  TRAINING.md

tests/
  test_*.py
```

以下内容不提交到 GitHub，已通过 `.gitignore` 排除：

- 基座模型、LoRA adapter、checkpoint
- SWE-bench Docker 镜像包
- rollout trajectory、predictions、评测报告
- repo cache、workspace、临时文件
- 训练中间数据和缓存
- `third_party/` 下的外部项目源码
- `configs/*.local.yaml` 本机真实路径配置

## 环境安装

建议使用独立虚拟环境：

```bash
conda create -n swe-agent python=3.11 -y
conda activate swe-agent
pip install -r requirements.txt
```

如果需要本地开发 LLaMA-Factory 或 SWE-bench，可以单独 clone 到 `third_party/` 后安装 editable 版本：

```bash
pip install -e third_party/LLaMA-Factory
pip install -e third_party/SWE-bench
```

`third_party/` 只是本地开发目录，不属于本仓库提交内容。

## 模型配置

公开配置模板位于：

```text
configs/qwen_local.yaml
```

示例：

```yaml
model:
  path: /path/to/Qwen2.5-Coder-7B-Instruct
  torch_dtype: auto
  device_map: auto

generation:
  max_new_tokens: 2048
  temperature: 0.2
  top_p: 0.95
  do_sample: false
```

本机真实路径可以复制到 `configs/qwen_local.local.yaml`，该文件不会被 Git 提交。

## 数据构造

从 SWE-smith trajectories 构造接近 SWE-agent-LM 路线的 SFT 数据：

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct

python scripts/build_swesmith_official_sft.py \
  --split xml \
  --max-tokens 16384 \
  --dataset-name swesmith_official_xml_resolved_16k
```

构造二阶段 key-action 数据：

```bash
python scripts/build_swesmith_stage2_key_actions.py
```

生成的数据默认写入 `data/`，不进入 Git。

## LoRA 训练

训练配置模板在 `configs/` 下。启动示例：

```bash
bash scripts/train_swesmith_official_xml_lora_sft_16k_r128.sh
```

训练产物默认写入 `saves/`，不进入 Git。

## Rollout

使用共享 Docker 环境运行 SWE-bench Lite rollout：

```bash
python -u scripts/run_swebench_shared_docker_batch.py \
  --start-index 0 \
  --end-index 300 \
  --max-steps 150 \
  --spec-cache data/cache/swebench/lite_test_specs_0_300.json \
  --adapter-path /path/to/lora-adapter \
  --workspace-root "${HOME}/.cache/swe-agent/workspaces/shared_docker_v2_0_300" \
  --trajectory-dir data/trajectories/shared_docker_v2_0_300 \
  --summary-file data/runs/shared_docker_v2_0_300.jsonl \
  --skip-existing
```

## 导出 Predictions

```bash
python scripts/export_swebench_predictions.py \
  --input-glob 'data/trajectories/shared_docker_v2_0_300/*.jsonl' \
  --output-file data/predictions/shared_docker_v2_0_300.predictions.jsonl \
  --model-name qwen2.5-coder-7b-swe-agent
```

## 本地评测

本地 official eval 需要提前准备对应 SWE-bench Docker 镜像：

```bash
python scripts/run_official_eval_cached_specs.py \
  --predictions-path data/predictions/shared_docker_v2_0_300.predictions.jsonl \
  --test-specs-path data/cache/swebench/lite_test_specs_0_300.json \
  --report-dir data/official_eval/local_eval \
  --run-id local-eval \
  --max-workers 2 \
  --timeout 1800 \
  --cache-level instance
```

## GitHub 开源建议

首次发布前建议检查：

```bash
git status --short
git check-ignore -v saves/ data/trajectories/ data/predictions/ third_party/ configs/qwen_local.local.yaml
```

只提交源码、配置模板、文档和测试：

```bash
git add README.md .gitignore requirements.txt agent bench tools scripts configs docs tests TECHNICAL_ROADMAP.md
git commit -m "Initial open-source SWE agent workflow"
```

创建远程仓库后推送：

```bash
git remote add origin git@github.com:<user>/<repo>.git
git branch -M main
git push -u origin main
```
