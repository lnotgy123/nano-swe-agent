# Nano SWE Agent

Nano SWE Agent 是一个面向 SWE-bench 的轻量级软件工程智能体实验项目。项目围绕本地代码模型构建完整的 issue 修复链路：读取问题描述、定位仓库代码、执行工具、编辑文件、生成 patch，并导出可用于 SWE-bench harness 评测的 predictions。

当前实现主要参考 SWE-agent / mini-swe-agent 的工作方式，但保留了更小、更容易实验和改造的代码结构。

```text
SWE-bench instance
  -> XML agent workflow
  -> bash / str_replace_editor / submit
  -> repository patch
  -> SWE-bench prediction
  -> official harness evaluation
```

## 特性

- 支持本地 Qwen2.5-Coder-7B-Instruct 推理。
- 支持 LoRA adapter 挂载与 SWE-smith trajectory SFT 数据构造。
- 使用 XML action 协议驱动工具调用。
- 提供 Docker executor，用于隔离执行仓库命令。
- 支持 SWE-bench Lite rollout、trajectory 保存、patch 导出。
- 支持基于 SWE-bench official harness 的本地评测。

## 技术栈

- Base model：Qwen2.5-Coder-7B-Instruct
- Inference：Transformers / PyTorch
- Fine-tuning：LLaMA-Factory / LoRA SFT
- Training data：SWE-bench/SWE-smith-trajectories
- Benchmark：SWE-bench / SWE-bench Lite
- Runtime isolation：Docker
- Agent protocol：XML actions

主要工具动作：

- `bash`：执行 shell 命令、搜索代码、运行测试。
- `str_replace_editor`：查看、替换、插入、创建文件。
- `submit`：完成当前 patch。

## 文件结构

```text
agent/
  llm_client.py                  # 本地模型推理封装
  sweagent_xml_workflow.py        # 核心 XML workflow
  trajectory.py                   # trajectory 记录与保存
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
  run_swebench_shared_docker_batch.py   # 批量 rollout
  run_official_eval_cached_specs.py     # 本地 official eval
  export_swebench_predictions.py        # 导出 SWE-bench predictions
  cache_swebench_test_specs.py          # 缓存 SWE-bench TestSpec
  cache_swebench_images.py              # 缓存 SWE-bench Docker 镜像
  build_swesmith_official_sft.py        # 构造官方风格 SFT 数据
  build_swesmith_stage2_key_actions.py  # 构造二阶段 key-action 数据
  train_*.sh                            # LLaMA-Factory 训练入口

configs/
  qwen_local.yaml                       # 本地模型推理配置模板
  llamafactory_*.yaml                   # LLaMA-Factory 训练配置模板

docs/
  ENVIRONMENT.md                        # 环境说明
  SWEBENCH.md                           # SWE-bench 运行说明
  TRAINING.md                           # 训练说明

tests/
  test_*.py                             # 单元测试
```

## 安装

建议使用独立 Python 环境：

```bash
conda create -n nano-swe-agent python=3.11 -y
conda activate nano-swe-agent
pip install -r requirements.txt
```

如果需要使用本地开发版 LLaMA-Factory 或 SWE-bench，可以额外安装 editable clone：

```bash
pip install -e third_party/LLaMA-Factory
pip install -e third_party/SWE-bench
```

## 配置模型

编辑 `configs/qwen_local.yaml`：

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

训练和数据构造脚本也支持通过环境变量指定模型路径：

```bash
export QWEN_MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct
```

## 使用流程

### 1. 构造 SFT 数据

从 SWE-smith trajectories 构造 XML action 风格 SFT 数据：

```bash
python scripts/build_swesmith_official_sft.py \
  --split xml \
  --max-tokens 16384 \
  --dataset-name swesmith_official_xml_resolved_16k
```

构造二阶段 key-action 数据：

```bash
python scripts/build_swesmith_stage2_key_actions.py
```

### 2. LoRA 训练

使用 LLaMA-Factory 启动 LoRA SFT：

```bash
bash scripts/train_swesmith_official_xml_lora_sft_16k_r128.sh
```

二阶段训练入口：

```bash
bash scripts/train_swesmith_stage2_key_actions_lora_v2.sh
```

### 3. SWE-bench Rollout

使用共享 Docker 环境运行 SWE-bench Lite rollout：

```bash
python -u scripts/run_swebench_shared_docker_batch.py \
  --start-index 0 \
  --end-index 300 \
  --max-steps 150 \
  --spec-cache data/cache/swebench/lite_test_specs_0_300.json \
  --adapter-path /path/to/lora-adapter \
  --workspace-root "${HOME}/.cache/nano-swe-agent/workspaces/shared_docker_v2_0_300" \
  --trajectory-dir data/trajectories/shared_docker_v2_0_300 \
  --summary-file data/runs/shared_docker_v2_0_300.jsonl \
  --skip-existing
```

### 4. 导出 Predictions

```bash
python scripts/export_swebench_predictions.py \
  --input-glob 'data/trajectories/shared_docker_v2_0_300/*.jsonl' \
  --output-file data/predictions/shared_docker_v2_0_300.predictions.jsonl \
  --model-name qwen2.5-coder-7b-nano-swe-agent
```

### 5. 本地 Official Eval

本地 official eval 需要准备对应 SWE-bench Docker 镜像：

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

评测逻辑与 SWE-bench official harness 一致：

```text
apply model_patch -> run FAIL_TO_PASS -> run PASS_TO_PASS -> judge resolved
```

## 实验效果

当前公开仓库中的 workflow 已完成 SWE-bench Lite 链路打通，包括模型推理、工具调用、patch 生成、prediction 导出和本地 official eval。

在已有 exact-image 环境覆盖的本地子集上，当前 LoRA workflow 的实验结果为：

```text
evaluation subset: 63 SWE-bench Lite instances
completed: 62
resolved: 10
error / timeout: 1
resolved / total: 10 / 63 = 15.87%
```

该结果是本地 exact-image 子集评测，不等价于完整 SWE-bench Lite 300 条官方分数。完整评测需要准备全部 instance 对应的 SWE-bench eval images。

## 测试

```bash
pytest -q
```

当前单元测试覆盖 workflow guard、数据构造采样和测试命令解析等核心逻辑。

## License

MIT License
