# SWE-bench 接入说明

当前已经接入 SWE-bench Lite 的单样本加载、仓库准备、测试补丁应用、隔离环境安装、agent 运行和轻量 resolved 评测链路。

本阶段目标是先跑通真实仓库上的 agent 轨迹采集与本地轻量评测，不等同于官方 SWE-bench Docker 评测。官方评测仍建议作为最终成绩来源。

## 当前机器分工

AutoDL 当前不适合作为官方 SWE-bench Docker 评测机使用。因此项目按两段链路拆开：

```text
AutoDL 训练机：
  跑 Qwen agent
  生成 patch
  保存 trajectory
  执行 venv 轻量评测初筛
  导出 SWE-bench predictions
  做 LoRA / QLoRA 训练

Docker 评测机：
  接收 predictions JSONL
  运行官方 SWE-bench harness
  生成官方 resolved 结果
  把结果回传给 AutoDL
```

这个拆分能避免在 AutoDL 上折腾 Docker 权限，同时保证最终训练标签来自官方评测。

## 查看单个样本

```bash
python scripts/inspect_swebench_instance.py --index 0
```

默认加载：

```text
dataset: princeton-nlp/SWE-bench_Lite
split: test
```

输出内容包括：

```text
instance_id
repo
base_commit
problem_statement
FAIL_TO_PASS
PASS_TO_PASS
```

## 准备 Workspace

```bash
python scripts/prepare_swebench_instance.py --index 0
```

脚本会：

```text
clone GitHub 仓库
checkout 到 base_commit
应用 SWE-bench test_patch
把 test_patch 提交为本地基线 commit
创建独立 workspace
```

默认 workspace 目录：

```text
data/workspaces/swebench_lite/
```

这样做的目的是让 agent 能看到 SWE-bench 新增的失败测试，同时保持 `git diff` 只包含 agent 对源码的修复，不混入测试补丁。

## 安装隔离测试环境

每个 SWE-bench 样本使用一个独立 venv，默认目录：

```text
data/envs/swebench_lite/<instance_id>/
```

安装 index 0 的环境：

```bash
python scripts/install_swebench_env.py \
  --index 0 \
  --recreate-env \
  --env-python /root/miniconda3/bin/python \
  --timeout 1800
```

这里的 `--env-python /root/miniconda3/bin/python` 使用 Python 3.10 创建样本 venv。不要直接用项目主环境的 Python 3.11 跑老版本 astropy，否则会遇到 C 扩展编译兼容问题。

安装日志：

```text
data/envs/swebench_lite/<instance_id>.install.log
```

当前 fallback 策略会先尝试：

```text
pip install -e .[test]
```

失败后对老式 `setup.py` 项目使用：

```text
pip<24
setuptools<60
wheel<0.38
cython==0.29.22
oldest-supported-numpy
extension-helpers
python setup.py develop
pytest==7.4.4
hypothesis
pytest-astropy
pytest-astropy-header
pytest-xdist
```

## 运行单样本 Agent

```bash
python scripts/run_swebench_instance.py \
  --index 0 \
  --max-steps 12 \
  --agent-mode mini \
  --install-env \
  --env-python /root/miniconda3/bin/python \
  --install-timeout 1800
```

运行内容：

```text
加载 SWE-bench Lite 样本
准备 repo workspace
加载本地 Qwen2.5-Coder-7B-Instruct
安装或复用该样本隔离 venv
运行 mini-swe-agent 风格 shell-only workflow
运行轻量 evaluator
保存 trajectory
输出每一步工具调用和工具结果
```

`--agent-mode mini` 是当前默认模式。它参考 mini-swe-agent，把动作收敛为单一 shell 命令协议，并在提交前强制检查 `git diff`，避免无 patch 完成。旧的自由探索模式仍保留：

```bash
python scripts/run_swebench_instance.py \
  --index 0 \
  --agent-mode free
```

轨迹默认保存到：

```text
data/trajectories/swebench_lite/
```

## 批量运行 Agent

小批量采样使用：

```bash
python scripts/run_swebench_batch.py \
  --start-index 0 \
  --end-index 5 \
  --max-steps 12 \
  --agent-mode mini \
  --dataset-offline \
  --install-env \
  --reuse-existing-env \
  --reuse-existing-workspace \
  --reset-existing-workspace \
  --env-python /root/miniconda3/bin/python \
  --install-timeout 1800
```

`--end-index` 是左闭右开区间的结束位置，所以上面的命令会运行 index 0 到 4。

批量脚本会：

```text
启动时加载一次 Qwen 模型
逐个准备 workspace
逐个准备或复用 instance venv
逐个运行 SWE-bench workflow agent
逐个执行轻量 evaluator
逐个保存 trajectory
把每个样本的运行状态写入 summary JSONL
```

默认 summary 路径：

```text
data/runs/swebench_lite_batch_<timestamp>.jsonl
```

如果某个样本失败，脚本会把错误写入 summary，然后继续后面的样本。常用参数：

```text
--skip-existing        已经有 trajectory 的 instance 直接跳过
--prepare-only         只准备 workspace 和 venv，不加载模型、不运行 agent
--skip-eval            不做本地轻量评测
--recreate-env         删除并重建 instance venv
--reuse-existing-env   venv 已存在时跳过安装
--reuse-existing-workspace   workspace 已存在时不重新 clone
--reset-existing-workspace   复用 workspace 前先清理上一次 agent 生成的 patch
--dataset-offline      只使用本地 Hugging Face 数据集缓存
```

注意：对 astropy 这类有 C 扩展的项目，venv 里的 editable install 会指向 workspace 源码目录。如果删除并重建 workspace，却直接 `--reuse-existing-env`，可能会丢失已编译扩展。因此复用环境时，要么同时复用 workspace，要么不要加 `--reuse-existing-env`，让脚本重新安装依赖。

建议先用 `--prepare-only` 做一小段环境安装测试：

```bash
python scripts/run_swebench_batch.py \
  --start-index 0 \
  --end-index 3 \
  --install-env \
  --env-python /root/miniconda3/bin/python \
  --install-timeout 1800 \
  --prepare-only
```

确认 summary 中环境安装没有大面积失败后，再加载模型跑 agent。

批量跑完后可以直接分析失败类型：

```bash
python scripts/analyze_swebench_runs.py \
  data/runs/swebench_lite_batch_<timestamp>.jsonl
```

分析脚本会输出 resolved 数量、是否产生 patch、`patch_review` 拒绝次数、`syntax_check` 失败次数，以及 `repeated_bad_patch`、`syntax_error`、`regression_or_bad_patch` 等分类。

## 轻量 Evaluator

`run_swebench_instance.py` 默认会在 agent 运行结束后执行轻量 evaluator：

```text
读取 git diff
逐条运行 FAIL_TO_PASS 测试
逐条运行 PASS_TO_PASS 测试
写入 evaluation 字段
```

resolved 判断逻辑：

```text
resolved = patch 非空 && FAIL_TO_PASS 全通过 && PASS_TO_PASS 全通过
```

轨迹中会包含：

```json
{
  "evaluation": {
    "resolved": false,
    "patch": "...",
    "fail_to_pass": [],
    "pass_to_pass": [],
    "fail_to_pass_passed": false,
    "pass_to_pass_passed": false
  }
}
```

如果只想运行 agent，不执行轻量评测：

```bash
python scripts/run_swebench_instance.py \
  --index 0 \
  --max-steps 8 \
  --skip-eval
```

## 导出官方 Predictions

在 AutoDL 上跑完一批轨迹后，导出 SWE-bench 官方 predictions JSONL：

```bash
python scripts/export_swebench_predictions.py \
  --input-glob 'data/trajectories/swebench_lite/*.jsonl' \
  --output-file data/predictions/swebench_lite_predictions.jsonl \
  --model-name qwen2.5-coder-7b-swe-agent
```

导出的每一行格式：

```json
{
  "instance_id": "astropy__astropy-12907",
  "model_name_or_path": "qwen2.5-coder-7b-swe-agent",
  "model_patch": "diff --git ..."
}
```

默认会跳过空 patch。调试时如果想导出空 patch：

```bash
python scripts/export_swebench_predictions.py --allow-empty-patch
```

如果同一个 instance 有多条轨迹，默认只导出最新一条，避免官方评测输入里出现重复 `instance_id`。调试时可以关闭去重：

```bash
python scripts/export_swebench_predictions.py --dedupe none
```

把 `data/predictions/swebench_lite_predictions.jsonl` 拷贝到支持 Docker 的机器后，用官方 SWE-bench harness 评测。

## 导入官方评测结果

Docker 机器评测完成后，把官方结果目录或结果文件拷回 AutoDL，然后导入到轨迹：

```bash
python scripts/import_swebench_results.py \
  --trajectory-glob 'data/trajectories/swebench_lite/*.jsonl' \
  --results-path /path/to/official/results \
  --output-dir data/trajectories_with_official_eval
```

脚本会给匹配到的轨迹添加：

```json
{
  "official_evaluation": {
    "resolved": true,
    "source": "/path/to/official/results"
  }
}
```

如果确认要直接改原始轨迹文件，可以使用：

```bash
python scripts/import_swebench_results.py \
  --results-path /path/to/official/results \
  --in-place
```

## 生成正式 SFT 数据

正式训练时，只使用官方 Docker 评测为 resolved 的轨迹：

```bash
python scripts/convert_trajectories_to_sft.py \
  --input-glob 'data/trajectories_with_official_eval/*.jsonl' \
  --output-file data/llamafactory/swe_agent_sft_official.jsonl \
  --dataset-name swe_agent_sft_official \
  --require-official-resolved
```

这一步和前面的 dry run 不同：dry run 只验证训练链路；这里开始才是可信训练数据。

如果只想准备 workspace 和隔离环境，不加载模型：

```bash
python scripts/run_swebench_instance.py \
  --index 0 \
  --install-env \
  --env-python /root/miniconda3/bin/python \
  --install-timeout 1800 \
  --prepare-only
```

## 当前已验证样本

已验证 index 0：

```text
instance_id: astropy__astropy-12907
repo: astropy/astropy
base_commit: d16bfe05a744909de4b27f5875fe0d4ed41ce607
```

该样本已经完成：

```text
数据加载成功
仓库 clone 成功
base_commit checkout 成功
test_patch 应用并提交为本地基线成功
隔离 venv 创建成功
项目依赖和测试依赖安装成功
FAIL_TO_PASS 测试可收集并在未修复源码上失败
PASS_TO_PASS 抽样测试通过
agent 真实运行成功
trajectory 保存成功
evaluation 字段保存成功
```

但当前 agent 没有解决该问题。这个结果是正常的：本阶段验证的是“真实任务链路”，不是模型能力。

当前 index 0 已验证的关键命令：

```bash
data/envs/swebench_lite/astropy__astropy-12907/bin/python -m pytest --collect-only -q \
  astropy/modeling/tests/test_separable.py
```

可收集到 `compound_model6` 和 `compound_model9`。未修复源码下，下面两个 fail-to-pass 测试会失败：

```bash
data/envs/swebench_lite/astropy__astropy-12907/bin/python -m pytest -q \
  'astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]' \
  'astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]'
```

## 注意事项

- 运行命令前建议先激活统一环境：

```bash
conda activate swe-agent-lf
```

- 工具层会把样本 venv 的 `bin` 目录加到 `PATH` 前面，确保 agent 内部执行 `python` / `pip` 时优先使用该样本隔离环境。
- 当前没有做官方 Docker 评测。
- 轻量 evaluator 可以帮助我们先区分 patch 是否为空、目标测试是否通过、回归测试是否失败。
- 后续需要批量化 SWE-bench Lite 运行，并把成功 resolved 的轨迹清洗成 LoRA SFT 数据。
