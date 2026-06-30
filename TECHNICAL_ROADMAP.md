# SWE-Agent 技术路径

本文档作为本项目后续实现参考。项目目标是基于 `Qwen2.5-Coder-7B-Instruct` 开发一个面向 SWE-bench 的软件工程智能体，并逐步构建数据采集、监督微调、评测和自我改进闭环。

## 1. 项目目标

最终系统应具备以下能力：

- 接收 SWE-bench 样本中的 issue/problem statement。
- 自动拉取目标仓库并 checkout 到指定 base commit。
- 通过搜索、阅读、编辑代码和运行测试完成 bug 修复。
- 输出可应用的 patch。
- 在 SWE-bench 评测环境中验证 patch 是否解决问题且不引入回归。
- 记录完整 agent 轨迹，用于后续 SFT、DPO 或 RL 训练。

当前选用基座模型：

- `Qwen/Qwen2.5-Coder-7B-Instruct`

初期建议优先使用 LoRA/QLoRA 训练，先完成小规模闭环，再考虑更重的训练方案。

## 2. 总体阶段

### Phase 0: 项目初始化

目标：建立清晰的工程结构，保证后续 agent、benchmark、training 可以独立演进。

建议目录：

```text
swe_agent/
  agent/              # Agent 主循环、策略、状态管理
  tools/              # Shell、文件读写、搜索、测试运行等工具
  bench/              # SWE-bench 数据加载、环境构建、评测入口
  train/              # SFT、DPO、RL 训练脚本
  configs/            # 模型、agent、评测、训练配置
  scripts/            # 常用运行脚本
  data/               # 轨迹数据、训练数据、评测结果
  models/             # 本地模型或适配器
  docs/               # 设计文档
```

第一步先不要追求复杂框架，重点是让每个模块边界清楚。

### Phase 1: 本地模型推理

目标：让 `Qwen2.5-Coder-7B-Instruct` 可以在本地稳定推理。

需要实现：

- 模型下载脚本。
- 本地推理入口。
- chat template 适配。
- generation config 管理。
- 单轮和多轮对话测试。

推荐接口：

```python
class LLMClient:
    def generate(self, messages: list[dict], **kwargs) -> str:
        ...
```

初期可以直接使用 `transformers`，后续如果需要更高吞吐，可切换到 `vLLM`。

### Phase 2: 最小可用 Agent

目标：先做一个不训练也能运行的 SWE-Agent。

Agent 循环：

```text
读取 problem statement
  -> 观察仓库结构
  -> 搜索相关代码
  -> 阅读候选文件
  -> 制定修复方案
  -> 修改代码
  -> 运行测试
  -> 根据测试结果迭代
  -> 输出 git diff
```

核心模块：

- `AgentLoop`: 控制多轮推理和工具调用。
- `AgentState`: 保存当前任务、已读文件、命令历史、测试结果。
- `ToolExecutor`: 执行工具并返回观察结果。
- `PatchManager`: 管理代码修改和最终 diff。

初期工具集合：

- `list_files`
- `search_text`
- `read_file`
- `edit_file`
- `run_shell`
- `run_tests`
- `git_diff`

工具调用格式需要固定下来，因为后续训练数据会依赖该格式。

### Phase 3: 接入 SWE-bench

目标：能加载 SWE-bench 样本，并对单个样本完成环境构建和评测。

每个样本通常需要处理：

- `instance_id`
- `repo`
- `base_commit`
- `problem_statement`
- `test_patch`
- `FAIL_TO_PASS`
- `PASS_TO_PASS`

流程：

```text
加载样本
  -> clone 仓库
  -> checkout base commit
  -> 安装依赖
  -> 运行基线测试
  -> agent 生成 patch
  -> apply patch
  -> 运行 fail-to-pass/pass-to-pass tests
  -> 输出 resolved 结果
```

第一阶段建议只跑 SWE-bench Lite，减少环境构建和评测成本。

### Phase 4: 轨迹采集

目标：把 agent 解题过程完整保存下来，为训练提供数据。

推荐数据格式：

```json
{
  "instance_id": "repo__issue_id",
  "repo": "owner/name",
  "base_commit": "...",
  "problem_statement": "...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "name": "search_text", "content": "..."}
  ],
  "final_patch": "...",
  "resolved": true,
  "test_result": {
    "fail_to_pass": true,
    "pass_to_pass": true
  }
}
```

轨迹分三类保存：

- 成功轨迹：可直接用于 SFT。
- 失败轨迹：用于错误分析和负样本构造。
- 修正轨迹：失败样本经人工或强模型修复后，用作高价值训练数据。

### Phase 5: SFT 训练

目标：让模型学会 SWE-Agent 的工具调用、代码定位、修改和测试迭代模式。

训练策略：

- 首选 QLoRA，降低显存占用。
- 训练数据优先使用成功轨迹。
- 保留工具调用和工具返回结果。
- 不只训练最终 patch，也训练中间推理和操作步骤。

训练样本应覆盖：

- issue 理解。
- 代码搜索。
- 文件阅读。
- patch 生成。
- 测试失败后的修复迭代。
- 最终总结。

需要注意：

- 过长轨迹需要截断或分段。
- 工具输出要控制长度，避免上下文被无关日志污染。
- 训练集和评测集必须按 SWE-bench instance 隔离，避免泄漏。

### Phase 6: 评测闭环

目标：建立稳定、可复现的评测体系。

核心指标：

- `resolved_rate`: 最终解决率。
- `apply_success_rate`: patch 可应用比例。
- `fail_to_pass_rate`: 目标失败测试变为通过的比例。
- `pass_to_pass_rate`: 原有通过测试保持通过的比例。
- 平均工具调用次数。
- 平均 token 消耗。
- 平均运行时间。

每次模型或 agent 策略变更后，都需要保存：

- 配置文件。
- 模型或 adapter 路径。
- SWE-bench 子集。
- 每个 instance 的轨迹。
- 每个 instance 的最终 patch。
- 汇总指标。

### Phase 7: DPO / RL / 自改进

目标：在 SFT 基础上继续提升解决率。

可选路线：

1. Rejection Sampling
   - 同一题采样多条解法。
   - 只保留成功轨迹。
   - 周期性加入 SFT 数据。

2. DPO
   - 构造成对偏好数据。
   - chosen: 成功 patch 或更优轨迹。
   - rejected: 无效 patch、测试失败 patch、过度修改 patch。

3. RL
   - 使用测试结果作为 outcome reward。
   - patch 可应用给小奖励。
   - fail-to-pass 通过给大奖励。
   - pass-to-pass 保持通过给大奖励。
   - 语法错误、超时、无效工具调用给负奖励。

建议顺序：

```text
SFT -> Rejection Sampling -> DPO -> RL
```

不要一开始直接做 RL。先把 agent 框架、评测和数据闭环做稳，否则 RL 很难定位问题。

## 3. 推荐实现顺序

近期可以按以下里程碑推进：

### M1: 建立项目骨架

- 创建标准目录。
- 添加 Python 项目配置。
- 统一日志、配置和运行入口。

### M2: 跑通本地 Qwen 推理

- 完成模型下载。
- 完成本地 chat 推理。
- 支持从配置选择模型路径和推理参数。

### M3: 实现最小 Agent Loop

- 支持 problem statement 输入。
- 支持搜索、读文件、运行命令。
- 支持生成 patch。
- 先在人造小仓库上验证。

### M4: 接入单个 SWE-bench Lite 样本

- 加载一个 instance。
- 构建 repo 环境。
- 运行 agent。
- 应用 patch。
- 输出评测结果。

### M5: 批量评测 SWE-bench Lite

- 支持并发或队列执行。
- 保存每个样本轨迹和 patch。
- 输出汇总指标。

### M6: 构建 SFT 数据集

- 从成功轨迹生成 chat-format 数据。
- 清洗超长输出和无效轨迹。
- 划分 train/eval。

### M7: QLoRA SFT

- 使用 `Qwen2.5-Coder-7B-Instruct` 作为基座。
- 训练 agent 行为。
- 保存 adapter。
- 用同一评测脚本比较训练前后结果。

### M8: 数据迭代和偏好训练

- 分析失败样本。
- 生成 chosen/rejected 数据。
- 尝试 DPO 或更简单的 rejection sampling。

## 4. 关键设计原则

- 先评测，后训练：没有稳定评测，训练结果不可解释。
- 先小闭环，后大规模：先让 1 个样本跑通，再跑 10 个，再跑 Lite。
- 轨迹即资产：所有 agent 行为都要记录，成功和失败都有价值。
- 工具接口要稳定：训练数据依赖工具格式，频繁改格式会增加数据成本。
- 修改要可回滚：每个 instance 都应该在独立 workspace 中运行。
- 避免 prompt-only 复杂化：能用代码保证的流程，不要全部压到 prompt 里。

## 5. 初始技术栈建议

- Python 3.10+
- PyTorch
- transformers
- accelerate
- peft
- datasets
- bitsandbytes
- SWE-bench
- pytest
- gitpython 或 shell git
- pydantic / dataclasses
- yaml 配置

可选：

- vLLM：提升推理吞吐。
- wandb / tensorboard：记录训练和评测。
- Docker：隔离 SWE-bench 运行环境。

## 6. 下一步

建议下一步从 M1 开始：

1. 初始化项目结构。
2. 修正模型下载目录命名。
3. 写一个最小 `LLMClient`。
4. 写一个 `hello_agent.py`，验证本地 Qwen 可以基于输入生成修复计划。

完成这些后，再进入 SWE-bench 单样本接入。
