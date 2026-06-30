# 训练路线

本项目后续不再使用本地弱模型 rollout 轨迹作为冷启动 SFT 数据。上一轮实验表明，自采失败轨迹会把模型推向过度保守、格式崩坏或低质量编辑。

当前训练路线：

```text
SWE-bench/SWE-smith-trajectories 转为 mini shell-only SFT
→ 冷启动 LoRA
→ 在 mini workflow 下 rollout / 测评 / 采样
→ 过滤并重构 mini workflow trajectory
→ 二阶段 LoRA / DPO / verifier
→ 再测试
```

## 数据源

第一阶段使用：

```text
SWE-bench/SWE-smith-trajectories
```

当前已下载本地 `tool` split：

```text
data/open_trajectories/swesmith/raw/data/tool-00000-of-00008.parquet
...
data/open_trajectories/swesmith/raw/data/tool-00007-of-00008.parquet
```

本地统计：

```text
tool split: 8 个 parquet，约 1.1GB，共 24100 条 trajectory
schema: messages, instance_id, resolved, model, traj_id, patch
```

原因：

```text
公开轨迹规模更大
数据来自成熟 SWE-agent 训练路线
已在 Qwen2.5-Coder 系列路径上验证过
比本地弱模型生成的失败轨迹更适合冷启动
```

## 保留内容

当前保留：

```text
Qwen2.5-Coder-7B-Instruct 基座模型
LLaMA-Factory 环境
SWE-bench Lite conda 环境缓存
repo cache
agent / bench / tools 源码
通用运行、分析、转换脚本
```

当前已清理：

```text
本地自采 trajectories
旧 batch run 记录
旧 A/B workspace
旧 recovery LoRA 权重
旧 recovery SFT JSONL
旧 recovery 训练配置和入口脚本
旧 structured tool workflow
旧 SWE-smith structured tool SFT 转换脚本
```

## Workflow 约束

旧 structured tool workflow 已废弃，当前主线切换到 mini-swe-agent 风格 shell-only workflow。当前保留的硬约束是：

```text
finish 不能在无 patch 时完成：已在 agent/mini_workflow.py 中强制检查 git_diff
只打印建议但不实际读写仓库的 no-op shell 命令会被拒绝
```

第一条用于避免模型逃避编辑。第二条用于避免模型在失败后反复输出“请确保修改”这类空转命令。

## 下一步

旧 structured tool LoRA 已删除。下一步需要重新转换开源 trajectory，使训练目标匹配 mini workflow 的单 shell 命令协议：

```text
输入：data/open_trajectories/swesmith/raw/data/tool-*.parquet
过滤：model=claude-3-7-sonnet-20250219, resolved=True, patch 非空
格式：step-level ShareGPT SFT
动作协议：输出 <mswea_bash_command>...</mswea_bash_command>
保护：过滤无 patch finish、超长 shell、测试文件编辑、明显 no-op 命令
```

旧 structured tool 转换结果仅作为历史参考，不再用于训练：

```text
过滤后轨迹：5535 条
SFT step 样本：158124 条
输出文件：data/llamafactory/swesmith_tool_sft.jsonl
统计文件：data/llamafactory/swesmith_tool_sft_stats.json
LLaMA-Factory 数据集名：swe_agent_sft（已废弃）
```

进入训练前，下一步是先实现 mini 协议转换器并做 SFT dry run：

```text
1. 将 SWE-smith tool trajectories 映射为 <mswea_bash_command> 单命令样本
2. 用 max_samples 小批量跑通 LLaMA-Factory 训练
3. 检查 loss 是否正常下降、是否 OOM
4. 再启动第一阶段 mini workflow 冷启动 LoRA
```

当前探针命令：

```bash
pip install -r requirements.txt
python scripts/inspect_swesmith_trajectories.py --limit 5
python scripts/inspect_swesmith_trajectories.py \
  --local-data-files 'data/open_trajectories/swesmith/raw/data/tool-*.parquet' \
  --limit 5
```

默认使用 streaming 读取，不会为了检查 schema 下载完整数据。输出位置：

```text
data/open_trajectories/swesmith/sample.jsonl
data/open_trajectories/swesmith/schema.json
```
