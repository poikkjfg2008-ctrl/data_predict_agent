---
name: agentic-stage-context
description: 将 data_predict_agent 项目的阶段化 AgentMessage（exploration/preprocessing/feature_engineering/model_selection/training/evaluation/prediction）组装成可直接注入部署智能体的上下文。用户提到“阶段消息”“pipeline_results.json”“给现有智能体喂 context”“把中间结果串成提示词”时应优先使用本技能，即便用户没有明确说“skill”。
---

# Agentic Stage Context Assembler

用于把 `agentic_prediction_pipeline.py` 产出的多阶段消息转成**稳定、可复用、可审计**的上下文包，便于集成到你部署好的智能体。

## 何时使用

当用户出现以下诉求时直接触发：
- 想把训练流程的阶段结果喂给另一个智能体
- 想把 `pipeline_results.json` 变成 prompt/context
- 想按阶段控制后续动作（例如失败短路、条件分支）
- 想把 `AgentLoop` 决策日志和阶段消息统一成一份输入

## 输入

优先使用以下任一输入：
1. `output/pipeline_results.json`（推荐）
2. Python 运行返回的 `results` 字典（结构同上）

预期包含：
- `stages`: 每阶段 `AgentMessage` 字典
- `summary`: 最终摘要
- `agent_loop`（可选）: 决策日志

## 输出

输出一个 JSON 上下文包，核心字段：
- `meta`: 版本/来源/生成时间
- `pipeline_status`: `success` 或 `failed`
- `summary`: 最终模型与关键指标
- `stage_contexts`: 按阶段展开的可读上下文块
- `agent_handoff`: 给下游智能体的执行提示（下一步建议、关键风险）

## 标准工作流

1. 读取 `pipeline_results.json`。
2. 逐阶段提取 `status/message/suggestions/next_actions/agent_hints`。
3. 生成“阶段简报 + 可执行动作”格式。
4. 若出现 `error` 或 `summary.status=failed`，显式写出 `failed_stage` 与恢复建议。
5. 输出 JSON 到目标路径，供部署智能体直接加载。

## 使用脚本

优先使用内置脚本：

```bash
python skills/agentic-stage-context/scripts/assemble_context.py \
  --input output/pipeline_results.json \
  --output output/agent_context.json
```

可选参数：
- `--max-suggestions N`：每阶段最多保留 N 条建议（默认 3）
- `--max-next-actions N`：每阶段最多保留 N 个下一步动作（默认 3）
- `--include-raw`：将阶段原始 payload 放入 `raw_stage_payload`（便于调试）

## 集成模式建议

### 模式A：一次性离线构建（推荐）
- 训练完成后执行脚本，生成 `agent_context.json`
- 部署智能体启动时加载该文件到系统上下文

### 模式B：在线增量拼接
- 每个阶段完成后只更新对应 `stage_contexts[i]`
- 下游智能体每轮仅消费“最新阶段 + 摘要”

### 模式C：失败优先策略
- 若 `pipeline_status=failed`，下游智能体进入“恢复模式”
- 优先执行 `agent_handoff.recovery_actions`

## 输出模板约束（给下游智能体）

当你把结果再次转述给用户时，保持以下结构：

```text
[Pipeline状态]
- status: ...
- failed_stage: ... (if any)

[阶段摘要]
1) exploration: ...
2) preprocessing: ...
...

[可执行动作]
- high_priority: ...
- optional: ...

[风险与回退]
- risk: ...
- fallback: ...
```

## 质量门槛

- 不丢失失败阶段信息
- 不删除 `next_actions.required=true` 的动作
- 不输出不可序列化对象
- 对未知字段保持向前兼容（保留在 `extras` 或 `raw_stage_payload`）
