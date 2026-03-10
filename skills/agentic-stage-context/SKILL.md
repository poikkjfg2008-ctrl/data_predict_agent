---
name: agentic-stage-context
description: 将 data_predict_agent 项目的分阶段 AgentMessage 结果自动组装成可直接注入智能体上下文的结构化摘要。只要用户提到“阶段消息”“pipeline中间结果”“把训练流程塞进agent context”“多阶段决策日志”或想把 agentic_prediction_pipeline.py 输出接入已部署智能体，就应触发本技能。
---

# Agentic Stage Context

用于把 `agentic_prediction_pipeline.py` 的全流程阶段消息（exploration/preprocessing/feature_engineering/model_selection/model_training/evaluation/prediction）转换成稳定、可控、低噪音的智能体上下文。

## 何时使用

在以下请求里优先使用本技能：

- 用户希望把训练流程的中间结果接入到现有智能体。
- 用户要“保留每个阶段的建议/下一步动作”，用于 agent loop。
- 用户需要把 `run_full_pipeline` 的返回值变成 prompt/context。
- 用户想把失败阶段、告警阶段单独高亮，做自动重试与路由。

## 输入约定

优先支持两种输入：

1. **完整 pipeline 输出 JSON**（`run_full_pipeline` 返回结构）
2. **阶段消息数组/字典**（每条包含 `stage/status/message/data/suggestions/next_actions/agent_hints`）

## 工作流

1. 读取阶段结果，按固定顺序排序。
2. 生成轻量摘要：每阶段一句结论 + 风险标签。
3. 提取关键决策信号：
   - `status`
   - `suggestions`（限制条数）
   - `next_actions`
   - `agent_hints`
4. 生成两个输出：
   - 机器可消费 JSON（给上游编排器）
   - 人类可读 Markdown（给 prompt/context 注入）
5. 若存在失败阶段，自动把失败原因和可执行恢复动作放到 `recovery_plan`。

## 脚本

### `scripts/run_and_capture_messages.py`

用途：直接运行 `AgenticPredictionPipeline`，并把全阶段消息持久化为标准 JSON 文件。

示例：

```bash
python skills/agentic-stage-context/scripts/run_and_capture_messages.py \
  --data-file ./test_data/train_data.csv \
  --output-json ./artifacts/pipeline_results.json \
  --output-dir ./artifacts/pipeline_output
```

### `scripts/assemble_context.py`

用途：把阶段消息转成“部署智能体可直接注入”的 context 包。

示例：

```bash
python skills/agentic-stage-context/scripts/assemble_context.py \
  --input ./artifacts/pipeline_results.json \
  --context-json ./artifacts/context_package.json \
  --context-md ./artifacts/context_package.md
```

## 输出格式

`context_json` 核心结构：

- `pipeline_status`: overall 状态
- `stage_order`: 标准阶段顺序
- `stage_summaries`: 每阶段摘要
- `critical_findings`: 错误/告警列表
- `recommended_actions`: 跨阶段聚合动作
- `recovery_plan`: 失败恢复建议
- `agent_loop_brief`: 可直接注入系统提示词的简版文本

## 集成建议

- 在你的部署智能体中，把 `agent_loop_brief` 放进 system 或 tool context。
- 把 `recommended_actions` 映射到工具调用策略（例如：重跑预处理、切换模型偏好）。
- 当 `pipeline_status=failed` 时，优先执行 `recovery_plan`，不要直接继续预测。
