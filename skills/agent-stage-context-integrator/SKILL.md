---
name: agent-stage-context-integrator
description: 将 `agentic_prediction_pipeline.py` 各阶段（exploration/preprocessing/feature_engineering/model_selection/training/evaluation/prediction）的 AgentMessage 组装为可部署智能体上下文。只要用户提到“阶段消息整合”、“把 pipeline 中间结果喂给另一个 agent”、“做 context bundle/提示词拼装/阶段决策串联”，就应该优先使用本技能，即使用户没有明确说“skill”。
---

# Agent Stage Context Integrator

## 这个技能解决什么问题

把项目里分阶段产出的结构化消息（`status/message/data/suggestions/next_actions/agent_hints`）转成两类可直接投喂部署智能体的材料：

1. **机器可读 context bundle（JSON）**：适合 API 透传与会话编排。
2. **模型可读 prompt context（Markdown）**：适合拼接到 system/developer/user prompt 中。

## 何时触发

当用户出现以下意图时，直接触发本技能：

- “把完整流程消息串起来给另一个 agent 用”
- “我已经有 `pipeline_results.json`，想转成统一上下文”
- “按阶段失败回溯 / 决策重试 / 下一步 action 自动化”
- “想把训练阶段和评估阶段结论沉淀成部署侧上下文”

## 输入约定

默认输入文件是 `run_full_pipeline` 导出的 `pipeline_results.json`，核心结构：

- `stages`: 每个阶段的 AgentMessage 字典
- `summary`: 全局摘要（可选但推荐）

## 执行步骤

1. 读取 `pipeline_results.json`。
2. 调用脚本：

```bash
python skills/agent-stage-context-integrator/scripts/assemble_stage_context.py \
  --input output/pipeline_results.json \
  --output-json output/context/context_bundle.json \
  --output-md output/context/prompt_context.md \
  --run-id run-2026-03-10-001
```

3. 将输出接入你部署的智能体：
   - `context_bundle.json` → 编排层（router/orchestrator）
   - `prompt_context.md` → 模型上下文（system 或 developer 附加块）

## 输出解释

### 1) context_bundle.json

- `overall_status`: 按严重性聚合（error > needs_decision > warning > success）
- `context_blocks`: 按阶段排序后的结构化块
- `summary`: 保留原始流程摘要，供路由策略使用

### 2) prompt_context.md

- 压缩后的阶段时间线
- 每个阶段的 `status/message/suggestions/next_actions/agent_hints`
- 内置执行策略（先处理 error，再处理 needs_decision）

## 集成建议（部署侧）

- 将 `overall_status` 作为智能体第一跳路由条件：
  - `error`：进入修复/追问分支
  - `needs_decision`：进入用户澄清分支
  - 其他：进入训练完成后的预测/汇报分支
- 限制上下文体积：优先传递 `critical_data` 的摘要，不传整表数据。
- 如果你有多次运行记录，按 `run_id` 分桶存储并做回放追踪。

## 快速检查清单

- 输入 JSON 是否包含 `stages` 对象
- 阶段顺序是否正确（exploration → ... → prediction）
- `context_bundle.json` 与 `prompt_context.md` 是否同步生成
- 部署端是否使用 `overall_status` 做了分支处理
