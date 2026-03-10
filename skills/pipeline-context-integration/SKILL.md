---
name: pipeline-context-integration
description: 将 agentic_prediction_pipeline.py 的分阶段 AgentMessage 结果组装为可直接注入外部智能体的上下文包（system/context/next-actions）。当用户提到“阶段消息整合”“上下文组装”“把 pipeline 输出喂给另一个智能体”“将中间状态转成 prompt/context”时务必触发此技能；即使用户没明确说 skill，也应在涉及多阶段预测流程上下文拼装时使用。
---

# Pipeline Context Integration

将项目中的阶段化输出（`AgentMessage`）稳定转换为可复用、可追踪、可裁剪的智能体上下文。

## 你要完成的目标

1. 读取 `run_full_pipeline` 的结果 JSON（通常是 `pipeline_results.json`）。
2. 从每个阶段提取：状态、核心 message、关键 data、suggestions、next_actions、agent_hints。
3. 生成两类输出：
   - **machine-friendly JSON**（便于程序注入到你部署的智能体）
   - **LLM-friendly Markdown**（便于直接放进 system / developer / context prompt）
4. 在失败阶段自动“截断后续流程”，并突出恢复动作。

## 输入要求

- 输入文件必须包含 `stages` 字段，且每个 stage 的内容符合 `AgentMessage.to_dict()` 结构。
- 支持 stage 键（常见）：
  - `exploration`
  - `preprocessing`
  - `feature_engineering`
  - `model_selection`
  - `training`
  - `evaluation`

## 统一执行步骤

按顺序执行：

1. 运行脚本：
   - `python skills/pipeline-context-integration/scripts/context_assembler.py --input <pipeline_results.json> --output-dir <dir>`
2. 检查输出文件：
   - `agent_context_bundle.json`
   - `agent_context_bundle.md`
3. 将 `agent_context_bundle.md` 作为外部智能体的上下文模板。
4. 将 `agent_context_bundle.json` 作为服务端结构化输入（可用于路由、策略选择、动作执行）。

## 输出结构（JSON）

脚本输出的 `agent_context_bundle.json` 含以下关键字段：

- `meta`: 生成时间、输入路径、阶段数量
- `summary`: `status`、`failed_stage`、`final_model`、关键指标
- `stage_contexts[]`:
  - `stage`
  - `status`
  - `message`
  - `important_data`
  - `suggestions`
  - `next_actions`
  - `agent_hints`
- `decision_context`:
  - `recommended_next_actions`（按 required 优先）
  - `risk_flags`（error/warning）

## 集成建议（部署到你的智能体）

### 模式 A：单轮注入

- 每次预测任务完成后，重新生成 context bundle。
- 在新任务轮次将 markdown 直接注入到系统上下文中。

### 模式 B：阶段驱动注入（推荐）

- 在每个阶段完成后更新一次 bundle。
- 智能体在每轮推理只加载“当前阶段+上一阶段+summary”，降低 token 开销。

### 模式 C：失败恢复路由

- 若 `summary.status == failed`，读取 `decision_context.recommended_next_actions` 触发恢复 agent（如数据修复、列对齐、重试训练）。

## 质量门槛

在交付前确认：

1. 输出 JSON 可被 `json.load` 成功解析。
2. Markdown 至少包含：
   - 全局摘要
   - 各阶段状态
   - 可执行 next actions
3. 当某阶段报错时，风险标记必须出现该阶段。

## 常见问题

- **Q: 用户没有 `summary.final_model` 怎么办？**  
  A: 正常回退为 `null`，仍可继续上下文组装。

- **Q: data 很大（如完整特征重要性表）怎么办？**  
  A: 使用 `--max-data-items` 控制每阶段保留条目数，默认只保留最有决策价值的片段。

## 最小命令示例

```bash
python skills/pipeline-context-integration/scripts/context_assembler.py \
  --input ./output/pipeline_results.json \
  --output-dir ./output/context
```
