---
name: ollama-agent-integration
description: 将 Ollama 本地推理接口接入任意 Agent 项目（Python/Node 通用）的标准流程与配置模板。只要用户提到“本地大模型”“Ollama”“替换 OpenAI 兼容接口”“离线推理”“agent 模型路由/回退”，都应优先使用此 skill，直接产出可落地的接入步骤、配置文件和连通性验证命令。
compatibility:
  tools: ["shell", "python"]
  dependencies: ["ollama", "httpx|requests", "yaml"]
---

# Ollama Agent Integration Skill

按下面流程完成接入，避免“只改模型名但没有真正走本地推理”的伪接入。

## 1. 先确认输入信息（缺一不可）

向用户确认：
1. Agent 框架（自研 / LangChain / LlamaIndex / AutoGen / 其他）
2. Ollama 部署地址（默认 `http://127.0.0.1:11434`）
3. 目标模型（如 `qwen2.5:7b`, `llama3.1:8b`）
4. 是否需要回退模型（本地失败后切换云端）
5. 运行环境（本机 / Docker / 远程主机）

若信息不全，先产出模板并把待填项显式标记为 `TODO`。

## 2. 配置优先，不把参数写死在代码里

优先引导用户维护：
- `.env`（敏感值、端口、主机）
- `ollama.yaml`（模型参数、超时、重试）
- `model-routing.yaml`（任务到模型的路由规则）

不要把 `base_url`、`model`、`timeout` 写死在主业务逻辑。

## 3. 推荐接入层设计

实现一个轻量适配器（例如 `OllamaClient`），只暴露：
- `generate(prompt, system=None, options=None)`
- `chat(messages, options=None)`
- 统一错误类型（超时、连接失败、模型不存在）

关键要求：
- 所有调用走 `OLLAMA_BASE_URL`
- 每次请求可覆盖 `temperature/top_p/num_ctx`
- 对 5xx / 连接错误做指数退避重试
- 记录最小可观测日志（模型名、耗时、token 估计）

## 4. Agent 侧路由策略

默认三层：
1. `planner_model`：低温度，稳定规划
2. `executor_model`：中温度，执行任务
3. `fallback_model`：主模型失败时接管

路由顺序：
- 优先使用任务专属模型
- 不存在则回退默认执行模型
- 失败次数超阈值触发 fallback

## 5. 连通性和健康检查（必须执行）

至少执行：
1. `ollama list`
2. `curl $OLLAMA_BASE_URL/api/tags`
3. `curl $OLLAMA_BASE_URL/api/generate`（最小 prompt）
4. 在 agent 内跑一次端到端最小任务

若失败，先定位在“服务不可达 / 模型未拉取 / 参数不兼容 / agent 适配层 bug”哪个层级，再给修复建议。

## 6. 输出格式

每次都按下面结构输出，方便用户直接执行：

1. **接入方案摘要**（3-6 条）
2. **需要创建或修改的文件清单**
3. **完整配置模板**（可复制）
4. **最小可运行代码片段**（可直接粘贴）
5. **验证命令**（按顺序）
6. **常见故障排查**（至少 3 条）

## 7. 约束

- 不生成任何恶意用途内容。
- 不建议暴露未鉴权的公网 Ollama 端口。
- 用户未要求时，不自动引入重量级框架改造整个项目。
