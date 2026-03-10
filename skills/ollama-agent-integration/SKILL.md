---
name: ollama-agent-integration
description: 将 Ollama 本地推理接口接入任意 Agent（ReAct / Tool-calling / Planner-Executor）工作流。只要用户提到“接入 Ollama”、“本地模型做 agent”、“替换 OpenAI API 为 Ollama”、“给 agent 配置本地 LLM”，都应优先使用本技能，即使用户没有明确说“skill”。
compatibility:
  - requires: [bash, curl]
  - optional: [python>=3.10]
---

# Ollama Agent Integration Skill

## 目标
把 Agent 的模型调用层抽象成 `LLMAdapter`，并通过配置把 provider 切到 Ollama，避免业务逻辑和供应商 SDK 强耦合。

## 何时使用
当用户需要以下任一事项时触发：
- 把现有 agent 从 OpenAI/Anthropic 切换到 Ollama
- 在本地离线环境运行 agent
- 为不同任务设置不同 Ollama 模型（如规划模型/执行模型分离）
- 需要可填充的配置模板（endpoint/model/temperature/auth）

## 工作流程
1. **识别 agent 框架**：先确认项目是自研 loop、LangChain、LlamaIndex 还是其他。
2. **建立统一接口**：要求项目具备统一方法：
   - `chat(messages, tools=None, stream=False)`
   - `embed(texts)`（可选）
3. **落配置，不写死**：把下列参数放入配置文件，不硬编码：
   - `base_url`、`model`、`timeout`、`temperature`、`max_tokens`
   - `auth`（可选，反向代理场景）
4. **连通性检查**：先测 `/api/tags`，再做最小 chat 调用。
5. **Agent 回归测试**：至少验证
   - 无工具推理
   - 需要工具调用的任务
   - 长上下文任务（验证 token/超时设置）

## 最小接入契约

### HTTP 接口（推荐）
- `POST /api/chat`
- `POST /api/generate`
- `GET /api/tags`

### Python 伪代码
```python
class OllamaAdapter:
    def __init__(self, cfg):
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.timeout = cfg.timeout

    def chat(self, messages, tools=None, stream=False):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": cfg.temperature,
                "num_ctx": cfg.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        return http_post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
```

## 故障排查优先级
1. **连接失败**：检查 `OLLAMA_BASE_URL`、端口映射、容器网络。
2. **模型不存在**：执行 `ollama pull <model>`，并确认拼写一致。
3. **响应慢/超时**：调大 timeout，降低 `num_ctx`，或换小模型。
4. **工具调用不稳定**：先禁用 streaming，固定低 temperature（如 0.1-0.3）。

## 输出要求
交付内容应至少包含：
- 一份接入说明（你改了什么，在哪里改）
- 一份可直接填写的配置模板
- 一套健康检查命令（curl）
- 一个回滚方案（如何切回原 provider）
