# Ollama 推理接口接入 Agent（实战版）

本文给你一套“可直接改造现有项目”的方式，把 Ollama 变成 Agent 的可选推理后端。

## 1. 接入架构

推荐分三层：

- **配置层**：环境变量 + YAML（便于多环境切换）
- **Provider 层**：统一 `chat(messages, **kwargs)` 接口
- **Agent 业务层**：只依赖 Provider，不关心具体是 OpenAI 还是 Ollama

## 2. 目录建议

```text
project/
├── configs/
│   └── ollama-agent/
│       ├── ollama.env.example
│       └── ollama-agent.config.yaml
├── skills/
│   └── ollama-agent-integration/
│       └── SKILL.md
└── docs/
    └── OLLAMA_AGENT_INTEGRATION.md
```

## 3. 配置文件

### 3.1 环境变量模板
复制 `configs/ollama-agent/ollama.env.example` 为 `.env`，填入你的地址与模型名。

### 3.2 YAML 模板
使用 `configs/ollama-agent/ollama-agent.config.yaml` 管理默认参数，并支持被环境变量覆盖。

## 4. Python 最小接入代码

```python
import os
from openai import OpenAI


def build_ollama_client() -> OpenAI:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")
    return OpenAI(base_url=base_url, api_key=api_key)


def ollama_chat(messages: list[dict]) -> str:
    client = build_ollama_client()
    response = client.chat.completions.create(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        messages=messages,
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "1024")),
    )
    return response.choices[0].message.content or ""
```

## 5. Agent 内路由建议

- `provider=ollama` 时，走本地接口
- `provider=openai` 时，走云端
- 本地失败且 `ENABLE_FALLBACK=true` 时，自动回退云端

## 6. 验证步骤

```bash
# 1) 服务可用性
curl http://127.0.0.1:11434/api/tags

# 2) 模型已安装（应看到你配置的模型）
ollama list

# 3) 本地推理冒烟
curl http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hello"}],"stream":false}'
```

## 7. 常见故障

- `connection refused`：Ollama 服务未启动或地址错误。
- `model not found`：执行 `ollama pull <model>` 下载模型。
- 返回慢：减小模型、缩短上下文、关闭不必要工具调用。
