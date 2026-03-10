---
name: ollama-agent-integration
description: 将 Ollama 本地推理接口接入 agent 项目（含 OpenAI 兼容接口与原生 /api/chat）。当用户提到“本地大模型、离线推理、Ollama、私有部署、把 LLM 接进 agent、替换 OpenAI 接口、模型路由、推理配置”时都应优先使用本技能，即使用户没有明确说“写 skill”。
---

# Ollama Agent Integration

参考风格：工程化落地优先（借鉴 LangChain / OpenAI SDK 的 provider 抽象思路）。

## 目标

把 agent 的模型调用层做成 **provider 可替换**：
- 默认支持 Ollama（本地/内网）。
- 兼容 OpenAI 风格调用，便于以后切换到其他模型服务。
- 通过配置文件注入参数，避免把 endpoint/token/model 写死在代码里。

## 什么时候使用这个技能

当用户出现以下需求时直接触发：
- “把 ollama 接到 agent”
- “离线部署 / 本地模型 / 内网推理”
- “把 OpenAI 接口换成私有 LLM”
- “想要可切换的 LLM provider 配置”

## 实施步骤

1. **识别现有调用点**
   - 搜索所有模型调用入口（如 `chat`, `generate`, `client.responses.create`）。
   - 找出硬编码的 model/base_url/api_key。

2. **抽象 provider 层**
   - 新增统一接口（例如 `LLMClient`）：`chat(messages, **kwargs)`。
   - 至少实现：
     - `OllamaNativeClient`（`/api/chat`）
     - `OpenAICompatibleClient`（`/v1/chat/completions`）

3. **配置驱动**
   - 从 `config.yaml` + `.env` 读取：
     - provider（`ollama` / `openai_compatible`）
     - base_url
     - model
     - timeout
     - temperature / top_p / max_tokens
   - 不在代码中写死密钥。

4. **最小可验证调用**
   - 先用一条 smoke prompt 验证连通性，再跑 agent 主流程。
   - 若失败，先打印请求参数摘要（脱敏）与 endpoint，再重试。

5. **回退策略**
   - 连接失败时给出清晰错误：
     - 服务未启动
     - 模型未拉取
     - endpoint 写错
     - 认证失败

## 推荐目录约定

```text
configs/
  ollama/
    ollama.env.example
    agent-llm.config.example.yaml
    docker-compose.ollama.example.yml
skills/
  ollama-agent-integration/
    SKILL.md
```

## 配置示例说明

优先读取顺序：环境变量 > YAML > 默认值。

必填：
- `LLM_PROVIDER`
- `LLM_BASE_URL`
- `LLM_MODEL`

可选：
- `LLM_API_KEY`
- `LLM_TIMEOUT`
- `LLM_TEMPERATURE`
- `LLM_TOP_P`
- `LLM_MAX_TOKENS`

## 联调检查清单

- `ollama serve` 正在运行。
- `ollama list` 可见目标模型。
- `curl $LLM_BASE_URL/api/tags` 返回 200。
- agent 配置已切换到 `provider=ollama`。
- 首次请求延迟可接受（冷启动较慢属正常）。

## 输出要求

交付内容应至少包含：
1. 可直接填写的 `.env.example`。
2. 可直接填写的 `config.example.yaml`。
3. （可选）本地启动 ollama 的 `docker-compose` 样例。
4. 一段可复制的 smoke test 命令。

## Smoke Test

### 原生 Ollama API

```bash
curl -sS http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "ping"}],
    "stream": false
  }'
```

### OpenAI 兼容模式（如果网关支持）

```bash
curl -sS http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${LLM_API_KEY:-dummy}" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "ping"}],
    "temperature": 0.2
  }'
```
