---
name: ollama-agent-integration
description: 将 Ollama 本地推理接口接入任意 Agent 项目（CLI Agent、工作流 Agent、工具调用 Agent）。当用户提到 Ollama、llama3/qwen/deepseek 本地模型、OpenAI 兼容接口替换、私有化部署、模型路由、温度/上下文长度/流式输出配置时，务必优先使用这个技能，给出可直接落地的接入步骤、配置文件模板与最小可运行代码。
---

# Ollama Agent Integration

## 目标
把一个已有 Agent 项目的模型调用层从“写死某个云 API”改造成“可切换 Ollama 的推理后端”，并保持：
- 配置驱动（不要硬编码地址和模型名）
- 可观测（日志里能看到模型、耗时、token）
- 可降级（Ollama 失败时给出明确错误，或走备用后端）

## 触发信号
当用户出现下列需求，直接应用本技能：
- “把 Ollama 接到我的 Agent / 工作流里”
- “本地模型替换 OpenAI API”
- “离线推理 / 私有部署 / 内网大模型”
- “给我 Ollama 的配置模板（env/yaml/docker）”

## 标准工作流

### 1) 识别当前项目调用模式
先判断用户项目属于哪种：
1. **OpenAI SDK 风格**：`client.chat.completions.create(...)`
2. **HTTP 直连风格**：自己 `POST /v1/chat/completions`
3. **框架封装风格**：LangChain/LlamaIndex/自研 Provider

然后选择最小改造方案：
- 若是 OpenAI SDK 风格：优先走 Ollama 的 OpenAI 兼容接口
- 若是 HTTP 直连风格：统一 `base_url` 与 `api_key` 来源于配置
- 若是框架封装：补一个 `OllamaProvider` 适配层

### 2) 固化配置层（必须）
至少提供三类可编辑项：
- 连接信息：`OLLAMA_BASE_URL`、`OLLAMA_API_KEY`（可空）
- 生成参数：`MODEL_NAME`、`TEMPERATURE`、`MAX_TOKENS`、`TOP_P`
- 运行策略：`REQUEST_TIMEOUT`、`ENABLE_FALLBACK`、`FALLBACK_PROVIDER`

要求：
- 先读环境变量，再读 yaml/json，最后使用安全默认值
- 对关键字段做启动时校验，缺失直接报错

### 3) 建立 Provider 抽象
使用统一接口，避免业务代码绑定某家模型服务：

```python
class LLMProvider(Protocol):
    def chat(self, messages: list[dict], **kwargs) -> dict: ...
```

实现 `OllamaProvider`，并在工厂中按 `provider=ollama` 装配。

### 4) 接入错误处理与可观测性
必须包含：
- 连接失败（host 不可达）
- 模型不存在（提示 `ollama pull <model>`）
- 超时（建议调大 timeout 或减小上下文）

日志至少记录：
- provider/model
- latency_ms
- prompt_tokens/completion_tokens（若上游返回）
- request_id（便于排查）

### 5) 给出最小验证步骤
完成改造后，提供可复制命令：
1. 健康检查：`curl $OLLAMA_BASE_URL/api/tags`
2. 推理冒烟：发一个 1 轮对话
3. Agent 端到端：跑一次用户主流程

## 输出格式（默认）
对用户输出时，按以下结构组织：
1. 改造思路（1段）
2. 需要新增/修改的文件清单
3. 可直接粘贴的配置模板
4. 最小可运行代码片段
5. 验证命令与常见故障排查

## 代码约束
- 不要在 import 外包 try/catch。
- 不要把密钥写入仓库，统一放 `.env`。
- 能用配置解决的，不要写死在业务逻辑里。
- 优先保持对现有 Agent API 的向后兼容。

## 示例：OpenAI 兼容模式（Ollama）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"  # 占位值，某些 SDK 必填
)

resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "你好，做个自我介绍"}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
```

## 常见坑提醒
- `11434` 端口未暴露：先确认服务监听地址。
- 模型标签写错：以 `ollama list` 输出为准。
- 长上下文 OOM：先降 `num_ctx` 或切小模型。
- Windows/WSL 网络隔离：优先使用宿主机可达地址，不要盲写 `localhost`。
