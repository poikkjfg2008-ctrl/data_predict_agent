# 在本项目接入 Ollama 作为 Agent 推理后端

## 1) 拷贝模板文件

把以下文件拷贝到你的运行目录（可按需改名）：

- `skills/ollama-agent-integration/assets/.env.example` → `.env`
- `skills/ollama-agent-integration/assets/ollama.yaml`
- `skills/ollama-agent-integration/assets/model-routing.yaml`
- `skills/ollama-agent-integration/assets/ollama_client.py`

## 2) 填写关键配置

至少填写：

- `OLLAMA_BASE_URL`
- `OLLAMA_PLANNER_MODEL`
- `OLLAMA_EXECUTOR_MODEL`
- `OLLAMA_FALLBACK_MODEL`

## 3) 启动 Ollama

本机安装方式（示例）：

```bash
ollama serve
ollama pull qwen2.5:7b
```

或使用容器：

```bash
docker compose -f skills/ollama-agent-integration/assets/docker-compose.ollama.yml up -d
```

## 4) 在 Agent Loop 中接入

示意：

```python
from ollama_client import OllamaClient

client = OllamaClient()

planner_result = client.chat(
    messages=[{"role": "user", "content": "请先给出任务规划"}],
    model="qwen2.5:7b",
)

executor_result = client.chat(
    messages=[{"role": "user", "content": "根据规划执行"}],
    model="qwen2.5:7b",
)
```

> 注意：如果你将模板文件复制到别的位置，请同步修改 import 路径。

## 5) 验证命令

```bash
ollama list
curl -s ${OLLAMA_BASE_URL}/api/tags
curl -s ${OLLAMA_BASE_URL}/api/generate -d '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
python -c "from ollama_client import OllamaClient;print(OllamaClient().generate('ping')['response'][:80])"
```
