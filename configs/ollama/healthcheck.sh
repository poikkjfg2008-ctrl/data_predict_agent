#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${OLLAMA_BASE_URL:?OLLAMA_BASE_URL is required}"
: "${OLLAMA_CHAT_MODEL:?OLLAMA_CHAT_MODEL is required}"

echo "[1/2] GET ${OLLAMA_BASE_URL}/api/tags"
curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/tmp/ollama_tags.json
jq '.models | length' /tmp/ollama_tags.json >/dev/null || true

echo "[2/2] POST ${OLLAMA_BASE_URL}/api/chat"
curl -fsS "${OLLAMA_BASE_URL}/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${OLLAMA_CHAT_MODEL}\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK only\"}]}" \
  >/tmp/ollama_chat.json

echo "Healthcheck passed."
cat /tmp/ollama_chat.json
