import os
import time
from typing import Any, Dict, List, Optional

import requests


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
        self.backoff = int(os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "2"))

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or os.getenv("OLLAMA_EXECUTOR_MODEL", "qwen2.5:7b"),
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options or {},
        }
        return self._post_with_retry("/api/generate", payload)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": model or os.getenv("OLLAMA_EXECUTOR_MODEL", "qwen2.5:7b"),
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        return self._post_with_retry("/api/chat", payload)

    def _post_with_retry(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        last_err = None
        start = time.time()

        for i in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                elapsed = time.time() - start
                print(f"[ollama] model={payload.get('model')} endpoint={endpoint} elapsed={elapsed:.2f}s")
                return data
            except requests.RequestException as err:
                last_err = err
                if i >= self.max_retries:
                    break
                time.sleep(self.backoff * (2**i))

        raise RuntimeError(f"Ollama request failed after retries: {last_err}")
