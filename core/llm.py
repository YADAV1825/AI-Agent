
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List
import requests

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Lightning AI API env vars
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "")
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

DEFAULT_MODEL = MODEL_NAME or "lightning-ai/gpt-oss-120b"
LIGHTNING_CHAT_URL = "https://lightning.ai/api/v1/chat/completions"


@dataclass
class OllamaClient:
    base_url: str
    model: str = DEFAULT_MODEL
    stop_check: Any = None  # callable returning True if stop requested

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, timeout: int = 120) -> str:
        if not self.base_url.strip():
            raise ValueError("Ollama API URL is empty")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        resp = requests.post(self.base_url, json=payload, timeout=timeout, stream=True)
        resp.raise_for_status()
        chunks = []
        for line in resp.iter_lines(decode_unicode=True):
            if self.stop_check and self.stop_check():
                resp.close()
                raise InterruptedError("Stop requested during model response")
            if line:
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    chunks.append(token)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        return "".join(chunks)


class LightningClient:
    def __init__(self, api_key=None, teamspace=None, model=None, stop_check=None):
        self.api_key = api_key or LIGHTNING_API_KEY
        self.teamspace = teamspace or LIGHTNING_TEAMSPACE
        self.model = model or MODEL_NAME
        self.stop_check = stop_check
        self.base_url = LIGHTNING_CHAT_URL

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1200, timeout: int = 120) -> str:
        """Calls Lightning AI chat completions and returns text."""
        if not self.api_key.strip():
            raise ValueError("Lightning API key is empty")
        if not self.teamspace.strip():
            raise ValueError("Lightning teamspace/project path is empty")

        # Use the correct auth format: Bearer {api_key}/{teamspace}
        headers = {
            "Authorization": f"Bearer {self.api_key}/{self.teamspace}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Handle both 'choices' and 'candidates' response formats
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            return msg.get("content", "") or ""

        if "candidates" in data and data["candidates"]:
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            if parts and isinstance(parts, list):
                return parts[0].get("text", "") or ""

        return json.dumps(data, ensure_ascii=False, indent=2)
