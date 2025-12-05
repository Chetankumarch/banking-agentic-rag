from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ollama


@dataclass
class LLMConfig:
    """
    Singleton wrapper for local LLM settings and chat calls.
    """

    model: str = os.getenv("BANK_RAG_MODEL", "llama3.2")
    temperature: float = float(os.getenv("BANK_RAG_TEMP", "0.0"))
    base_url: Optional[str] = os.getenv("BANK_RAG_OLLAMA_BASE_URL")

    _instance: "LLMConfig" = None  # type: ignore

    @classmethod
    def get(cls) -> "LLMConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def call_llm_prompt(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Call Ollama with an optional system prompt and user content.
        """
        messages: List[Dict[str, Any]] = []
        messages.append({"role": "system", "content": system or "You are a helpful assistant."})
        messages.append({"role": "user", "content": prompt})

        client_kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        response = ollama.chat(**client_kwargs)
        return response["message"]["content"]
