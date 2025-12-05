from __future__ import annotations

import os
from dataclasses import dataclass

from banking_rag.memory.manager import MemoryManager


@dataclass
class MemoryConfig:
    """
    Singleton wrapper for memory settings and manager access.
    """

    short_tokens: int = int(os.getenv("BANK_RAG_SHORT_TOKENS", "1200"))
    summarize_every: int = int(os.getenv("BANK_RAG_SUMMARY_EVERY", "5"))
    ltm_limit: int = int(os.getenv("BANK_RAG_LTM_LIMIT", "2"))
    history_path: str = ".memory/history.json"

    _instance: "MemoryConfig" = None  # type: ignore

    @classmethod
    def get(cls) -> "MemoryConfig":
        if cls._instance is None:
            cfg = cls()
            MemoryManager.get(
                short_tokens=cfg.short_tokens,
                history_path=cfg.history_path,
                summarize_every=cfg.summarize_every,
            )
            cls._instance = cfg
        return cls._instance

    def manager(self) -> MemoryManager:
        return MemoryManager.get(
            short_tokens=self.short_tokens,
            history_path=self.history_path,
            summarize_every=self.summarize_every,
        )
