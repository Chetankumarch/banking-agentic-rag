from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class MemoryBlocks:
    short_block: str
    long_blocks: List[str]


class ShortTermMemory:
    def __init__(self, max_tokens: int = 1200):
        self.turns: Deque[Turn] = deque()
        self.max_tokens = max_tokens

    def add(self, role: str, text: str):
        # Minimal redaction for long number-like strings
        safe_text = re.sub(r"\b\d{9,}\b", "[[REDACTED]]", text)
        self.turns.append(Turn(role, safe_text))
        self._trim_to_budget()

    def get_formatted(self, max_chars: int = 4000) -> str:
        """Return recent turns as readable block."""
        buf = []
        for t in self.turns:
            buf.append(f"{t.role.capitalize()}: {t.text}")
        s = "\n".join(buf)
        return s[:max_chars]

    def _trim_to_budget(self):
        """Trim oldest turns using char proxy for tokens."""
        def _len_chars() -> int:
            return len("\n".join(f"{t.role}:{t.text}" for t in self.turns))

        while self.turns and _len_chars() > self.max_tokens * 4:
            self.turns.popleft()


class LongTermMemory:
    def __init__(self, path: str = ".memory/history.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.summaries: List[str] = self._load()

    def _load(self) -> List[str]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.summaries, f, ensure_ascii=False, indent=2)

    def add_summary(self, text: str):
        self.summaries.append(text)
        self._save()

    def retrieve_relevant(self, query: str, limit: int = 2) -> List[str]:
        """Lexical overlap retrieval over summaries."""
        scores = []
        q_terms = set(query.lower().split())
        for s in self.summaries:
            score = sum(1 for w in s.lower().split() if w in q_terms)
            scores.append((score, s))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [s for score, s in scores[:limit] if score > 0]

    def all(self) -> List[str]:
        return list(self.summaries)


class MemoryManager:
    _instance: Optional["MemoryManager"] = None

    def __init__(
        self,
        short_tokens: int = 1200,
        history_path: str = ".memory/history.json",
        summarize_every: int = 5,
    ):
        self.short = ShortTermMemory(max_tokens=short_tokens)
        self.long = LongTermMemory(path=history_path)
        self.summarize_every = summarize_every
        self.turn_count = 0

    @classmethod
    def get(
        cls,
        short_tokens: int = 1200,
        history_path: str = ".memory/history.json",
        summarize_every: int = 5,
    ) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = MemoryManager(
                short_tokens=short_tokens,
                history_path=history_path,
                summarize_every=summarize_every,
            )
        return cls._instance

    def add_turns(self, user_text: str, assistant_text: str):
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)
        self.turn_count += 1

    def get_memory_blocks(self, query: str, max_summary: int = 2) -> MemoryBlocks:
        short_block = self.short.get_formatted()
        long_blocks = self.long.retrieve_relevant(query, limit=max_summary)
        return MemoryBlocks(short_block=short_block or "None", long_blocks=long_blocks)

    def summarize_if_needed(self, summarize_fn: Callable[[str], str]):
        if self.turn_count < self.summarize_every:
            return
        source = self.short.get_formatted(max_chars=4000)
        prompt = (
            "Summarize the following conversation for future recall in 5-8 concise bullet points.\n"
            "Focus on user intent, preferences, decisions, and specific policy conclusions.\n"
            "Avoid speculation; keep it factual and neutral.\n\n"
            f"Conversation:\n{source}\n"
        )
        summary = summarize_fn(prompt).strip()
        if summary:
            self.long.add_summary(summary)
        self.turn_count = 0
