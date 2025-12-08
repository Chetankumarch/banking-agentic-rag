from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


LOG_PATH = os.path.join("logs", "banking_rag_conversations.jsonl")


def _ensure_logs_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log_turn(record: Dict[str, Any]) -> None:
    """
    Append a single JSON record for a turn to a JSONL file.
    Errors are swallowed to avoid breaking the app.
    """
    try:
        _ensure_logs_dir()
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            json_record = json.dumps(record, ensure_ascii=False)
            f.write(json_record + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to log turn: {exc}")


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
