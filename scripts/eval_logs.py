import json
import os
from collections import Counter, defaultdict
from statistics import mean

LOG_PATH = os.path.join("logs", "banking_rag_conversations.jsonl")


def read_logs(path: str):
    if not os.path.exists(path):
        print(f"No log file found at {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Skipping malformed line: {exc}")
                continue
    return records


def summarize(records):
    total = len(records)
    print(f"Total turns: {total}")
    if total == 0:
        return

    qtype_counter = Counter(r.get("query_type", "unknown") for r in records)
    print("Query types:", dict(qtype_counter))

    compliance_counter = Counter(r.get("compliance_status", "unknown") for r in records)
    print("Compliance status:", dict(compliance_counter))

    sentiment_counter = Counter(r.get("sentiment", "unknown") for r in records)
    print("Sentiment:", dict(sentiment_counter))

    # Average latency by mode and query_type
    lat_by_mode = defaultdict(list)
    lat_by_qtype = defaultdict(list)
    for r in records:
        lat = r.get("latency_ms")
        if lat is None:
            continue
        lat_by_mode[r.get("mode", "unknown")].append(lat)
        lat_by_qtype[r.get("query_type", "unknown")].append(lat)

    if lat_by_mode:
        print("Avg latency by mode (ms):", {k: round(mean(v), 1) for k, v in lat_by_mode.items()})
    if lat_by_qtype:
        print("Avg latency by query_type (ms):", {k: round(mean(v), 1) for k, v in lat_by_qtype.items()})

    # Samples for compliance != ok
    bad_compliance = [r for r in records if str(r.get("compliance_status", "")).upper() not in ("OK", "SKIPPED")]
    if bad_compliance:
        print("\nSample records with compliance issues:")
        for r in bad_compliance[:3]:
            print(json.dumps(r, ensure_ascii=False)[:500])

    # Samples for negative sentiment
    neg_sent = [r for r in records if str(r.get("sentiment", "")).upper() in ("FRUSTRATED", "ANGRY")]
    if neg_sent:
        print("\nSample records with negative sentiment:")
        for r in neg_sent[:3]:
            print(json.dumps(r, ensure_ascii=False)[:500])


if __name__ == "__main__":
    recs = read_logs(LOG_PATH)
    summarize(recs)
