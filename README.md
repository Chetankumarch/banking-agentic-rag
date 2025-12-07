# LangGraph Banking RAG (Local-Only)

Local terminal Q&A assistant for overdraft/fees. It scrapes public banking pages, builds embeddings + BM25, reranks with a cross-encoder, and answers via an Ollama-served LLM. Memory is local (short-term buffer + long-term JSON summaries). No cloud APIs.

## What’s inside
- LangGraph agent with nodes: query guard → policy retrieval + memory prep → merger (LLM) or refusal for transactional/PII.
- Retrieval: sentence-transformers (`BAAI/bge-small-en`) + BM25, hybrid scoring, cross-encoder reranker (`BAAI/bge-reranker-base`).
- LLM: Ollama (e.g., `llama3.2`).
- Memory: short-term rolling buffer + long-term summaries in `.memory/history.json`.
- All local: requests + BeautifulSoup, NumPy; no external services.

## Setup
- Install deps: `pip3 install -r requirements.txt`
- Install/run Ollama: https://ollama.com
- Pull a model: `ollama pull llama3.2`

## Run
- Default (parallel policy path, no per-node timing):  
  `python3 banking_agent.py`
- Toggle modes:  
  - Parallel (default): `BANK_RAG_PARALLEL_MODE=1 python3 banking_agent.py`  
  - Sequential: `BANK_RAG_PARALLEL_MODE=0 python3 banking_agent.py`
- Optional debug:  
  - Per-node timing: `BANK_RAG_DEBUG_TIMING=1 …`  
  - Retrieval debug (chunks/scores): `BANK_RAG_DEBUG_RETRIEVAL=1 …`  
  - Memory debug (sizes): `BANK_RAG_DEBUG_MEMORY=1 …`  
  - Disable memory: `BANK_RAG_ENABLE_MEMORY=0 …`

## Usage
- Ask: “What are overdraft fees?” or “How can I avoid overdrafts?”  
- Transactional/PII requests (e.g., transfers, account numbers) get a safe refusal.

## Notes
- Latency printed each turn: `[mode=parallel|sequential latency=XX.X ms]`.  
- Per-node timings print only if `BANK_RAG_DEBUG_TIMING=1`.  
- Long-term summaries persist in `.memory/history.json`; short-term memory resets per run.
