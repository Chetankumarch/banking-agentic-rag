import os
import re
import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from banking_rag.configs.llm_config import LLMConfig
from banking_rag.configs.rag_config import RAGConfig
from banking_rag.configs.memory_config import MemoryConfig
from banking_rag.memory.manager import MemoryBlocks
from banking_rag.utils.logging_utils import log_turn, iso_timestamp

# Debug toggles and memory settings (default to previous behavior)
DEBUG_RETRIEVAL = os.getenv("BANK_RAG_DEBUG_RETRIEVAL", "1") == "1"
ENABLE_MEMORY = os.getenv("BANK_RAG_ENABLE_MEMORY", "1") == "1"
DEBUG_MEMORY = os.getenv("BANK_RAG_DEBUG_MEMORY", "1") == "1"
PARALLEL_MODE = os.getenv("BANK_RAG_PARALLEL_MODE", "1") == "1"
DEBUG_TIMING = os.getenv("BANK_RAG_DEBUG_TIMING", "0") == "1"
DEBUG_SENTIMENT = os.getenv("BANK_RAG_DEBUG_SENTIMENT", "0") == "1"
ENABLE_COMPLIANCE = os.getenv("BANK_RAG_ENABLE_COMPLIANCE_CHECK", "1") == "1"
DEBUG_COMPLIANCE = os.getenv("BANK_RAG_DEBUG_COMPLIANCE", "0") == "1"
ENABLE_LOGGING = os.getenv("BANK_RAG_ENABLE_LOGGING", "1") == "1"

# Singleton configs
llm_cfg = LLMConfig.get()
rag_cfg = RAGConfig.get()
mem_cfg = MemoryConfig.get()
mm = mem_cfg.manager()
MEM_LTM_LIMIT = mem_cfg.ltm_limit


class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph."""

    user_input: str
    query_type: str
    context: str
    policy_context: str
    conversation_memory: str
    session_summary: str
    answer: str
    grounded_answer: str
    compliance_status: str
    chunks: Optional[List]  # present only in debug mode; may be dataclasses or dicts
    sentiment: Optional[str]


def build_prompt(
    short_block: str,
    long_blocks: List[str],
    policy_context: str,
    user_question: str,
) -> str:
    """Assemble the final prompt with memory + policy context."""
    long_section = "\n\n".join(long_blocks) if long_blocks else "None"
    return (
        "### Conversation Memory (recent turns)\n"
        f"{short_block or 'None'}\n\n"
        "### Session Summary (selected history)\n"
        f"{long_section}\n\n"
        "### Retrieved Policy Context\n"
        f"{policy_context}\n\n"
        "### User Question\n"
        f"{user_question}\n\n"
        "### Instructions\n"
        "- Ground answers in the Retrieved Policy Context.\n"
        "- If a claim is not supported by the context, say so and suggest a clarifying question.\n"
        "- Keep answers concise and specific to overdrafts/fees unless asked otherwise."
    )


def classify_query(text: str) -> str:
    """Lightweight guardrail to detect transactional/PII vs policy questions."""
    lowered = text.lower()
    transactional_keywords = [
        "transfer",
        "send money",
        "pay",
        "move money",
        "deposit",
        "withdraw",
        "balance",
        "my account",
        "card number",
        "routing",
        "account number",
        "ssn",
        "social security",
        "wire",
        "payment",
    ]
    if any(k in lowered for k in transactional_keywords):
        return "transactional_or_pii"
    if re.search(r"\b\d{9,}\b", text):
        return "transactional_or_pii"
    # Default to policy/informational
    return "policy"


def query_guard(state: AgentState) -> AgentState:
    """Classify the incoming user query for routing."""
    start = time.perf_counter()
    user_text = state.get("user_input") or ""
    result = {"user_input": user_text, "query_type": classify_query(user_text)}
    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] query_guard: {elapsed:.1f} ms")
    return result


def policy_answer_node(state: AgentState) -> AgentState:
    """Placeholder retained for compatibility; not used in new parallel flow."""
    return state


def policy_retrieval_node(state: AgentState) -> AgentState:
    """Retrieve policy context (RAG) without answering."""
    start = time.perf_counter()
    if state.get("query_type") == "transactional_or_pii":
        return {}
    question = state.get("user_input") or ""
    try:
        if DEBUG_RETRIEVAL:
            debug_result = rag_cfg.get_context_with_debug(question, k=5)
            if isinstance(debug_result, dict):
                context = debug_result.get("context", "")
                chunks = debug_result.get("chunks", [])
            else:
                context, chunks = debug_result
            return {"policy_context": context, "chunks": chunks}
        else:
            context = rag_cfg.get_context(question, k=5)
            return {"policy_context": context, "chunks": None}
    finally:
        if DEBUG_TIMING:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"[timing] policy_retrieval_node: {elapsed:.1f} ms")


def memory_context_node(state: AgentState) -> AgentState:
    """Prepare memory blocks (STM/LTM) without answering."""
    start = time.perf_counter()
    if not ENABLE_MEMORY or state.get("query_type") == "transactional_or_pii":
        return {"conversation_memory": "None", "session_summary": "None"}

    question = state.get("user_input") or ""
    blocks: MemoryBlocks = mm.get_memory_blocks(question, max_summary=MEM_LTM_LIMIT)
    conversation_memory = blocks.short_block or "None"
    session_summary = "\n\n".join(blocks.long_blocks) if blocks.long_blocks else "None"

    result = {
        "conversation_memory": conversation_memory,
        "session_summary": session_summary,
    }
    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] memory_context_node: {elapsed:.1f} ms")
    return result


def merger_node(state: AgentState) -> AgentState:
    """Combine retrieval + memory context, call LLM, and update memory."""
    start = time.perf_counter()
    question = state.get("user_input") or ""
    policy_context = state.get("policy_context", "")
    short_block = state.get("conversation_memory", "None")
    long_block = state.get("session_summary", "None")
    long_blocks = [long_block] if long_block and long_block != "None" else []

    prompt = build_prompt(
        short_block=short_block,
        long_blocks=long_blocks,
        policy_context=policy_context,
        user_question=question,
    )

    answer = llm_cfg.call_llm_prompt(
        prompt,
        system=(
            "You are a banking policy assistant. Prefer retrieved policy text over memory "
            "if there is any conflict. Do not perform transactions. Do not request or process "
            "personal account information."
        ),
    )
    state["answer"] = answer

    if ENABLE_MEMORY and state.get("query_type") != "transactional_or_pii":
        mm.add_turns(question, answer)

        def summarize_fn(src_prompt: str) -> str:
            return llm_cfg.call_llm_prompt(
                src_prompt,
                system="You summarize conversations for future recall.",
            )

        mm.summarize_if_needed(summarize_fn)

    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] merger_node: {elapsed:.1f} ms")
    return state


def transaction_refusal_node(state: AgentState) -> AgentState:
    """Refuse transactional/PII requests safely."""
    start = time.perf_counter()
    user_text = state.get("user_input") or ""
    refusal = (
        "I’m a policy-only assistant and cannot perform transactions, check balances, "
        "or handle sensitive personal data. I can explain how the bank’s overdraft and fee "
        "policies work, if you’d like."
    )
    # Optional: tailor slightly with the LLM while keeping safety guardrails
    prompt = (
        "You are a cautious banking assistant. Given the user message, respond with a short, "
        "safe refusal making clear you cannot transact or handle PII, and offer policy help instead.\n\n"
        f"User message:\n{user_text}"
    )
    answer = llm_cfg.call_llm_prompt(
        prompt,
        system=(
            "Do NOT perform transactions. Do NOT process PII. Be brief and polite. "
            "Offer to answer policy questions instead."
        ),
    )
    state["answer"] = answer or refusal
    state["grounded_answer"] = state["answer"]
    state["compliance_status"] = "skipped"
    state["context"] = state.get("context", "")
    state["chunks"] = None
    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] transaction_refusal_node: {elapsed:.1f} ms")
    return state


def compliance_check_node(state: AgentState) -> AgentState:
    """Check grounding of the answer against policy context and adjust if needed."""
    start = time.perf_counter()
    if not ENABLE_COMPLIANCE:
        state["grounded_answer"] = state.get("answer", "")
        state["compliance_status"] = "skipped"
        return state

    question = state.get("user_input", "")
    policy_context = state.get("policy_context", "")
    draft_answer = state.get("answer", "")
    prompt = (
        "You are a strict banking policy compliance checker. You will receive a user question, "
        "retrieved policy context, and a draft answer.\n\n"
        "Tasks:\n"
        "- Check if each claim in the draft answer is supported by the policy context.\n"
        "- If something is unsupported or unclear, remove or soften it and say you don't see it in the context.\n"
        "- Do not invent details not present in the policy context.\n"
        "- Respond with this format:\n"
        "COMPLIANCE_STATUS: OK | UNCERTAIN | UNSUPPORTED\n"
        "GROUNDED_ANSWER:\n"
        "<rewritten answer>\n\n"
        f"User question:\n{question}\n\n"
        f"Policy context:\n{policy_context}\n\n"
        f"Draft answer:\n{draft_answer}\n"
    )

    response = llm_cfg.call_llm_prompt(
        prompt,
        system=(
            "Return only the compliance block. "
            "COMPLIANCE_STATUS must be one of: OK, UNCERTAIN, UNSUPPORTED."
        ),
    )
    grounded = draft_answer
    status = "uncertain"
    if response:
        lines = response.splitlines()
        for line in lines:
            if line.upper().startswith("COMPLIANCE_STATUS"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    status = parts[1].strip().upper()
            if line.strip().upper().startswith("GROUNDED_ANSWER"):
                # Everything after this line is the grounded answer
                idx = lines.index(line)
                grounded = "\n".join(lines[idx + 1 :]).strip() or draft_answer
                break

    state["grounded_answer"] = grounded
    state["compliance_status"] = status or "uncertain"

    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] compliance_check_node: {elapsed:.1f} ms")
    return state


def sentiment_node(state: AgentState) -> AgentState:
    """Classify sentiment after an answer is produced."""
    start = time.perf_counter()
    user_text = state.get("user_input", "")
    answer = state.get("grounded_answer") or state.get("answer", "")
    prompt = (
        "You are a sentiment classifier for banking support conversations.\n"
        "Classify the user's sentiment as one of: CALM, CONFUSED, FRUSTRATED, ANGRY, CURIOUS, GRATEFUL.\n"
        "Respond with exactly one label word.\n\n"
        f"User message:\n{user_text}\n\nAssistant answer:\n{answer}\n"
    )
    label = llm_cfg.call_llm_prompt(
        prompt,
        system="Return exactly one word: CALM, CONFUSED, FRUSTRATED, ANGRY, CURIOUS, or GRATEFUL.",
    )
    sentiment = (label or "").strip().split()[0] if label else ""
    state["sentiment"] = sentiment

    if DEBUG_SENTIMENT:
        print(f"[sentiment] user_sentiment={sentiment or 'UNKNOWN'}")

    if DEBUG_TIMING:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timing] sentiment_node: {elapsed:.1f} ms")
    return state


def build_graph():
    """Build a multi-node LangGraph pipeline with query guard + routing."""
    graph = StateGraph(AgentState)
    graph.add_node("query_guard", query_guard)
    graph.add_node("policy_retrieval", policy_retrieval_node)
    graph.add_node("memory_context", memory_context_node)
    graph.add_node("merger", merger_node)
    graph.add_node("compliance_check", compliance_check_node)
    graph.add_node("transaction_refusal", transaction_refusal_node)
    graph.add_node("sentiment", sentiment_node)

    # Entry
    graph.set_entry_point("query_guard")
    # Route based on query_type
    def route(state: AgentState) -> str:
        qt = state.get("query_type", "policy")
        if qt == "transactional_or_pii":
            return "transaction_refusal"
        if qt == "policy":
            return "policy"
        # default to policy path
        return "policy"

    if PARALLEL_MODE:
        graph.add_conditional_edges(
            "query_guard",
            route,
            {
                "transaction_refusal": "transaction_refusal",
                "policy": "policy_retrieval",
            },
        )
        # Parallel fan-out
        graph.add_edge("query_guard", "memory_context")
        graph.add_edge("policy_retrieval", "merger")
        graph.add_edge("memory_context", "merger")
    else:
        graph.add_conditional_edges(
            "query_guard",
            route,
            {
                "transaction_refusal": "transaction_refusal",
                "policy": "memory_context",
            },
        )
        # Sequential: guard -> memory -> retrieval -> merger
        graph.add_edge("memory_context", "policy_retrieval")
        graph.add_edge("policy_retrieval", "merger")
    graph.add_edge("merger", "compliance_check")
    graph.add_edge("compliance_check", "sentiment")
    graph.add_edge("transaction_refusal", "sentiment")
    graph.add_edge("sentiment", END)
    return graph.compile()


def chat_loop():
    """Simple terminal chat loop for banking questions."""
    app = build_graph()
    print("Open-source RAG banking assistant ready.")
    print("Type your banking questions (e.g., about overdrafts or account fees).")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        start_time = time.perf_counter()
        state: AgentState = {
            "user_input": user_input,
            "context": "",
            "policy_context": "",
            "conversation_memory": "None",
            "session_summary": "None",
            "answer": "",
            "chunks": None,
        }
        result = app.invoke(state)
        elapsed = (time.perf_counter() - start_time) * 1000
        # Choose grounded answer if compliance is enabled
        if ENABLE_COMPLIANCE:
            answer = result.get("grounded_answer") or result.get("answer") or "Sorry, I could not generate an answer."
        else:
            answer = result.get("answer") or "Sorry, I could not generate an answer."
        mode_str = "parallel" if PARALLEL_MODE else "sequential"
        print(f"\n[mode={mode_str} latency={elapsed:.1f} ms]")
        print(f"Agent: {answer}\n")

        if DEBUG_SENTIMENT and result.get("sentiment"):
            print(f"[sentiment] {result['sentiment']}")
        if DEBUG_COMPLIANCE and result.get("compliance_status"):
            print(f"[compliance] status={result['compliance_status']}")

        if ENABLE_LOGGING:
            try:
                record = {
                    "timestamp": iso_timestamp(),
                    "mode": mode_str,
                    "user_input": user_input,
                    "query_type": result.get("query_type"),
                    "policy_context_sample": (result.get("policy_context") or "")[:300],
                    "answer": result.get("answer"),
                    "grounded_answer": result.get("grounded_answer"),
                    "compliance_status": result.get("compliance_status"),
                    "sentiment": result.get("sentiment"),
                    "latency_ms": elapsed,
                }
                log_turn(record)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Failed to log turn: {exc}")

        if DEBUG_RETRIEVAL and "chunks" in result and result["chunks"] is not None:
            chunks = result["chunks"]  # type: ignore[index]
            print("--- Retrieval Debug Info ---")
            print(f"Selected top {len(chunks)} chunks:\n")
            for idx, chunk in enumerate(chunks, start=1):
                # Handle either dict or dataclass with attributes
                if isinstance(chunk, dict):
                    text = chunk.get("text", "")
                    hybrid = chunk.get("hybrid_score", 0.0)
                    dense = chunk.get("dense_score", 0.0)
                    bm25 = chunk.get("bm25_norm", chunk.get("bm25_score", 0.0))
                    rerank = chunk.get("reranker_score", 0.0) or 0.0
                else:
                    text = getattr(chunk, "text", "")
                    hybrid = getattr(chunk, "hybrid_score", 0.0)
                    dense = getattr(chunk, "dense_score", 0.0)
                    bm25 = getattr(chunk, "bm25_norm", None)
                    if bm25 is None:
                        bm25 = getattr(chunk, "bm25_score", 0.0)
                    rerank = getattr(chunk, "reranker_score", 0.0) or 0.0

                preview = (text[:300] + "...") if len(text) > 300 else text
                print(
                    f"[{idx}] hybrid={hybrid:.2f} dense={dense:.2f} bm25={bm25:.2f} reranker={rerank:.2f}"
                )
                print(f"    \"{preview}\"\n")

        if DEBUG_MEMORY and ENABLE_MEMORY:
            print("--- Memory Debug (live) ---")
            print("STM chars:", len(mm.short.get_formatted()))
            print("LTM summaries:", len(mm.long.summaries))


if __name__ == "__main__":
    print(
        "Note: Ensure Ollama is running and a model like "
        f"'{llm_cfg.model}' is pulled."
    )
    chat_loop()
