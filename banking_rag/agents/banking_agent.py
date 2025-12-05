import os
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from banking_rag.configs.llm_config import LLMConfig
from banking_rag.configs.rag_config import RAGConfig
from banking_rag.configs.memory_config import MemoryConfig
from banking_rag.memory.manager import MemoryBlocks

# Debug toggles and memory settings (default to previous behavior)
DEBUG_RETRIEVAL = os.getenv("BANK_RAG_DEBUG_RETRIEVAL", "1") == "1"
ENABLE_MEMORY = os.getenv("BANK_RAG_ENABLE_MEMORY", "1") == "1"
DEBUG_MEMORY = os.getenv("BANK_RAG_DEBUG_MEMORY", "1") == "1"

# Singleton configs
llm_cfg = LLMConfig.get()
rag_cfg = RAGConfig.get()
mem_cfg = MemoryConfig.get()
mm = mem_cfg.manager()
MEM_LTM_LIMIT = mem_cfg.ltm_limit


class AgentState(TypedDict):
    """State flowing through the LangGraph."""

    question: str
    context: str
    answer: str
    chunks: Optional[List]  # present only in debug mode; may be dataclasses or dicts


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


def retrieve_and_answer_node(state: AgentState) -> AgentState:
    """
    LangGraph node: run RAG retrieval, incorporate memory, then answer via LLM.
    """
    question = state["question"]
    if DEBUG_RETRIEVAL:
        debug_result = rag_cfg.get_context_with_debug(question, k=5)
        # Accept either dict with context/chunks or tuple (context, chunks)
        if isinstance(debug_result, dict):
            context = debug_result.get("context", "")
            chunks = debug_result.get("chunks", [])
        else:
            context, chunks = debug_result
        state["context"] = context
        state["chunks"] = chunks  # type: ignore[assignment]
    else:
        context = rag_cfg.get_context(question, k=5)
        state["context"] = context
        state["chunks"] = None

    if ENABLE_MEMORY:
        blocks: MemoryBlocks = mm.get_memory_blocks(
            question, max_summary=MEM_LTM_LIMIT
        )
        short_block = blocks.short_block
        long_blocks = blocks.long_blocks
    else:
        short_block, long_blocks = "None", []

    prompt = build_prompt(
        short_block=short_block,
        long_blocks=long_blocks,
        policy_context=context,
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

    if ENABLE_MEMORY:
        mm.add_turns(question, answer)

        def summarize_fn(src_prompt: str) -> str:
            return llm_cfg.call_llm_prompt(
                src_prompt,
                system="You summarize conversations for future recall.",
            )

        mm.summarize_if_needed(summarize_fn)

    return state


def build_graph():
    """Build a single-node LangGraph pipeline using local components."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_and_answer", retrieve_and_answer_node)
    graph.set_entry_point("retrieve_and_answer")
    graph.add_edge("retrieve_and_answer", END)
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

        state: AgentState = {
            "question": user_input,
            "context": "",
            "answer": "",
            "chunks": None,
        }
        result = app.invoke(state)
        answer = result.get("answer") or "Sorry, I could not generate an answer."
        print(f"\nAgent: {answer}\n")

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
