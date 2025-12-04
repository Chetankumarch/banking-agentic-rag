import os
from typing import TypedDict, List, Optional

from langgraph.graph import END, StateGraph
import ollama

from rag_docs import RetrievedChunk, rag_retrieve, rag_retrieve_debug
from memory import MemoryManager, MemoryBlocks

# Toggle to print retrieval debug info (scores and snippets)
DEBUG_RETRIEVAL = True

# Memory configuration
ENABLE_MEMORY = True
DEBUG_MEMORY = True
MEM_SUMMARIZE_EVERY = int(os.getenv("BANK_RAG_SUMMARY_EVERY", "5"))
MEM_SHORT_TOKENS = int(os.getenv("BANK_RAG_SHORT_TOKENS", "1200"))
MEM_LTM_LIMIT = int(os.getenv("BANK_RAG_LTM_LIMIT", "2"))

# Local model name to use with Ollama (ensure it's pulled)
OLLAMA_MODEL = "llama3.2"

# Initialize memory manager
mm = MemoryManager.get()
mm.summarize_every = MEM_SUMMARIZE_EVERY
mm.short.max_tokens = MEM_SHORT_TOKENS


class AgentState(TypedDict):
    """State flowing through the LangGraph."""

    question: str
    context: str
    answer: str
    chunks: Optional[List[RetrievedChunk]]  # present only in debug mode


def call_llm_prompt(prompt: str, system: Optional[str] = None) -> str:
    """
    Calls the local LLM with a prepared prompt. Optional system message overrides default.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    else:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    return response["message"]["content"]


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
        context, chunks = rag_retrieve_debug(question, top_k=5)
        state["context"] = context
        state["chunks"] = chunks  # type: ignore[assignment]
    else:
        context = rag_retrieve(question)
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

    answer = call_llm_prompt(
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
            return call_llm_prompt(
                src_prompt,
                system="You summarize conversations for future recall.",
            )

        mm.summarize_if_needed(summarize_fn)

        if DEBUG_MEMORY:
            print("\n--- Memory Debug ---")
            print("Short-term chars:", len(mm.short.get_formatted()))
            print("Long-term summaries:", len(mm.long.summaries))
            if long_blocks:
                print("Selected LTM this turn:", len(long_blocks))

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
                preview = (chunk.text[:300] + "...") if len(chunk.text) > 300 else chunk.text
                print(
                    f"[{idx}] hybrid={chunk.hybrid_score:.2f} "
                    f"dense={chunk.dense_score:.2f} "
                    f"bm25={chunk.bm25_score:.2f} "
                    f"reranker={(chunk.reranker_score or 0.0):.2f}"
                )
                print(f"    \"{preview}\"\n")

        if DEBUG_MEMORY and ENABLE_MEMORY:
            print("--- Memory Debug (live) ---")
            print("STM chars:", len(mm.short.get_formatted()))
            print("LTM summaries:", len(mm.long.summaries))


if __name__ == "__main__":
    print(
        "Note: Ensure Ollama is running and a model like "
        f"'{OLLAMA_MODEL}' is pulled."
    )
    chat_loop()
