from typing import TypedDict

from langgraph.graph import END, StateGraph
import ollama

from rag_docs import rag_retrieve

# Local model name to use with Ollama (ensure it's pulled)
OLLAMA_MODEL = "llama3.2"


class AgentState(TypedDict):
    """State flowing through the LangGraph."""

    question: str
    context: str
    answer: str


def call_llm(question: str, context: str) -> str:
    """
    Calls a local LLM via Ollama, grounding answers in retrieved context.
    """
    prompt = (
        "You are a helpful virtual assistant for a bank. Use the provided context "
        "when relevant. If something is not covered, say so instead of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Customer question:\n{question}"
    )
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def retrieve_and_answer_node(state: AgentState) -> AgentState:
    """
    LangGraph node: run RAG retrieval over web docs, then answer via LLM.
    """
    question = state["question"]
    context = rag_retrieve(question)
    answer = call_llm(question, context)
    state["context"] = context
    state["answer"] = answer
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

        state: AgentState = {"question": user_input, "context": "", "answer": ""}
        result = app.invoke(state)
        answer = result.get("answer") or "Sorry, I could not generate an answer."
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    print(
        "Note: Ensure Ollama is running and a model like "
        f"'{OLLAMA_MODEL}' is pulled."
    )
    chat_loop()
