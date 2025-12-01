# LangGraph Banking Assistant (Open-Source)

Small terminal-based banking assistant that uses LangGraph with only local, open-source components. It scrapes public banking documents (overdrafts and account fees), builds embeddings with sentence-transformers, and answers questions via an Ollama-served LLM.

## Setup
- Install Python deps: `pip install -r requirements.txt`
- Install and run Ollama: https://ollama.com
- Pull a model (example): `ollama pull llama3.2`

## Run
`python banking_agent.py`

Ask banking questions like “What are overdraft fees?” and the assistant will retrieve relevant web text, ground the answer in that context, and respond locally.
