from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# Use a small but capable open-source embedding model
_EMBED_MODEL_NAME = "BAAI/bge-small-en"
_embedder = SentenceTransformer(_EMBED_MODEL_NAME)

# Banking/overdraft pages to retrieve from
RAG_URLS: List[str] = [
    "https://www.consumerfinance.gov/consumer-tools/bank-accounts/know-your-overdraft-options/",
    "https://www.fdic.gov/consumer-resource-center/2021-12/overdraft-and-account-fees",
    "https://www.bankofamerica.com/deposits/overdrafts-and-overdraft-protection/",
    "https://www.machiassavings.bank/msb-general-service-fee-schedule/",
]


def fetch_url_text(url: str) -> str:
    """
    Fetches a URL and returns visible text as a single normalized string.
    Removes script/style/noscript tags before extracting text.
    """
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """
    Naively chunk long text, preferring to cut on sentence boundaries.
    """
    chunks: List[str] = []
    cursor = 0
    while cursor < len(text):
        window_end = min(cursor + max_chars, len(text))
        window = text[cursor:window_end]
        cut = window.rfind(".")
        if cut == -1 or cut < max_chars * 0.5:
            cut = len(window)
        else:
            cut = cut + 1  # include the period
        chunk = text[cursor : cursor + cut].strip()
        if chunk:
            chunks.append(chunk)
        cursor = cursor + cut
    return chunks


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed texts with sentence-transformers and L2-normalize rows for cosine.
    """
    return _embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)


@lru_cache
def _build_corpus_and_index() -> Tuple[List[Dict[str, str]], np.ndarray]:
    """
    Download URLs, chunk them, and embed all chunks.
    Returns:
        corpus: list of dicts {id, url, text}
        embeddings: np.ndarray [num_chunks, dim]
    """
    corpus: List[Dict[str, str]] = []
    for url in RAG_URLS:
        try:
            text = fetch_url_text(url)
            for idx, chunk in enumerate(chunk_text(text)):
                corpus.append({"id": f"{url}#chunk-{idx}", "url": url, "text": chunk})
        except Exception as exc:  # noqa: BLE001 - demo-friendly warning
            print(f"[warn] Failed to process {url}: {exc}")
            continue

    if not corpus:
        return corpus, np.zeros((0, 1), dtype=np.float32)

    embeddings = _embed_texts([doc["text"] for doc in corpus])
    return corpus, embeddings


def rag_retrieve(query: str, top_k: int = 5) -> str:
    """
    Retrieve most relevant chunks using cosine similarity over embeddings.
    Returns concatenated context text.
    """
    corpus, embeddings = _build_corpus_and_index()
    if not corpus or embeddings.shape[0] == 0:
        return ""

    q_vec = _embed_texts([query])[0]
    sims = embeddings @ q_vec
    top_indices = np.argsort(-sims)[:top_k]
    chunks = [corpus[i]["text"] for i in top_indices]
    return "\n\n".join(chunks)
