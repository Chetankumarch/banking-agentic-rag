from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from banking_rag.retrieval.hybrid_retrievers import (
    RetrievedChunk,
    CrossEncoderReranker,
    EnsembleRetriever,
    HybridCompressionRetriever,
)

# Banking/overdraft pages to retrieve from
RAG_URLS: List[str] = [
    "https://www.consumerfinance.gov/consumer-tools/bank-accounts/know-your-overdraft-options/",
    "https://www.fdic.gov/consumer-resource-center/2021-12/overdraft-and-account-fees",
    "https://www.bankofamerica.com/deposits/overdrafts-and-overdraft-protection/",
    "https://www.machiassavings.bank/msb-general-service-fee-schedule/",
]

_retriever: HybridCompressionRetriever | None = None
_embed_model: SentenceTransformer | None = None


def fetch_url_text(url: str) -> str:
    """Fetch a URL and return visible text as a normalized string."""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Naively chunk long text, preferring to cut on sentence boundaries."""
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


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def _embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Embed texts with sentence-transformers and L2-normalize rows."""
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(
        np.float32
    )


def _build_corpus() -> List[str]:
    """Download and chunk all pages."""
    corpus: List[str] = []
    for url in RAG_URLS:
        try:
            text = fetch_url_text(url)
            corpus.extend(chunk_text(text))
        except Exception as exc:  # noqa: BLE001 - demo-friendly warning
            print(f"[warn] Failed to process {url}: {exc}")
            continue
    return corpus


@lru_cache
def initialize_indices(
    alpha_dense: float = 0.6,
    alpha_bm25: float = 0.4,
    initial_k: int = 30,
    final_k: int = 5,
) -> HybridCompressionRetriever:
    """
    Build embeddings, BM25, ensemble retriever, reranker, and hybrid retriever once.
    """
    global _retriever, _embed_model

    corpus = _build_corpus()
    _embed_model = SentenceTransformer("BAAI/bge-small-en")

    if not corpus:
        bm25 = BM25Okapi([[]])
        embeddings = np.zeros((0, 0), dtype=np.float32)
    else:
        embeddings = _embed_texts(_embed_model, corpus)
        bm25 = BM25Okapi([_tokenize(c) for c in corpus])

    ensemble = EnsembleRetriever(
        chunks=corpus,
        embeddings=embeddings,
        bm25=bm25,
        embed_model=_embed_model,
        alpha_dense=alpha_dense,
        alpha_bm25=alpha_bm25,
        default_top_n=initial_k,
    )
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    _retriever = HybridCompressionRetriever(
        ensemble_retriever=ensemble,
        reranker=reranker,
        initial_k=initial_k,
        final_k=final_k,
    )
    return _retriever


def _get_retriever(final_k: int) -> HybridCompressionRetriever:
    retriever = _retriever or initialize_indices(final_k=final_k)
    return retriever


def rag_retrieve(query: str, top_k: int = 5) -> str:
    """
    Backward-compatible retrieval: return concatenated context string.
    """
    retriever = _get_retriever(final_k=top_k)
    chunks = retriever.retrieve(query, final_k=top_k)
    return "\n\n".join(c.text for c in chunks)


def rag_retrieve_debug(query: str, top_k: int = 5):
    """
    Debug retrieval: return context and per-chunk score details.
    """
    retriever = _get_retriever(final_k=top_k)
    return retriever.retrieve_with_debug(query, final_k=top_k)
