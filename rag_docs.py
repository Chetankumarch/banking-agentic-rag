from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# Use a small but capable open-source embedding model
_EMBED_MODEL_NAME = "BAAI/bge-small-en"
_embedder = SentenceTransformer(_EMBED_MODEL_NAME)
_reranker_model: Optional[CrossEncoder] = None

# Banking/overdraft pages to retrieve from
RAG_URLS: List[str] = [
    "https://www.consumerfinance.gov/consumer-tools/bank-accounts/know-your-overdraft-options/",
    "https://www.fdic.gov/consumer-resource-center/2021-12/overdraft-and-account-fees",
    "https://www.bankofamerica.com/deposits/overdrafts-and-overdraft-protection/",
    "https://www.machiassavings.bank/msb-general-service-fee-schedule/",
]


@dataclass
class RetrievedChunk:
    """Container for a chunk and all retrieval scores."""

    text: str
    dense_score: float
    bm25_score: float
    hybrid_score: float
    reranker_score: Optional[float] = None


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


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


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


@lru_cache
def _build_lexical_index() -> Tuple[List[Dict[str, str]], BM25Okapi]:
    """
    Build a BM25 index over the same corpus chunks for lexical matching.
    """
    corpus, _ = _build_corpus_and_index()
    tokenized = [_tokenize(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([[]])
    return corpus, bm25


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize scores to [0,1]; return zeros if constant to avoid NaNs.
    """
    if scores.size == 0:
        return scores
    s_min = scores.min()
    s_max = scores.max()
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def hybrid_retrieve_candidates_with_scores(
    query: str, top_n: int = 30, alpha: float = 0.6
) -> List[RetrievedChunk]:
    """
    Hybrid retrieval: combine dense (semantic) and BM25 (lexical) scores.
    Returns top_n candidates with all component scores.
    """
    corpus, embeddings = _build_corpus_and_index()
    if not corpus or embeddings.shape[0] == 0:
        return []

    # Dense scores (cosine because embeddings are normalized)
    q_vec = _embed_texts([query])[0]
    dense_scores = embeddings @ q_vec

    # Lexical scores via BM25
    corpus_for_bm25, bm25 = _build_lexical_index()
    sparse_scores = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)

    # Ensure both score vectors align on corpus order
    if len(corpus_for_bm25) != len(corpus):
        # Fallback to dense-only if something goes off
        combined_scores = dense_scores
        sparse_scores = np.zeros_like(dense_scores)
    else:
        dense_norm = _normalize_scores(dense_scores.astype(np.float32))
        sparse_norm = _normalize_scores(sparse_scores)
        alpha = float(alpha)
        alpha = max(0.0, min(1.0, alpha))
        combined_scores = alpha * dense_norm + (1.0 - alpha) * sparse_norm

    top_indices = np.argsort(-combined_scores)[:top_n]
    candidates: List[RetrievedChunk] = []
    for i in top_indices:
        candidates.append(
            RetrievedChunk(
                text=corpus[i]["text"],
                dense_score=float(dense_scores[i]),
                bm25_score=float(sparse_scores[i]) if sparse_scores.size else 0.0,
                hybrid_score=float(combined_scores[i]),
            )
        )
    return candidates


def get_reranker_model() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker_model


def rerank_and_compress(
    query: str, candidates: List[RetrievedChunk], k_final: int = 5
) -> Tuple[str, List[RetrievedChunk]]:
    """
    Use a cross-encoder to rerank candidates, keep top k_final, and build context.
    """
    if not candidates:
        return "", []

    model = get_reranker_model()
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c.reranker_score = float(score)

    reranked = sorted(candidates, key=lambda c: c.reranker_score or -1.0, reverse=True)
    selected = reranked[:k_final]
    context = "\n\n---\n\n".join(chunk.text for chunk in selected)
    return context, selected


def retrieve_with_rerank_and_debug(
    query: str, n_candidates: int = 30, k_final: int = 5, alpha: float = 0.6
) -> Tuple[str, List[RetrievedChunk]]:
    """
    Full pipeline: hybrid retrieval -> rerank -> compressed context, with debug data.
    """
    candidates = hybrid_retrieve_candidates_with_scores(
        query, top_n=n_candidates, alpha=alpha
    )
    return rerank_and_compress(query, candidates, k_final=k_final)


def rag_retrieve(query: str, top_k: int = 5) -> str:
    """
    Backward-compatible wrapper that now uses hybrid+rerank retrieval by default.
    """
    context, _ = retrieve_with_rerank_and_debug(query, n_candidates=30, k_final=top_k)
    return context


def rag_retrieve_debug(
    query: str, top_k: int = 5
) -> Tuple[str, List[RetrievedChunk]]:
    """
    Debug version: return context and selected chunks with scores.
    """
    return retrieve_with_rerank_and_debug(query, n_candidates=30, k_final=top_k)
