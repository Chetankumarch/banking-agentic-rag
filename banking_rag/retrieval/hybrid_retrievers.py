from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class RetrievedChunk:
    index: int
    text: str
    dense_score: float
    bm25_score: float
    dense_norm: float
    bm25_norm: float
    hybrid_score: float
    reranker_score: Optional[float] = None


class EnsembleRetriever:
    """
    Combines dense cosine scores and BM25 scores with weighting.
    """

    def __init__(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        embed_model: SentenceTransformer,
        alpha_dense: float = 0.6,
        alpha_bm25: float = 0.4,
        default_top_n: int = 30,
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = bm25
        self.embed_model = embed_model
        self.alpha_dense = alpha_dense
        self.alpha_bm25 = alpha_bm25
        self.default_top_n = default_top_n

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def retrieve(self, query: str, top_n: Optional[int] = None) -> List[RetrievedChunk]:
        top_n = top_n or self.default_top_n
        if len(self.chunks) == 0:
            return []

        # Dense scores (cosine on normalized embeddings)
        q_vec = self.embed_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        dense_scores = self.embeddings @ q_vec

        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.split()), dtype=np.float32)

        dense_norm = self._normalize(dense_scores.astype(np.float32))
        bm25_norm = self._normalize(bm25_scores)

        hybrid = (
            float(self.alpha_dense) * dense_norm
            + float(self.alpha_bm25) * bm25_norm
        )

        top_indices = np.argsort(-hybrid)[:top_n]
        results: List[RetrievedChunk] = []
        for idx in top_indices:
            results.append(
                RetrievedChunk(
                    index=int(idx),
                    text=self.chunks[idx],
                    dense_score=float(dense_scores[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    dense_norm=float(dense_norm[idx]),
                    bm25_norm=float(bm25_norm[idx]),
                    hybrid_score=float(hybrid[idx]),
                )
            )
        return results


class CrossEncoderReranker:
    """
    Cross-encoder reranker over query/chunk pairs.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.reranker_score = float(s)
        reranked = sorted(
            candidates,
            key=lambda c: c.reranker_score if c.reranker_score is not None else -1.0,
            reverse=True,
        )
        return reranked[:top_k]


class HybridCompressionRetriever:
    """
    Base retriever (ensemble) + reranker to return compressed top chunks.
    """

    def __init__(
        self,
        ensemble_retriever: EnsembleRetriever,
        reranker: CrossEncoderReranker,
        initial_k: int = 30,
        final_k: int = 5,
    ):
        self.ensemble = ensemble_retriever
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k

    def retrieve(self, query: str, final_k: Optional[int] = None) -> List[RetrievedChunk]:
        k_final = final_k or self.final_k
        initial = self.ensemble.retrieve(query, top_n=self.initial_k)
        return self.reranker.rerank(query, initial, top_k=k_final)

    def retrieve_with_debug(
        self, query: str, final_k: Optional[int] = None
    ) -> dict:
        selected = self.retrieve(query, final_k=final_k)
        context = "\n\n".join(chunk.text for chunk in selected)
        debug_chunks = []
        for c in selected:
            debug_chunks.append(
                {
                    "index": c.index,
                    "text": c.text,
                    "dense_score": c.dense_score,
                    "bm25_score": c.bm25_score,
                    "dense_norm": c.dense_norm,
                    "bm25_norm": c.bm25_norm,
                    "hybrid_score": c.hybrid_score,
                    "reranker_score": c.reranker_score,
                }
            )
        return {"context": context, "chunks": debug_chunks}
