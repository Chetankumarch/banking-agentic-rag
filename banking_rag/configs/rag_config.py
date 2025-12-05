from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from banking_rag.retrieval.rag_docs import (
    initialize_indices,
    rag_retrieve,
    rag_retrieve_debug,
)


@dataclass
class RAGConfig:
    """
    Singleton wrapper for retrieval access.
    """

    alpha_dense: float = 0.6
    alpha_bm25: float = 0.4
    initial_k: int = 30
    final_k: int = 5
    _instance: "RAGConfig" = None  # type: ignore
    _initialized: bool = False

    @classmethod
    def get(cls) -> "RAGConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(
        self,
        alpha_dense: float | None = None,
        alpha_bm25: float | None = None,
        initial_k: int | None = None,
        final_k: int | None = None,
    ):
        if self._initialized:
            return
        if alpha_dense is not None:
            self.alpha_dense = alpha_dense
        if alpha_bm25 is not None:
            self.alpha_bm25 = alpha_bm25
        if initial_k is not None:
            self.initial_k = initial_k
        if final_k is not None:
            self.final_k = final_k

        initialize_indices(
            alpha_dense=self.alpha_dense,
            alpha_bm25=self.alpha_bm25,
            initial_k=self.initial_k,
            final_k=self.final_k,
        )
        self._initialized = True

    def get_context(self, query: str, k: int = 5) -> str:
        if not self._initialized:
            self.init(final_k=k)
        return rag_retrieve(query, top_k=k)

    def get_context_with_debug(self, query: str, k: int = 5) -> Any:
        if not self._initialized:
            self.init(final_k=k)
        return rag_retrieve_debug(query, top_k=k)
