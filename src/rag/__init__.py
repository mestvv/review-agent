"""RAG модуль для работы с научными статьями."""

from src.rag.indexer import (
    index_pdf,
    index_all_pdfs,
    clear_database,
    show_stats,
    list_dbs,
)
from src.rag.retriever import (
    retrieve_chunks,
    retrieve_with_reranking,
    get_neighbor_chunks,
    format_context_with_citations,
    calculate_confidence,
    RetrievedChunk,
    ConfidenceScore,
    ConfidenceLevel,
)

__all__ = [
    "index_pdf",
    "index_all_pdfs",
    "clear_database",
    "show_stats",
    "list_dbs",
    "retrieve_chunks",
    "retrieve_with_reranking",
    "get_neighbor_chunks",
    "format_context_with_citations",
    "calculate_confidence",
    "RetrievedChunk",
    "ConfidenceScore",
    "ConfidenceLevel",
]
