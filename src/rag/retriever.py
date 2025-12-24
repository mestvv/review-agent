"""Поиск и извлечение чанков из RAG базы."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    COLLECTION_NAME,
    SENTENCE_TRANSFORMER_MODEL,
    SECTION_WEIGHTS,
    QUERY_TYPE_KEYWORDS,
    CONFIDENCE_THRESHOLDS,
    MIN_CHUNKS_FOR_CONFIDENT_ANSWER,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    USE_RERANKER,
    INITIAL_FETCH_COUNT,
    get_db_path,
)

logger = logging.getLogger(__name__)

# Глобальные компоненты
_embedding_model = None
_clients = {}  # Словарь для хранения клиентов для разных БД
_collections = {}  # Словарь для хранения коллекций для разных БД
_reranker_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _embedding_model


def _get_collection(db_name: str):
    """Получить коллекцию для указанной БД."""
    global _clients, _collections
    if db_name not in _collections:
        db_path = get_db_path(db_name)
        _clients[db_name] = chromadb.PersistentClient(path=db_path)
        _collections[db_name] = _clients[db_name].get_or_create_collection(
            name=COLLECTION_NAME
        )
    return _collections[db_name]


def _get_reranker():
    """Lazy loading reranker модели."""
    global _reranker_model
    if not USE_RERANKER or not RERANKER_MODEL:
        return None
    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"🔄 Загрузка reranker: {RERANKER_MODEL}")
            _reranker_model = CrossEncoder(RERANKER_MODEL, trust_remote_code=True)
            logger.info("✅ Reranker загружен")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить reranker: {e}")
            return None
    return _reranker_model


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class RetrievedChunk:
    """Извлечённый чанк с метаданными."""

    text: str
    file_name: str
    file_hash: str
    chunk_id: int
    page: int
    section: str
    distance: float
    reranked_score: float = 0.0

    def citation(self) -> str:
        return f"[{self.file_name}, стр. {self.page}]"

    def full_citation(self) -> str:
        return f"{self.file_name} (стр. {self.page}, {self.section})"


@dataclass
class ConfidenceScore:
    """Оценка уверенности ответа."""

    level: ConfidenceLevel
    score: float
    avg_distance: float
    min_distance: float
    max_distance: float
    num_chunks: int
    num_sources: int
    coverage_by_section: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "score": round(self.score, 3),
            "avg_distance": round(self.avg_distance, 3),
            "min_distance": round(self.min_distance, 3),
            "max_distance": round(self.max_distance, 3),
            "num_chunks": self.num_chunks,
            "num_sources": self.num_sources,
            "coverage_by_section": self.coverage_by_section,
            "warnings": self.warnings,
        }


def detect_query_type(query: str) -> str:
    """Определяет тип запроса по ключевым словам."""
    query_lower = query.lower()
    type_scores = {}
    for query_type, keywords in QUERY_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            type_scores[query_type] = score
    if not type_scores:
        return "default"
    return max(type_scores, key=type_scores.get)


def retrieve_chunks(
    query: str,
    db_name: str,
    n_results: int = 5,
    section_filter: Optional[str] = None,
) -> list[RetrievedChunk]:
    """Извлекает релевантные чанки из базы.
    
    Args:
        query: Поисковый запрос
        db_name: Имя базы данных
        n_results: Количество результатов
        section_filter: Фильтр по секции (опционально)
        
    Returns:
        Список извлеченных чанков
    """
    logger.info(f"🔍 Поиск в БД '{db_name}': '{query[:60]}...'")

    model = _get_embedding_model()
    collection = _get_collection(db_name)

    embedding = model.encode([query]).tolist()
    where_filter = {"section": section_filter} if section_filter else None

    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            RetrievedChunk(
                text=doc,
                file_name=meta.get("file_name", "unknown"),
                file_hash=meta.get("file_hash", ""),
                chunk_id=meta.get("chunk_id", 0),
                page=meta.get("page", 0),
                section=meta.get("section", "unknown"),
                distance=dist,
            )
        )

    sources = sorted(set(c.file_name for c in chunks))
    logger.info(f"📚 Найдено: {len(chunks)} чанков из {len(sources)} источников")
    return chunks


def get_neighbor_chunks(
    chunk: RetrievedChunk, db_name: str, window: int = 1, query: Optional[str] = None
) -> list[RetrievedChunk]:
    """Извлекает соседние чанки для расширения контекста.
    
    Args:
        chunk: Чанк, для которого ищем соседей
        db_name: Имя базы данных
        window: Размер окна (количество соседних чанков с каждой стороны)
        query: Опциональный запрос для вычисления реального distance
        
    Returns:
        Список соседних чанков
    """
    collection = _get_collection(db_name)
    neighbor_ids = [
        f"{chunk.file_hash}_{chunk.chunk_id + offset}"
        for offset in range(-window, window + 1)
    ]

    results = collection.get(ids=neighbor_ids, include=["documents", "metadatas"])

    neighbors = []
    # Если передан query, вычисляем реальное distance для каждого соседнего чанка
    query_embedding = None
    if query:
        model = _get_embedding_model()
        query_embedding = model.encode([query])[0]

    for doc, meta in zip(results["documents"], results["metadatas"]):
        if doc and meta:
            chunk_id_val = meta.get("chunk_id", 0)
            
            # Вычисляем реальное distance, если query передан
            if query and query_embedding is not None:
                model = _get_embedding_model()
                doc_embedding = model.encode([doc])[0]
                # Вычисляем косинусное сходство
                similarity = float(
                    np.dot(query_embedding, doc_embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                )
                # Конвертируем в distance (1 - similarity для ChromaDB)
                distance = 1.0 - similarity
            else:
                # Используем большое значение для expanded chunks без реального distance
                # чтобы они были в конце при сортировке по distance
                distance = 999.0
            
            neighbors.append(
                RetrievedChunk(
                    text=doc,
                    file_name=meta.get("file_name", "unknown"),
                    file_hash=meta.get("file_hash", ""),
                    chunk_id=chunk_id_val,
                    page=meta.get("page", 0),
                    section=meta.get("section", "unknown"),
                    distance=distance,
                )
            )
    return sorted(neighbors, key=lambda c: c.chunk_id)


def format_context_with_citations(chunks: list[RetrievedChunk]) -> str:
    """Форматирует контекст с источниками."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(
            f"[{i}] {chunk.citation()}\nСекция: {chunk.section}\n---\n{chunk.text}\n"
        )
    return "\n".join(formatted)


def rerank_chunks_with_model(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """Ре-ранжирует чанки с помощью CrossEncoder."""
    reranker = _get_reranker()
    if reranker is None or not chunks:
        return chunks

    documents = [chunk.text for chunk in chunks]
    try:
        results = reranker.rank(
            query, documents, return_documents=False, top_k=top_k or len(chunks)
        )
        reranked_chunks = []
        for result in results:
            idx = result["corpus_id"]
            chunk = chunks[idx]
            chunk.reranked_score = result["score"]
            reranked_chunks.append(chunk)
        logger.info(f"🎯 Reranker: топ score={reranked_chunks[0].reranked_score:.3f}")
        return reranked_chunks
    except Exception as e:
        logger.warning(f"⚠️ Ошибка reranker: {e}")
        return chunks


def rerank_chunks_heuristic(
    chunks: list[RetrievedChunk], query_type: str
) -> list[RetrievedChunk]:
    """Ре-ранжирует чанки эвристикой с весами секций."""
    weights = SECTION_WEIGHTS.get(query_type, SECTION_WEIGHTS["default"])
    for chunk in chunks:
        section_weight = weights.get(chunk.section, 1.0)
        chunk.reranked_score = (1.0 - chunk.distance) * section_weight
    return sorted(chunks, key=lambda c: c.reranked_score, reverse=True)


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    query_type: str,
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """Ре-ранжирует чанки (ML модель или эвристика)."""
    if not chunks:
        return chunks

    reranker = _get_reranker()
    if reranker is not None:
        reranked = rerank_chunks_with_model(query, chunks, top_k)
        if reranked and reranked[0].reranked_score > 0:
            return reranked[:top_k] if top_k else reranked

    logger.info("📊 Используем эвристику с весами секций")
    reranked = rerank_chunks_heuristic(chunks, query_type)
    return reranked[:top_k] if top_k else reranked


def calculate_confidence(
    chunks: list[RetrievedChunk], query_type: str
) -> ConfidenceScore:
    """Вычисляет оценку уверенности."""
    warnings = []

    if not chunks:
        return ConfidenceScore(
            level=ConfidenceLevel.VERY_LOW,
            score=0.0,
            avg_distance=1.0,
            min_distance=1.0,
            max_distance=1.0,
            num_chunks=0,
            num_sources=0,
            warnings=["Не найдено релевантных чанков"],
        )

    distances = [c.distance for c in chunks]
    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)
    max_distance = max(distances)
    sources = set(c.file_name for c in chunks)
    num_sources = len(sources)

    section_coverage = {}
    for chunk in chunks:
        section_coverage[chunk.section] = section_coverage.get(chunk.section, 0) + 1

    if avg_distance < CONFIDENCE_THRESHOLDS["high"]:
        level, base_score = ConfidenceLevel.HIGH, 0.9
    elif avg_distance < CONFIDENCE_THRESHOLDS["medium"]:
        level, base_score = ConfidenceLevel.MEDIUM, 0.7
    elif avg_distance < CONFIDENCE_THRESHOLDS["low"]:
        level, base_score = ConfidenceLevel.LOW, 0.5
    else:
        level, base_score = ConfidenceLevel.VERY_LOW, 0.3

    score = base_score

    if len(chunks) >= MIN_CHUNKS_FOR_CONFIDENT_ANSWER:
        score += 0.05
    else:
        warnings.append(f"Мало контекста: {len(chunks)} чанков")
        score -= 0.1

    if num_sources >= 2:
        score += 0.05
    else:
        warnings.append("Ответ основан только на одном источнике")

    distance_spread = max_distance - min_distance
    if distance_spread > 0.4:
        warnings.append(f"Высокий разброс релевантности: {distance_spread:.2f}")
        score -= 0.05

    important_sections = {
        "results": ["results", "conclusion", "discussion"],
        "methods": ["methods"],
        "definitions": ["introduction", "methods"],
        "overview": ["abstract", "introduction", "conclusion"],
    }
    if query_type in important_sections:
        covered = [s for s in important_sections[query_type] if s in section_coverage]
        if not covered:
            warnings.append(f"Нет чанков из ключевых секций для '{query_type}'")
            score -= 0.1

    score = max(0.0, min(1.0, score))

    return ConfidenceScore(
        level=level,
        score=score,
        avg_distance=avg_distance,
        min_distance=min_distance,
        max_distance=max_distance,
        num_chunks=len(chunks),
        num_sources=num_sources,
        coverage_by_section=section_coverage,
        warnings=warnings,
    )


def retrieve_with_reranking(
    query: str,
    db_name: str,
    n_results: int = 5,
    section_filter: Optional[str] = None,
    fetch_multiplier: int = 3,
) -> tuple[list[RetrievedChunk], str, ConfidenceScore]:
    """Извлекает чанки с re-ranking.
    
    Args:
        query: Поисковый запрос
        db_name: Имя базы данных
        n_results: Количество результатов после re-ranking
        section_filter: Фильтр по секции (опционально)
        fetch_multiplier: Множитель для первичной выборки
        
    Returns:
        Кортеж из списка чанков, типа запроса и оценки уверенности
    """
    query_type = detect_query_type(query)
    logger.info(f"🎯 Тип запроса: {query_type}")

    # Используем INITIAL_FETCH_COUNT как минимум, чтобы не пропустить релевантные чанки
    fetch_count = max(n_results * fetch_multiplier, RERANKER_TOP_K, INITIAL_FETCH_COUNT)
    initial_chunks = retrieve_chunks(query, db_name, fetch_count, section_filter)

    if not initial_chunks:
        return [], query_type, calculate_confidence([], query_type)

    top_chunks = rerank_chunks(query, initial_chunks, query_type, top_k=n_results)
    confidence = calculate_confidence(top_chunks, query_type)

    changes = sum(
        1
        for i, c in enumerate(top_chunks)
        if i < len(initial_chunks) and c != initial_chunks[i]
    )
    logger.info(f"📊 Re-ranking: изменений в топ-{n_results}: {changes}")

    return top_chunks, query_type, confidence
