"""
Агент для работы с RAG базой научных статей.

Возможности:
- Ответы на вопросы с цитированием источников
- Обзоры литературы по теме
- Извлечение контекста с соседними чанками
- Section-aware re-ranking для лучшей релевантности
- Confidence score для оценки надёжности ответа
- Cross-chunk synthesis контроль (citation validation)
- Экспорт в LaTeX формате
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

console = Console()

# ============ Конфигурация ============

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNKS_LOG_DIR,
    RESPONSES_LOG_DIR,
    LATEX_OUTPUT_DIR,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    SENTENCE_TRANSFORMER_MODEL,
    SECTION_WEIGHTS,
    QUERY_TYPE_KEYWORDS,
    CONFIDENCE_THRESHOLDS,
    MIN_CHUNKS_FOR_CONFIDENT_ANSWER,
    MAX_AVG_DISTANCE_FOR_CONFIDENT,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    USE_RERANKER,
)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    extra_body={"thinking": {"type": "enabled"}},
)

# ============ Reranker (lazy loading) ============

_reranker_model = None


def get_reranker():
    """Lazy loading reranker модели."""
    global _reranker_model

    if not USE_RERANKER or not RERANKER_MODEL:
        return None

    if _reranker_model is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"🔄 Загрузка reranker модели: {RERANKER_MODEL}")
            _reranker_model = CrossEncoder(
                RERANKER_MODEL,
                trust_remote_code=True,
                # Для CPU или GPU без flash attention:
                # model_kwargs={"use_flash_attn": False}
            )
            logger.info("✅ Reranker загружен")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить reranker: {e}")
            logger.warning("   Используется fallback на эвристику с весами секций")
            return None

    return _reranker_model


# ============ Структуры данных ============


class ConfidenceLevel(Enum):
    """Уровни уверенности в ответе."""

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
    reranked_score: float = 0.0  # Score после re-ranking

    def citation(self) -> str:
        """Формирует краткую цитату для ссылки."""
        return f"[{self.file_name}, стр. {self.page}]"

    def full_citation(self) -> str:
        """Формирует полную цитату."""
        return f"{self.file_name} (стр. {self.page}, {self.section})"

    def latex_citation(self, cite_key: str) -> str:
        """Формирует LaTeX цитату."""
        return f"\\cite{{{cite_key}}}"


@dataclass
class ConfidenceScore:
    """Оценка уверенности ответа."""

    level: ConfidenceLevel
    score: float  # 0.0 - 1.0
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


@dataclass
class CitationValidation:
    """Результат валидации цитирования."""

    is_valid: bool
    claims_without_citation: list = field(default_factory=list)
    mixed_sources_in_claim: list = field(default_factory=list)
    citation_mapping: dict = field(default_factory=dict)  # claim -> [citations]

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "claims_without_citation": self.claims_without_citation,
            "mixed_sources_in_claim": self.mixed_sources_in_claim,
            "citation_mapping": self.citation_mapping,
        }


# ============ Retrieval функции ============


def save_chunks_to_json(
    chunks: list[RetrievedChunk],
    query: str,
    expanded_chunks: Optional[list[RetrievedChunk]] = None,
) -> None:
    """
    Сохраняет информацию о использованных чанках в JSON файл.

    Args:
        chunks: Список чанков для сохранения
        query: Поисковый запрос
        expanded_chunks: Расширенные чанки (если есть)
    """
    # Создаём директорию для логов, если её нет
    CHUNKS_LOG_DIR.mkdir(exist_ok=True)

    # Формируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Очищаем query для имени файла (убираем спецсимволы)
    safe_query = re.sub(r"[^\w\s-]", "", query[:50]).strip().replace(" ", "_")
    filename = f"chunks_{timestamp}_{safe_query}.json"
    filepath = CHUNKS_LOG_DIR / filename

    # Определяем, какие чанки были расширены
    expanded_ids = set()
    if expanded_chunks:
        expanded_ids = {f"{c.file_hash}_{c.chunk_id}" for c in expanded_chunks}

    # Формируем данные для сохранения
    chunks_data = []
    for chunk in chunks:
        chunk_key = f"{chunk.file_hash}_{chunk.chunk_id}"
        chunks_data.append(
            {
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.file_name,
                "file_hash": chunk.file_hash,
                "page": chunk.page,
                "section": chunk.section,
                "distance": chunk.distance,
                "text": chunk.text,
                "is_expanded": chunk_key in expanded_ids,
            }
        )

    data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "expanded_chunks_count": len(expanded_chunks) if expanded_chunks else 0,
        "sources": sorted(set(c.file_name for c in chunks)),
        "chunks": chunks_data,
    }

    # Сохраняем в JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Чанки сохранены в {filepath}")


def serialize_response(response) -> dict:
    """
    Сериализует объект ответа LLM (AIMessage) в словарь для JSON.

    Args:
        response: Объект ответа от LLM

    Returns:
        Словарь с данными ответа
    """
    # Если это объект с методом dict() (LangChain сообщения)
    if hasattr(response, "dict"):
        return response.dict()
    # Если это объект с методом model_dump() (Pydantic v2)
    elif hasattr(response, "model_dump"):
        return response.model_dump()
    # Если это объект с __dict__
    elif hasattr(response, "__dict__"):
        result = {}
        for key, value in response.__dict__.items():
            # Рекурсивно сериализуем вложенные объекты
            if hasattr(value, "dict"):
                result[key] = value.dict()
            elif hasattr(value, "model_dump"):
                result[key] = value.model_dump()
            elif hasattr(value, "__dict__"):
                result[key] = serialize_response(value)
            else:
                result[key] = value
        return result
    # Если это уже словарь или примитив
    else:
        return response


def save_response_to_json(query, response) -> None:
    """
    Сохраняет полный ответ LLM в JSON файл со всеми служебными полями.

    Args:
        query: Исходный запрос (строка или словарь)
        response: Полный объект ответа от LLM (AIMessage)
    """
    # Создаём директорию для логов, если её нет
    RESPONSES_LOG_DIR.mkdir(exist_ok=True)

    # Формируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Извлекаем строку для имени файла
    if isinstance(query, dict):
        # Берём "question" или "topic" из словаря
        query_str = query.get("question") or query.get("topic") or str(query)
    else:
        query_str = query

    # Очищаем query для имени файла (убираем спецсимволы)
    safe_query = re.sub(r"[^\w\s-]", "", str(query_str)[:50]).strip().replace(" ", "_")
    filename = f"response_{timestamp}_{safe_query}.json"
    filepath = RESPONSES_LOG_DIR / filename

    # Сериализуем response в словарь
    response_dict = serialize_response(response)

    # Формируем данные для сохранения
    data = {
        "query": query,
        "response": response_dict,
        "timestamp": datetime.now().isoformat(),
    }

    # Сохраняем в JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"💾 Ответ сохранён в {filepath}")


def retrieve_chunks(
    query: str,
    n_results: int = 5,
    section_filter: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Извлекает релевантные чанки из базы.

    Args:
        query: Поисковый запрос
        n_results: Количество результатов
        section_filter: Фильтр по секции (introduction, methods, results, discussion, conclusion)

    Returns:
        Список извлечённых чанков с метаданными
    """
    logger.info(f"🔍 Поиск: '{query[:60]}...'")

    embedding = embedding_model.encode([query]).tolist()

    where_filter = None
    if section_filter:
        where_filter = {"section": section_filter}

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


def get_neighbor_chunks(chunk: RetrievedChunk, window: int = 1) -> list[RetrievedChunk]:
    """
    Извлекает соседние чанки для расширения контекста.

    Args:
        chunk: Базовый чанк
        window: Количество соседей с каждой стороны

    Returns:
        Список соседних чанков (включая исходный)
    """
    neighbor_ids = []
    for offset in range(-window, window + 1):
        neighbor_id = f"{chunk.file_hash}_{chunk.chunk_id + offset}"
        neighbor_ids.append(neighbor_id)

    results = collection.get(
        ids=neighbor_ids,
        include=["documents", "metadatas"],
    )

    neighbors = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if doc and meta:
            neighbors.append(
                RetrievedChunk(
                    text=doc,
                    file_name=meta.get("file_name", "unknown"),
                    file_hash=meta.get("file_hash", ""),
                    chunk_id=meta.get("chunk_id", 0),
                    page=meta.get("page", 0),
                    section=meta.get("section", "unknown"),
                    distance=0.0,
                )
            )

    # Сортируем по chunk_id
    return sorted(neighbors, key=lambda c: c.chunk_id)


def format_context_with_citations(chunks: list[RetrievedChunk]) -> str:
    """Форматирует контекст с указанием источников."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(
            f"[{i}] {chunk.citation()}\n"
            f"Секция: {chunk.section}\n"
            f"---\n{chunk.text}\n"
        )
    return "\n".join(formatted)


# ============ Section-aware Re-ranking ============


def detect_query_type(query: str) -> str:
    """
    Определяет тип запроса по ключевым словам.

    Args:
        query: Текст запроса

    Returns:
        Тип запроса: 'results', 'methods', 'definitions', 'overview' или 'default'
    """
    query_lower = query.lower()

    # Подсчитываем совпадения для каждого типа
    type_scores = {}
    for query_type, keywords in QUERY_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            type_scores[query_type] = score

    if not type_scores:
        return "default"

    # Возвращаем тип с максимальным количеством совпадений
    return max(type_scores, key=type_scores.get)


def rerank_chunks_with_model(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """
    Ре-ранжирует чанки с помощью CrossEncoder reranker модели.

    Args:
        query: Поисковый запрос
        chunks: Список чанков для реранкинга
        top_k: Количество топ результатов (None = все)

    Returns:
        Отсортированный список чанков по reranked_score
    """
    reranker = get_reranker()

    if reranker is None or not chunks:
        return chunks

    # Подготавливаем документы для reranker
    documents = [chunk.text for chunk in chunks]

    try:
        # Используем метод rank() для получения отсортированных результатов
        results = reranker.rank(
            query,
            documents,
            return_documents=False,  # Возвращаем только индексы и scores
            top_k=top_k or len(chunks),
        )

        # results - это список словарей с 'corpus_id' и 'score'
        reranked_chunks = []
        for result in results:
            idx = result["corpus_id"]
            score = result["score"]
            chunk = chunks[idx]
            chunk.reranked_score = score
            reranked_chunks.append(chunk)

        logger.info(
            f"🎯 Reranker: топ score={reranked_chunks[0].reranked_score:.3f}, "
            f"мин score={reranked_chunks[-1].reranked_score:.3f}"
        )
        return reranked_chunks

    except Exception as e:
        logger.warning(f"⚠️ Ошибка reranker: {e}, используем fallback")
        return chunks


def rerank_chunks_heuristic(
    chunks: list[RetrievedChunk],
    query_type: str,
) -> list[RetrievedChunk]:
    """
    Ре-ранжирует чанки эвристикой с весами секций (fallback).

    Args:
        chunks: Список чанков с базовым distance
        query_type: Тип запроса ('results', 'methods', etc.)

    Returns:
        Отсортированный список чанков по reranked_score
    """
    weights = SECTION_WEIGHTS.get(query_type, SECTION_WEIGHTS["default"])

    for chunk in chunks:
        section_weight = weights.get(chunk.section, 1.0)
        # Инвертируем distance (меньше = лучше) и применяем вес секции
        chunk.reranked_score = (1.0 - chunk.distance) * section_weight

    return sorted(chunks, key=lambda c: c.reranked_score, reverse=True)


def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    query_type: str,
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """
    Ре-ранжирует чанки — сначала пробует ML модель, затем fallback на эвристику.

    Args:
        query: Поисковый запрос
        chunks: Список чанков для реранкинга
        query_type: Тип запроса для fallback эвристики
        top_k: Количество топ результатов

    Returns:
        Отсортированный список чанков
    """
    if not chunks:
        return chunks

    reranker = get_reranker()

    if reranker is not None:
        # Используем ML reranker
        reranked = rerank_chunks_with_model(query, chunks, top_k)
        if reranked and reranked[0].reranked_score > 0:
            return reranked[:top_k] if top_k else reranked

    # Fallback на эвристику
    logger.info("📊 Используем эвристику с весами секций")
    reranked = rerank_chunks_heuristic(chunks, query_type)
    return reranked[:top_k] if top_k else reranked


def retrieve_with_reranking(
    query: str,
    n_results: int = 5,
    section_filter: Optional[str] = None,
    fetch_multiplier: int = 3,
) -> tuple[list[RetrievedChunk], str, ConfidenceScore]:
    """
    Извлекает чанки с re-ranking (ML модель или эвристика).

    Args:
        query: Поисковый запрос
        n_results: Количество результатов после re-ranking
        section_filter: Фильтр по секции
        fetch_multiplier: Множитель для начального fetch

    Returns:
        (reranked_chunks, query_type, confidence_score)
    """
    # Определяем тип запроса (для fallback эвристики и confidence)
    query_type = detect_query_type(query)
    logger.info(f"🎯 Тип запроса: {query_type}")

    # Извлекаем больше чанков для последующего re-ranking
    # Для ML reranker важно иметь достаточно кандидатов
    fetch_count = max(n_results * fetch_multiplier, RERANKER_TOP_K)
    initial_chunks = retrieve_chunks(query, fetch_count, section_filter)

    if not initial_chunks:
        return [], query_type, calculate_confidence([], query_type)

    # Применяем re-ranking
    top_chunks = rerank_chunks(
        query=query,
        chunks=initial_chunks,
        query_type=query_type,
        top_k=n_results,
    )

    # Вычисляем confidence score
    confidence = calculate_confidence(top_chunks, query_type)

    # Логируем изменения в ranking
    changes = sum(
        1
        for i, c in enumerate(top_chunks)
        if i < len(initial_chunks) and c != initial_chunks[i]
    )
    logger.info(f"📊 Re-ranking: изменений в топ-{n_results}: {changes}")

    return top_chunks, query_type, confidence


# ============ Confidence Score ============


def calculate_confidence(
    chunks: list[RetrievedChunk],
    query_type: str,
) -> ConfidenceScore:
    """
    Вычисляет оценку уверенности на основе retrieval metrics.

    Args:
        chunks: Список извлечённых чанков
        query_type: Тип запроса

    Returns:
        ConfidenceScore с детальной информацией
    """
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

    # Количество уникальных источников
    sources = set(c.file_name for c in chunks)
    num_sources = len(sources)

    # Покрытие по секциям
    section_coverage = {}
    for chunk in chunks:
        section_coverage[chunk.section] = section_coverage.get(chunk.section, 0) + 1

    # Определяем уровень уверенности
    if avg_distance < CONFIDENCE_THRESHOLDS["high"]:
        level = ConfidenceLevel.HIGH
        base_score = 0.9
    elif avg_distance < CONFIDENCE_THRESHOLDS["medium"]:
        level = ConfidenceLevel.MEDIUM
        base_score = 0.7
    elif avg_distance < CONFIDENCE_THRESHOLDS["low"]:
        level = ConfidenceLevel.LOW
        base_score = 0.5
    else:
        level = ConfidenceLevel.VERY_LOW
        base_score = 0.3

    # Корректировки score
    score = base_score

    # Бонус за количество чанков
    if len(chunks) >= MIN_CHUNKS_FOR_CONFIDENT_ANSWER:
        score += 0.05
    else:
        warnings.append(
            f"Мало контекста: {len(chunks)} чанков (рекомендуется ≥{MIN_CHUNKS_FOR_CONFIDENT_ANSWER})"
        )
        score -= 0.1

    # Бонус за разнообразие источников
    if num_sources >= 2:
        score += 0.05
    else:
        warnings.append("Ответ основан только на одном источнике")

    # Штраф за высокий разброс distances
    distance_spread = max_distance - min_distance
    if distance_spread > 0.4:
        warnings.append(f"Высокий разброс релевантности: {distance_spread:.2f}")
        score -= 0.05

    # Проверка покрытия важных секций для типа запроса
    important_sections = {
        "results": ["results", "conclusion", "discussion"],
        "methods": ["methods"],
        "definitions": ["introduction", "methods"],
        "overview": ["abstract", "introduction", "conclusion"],
    }

    if query_type in important_sections:
        covered = [s for s in important_sections[query_type] if s in section_coverage]
        if not covered:
            warnings.append(
                f"Нет чанков из ключевых секций для запроса типа '{query_type}'"
            )
            score -= 0.1

    # Ограничиваем score в [0, 1]
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


# ============ Cross-chunk Synthesis Control ============


def validate_citations_in_response(
    response_text: str,
    available_chunks: list[RetrievedChunk],
) -> CitationValidation:
    """
    Валидирует цитирование в ответе LLM.

    Проверяет:
    - Каждое утверждение имеет хотя бы одну цитату
    - Нет логического смешивания разных источников

    Args:
        response_text: Текст ответа LLM
        available_chunks: Список чанков, которые были предоставлены LLM

    Returns:
        CitationValidation с результатами проверки
    """
    claims_without_citation = []
    mixed_sources_in_claim = []
    citation_mapping = {}

    # Паттерн для поиска цитат в формате [Файл, стр. X] или [N]
    citation_pattern = r"\[([^\]]+(?:стр\.|стр|p\.|pp\.)[^\]]*)\]|\[(\d+)\]"

    # Разбиваем ответ на предложения/утверждения
    # Используем точку, но избегаем разбиения на сокращениях
    sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z])", response_text)

    # Собираем доступные источники для проверки
    available_sources = {c.file_name for c in available_chunks}
    source_to_chunks = {}
    for chunk in available_chunks:
        if chunk.file_name not in source_to_chunks:
            source_to_chunks[chunk.file_name] = []
        source_to_chunks[chunk.file_name].append(chunk)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:
            continue

        # Пропускаем служебные фразы
        skip_phrases = [
            "информации недостаточно",
            "в предоставленном контексте",
            "не содержит",
            "не указано",
            "не найдено",
        ]
        if any(phrase in sentence.lower() for phrase in skip_phrases):
            continue

        # Ищем цитаты в предложении
        citations_in_sentence = re.findall(citation_pattern, sentence)

        # Если это утверждение с фактами (содержит числа или ключевые слова)
        is_factual = bool(re.search(r"\d+", sentence)) or any(
            kw in sentence.lower()
            for kw in [
                "составляет",
                "равен",
                "показывает",
                "демонстрирует",
                "наблюдается",
            ]
        )

        if is_factual and not citations_in_sentence:
            claims_without_citation.append(
                sentence[:100] + "..." if len(sentence) > 100 else sentence
            )

        # Анализируем источники в цитатах
        sources_in_sentence = set()
        for cite_match in citations_in_sentence:
            cite_text = cite_match[0] or cite_match[1]
            for source in available_sources:
                if source in cite_text or any(
                    part in cite_text for part in source.replace(".pdf", "").split("_")
                ):
                    sources_in_sentence.add(source)

        # Проверяем смешивание источников в одном утверждении
        if len(sources_in_sentence) > 1 and is_factual:
            # Это может быть нормально для сравнения, но отмечаем для проверки
            mixed_sources_in_claim.append(
                {
                    "claim": (
                        sentence[:100] + "..." if len(sentence) > 100 else sentence
                    ),
                    "sources": list(sources_in_sentence),
                }
            )

        # Сохраняем mapping
        if citations_in_sentence:
            claim_key = sentence[:50]
            citation_mapping[claim_key] = [
                cite_match[0] or cite_match[1] for cite_match in citations_in_sentence
            ]

    # Определяем валидность
    is_valid = len(claims_without_citation) == 0

    return CitationValidation(
        is_valid=is_valid,
        claims_without_citation=claims_without_citation,
        mixed_sources_in_claim=mixed_sources_in_claim,
        citation_mapping=citation_mapping,
    )


# ============ LaTeX Export ============


def generate_latex_document(
    title: str,
    content: str,
    chunks: list[RetrievedChunk],
    confidence: Optional[ConfidenceScore] = None,
    query: str = "",
) -> str:
    """
    Генерирует LaTeX документ из ответа агента.

    Args:
        title: Заголовок документа
        content: Основной текст ответа
        chunks: Использованные чанки для библиографии
        confidence: Оценка уверенности (опционально)
        query: Исходный запрос

    Returns:
        Строка с LaTeX документом
    """
    # Собираем уникальные источники для библиографии
    sources = {}
    for chunk in chunks:
        if chunk.file_name not in sources:
            sources[chunk.file_name] = {
                "pages": set(),
                "sections": set(),
                "cite_key": f"source{len(sources) + 1}",
            }
        sources[chunk.file_name]["pages"].add(chunk.page)
        sources[chunk.file_name]["sections"].add(chunk.section)

    # Конвертируем markdown-подобный текст в LaTeX
    latex_content = convert_to_latex(content, sources)

    # Формируем документ
    document = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T2A]{{fontenc}}
\\usepackage[russian]{{babel}}
\\usepackage{{hyperref}}
\\usepackage{{geometry}}
\\usepackage{{natbib}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}

\\geometry{{margin=2.5cm}}

\\title{{{escape_latex(title)}}}
\\author{{RAG Literature Agent}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

"""

    # Добавляем информацию о запросе
    if query:
        document += f"""\\section*{{Запрос}}
\\textit{{{escape_latex(query)}}}

"""

    # Добавляем оценку уверенности
    if confidence:
        confidence_text = {
            ConfidenceLevel.HIGH: "Высокая",
            ConfidenceLevel.MEDIUM: "Средняя",
            ConfidenceLevel.LOW: "Низкая",
            ConfidenceLevel.VERY_LOW: "Очень низкая",
        }.get(confidence.level, "Неизвестна")

        document += f"""\\section*{{Метаданные ответа}}
\\begin{{itemize}}
    \\item Уверенность: \\textbf{{{confidence_text}}} (score: {confidence.score:.2f})
    \\item Средняя релевантность: {confidence.avg_distance:.3f}
    \\item Количество источников: {confidence.num_sources}
    \\item Количество чанков: {confidence.num_chunks}
\\end{{itemize}}

"""
        if confidence.warnings:
            document += "\\textbf{Предупреждения:}\n\\begin{itemize}\n"
            for warning in confidence.warnings:
                document += f"    \\item {escape_latex(warning)}\n"
            document += "\\end{itemize}\n\n"

    # Основной контент
    document += f"""\\section*{{Ответ}}

{latex_content}

"""

    # Библиография
    document += """\\section*{Использованные источники}

\\begin{thebibliography}{99}

"""

    for fname, info in sorted(sources.items()):
        pages = ", ".join(map(str, sorted(info["pages"])))
        sections = ", ".join(sorted(info["sections"]))
        document += f"""\\bibitem{{{info['cite_key']}}}
{escape_latex(fname)}, стр. {pages}. Секции: {sections}.

"""

    document += """\\end{thebibliography}

\\end{document}
"""

    return document


def convert_to_latex(text: str, sources: dict) -> str:
    """
    Конвертирует текст в LaTeX формат.

    Args:
        text: Исходный текст (возможно с markdown)
        sources: Словарь источников для замены цитат

    Returns:
        Текст в LaTeX формате
    """
    result = text

    # Экранируем специальные символы LaTeX (кроме тех, что используем)
    special_chars = ["%", "&", "#", "_"]
    for char in special_chars:
        result = result.replace(char, f"\\{char}")

    # Конвертируем markdown заголовки
    result = re.sub(r"^### (.+)$", r"\\subsubsection*{\1}", result, flags=re.MULTILINE)
    result = re.sub(r"^## (.+)$", r"\\subsection*{\1}", result, flags=re.MULTILINE)
    result = re.sub(r"^# (.+)$", r"\\section*{\1}", result, flags=re.MULTILINE)

    # Конвертируем **bold** в \textbf{}
    result = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", result)

    # Конвертируем *italic* в \textit{}
    result = re.sub(r"\*(.+?)\*", r"\\textit{\1}", result)

    # Конвертируем списки
    lines = result.split("\n")
    in_list = False
    new_lines = []

    for line in lines:
        if re.match(r"^\s*[-•]\s+", line):
            if not in_list:
                new_lines.append("\\begin{itemize}")
                in_list = True
            item_text = re.sub(r"^\s*[-•]\s+", "", line)
            new_lines.append(f"    \\item {item_text}")
        else:
            if in_list:
                new_lines.append("\\end{itemize}")
                in_list = False
            new_lines.append(line)

    if in_list:
        new_lines.append("\\end{itemize}")

    result = "\n".join(new_lines)

    # Заменяем цитаты на LaTeX формат
    for fname, info in sources.items():
        # Заменяем [filename, стр. X] на \cite{key}
        pattern = re.escape(f"[{fname}")
        result = re.sub(
            pattern + r"[^\]]*\]", f'\\\\cite{{{info["cite_key"]}}}', result
        )

    return result


def escape_latex(text: str) -> str:
    """Экранирует специальные символы LaTeX."""
    replacements = {
        "\\": "\\textbackslash{}",
        "{": "\\{",
        "}": "\\}",
        "$": "\\$",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "_": "\\_",
        "^": "\\^{}",
        "~": "\\textasciitilde{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def save_latex_document(
    latex_content: str,
    query: str,
) -> Path:
    """
    Сохраняет LaTeX документ в файл.

    Args:
        latex_content: Содержимое LaTeX документа
        query: Запрос для формирования имени файла

    Returns:
        Путь к сохранённому файлу
    """
    LATEX_OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r"[^\w\s-]", "", query[:30]).strip().replace(" ", "_")
    filename = f"response_{timestamp}_{safe_query}.tex"
    filepath = LATEX_OUTPUT_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)

    logger.info(f"📄 LaTeX документ сохранён: {filepath}")
    return filepath


# ============ Промпты ============

QA_PROMPT = PromptTemplate(
    input_variables=["question", "context", "confidence_info"],
    template="""Ты эксперт-исследователь в области науки.
Ответь на вопрос, используя ТОЛЬКО предоставленный контекст.

КОНТЕКСТ (фрагменты из научных статей):
{context}

ИНФОРМАЦИЯ О КОНТЕКСТЕ:
{confidence_info}

ВОПРОС:
{question}

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ЦИТИРОВАНИЮ:
1. КАЖДОЕ фактическое утверждение (числа, данные, выводы) ДОЛЖНО иметь цитату
2. Формат цитаты: [Файл, стр. X]
3. НЕ СМЕШИВАЙ данные из разных источников в одном предложении без явного указания
4. Если данные из разных источников — укажи: "По данным [Источник1], ..., тогда как [Источник2] указывает..."

ТРЕБОВАНИЯ К ОТВЕТУ:
- Отвечай кратко и по существу
- Используй научный стиль изложения
- Если информации недостаточно или контекст слабый — ЧЕСТНО сообщи об этом
- Не выдумывай факты, которых нет в контексте
- При низкой уверенности используй формулировки: "согласно имеющимся данным", "в рамках предоставленного контекста"
""",
)

QA_PROMPT_SIMPLE = PromptTemplate(
    input_variables=["question", "context"],
    template="""Ты эксперт-исследователь в области науки.
Ответь на вопрос, используя ТОЛЬКО предоставленный контекст.

КОНТЕКСТ (фрагменты из научных статей):
{context}

ВОПРОС:
{question}

ТРЕБОВАНИЯ К ОТВЕТУ:
- Отвечай кратко и по существу
- Используй научный стиль изложения
- Обязательно указывай источники в формате [Файл, стр. X]
- Если информации недостаточно — честно сообщи об этом
- Не выдумывай факты, которых нет в контексте""",
)

REVIEW_PROMPT = PromptTemplate(
    input_variables=["topic", "context", "sources", "confidence_info"],
    template="""Ты научный исследователь, готовящий обзор литературы.

ТЕМА ОБЗОРА:
"{topic}"

ИЗВЛЕЧЁННЫЕ ФРАГМЕНТЫ ИЗ НАУЧНЫХ СТАТЕЙ:
{context}

ДОСТУПНЫЕ ИСТОЧНИКИ:
{sources}

ИНФОРМАЦИЯ О КОНТЕКСТЕ:
{confidence_info}

ЗАДАЧА:
Напиши структурированный обзор литературы по указанной теме.

СТРУКТУРА ОБЗОРА:
1. Введение — актуальность темы
2. Основные результаты исследований — ключевые находки из литературы
3. Обсуждение — тенденции, противоречия, пробелы в исследованиях
4. Выводы — краткое резюме состояния знаний
5. Список использованных источников

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К ЦИТИРОВАНИЮ:
1. КАЖДОЕ фактическое утверждение ДОЛЖНО иметь ссылку на конкретный источник
2. НЕ объединяй выводы разных статей без явного указания источников
3. Если статьи противоречат друг другу — укажи это явно
4. При сравнении данных: "Источник А указывает X, тогда как Источник Б показывает Y"

ТРЕБОВАНИЯ:
- Научный стиль изложения
- Синтез информации, а не простой пересказ
- Критический анализ представленных данных
- При низкой уверенности в контексте — отмечай это в тексте""",
)


# ============ Агент ============


class LiteratureAgent:
    """Агент для научного анализа литературы с улучшенным retrieval."""

    def __init__(self, llm):
        self.llm = llm
        self.last_confidence: Optional[ConfidenceScore] = None
        self.last_citation_validation: Optional[CitationValidation] = None

    def answer_question(
        self,
        question: str,
        n_results: int = 5,
        expand_context: bool = True,
        save_latex: bool = False,
        validate_citations: bool = True,
    ) -> None:
        """
        Отвечает на научный вопрос с цитированием.

        Args:
            question: Вопрос
            n_results: Количество чанков для поиска
            expand_context: Расширять ли контекст соседними чанками
            save_latex: Сохранять ли ответ в LaTeX формате
            validate_citations: Проверять ли корректность цитирования
        """
        # Используем улучшенный retrieval с re-ranking
        initial_chunks, query_type, confidence = retrieve_with_reranking(
            question, n_results, fetch_multiplier=2
        )

        if not initial_chunks:
            console.print("[red]Релевантный контекст не найден.[/red]")
            return

        self.last_confidence = confidence

        # Расширяем контекст соседями для лучшей связности
        expanded_chunks = []
        if expand_context:
            seen_ids = {f"{c.file_hash}_{c.chunk_id}" for c in initial_chunks}
            for chunk in initial_chunks[:3]:  # Расширяем только топ-3
                neighbors = get_neighbor_chunks(chunk, window=1)
                for n in neighbors:
                    key = f"{n.file_hash}_{n.chunk_id}"
                    if key not in seen_ids:
                        expanded_chunks.append(n)
                        seen_ids.add(key)
            chunks = expanded_chunks + initial_chunks[3:]
        else:
            chunks = initial_chunks

        # Сохраняем финальный список чанков (включая расширенные)
        save_chunks_to_json(chunks, question, expanded_chunks)

        context = format_context_with_citations(chunks[:10])

        # Формируем информацию о confidence для промпта
        confidence_info = self._format_confidence_for_prompt(confidence, query_type)

        response = (QA_PROMPT | self.llm).invoke(
            {
                "question": question,
                "context": context,
                "confidence_info": confidence_info,
            }
        )

        save_response_to_json(
            query={
                "question": question,
                "context": context,
                "confidence": confidence.to_dict(),
            },
            response=response,
        )

        # Валидация цитирования
        citation_validation = None
        if validate_citations:
            citation_validation = validate_citations_in_response(
                response.content, chunks
            )
            self.last_citation_validation = citation_validation

        self._print_answer(
            "Ответ на вопрос",
            response.content,
            chunks,
            confidence=confidence,
            citation_validation=citation_validation,
        )

        # Сохраняем в LaTeX
        if save_latex:
            latex_doc = generate_latex_document(
                title=f"Ответ: {question[:50]}...",
                content=response.content,
                chunks=chunks,
                confidence=confidence,
                query=question,
            )
            save_latex_document(latex_doc, question)

    def _format_confidence_for_prompt(
        self, confidence: ConfidenceScore, query_type: str
    ) -> str:
        """Форматирует информацию о confidence для промпта."""
        level_text = {
            ConfidenceLevel.HIGH: "ВЫСОКАЯ — контекст релевантен, можно давать уверенные ответы",
            ConfidenceLevel.MEDIUM: "СРЕДНЯЯ — контекст частично релевантен, используй осторожные формулировки",
            ConfidenceLevel.LOW: "НИЗКАЯ — контекст слабо релевантен, указывай на ограничения",
            ConfidenceLevel.VERY_LOW: "ОЧЕНЬ НИЗКАЯ — контекст ненадёжен, явно сообщи о недостатке информации",
        }

        info = f"""Уровень релевантности контекста: {level_text.get(confidence.level, 'неизвестен')}
Тип запроса: {query_type}
Количество источников: {confidence.num_sources}
Средняя дистанция: {confidence.avg_distance:.3f}"""

        if confidence.warnings:
            info += f"\nПредупреждения: {'; '.join(confidence.warnings)}"

        return info

    def review_topic(
        self,
        topic: str,
        n_results: int = 15,
        sections: Optional[list[str]] = None,
        save_latex: bool = False,
        validate_citations: bool = True,
    ) -> None:
        """
        Генерирует обзор литературы по теме.

        Args:
            topic: Тема обзора
            n_results: Количество чанков
            sections: Фильтр по секциям (опционально)
            save_latex: Сохранять ли в LaTeX формате
            validate_citations: Проверять ли корректность цитирования
        """
        all_chunks = []
        query_type = detect_query_type(topic)

        if sections:
            # Собираем чанки из указанных секций с re-ranking
            for section in sections:
                chunks, _, _ = retrieve_with_reranking(
                    topic, n_results // len(sections), section_filter=section
                )
                all_chunks.extend(chunks)
        else:
            # Используем re-ranking для обзора
            all_chunks, query_type, confidence = retrieve_with_reranking(
                topic, n_results, fetch_multiplier=2
            )
            self.last_confidence = confidence

        if not all_chunks:
            console.print("[red]Нет данных для обзора.[/red]")
            return

        # Вычисляем confidence для всех чанков
        confidence = calculate_confidence(all_chunks, query_type)
        self.last_confidence = confidence

        # Сохраняем чанки в JSON
        save_chunks_to_json(all_chunks, topic)

        context = format_context_with_citations(all_chunks)

        # Формируем список источников с деталями
        sources_detail = []
        seen = set()
        for chunk in all_chunks:
            key = chunk.file_name
            if key not in seen:
                sources_detail.append(f"• {chunk.file_name}")
                seen.add(key)

        # Формируем информацию о confidence
        confidence_info = self._format_confidence_for_prompt(confidence, query_type)

        response = (REVIEW_PROMPT | self.llm).invoke(
            {
                "topic": topic,
                "context": context[:8000],  # Ограничиваем контекст
                "sources": "\n".join(sources_detail),
                "confidence_info": confidence_info,
            }
        )

        save_response_to_json(
            query={
                "topic": topic,
                "context": context[:8000],
                "sources": "\n".join(sources_detail),
                "confidence": confidence.to_dict(),
            },
            response=response,
        )

        # Валидация цитирования
        citation_validation = None
        if validate_citations:
            citation_validation = validate_citations_in_response(
                response.content, all_chunks
            )
            self.last_citation_validation = citation_validation

        self._print_answer(
            "Обзор литературы",
            response.content,
            all_chunks,
            confidence=confidence,
            citation_validation=citation_validation,
        )

        # Сохраняем в LaTeX
        if save_latex:
            latex_doc = generate_latex_document(
                title=f"Обзор литературы: {topic}",
                content=response.content,
                chunks=all_chunks,
                confidence=confidence,
                query=topic,
            )
            save_latex_document(latex_doc, topic)

    def search_chunks(
        self,
        query: str,
        n_results: int = 10,
        section: Optional[str] = None,
    ) -> None:
        """
        Поиск и отображение релевантных чанков (без LLM).

        Args:
            query: Поисковый запрос
            n_results: Количество результатов
            section: Фильтр по секции
        """
        chunks = retrieve_chunks(query, n_results, section)

        if not chunks:
            console.print("[red]Ничего не найдено.[/red]")
            return

        console.print(Rule(f"[bold blue]Результаты поиска: {query}[/bold blue]"))

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Источник", width=30)
        table.add_column("Стр.", width=5)
        table.add_column("Секция", width=12)
        table.add_column("Dist", width=6)

        for i, chunk in enumerate(chunks, 1):
            table.add_row(
                str(i),
                chunk.file_name[:28],
                str(chunk.page),
                chunk.section,
                f"{chunk.distance:.3f}",
            )

        console.print(table)
        console.print()

        for i, chunk in enumerate(chunks, 1):
            console.print(
                Panel(
                    chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""),
                    title=f"[{i}] {chunk.citation()}",
                    subtitle=f"Секция: {chunk.section}",
                )
            )

    # ============ Вывод ============

    def _split_thinking_and_answer(self, text: str) -> Tuple[str, str]:
        """
        Разделяет текст на размышления (thinking) и ответ.

        Returns:
            (thinking, answer) - кортеж из размышлений и ответа
        """
        # Ищем явные теги <think>
        think_patterns = [
            r"<think>(.*?)</think>",
            r"<thinking>(.*?)</thinking>",
        ]

        thinking = ""
        answer = text

        # Пытаемся найти явные теги
        for pattern in think_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()
                answer = text[: match.start()] + text[match.end() :].strip()
                return thinking, answer

        # Если тегов нет, ищем паттерны размышлений
        # Размышления обычно содержат мета-комментарии о процессе
        thinking_phrases = [
            "пользователь",
            "кажется",
            "интересно",
            "важно отметить",
            "стоит отметить",
            "нужно",
            "должен",
            "ожидает",
            "работает с",
            "знает",
            "это подтверждает",
            "хотя в контексте",
            "в предоставленном",
        ]

        lines = text.split("\n")
        thinking_lines = []
        answer_lines = []
        found_answer_start = False

        for line in lines:
            line_stripped = line.strip()

            # Пустые строки сохраняем как есть
            if not line_stripped:
                if found_answer_start:
                    answer_lines.append(line)
                elif thinking_lines:
                    thinking_lines.append(line)
                continue

            # Проверяем, похоже ли на размышление
            is_thinking = any(
                phrase in line_stripped.lower() for phrase in thinking_phrases
            ) and (
                # Дополнительная проверка: размышления обычно длиннее и содержат мета-комментарии
                len(line_stripped) > 60
                or "пользователь" in line_stripped.lower()
                or "контекст" in line_stripped.lower()
            )

            # Проверяем, похоже ли на начало ответа (фактическая информация)
            is_answer_start = (
                re.match(r"^[А-ЯЁA-Z]", line_stripped)  # Начинается с заглавной
                and not any(
                    phrase in line_stripped.lower() for phrase in thinking_phrases
                )
                and (
                    "составляет" in line_stripped.lower()
                    or "равна" in line_stripped.lower()
                    or "является" in line_stripped.lower()
                    or "наблюдается" in line_stripped.lower()
                    or re.search(r"\d+", line_stripped)  # Содержит цифры (данные)
                )
            )

            if is_answer_start and not found_answer_start:
                # Начало ответа найдено
                found_answer_start = True
                answer_lines.append(line)
            elif found_answer_start:
                # После начала ответа - всё идёт в ответ
                answer_lines.append(line)
            elif is_thinking:
                # Размышление
                thinking_lines.append(line)
            elif not found_answer_start and not thinking_lines:
                # Если ещё не нашли thinking и не нашли answer - пробуем определить
                # Если содержит фактические данные - это answer
                if re.search(
                    r"\d+.*°[СC]|составляет|равна", line_stripped, re.IGNORECASE
                ):
                    found_answer_start = True
                    answer_lines.append(line)
                else:
                    thinking_lines.append(line)
            else:
                # Неопределённый случай - добавляем в thinking, если он уже начался
                if thinking_lines:
                    thinking_lines.append(line)
                else:
                    answer_lines.append(line)

        thinking = "\n".join(thinking_lines).strip()
        answer = "\n".join(answer_lines).strip()

        # Если не удалось разделить или thinking слишком большой - считаем весь текст ответом
        if not answer or (thinking and len(thinking) > len(answer) * 1.5):
            return "", text

        return thinking, answer

    def _print_answer(
        self,
        title: str,
        text: str,
        chunks: list[RetrievedChunk],
        confidence: Optional[ConfidenceScore] = None,
        citation_validation: Optional[CitationValidation] = None,
    ) -> None:
        """Форматированный вывод ответа с разделением thinking и ответа."""
        console.print(Rule(f"[bold blue]{title}[/bold blue]"))

        # Выводим Confidence Score
        if confidence:
            self._print_confidence(confidence)

        # Разделяем thinking и answer
        thinking, answer = self._split_thinking_and_answer(text)

        # Выводим thinking отдельно, если есть
        if thinking:
            console.print(
                Panel(
                    thinking,
                    title="[dim italic]Размышления модели[/dim italic]",
                    border_style="dim",
                    style="dim italic",
                    expand=False,
                )
            )
            console.print()

        # Выводим сам ответ более заметно
        if answer:
            console.print(
                Panel(
                    Markdown(answer),
                    title="[bold green]Ответ[/bold green]",
                    border_style="green",
                    expand=True,
                )
            )
        else:
            # Если не удалось разделить, выводим весь текст как ответ
            console.print(
                Panel(
                    Markdown(text),
                    title="[bold green]Ответ[/bold green]",
                    border_style="green",
                    expand=True,
                )
            )

        # Выводим результаты валидации цитирования
        if citation_validation:
            self._print_citation_validation(citation_validation)

        # Таблица источников
        if chunks:
            console.print(Rule("[bold]Использованные источники[/bold]"))

            seen = {}
            for chunk in chunks:
                key = chunk.file_name
                if key not in seen:
                    seen[key] = {"pages": set(), "sections": set(), "distances": []}
                seen[key]["pages"].add(chunk.page)
                seen[key]["sections"].add(chunk.section)
                seen[key]["distances"].append(chunk.distance)

            table = Table(show_header=True, header_style="bold")
            table.add_column("Источник", width=45)
            table.add_column("Страницы", width=12)
            table.add_column("Секции", width=18)
            table.add_column("Релев.", width=8)

            for fname, info in sorted(seen.items()):
                pages = ", ".join(map(str, sorted(info["pages"])))
                sections = ", ".join(sorted(info["sections"]))
                avg_dist = sum(info["distances"]) / len(info["distances"])
                relevance = f"{1-avg_dist:.2f}"
                table.add_row(fname, pages, sections, relevance)

            console.print(table)

    def _print_confidence(self, confidence: ConfidenceScore) -> None:
        """Выводит информацию о confidence score."""
        # Определяем цвет и стиль в зависимости от уровня
        level_styles = {
            ConfidenceLevel.HIGH: ("green", "✓"),
            ConfidenceLevel.MEDIUM: ("yellow", "◐"),
            ConfidenceLevel.LOW: ("orange1", "◔"),
            ConfidenceLevel.VERY_LOW: ("red", "✗"),
        }

        color, icon = level_styles.get(confidence.level, ("white", "?"))

        level_text = {
            ConfidenceLevel.HIGH: "Высокая",
            ConfidenceLevel.MEDIUM: "Средняя",
            ConfidenceLevel.LOW: "Низкая",
            ConfidenceLevel.VERY_LOW: "Очень низкая",
        }.get(confidence.level, "?")

        # Создаём компактную панель с информацией
        info_lines = [
            f"[{color}]{icon} Уверенность: {level_text}[/{color}] (score: {confidence.score:.2f})",
            f"   Источников: {confidence.num_sources} | Чанков: {confidence.num_chunks} | Ср. дистанция: {confidence.avg_distance:.3f}",
        ]

        # Добавляем секции
        if confidence.coverage_by_section:
            sections_str = ", ".join(
                f"{s}: {c}" for s, c in sorted(confidence.coverage_by_section.items())
            )
            info_lines.append(f"   Секции: {sections_str}")

        # Добавляем предупреждения
        if confidence.warnings:
            info_lines.append(f"[yellow]   ⚠ {'; '.join(confidence.warnings)}[/yellow]")

        console.print(
            Panel(
                "\n".join(info_lines),
                title="[bold]Оценка контекста[/bold]",
                border_style=color,
                expand=False,
            )
        )
        console.print()

    def _print_citation_validation(self, validation: CitationValidation) -> None:
        """Выводит результаты валидации цитирования."""
        if validation.is_valid and not validation.mixed_sources_in_claim:
            # Всё хорошо, не показываем ничего
            return

        lines = []

        if validation.claims_without_citation:
            lines.append("[yellow]⚠ Утверждения без цитат:[/yellow]")
            for claim in validation.claims_without_citation[:3]:  # Показываем первые 3
                lines.append(f"  • {claim}")
            if len(validation.claims_without_citation) > 3:
                lines.append(
                    f"  ... и ещё {len(validation.claims_without_citation) - 3}"
                )

        if validation.mixed_sources_in_claim:
            lines.append("[orange1]⚠ Возможное смешивание источников:[/orange1]")
            for mixed in validation.mixed_sources_in_claim[:2]:
                lines.append(f"  • {mixed['claim']}")
                lines.append(f"    Источники: {', '.join(mixed['sources'])}")

        if lines:
            console.print(
                Panel(
                    "\n".join(lines),
                    title="[bold yellow]Проверка цитирования[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                )
            )
            console.print()


# ============ CLI ============

if __name__ == "__main__":
    agent = LiteratureAgent(llm)

    # Пример: ответ на вопрос
    agent.answer_question(
        "Средняя скорость потепления для Земли?",
        n_results=5,
    )
    # agent.answer_question(
    #     "Как изменяется среднегодовая температуры воздуха и среднегодовая температура горных пород на глубине 1 и 4 м?",
    #     n_results=5,
    # )
    # agent.review_topic("Устойчивость зданий и инженерных сооружений")
