"""Инструменты для RAG-агента."""

import logging
from typing import Optional

from langchain_core.tools import tool

from src.config import list_existing_dbs
from src.rag.retriever import (
    retrieve_with_reranking,
    retrieve_chunks,
    get_neighbor_chunks,
    format_context_with_citations,
    RetrievedChunk,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


@tool
def search_vector_db(
    query: str,
    db_name: str,
    n_results: int = 5,
    expand_context: bool = True,
    section_filter: Optional[str] = None,
) -> str:
    """Поиск в векторной базе данных научных статей.

    Используй этот инструмент для поиска релевантной информации в научных статьях.
    Инструмент возвращает фрагменты текста с указанием источников, страниц и секций.

    Args:
        query: Поисковый запрос на естественном языке.
               Формулируй максимально конкретно.
        db_name: Имя базы данных для поиска (например, 'climate').
                 Доступные БД можно узнать через list_available_databases.
        n_results: Количество результатов (по умолчанию 5).
        expand_context: Добавлять соседние чанки для расширения контекста (по умолчанию True).
        section_filter: Фильтр по секции статьи (опционально).
                       Возможные значения: 'abstract', 'introduction', 'methods',
                       'results', 'discussion', 'conclusion'.

    Returns:
        Форматированный контекст с цитатами и информацией об уверенности.
    """
    logger.info("🔧 Tool search_vector_db: query='%s...', db='%s'", query[:50], db_name)

    # Проверяем существование БД
    existing_dbs = list_existing_dbs()
    if db_name not in existing_dbs:
        available = ", ".join(existing_dbs) if existing_dbs else "нет доступных БД"
        return f"❌ База данных '{db_name}' не найдена. Доступные БД: {available}"

    # Извлекаем чанки с реранкингом
    initial_chunks, query_type, confidence = retrieve_with_reranking(
        query=query,
        db_name=db_name,
        n_results=n_results,
        section_filter=section_filter,
        fetch_multiplier=3,
    )

    if not initial_chunks:
        return f"❌ Не найдено релевантных фрагментов для запроса: '{query}'"

    # Расширяем контекст соседними чанками, если нужно
    chunks = initial_chunks
    if expand_context:
        seen_ids = {f"{c.file_hash}_{c.chunk_id}" for c in initial_chunks}
        expanded_chunks = []

        # Расширяем топ-3 чанка
        for chunk in initial_chunks[:3]:
            neighbors = get_neighbor_chunks(chunk, db_name, window=1, query=query)
            for n in neighbors:
                key = f"{n.file_hash}_{n.chunk_id}"
                if key not in seen_ids:
                    expanded_chunks.append(n)
                    seen_ids.add(key)

        chunks = initial_chunks + expanded_chunks

    # Форматируем контекст
    context = format_context_with_citations(chunks[:10])

    # Добавляем информацию об уверенности
    level_text = {
        ConfidenceLevel.HIGH: "ВЫСОКАЯ",
        ConfidenceLevel.MEDIUM: "СРЕДНЯЯ",
        ConfidenceLevel.LOW: "НИЗКАЯ",
        ConfidenceLevel.VERY_LOW: "ОЧЕНЬ НИЗКАЯ",
    }.get(confidence.level, "НЕИЗВЕСТНА")

    # Собираем уникальные источники
    sources = sorted(set(c.file_name for c in chunks))

    result = f"""📊 РЕЗУЛЬТАТЫ ПОИСКА
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Запрос: {query}
Тип запроса: {query_type}
Уверенность: {level_text} (score: {confidence.score:.2f})
Средняя дистанция: {confidence.avg_distance:.3f}
Найдено чанков: {len(chunks)}
Источники: {len(sources)}
"""

    if confidence.warnings:
        result += f"⚠️ Предупреждения: {'; '.join(confidence.warnings)}\n"

    result += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:
{chr(10).join(f'• {s}' for s in sources)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

НАЙДЕННЫЕ ФРАГМЕНТЫ:

{context}
"""

    logger.info("✅ Tool search_vector_db: найдено %d чанков", len(chunks))
    return result


@tool
def list_available_databases() -> str:
    """Получить список доступных баз данных для поиска.

    Используй этот инструмент, чтобы узнать какие базы данных доступны
    для поиска научных статей.

    Returns:
        Список доступных баз данных с описанием.
    """
    logger.info("🔧 Tool list_available_databases called")

    existing_dbs = list_existing_dbs()

    if not existing_dbs:
        return (
            "❌ Нет доступных баз данных. Необходимо сначала проиндексировать статьи."
        )

    result = "📚 ДОСТУПНЫЕ БАЗЫ ДАННЫХ:\n\n"
    for db_name in existing_dbs:
        result += f"• {db_name}\n"

    result += (
        "\nИспользуй название БД в параметре db_name инструмента search_vector_db."
    )

    logger.info("✅ Tool list_available_databases: найдено %d БД", len(existing_dbs))
    return result


@tool
def search_by_section(
    query: str,
    db_name: str,
    sections: list[str],
    n_results_per_section: int = 3,
) -> str:
    """Поиск в нескольких секциях статей одновременно.

    Полезно когда нужно найти информацию из разных частей статей:
    например, методы И результаты одновременно.

    Args:
        query: Поисковый запрос на естественном языке.
        db_name: Имя базы данных для поиска.
        sections: Список секций для поиска.
                 Допустимые значения: 'abstract', 'introduction', 'methods',
                 'results', 'discussion', 'conclusion'.
        n_results_per_section: Количество результатов на секцию (по умолчанию 3).

    Returns:
        Форматированный контекст из указанных секций.
    """
    logger.info(
        "🔧 Tool search_by_section: query='%s...', sections=%s", query[:50], sections
    )

    # Проверяем существование БД
    existing_dbs = list_existing_dbs()
    if db_name not in existing_dbs:
        available = ", ".join(existing_dbs) if existing_dbs else "нет доступных БД"
        return f"❌ База данных '{db_name}' не найдена. Доступные БД: {available}"

    all_chunks: list[RetrievedChunk] = []
    section_results = {}

    for section in sections:
        chunks = retrieve_chunks(
            query=query,
            db_name=db_name,
            n_results=n_results_per_section,
            section_filter=section,
        )
        section_results[section] = len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        return f"❌ Не найдено релевантных фрагментов для запроса: '{query}'"

    # Сортируем по distance
    all_chunks.sort(key=lambda c: c.distance)

    # Форматируем результат
    context = format_context_with_citations(all_chunks[:10])
    sources = sorted(set(c.file_name for c in all_chunks))

    result = f"""📊 РЕЗУЛЬТАТЫ ПОИСКА ПО СЕКЦИЯМ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Запрос: {query}
Секции: {', '.join(sections)}
Найдено по секциям: {section_results}
Всего чанков: {len(all_chunks)}
Источники: {len(sources)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:
{chr(10).join(f'• {s}' for s in sources)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

НАЙДЕННЫЕ ФРАГМЕНТЫ:

{context}
"""

    logger.info("✅ Tool search_by_section: найдено %d чанков", len(all_chunks))
    return result


# Экспортируем все инструменты
ALL_TOOLS = [
    search_vector_db,
    list_available_databases,
    search_by_section,
]
