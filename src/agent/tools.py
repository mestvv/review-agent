"""Инструменты для RAG-агента."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import numpy as np
from langchain_core.tools import tool

from src.config import list_existing_dbs, CHUNKS_LOG_DIR
from src.rag.retriever import (
    retrieve_with_reranking,
    retrieve_chunks,
    get_neighbor_chunks,
    format_context_with_citations,
    get_article_titles,
    RetrievedChunk,
    ConfidenceLevel,
    ConfidenceScore,
)

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения текущей директории сессии агента
_current_agent_session_dir: Optional[Path] = None


def _get_agent_session_dir() -> Path:
    """Получить или создать директорию для текущей сессии агента."""
    global _current_agent_session_dir
    if _current_agent_session_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _current_agent_session_dir = CHUNKS_LOG_DIR / f"{timestamp}_agent"
        _current_agent_session_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "📁 Создана директория для логов агента: %s", _current_agent_session_dir
        )
    return _current_agent_session_dir


def reset_agent_session_dir() -> None:
    """Сбросить директорию сессии агента (для новой сессии)."""
    global _current_agent_session_dir
    _current_agent_session_dir = None


def _convert_numpy_types(obj: Any) -> Any:
    """Рекурсивно преобразует numpy типы в стандартные Python типы для JSON сериализации.

    Args:
        obj: Объект для преобразования

    Returns:
        Объект с преобразованными типами
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, set):
        return {_convert_numpy_types(item) for item in obj}
    else:
        return obj


def _save_agent_chunks(
    chunks: list[RetrievedChunk],
    query: str,
    tool_name: str,
    confidence: Optional[ConfidenceScore] = None,
    query_type: Optional[str] = None,
    expanded_chunks: Optional[list[RetrievedChunk]] = None,
) -> str:
    """Сохраняет чанки, полученные агентом, в JSON файл.

    Args:
        chunks: Список найденных чанков
        query: Поисковый запрос
        tool_name: Имя инструмента
        confidence: Оценка уверенности
        query_type: Тип запроса
        expanded_chunks: Список расширенных чанков (если есть)

    Returns:
        Путь к сохранённому файлу
    """
    session_dir = _get_agent_session_dir()

    # Безопасное имя файла из запроса
    safe_query = re.sub(r"[^\w\s-]", "", query[:50]).strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%H%M%S")
    filepath = session_dir / f"{timestamp}_{safe_query}.json"

    # Определяем какие чанки были расширены
    expanded_ids = set()
    if expanded_chunks:
        expanded_ids = {f"{c.file_hash}_{c.chunk_id}" for c in expanded_chunks}

    # Формируем данные о чанках
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
                "reranked_score": chunk.reranked_score,
                "text": chunk.text,
                "is_expanded": chunk_key in expanded_ids,
            }
        )

    # Формируем итоговые данные
    data = {
        "tool_name": tool_name,
        "query": query,
        "query_type": query_type,
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "expanded_chunks_count": len(expanded_chunks) if expanded_chunks else 0,
        "sources": sorted(set(c.file_name for c in chunks)),
        "chunks": chunks_data,
    }

    # Добавляем информацию о confidence если есть
    if confidence:
        data["confidence"] = {
            "level": confidence.level.value,
            "score": confidence.score,
            "avg_distance": confidence.avg_distance,
            "min_distance": confidence.min_distance,
            "max_distance": confidence.max_distance,
            "num_chunks": confidence.num_chunks,
            "num_sources": confidence.num_sources,
            "coverage_by_section": confidence.coverage_by_section,
            "warnings": confidence.warnings,
        }

    # Преобразуем numpy типы в стандартные Python типы для JSON сериализации
    data = _convert_numpy_types(data)

    # Сохраняем в файл
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("💾 Чанки агента сохранены: %s", filepath)
    return str(filepath)


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
        file_name_filter=None,
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

    # Сохраняем чанки в лог
    _save_agent_chunks(
        chunks=chunks,
        query=query,
        tool_name="search_vector_db",
        confidence=confidence,
        query_type=query_type,
        expanded_chunks=expanded_chunks if expand_context else None,
    )

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

    ВАЖНО: Секция 'unknown' ВСЕГДА автоматически добавляется к поиску,
    так как разбивка на секции не всегда происходит корректно.
    Это гарантирует, что не будут пропущены релевантные фрагменты.

    Args:
        query: Поисковый запрос на естественном языке.
        db_name: Имя базы данных для поиска.
        sections: Список секций для поиска.
                 Допустимые значения: 'abstract', 'introduction', 'methods',
                 'results', 'discussion', 'conclusion'.
                 Секция 'unknown' будет добавлена автоматически, если её нет в списке.
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

    # Всегда добавляем секцию 'unknown', если её нет в списке
    # Это важно, так как разбивка на секции не всегда происходит корректно
    sections_to_search = list(sections)
    if "unknown" not in sections_to_search:
        sections_to_search.append("unknown")

    all_chunks: list[RetrievedChunk] = []
    section_results = {}

    for section in sections_to_search:
        chunks = retrieve_chunks(
            query=query,
            db_name=db_name,
            n_results=n_results_per_section,
            section_filter=section,
            file_name_filter=None,
        )
        section_results[section] = len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        return f"❌ Не найдено релевантных фрагментов для запроса: '{query}'"

    # Сортируем по distance
    all_chunks.sort(key=lambda c: c.distance)

    # Сохраняем чанки в лог
    _save_agent_chunks(
        chunks=all_chunks,
        query=query,
        tool_name="search_by_section",
        query_type=f"sections:{','.join(sections_to_search)}",
    )

    # Форматируем результат
    context = format_context_with_citations(all_chunks[:10])
    sources = sorted(set(c.file_name for c in all_chunks))

    result = f"""📊 РЕЗУЛЬТАТЫ ПОИСКА ПО СЕКЦИЯМ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Запрос: {query}
Секции: {', '.join(sections_to_search)}
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


@tool
def list_article_titles(db_name: str) -> str:
    """Получить список названий всех статей в базе данных.

    Используй этот инструмент, чтобы узнать какие статьи доступны
    в указанной базе данных. Это поможет выбрать конкретную статью
    для поиска с фильтром по названию.

    Args:
        db_name: Имя базы данных для поиска (например, 'climate').

    Returns:
        Список названий статей с их количеством.
    """
    logger.info("🔧 Tool list_article_titles: db_name='%s'", db_name)

    # Проверяем существование БД
    existing_dbs = list_existing_dbs()
    if db_name not in existing_dbs:
        available = ", ".join(existing_dbs) if existing_dbs else "нет доступных БД"
        return f"❌ База данных '{db_name}' не найдена. Доступные БД: {available}"

    try:
        titles = get_article_titles(db_name)

        if not titles:
            return f"❌ В базе данных '{db_name}' не найдено статей."

        result = f"📚 СТАТЬИ В БАЗЕ ДАННЫХ '{db_name}':\n\n"
        result += f"Всего статей: {len(titles)}\n\n"

        for i, title in enumerate(titles, 1):
            result += f"{i}. {title}\n"

        result += (
            "\nИспользуй точное название статьи в инструменте "
            "search_vector_db_by_article для поиска только в этой статье."
        )

        logger.info("✅ Tool list_article_titles: найдено %d статей", len(titles))
        return result
    except Exception as e:
        logger.exception("Ошибка в list_article_titles")
        return f"❌ Ошибка при получении списка статей: {e}"


@tool
def search_vector_db_by_article(
    query: str,
    db_name: str,
    article_title: str,
    n_results: int = 5,
    expand_context: bool = True,
    section_filter: Optional[str] = None,
) -> str:
    """Поиск в векторной базе данных с фильтром по названию статьи.

    Используй этот инструмент для поиска информации в конкретной статье.
    Инструмент возвращает фрагменты текста только из указанной статьи
    с указанием источников, страниц и секций.

    Args:
        query: Поисковый запрос на естественном языке.
               Формулируй максимально конкретно.
        db_name: Имя базы данных для поиска (например, 'climate').
                 Доступные БД можно узнать через list_available_databases.
        article_title: Точное название статьи (файла) для фильтрации.
                      Список доступных статей можно получить через list_article_titles.
        n_results: Количество результатов (по умолчанию 5).
        expand_context: Добавлять соседние чанки для расширения контекста (по умолчанию True).
        section_filter: Фильтр по секции статьи (опционально).
                       Возможные значения: 'abstract', 'introduction', 'methods',
                       'results', 'discussion', 'conclusion'.

    Returns:
        Форматированный контекст с цитатами и информацией об уверенности.
    """
    logger.info(
        "🔧 Tool search_vector_db_by_article: query='%s...', db='%s', article='%s'",
        query[:50],
        db_name,
        article_title,
    )

    # Проверяем существование БД
    existing_dbs = list_existing_dbs()
    if db_name not in existing_dbs:
        available = ", ".join(existing_dbs) if existing_dbs else "нет доступных БД"
        return f"❌ База данных '{db_name}' не найдена. Доступные БД: {available}"

    # Извлекаем чанки с реранкингом и фильтром по названию статьи
    initial_chunks, query_type, confidence = retrieve_with_reranking(
        query=query,
        db_name=db_name,
        n_results=n_results,
        section_filter=section_filter,
        file_name_filter=article_title,
        fetch_multiplier=3,
    )

    if not initial_chunks:
        return (
            f"❌ Не найдено релевантных фрагментов для запроса: '{query}' "
            f"в статье '{article_title}'"
        )

    # Расширяем контекст соседними чанками, если нужно
    chunks = initial_chunks
    if expand_context:
        seen_ids = {f"{c.file_hash}_{c.chunk_id}" for c in initial_chunks}
        expanded_chunks = []

        # Расширяем топ-3 чанка
        for chunk in initial_chunks[:3]:
            neighbors = get_neighbor_chunks(chunk, db_name, window=1, query=query)
            for n in neighbors:
                # Проверяем, что соседний чанк из той же статьи
                if n.file_name == article_title:
                    key = f"{n.file_hash}_{n.chunk_id}"
                    if key not in seen_ids:
                        expanded_chunks.append(n)
                        seen_ids.add(key)

        chunks = initial_chunks + expanded_chunks

    # Форматируем контекст
    context = format_context_with_citations(chunks[:10])

    # Сохраняем чанки в лог
    _save_agent_chunks(
        chunks=chunks,
        query=query,
        tool_name="search_vector_db_by_article",
        confidence=confidence,
        query_type=query_type,
        expanded_chunks=expanded_chunks if expand_context else None,
    )

    # Добавляем информацию об уверенности
    level_text = {
        ConfidenceLevel.HIGH: "ВЫСОКАЯ",
        ConfidenceLevel.MEDIUM: "СРЕДНЯЯ",
        ConfidenceLevel.LOW: "НИЗКАЯ",
        ConfidenceLevel.VERY_LOW: "ОЧЕНЬ НИЗКАЯ",
    }.get(confidence.level, "НЕИЗВЕСТНА")

    # Собираем уникальные источники
    sources = sorted(set(c.file_name for c in chunks))

    result = f"""📊 РЕЗУЛЬТАТЫ ПОИСКА В СТАТЬЕ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Запрос: {query}
Статья: {article_title}
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

    logger.info("✅ Tool search_vector_db_by_article: найдено %d чанков", len(chunks))
    return result


# Экспортируем все инструменты
ALL_TOOLS = [
    search_vector_db,
    list_available_databases,
    search_by_section,
    list_article_titles,
    search_vector_db_by_article,
]

# Экспортируем вспомогательные функции
__all__ = [
    "ALL_TOOLS",
    "search_vector_db",
    "list_available_databases",
    "search_by_section",
    "list_article_titles",
    "search_vector_db_by_article",
    "reset_agent_session_dir",
]
