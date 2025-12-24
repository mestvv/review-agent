"""Агент для работы с научной литературой."""

import json
import re
import logging
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table

from src.config import (
    CHUNKS_LOG_DIR,
    RESPONSES_LOG_DIR,
    RESULTS_DIR,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    EXPAND_WINDOW,
    EXPAND_TOP_N,
)
from src.rag.retriever import (
    retrieve_chunks,
    retrieve_with_reranking,
    get_neighbor_chunks,
    format_context_with_citations,
    calculate_confidence,
    detect_query_type,
    RetrievedChunk,
    ConfidenceScore,
    ConfidenceLevel,
)
from src.agent.prompts import QA_PROMPT, REVIEW_PROMPT

logger = logging.getLogger(__name__)
console = Console()


def create_llm():
    """Создаёт LLM клиент."""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        extra_body={"thinking": {"type": "enabled"}},
    )


def _save_chunks_to_json(
    chunks: list[RetrievedChunk],
    query: str,
    expanded_chunks: Optional[list[RetrievedChunk]] = None,
) -> None:
    """Сохраняет чанки в JSON."""
    CHUNKS_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r"[^\w\s-]", "", query[:50]).strip().replace(" ", "_")
    filepath = CHUNKS_LOG_DIR / f"chunks_{timestamp}_{safe_query}.json"

    expanded_ids = set()
    if expanded_chunks:
        expanded_ids = {f"{c.file_hash}_{c.chunk_id}" for c in expanded_chunks}

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

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Чанки сохранены в {filepath}")


def _save_response_to_json(query, response) -> None:
    """Сохраняет ответ LLM в JSON."""
    RESPONSES_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(query, dict):
        query_str = query.get("question") or query.get("topic") or str(query)
    else:
        query_str = query

    safe_query = re.sub(r"[^\w\s-]", "", str(query_str)[:50]).strip().replace(" ", "_")
    filepath = RESPONSES_LOG_DIR / f"response_{timestamp}_{safe_query}.json"

    response_dict = response.dict() if hasattr(response, "dict") else str(response)

    data = {
        "query": query,
        "response": response_dict,
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"💾 Ответ сохранён в {filepath}")


def _save_to_markdown(
    query: str,
    response_content: str,
    chunks: list[RetrievedChunk],
    query_type: str = "ask",
    confidence: Optional[ConfidenceScore] = None,
) -> None:
    """Сохраняет ответ в Markdown формате.

    Args:
        query: Вопрос или тема
        response_content: Содержимое ответа от LLM
        chunks: Список использованных чанков
        query_type: Тип запроса ('ask' или 'review')
        confidence: Оценка уверенности (опционально)
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    safe_query = re.sub(r"[^\w\s-]", "", query[:50]).strip().replace(" ", "_")
    filepath = RESULTS_DIR / f"{query_type}_{timestamp}_{safe_query}.md"

    # Собираем уникальные источники
    sources = {}
    for chunk in chunks:
        if chunk.file_name not in sources:
            sources[chunk.file_name] = {
                "pages": set(),
                "sections": set(),
            }
        sources[chunk.file_name]["pages"].add(chunk.page)
        sources[chunk.file_name]["sections"].add(chunk.section)

    # Формируем Markdown документ
    md_doc = f"""# {query}

**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Тип запроса:** {query_type}

"""

    # Добавляем информацию о confidence, если есть
    if confidence:
        level_text = {
            ConfidenceLevel.HIGH: "Высокая",
            ConfidenceLevel.MEDIUM: "Средняя",
            ConfidenceLevel.LOW: "Низкая",
            ConfidenceLevel.VERY_LOW: "Очень низкая",
        }.get(confidence.level, "Неизвестна")

        md_doc += f"""## Метаданные

- **Тип запроса:** {query_type}
- **Уверенность:** {level_text} (score: {confidence.score:.2f})
- **Количество источников:** {confidence.num_sources}
- **Количество чанков:** {confidence.num_chunks}
- **Средняя дистанция:** {confidence.avg_distance:.3f}

"""

    # Основной контент
    md_doc += f"""## Ответ

{response_content}

"""

    # Добавляем источники, если они есть
    if sources:
        md_doc += """## Использованные источники

"""

        # Добавляем источники
        for source_name, info in sorted(sources.items()):
            pages = sorted(info["pages"])
            sections = sorted(info["sections"])
            pages_str = ", ".join(map(str, pages))
            sections_str = ", ".join(sections) if sections else None

            md_doc += f"- **{source_name}**\n"
            md_doc += f"  - Страницы: {pages_str}\n"
            if sections_str:
                md_doc += f"  - Секции: {sections_str}\n"
            md_doc += "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_doc)

    logger.info(f"💾 Результат сохранён в Markdown: {filepath}")


def _format_confidence_for_prompt(confidence: ConfidenceScore, query_type: str) -> str:
    """Форматирует confidence для промпта."""
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


def _print_confidence(confidence: ConfidenceScore) -> None:
    """Выводит информацию о confidence."""
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

    info_lines = [
        f"[{color}]{icon} Уверенность: {level_text}[/{color}] (score: {confidence.score:.2f})",
        f"   Источников: {confidence.num_sources} | Чанков: {confidence.num_chunks} | Ср. дистанция: {confidence.avg_distance:.3f}",
    ]

    if confidence.coverage_by_section:
        sections_str = ", ".join(
            f"{s}: {c}" for s, c in sorted(confidence.coverage_by_section.items())
        )
        info_lines.append(f"   Секции: {sections_str}")

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


def _print_sources_table(chunks: list[RetrievedChunk]) -> None:
    """Выводит таблицу источников."""
    if not chunks:
        return

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
        table.add_row(fname, pages, sections, f"{1-avg_dist:.2f}")

    console.print(table)


def answer_question(
    question: str,
    db_name: str,
    n_results: int = 5,
    expand_context: bool = True,
) -> None:
    """Отвечает на научный вопрос с цитированием.

    Args:
        question: Вопрос для ответа
        db_name: Имя базы данных для поиска
        n_results: Количество результатов
        expand_context: Расширять контекст соседними чанками
    """
    llm = create_llm()

    initial_chunks, query_type, confidence = retrieve_with_reranking(
        question, db_name, n_results, fetch_multiplier=2
    )

    if not initial_chunks:
        console.print("[red]Релевантный контекст не найден.[/red]")
        return

    # Расширяем контекст соседями
    expanded_chunks = []
    if expand_context:
        seen_ids = {f"{c.file_hash}_{c.chunk_id}" for c in initial_chunks}
        # Используем конфигурационный параметр для количества топ-чанков
        top_n_for_expansion = min(EXPAND_TOP_N, len(initial_chunks))
        for chunk in initial_chunks[:top_n_for_expansion]:
            # Передаем query для вычисления реального distance и используем конфигурационный window
            neighbors = get_neighbor_chunks(
                chunk, db_name, window=EXPAND_WINDOW, query=question
            )
            for n in neighbors:
                key = f"{n.file_hash}_{n.chunk_id}"
                if key not in seen_ids:
                    expanded_chunks.append(n)
                    seen_ids.add(key)
        chunks = expanded_chunks + initial_chunks[top_n_for_expansion:]
    else:
        chunks = initial_chunks

    _save_chunks_to_json(chunks, question, expanded_chunks)

    context = format_context_with_citations(chunks[:10])
    confidence_info = _format_confidence_for_prompt(confidence, query_type)

    response = (QA_PROMPT | llm).invoke(
        {
            "question": question,
            "context": context,
            "confidence_info": confidence_info,
        }
    )

    _save_response_to_json(
        {"question": question, "context": context, "confidence": confidence.to_dict()},
        response,
    )

    # Сохранение в Markdown
    _save_to_markdown(
        query=question,
        response_content=response.content,
        chunks=chunks,
        query_type="ask",
        confidence=confidence,
    )

    # Вывод
    console.print(Rule("[bold blue]Ответ на вопрос[/bold blue]"))
    _print_confidence(confidence)
    console.print(
        Panel(
            Markdown(response.content),
            title="[bold green]Ответ[/bold green]",
            border_style="green",
            expand=True,
        )
    )
    _print_sources_table(chunks)


def review_topic(
    topic: str,
    db_name: str,
    n_results: int = 15,
    sections: Optional[list[str]] = None,
) -> None:
    """Генерирует обзор литературы по теме.

    Args:
        topic: Тема для обзора
        db_name: Имя базы данных для поиска
        n_results: Количество результатов
        sections: Список секций для фильтрации (опционально)
    """
    llm = create_llm()
    query_type = detect_query_type(topic)

    if sections:
        all_chunks = []
        for section in sections:
            chunks, _, _ = retrieve_with_reranking(
                topic, db_name, n_results // len(sections), section_filter=section
            )
            all_chunks.extend(chunks)
    else:
        all_chunks, query_type, confidence = retrieve_with_reranking(
            topic, db_name, n_results, fetch_multiplier=2
        )

    if not all_chunks:
        console.print("[red]Нет данных для обзора.[/red]")
        return

    confidence = calculate_confidence(all_chunks, query_type)
    _save_chunks_to_json(all_chunks, topic)

    context = format_context_with_citations(all_chunks)

    sources_detail = []
    seen = set()
    for chunk in all_chunks:
        if chunk.file_name not in seen:
            sources_detail.append(f"• {chunk.file_name}")
            seen.add(chunk.file_name)

    confidence_info = _format_confidence_for_prompt(confidence, query_type)

    response = (REVIEW_PROMPT | llm).invoke(
        {
            "topic": topic,
            "context": context[:8000],
            "sources": "\n".join(sources_detail),
            "confidence_info": confidence_info,
        }
    )

    _save_response_to_json(
        {"topic": topic, "context": context[:8000], "confidence": confidence.to_dict()},
        response,
    )

    # Сохранение в Markdown
    _save_to_markdown(
        query=topic,
        response_content=response.content,
        chunks=all_chunks,
        query_type="review",
        confidence=confidence,
    )

    # Вывод
    console.print(Rule("[bold blue]Обзор литературы[/bold blue]"))
    _print_confidence(confidence)
    console.print(
        Panel(
            Markdown(response.content),
            title="[bold green]Обзор[/bold green]",
            border_style="green",
            expand=True,
        )
    )
    _print_sources_table(all_chunks)


def search_chunks(
    query: str,
    db_name: str,
    n_results: int = 10,
    section: Optional[str] = None,
) -> None:
    """Поиск и отображение чанков (без LLM).

    Args:
        query: Поисковый запрос
        db_name: Имя базы данных для поиска
        n_results: Количество результатов
        section: Фильтр по секции (опционально)
    """
    chunks = retrieve_chunks(query, db_name, n_results, section)

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
