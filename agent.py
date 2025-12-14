"""
Агент для работы с RAG базой научных статей.

Возможности:
- Ответы на вопросы с цитированием источников
- Обзоры литературы по теме
- Извлечение контекста с соседними чанками
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict

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
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
    SENTENCE_TRANSFORMER_MODEL,
)

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
)


# ============ Структуры данных ============


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

    def citation(self) -> str:
        """Формирует краткую цитату для ссылки."""
        return f"[{self.file_name}, стр. {self.page}]"

    def full_citation(self) -> str:
        """Формирует полную цитату."""
        return f"{self.file_name} (стр. {self.page}, {self.section})"


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


# ============ Промпты ============

QA_PROMPT = PromptTemplate(
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
    input_variables=["topic", "context", "sources"],
    template="""Ты научный исследователь, готовящий обзор литературы.

ТЕМА ОБЗОРА:
"{topic}"

ИЗВЛЕЧЁННЫЕ ФРАГМЕНТЫ ИЗ НАУЧНЫХ СТАТЕЙ:
{context}

ДОСТУПНЫЕ ИСТОЧНИКИ:
{sources}

ЗАДАЧА:
Напиши структурированный обзор литературы по указанной теме.

СТРУКТУРА ОБЗОРА:
1. Введение — актуальность темы
2. Основные результаты исследований — ключевые находки из литературы
3. Обсуждение — тенденции, противоречия, пробелы в исследованиях
4. Выводы — краткое резюме состояния знаний
5. Список использованных источников

ТРЕБОВАНИЯ:
- Научный стиль изложения
- Синтез информации, а не простой пересказ
- Каждое утверждение должно иметь ссылку на источник
- Критический анализ представленных данных""",
)


# ============ Агент ============


class LiteratureAgent:
    """Агент для научного анализа литературы."""

    def __init__(self, llm):
        self.llm = llm

    def answer_question(
        self,
        question: str,
        n_results: int = 5,
        expand_context: bool = True,
    ) -> None:
        """
        Отвечает на научный вопрос с цитированием.

        Args:
            question: Вопрос
            n_results: Количество чанков для поиска
            expand_context: Расширять ли контекст соседними чанками
        """
        initial_chunks = retrieve_chunks(question, n_results)

        if not initial_chunks:
            console.print("[red]Релевантный контекст не найден.[/red]")
            return

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

        response = (QA_PROMPT | self.llm).invoke(
            {"question": question, "context": context}
        )

        self._print_answer("Ответ на вопрос", response.content, chunks)

    def review_topic(
        self,
        topic: str,
        n_results: int = 15,
        sections: Optional[list[str]] = None,
    ) -> None:
        """
        Генерирует обзор литературы по теме.

        Args:
            topic: Тема обзора
            n_results: Количество чанков
            sections: Фильтр по секциям (опционально)
        """
        all_chunks = []

        if sections:
            # Собираем чанки из указанных секций
            for section in sections:
                chunks = retrieve_chunks(topic, n_results // len(sections), section)
                all_chunks.extend(chunks)
        else:
            all_chunks = retrieve_chunks(topic, n_results)

        if not all_chunks:
            console.print("[red]Нет данных для обзора.[/red]")
            return

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

        response = (REVIEW_PROMPT | self.llm).invoke(
            {
                "topic": topic,
                "context": context[:8000],  # Ограничиваем контекст
                "sources": "\n".join(sources_detail),
            }
        )

        self._print_answer("Обзор литературы", response.content, all_chunks)

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
    ) -> None:
        """Форматированный вывод ответа с разделением thinking и ответа."""
        console.print(Rule(f"[bold blue]{title}[/bold blue]"))

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

        # Таблица источников
        if chunks:
            console.print(Rule("[bold]Использованные источники[/bold]"))

            seen = {}
            for chunk in chunks:
                key = chunk.file_name
                if key not in seen:
                    seen[key] = {"pages": set(), "sections": set()}
                seen[key]["pages"].add(chunk.page)
                seen[key]["sections"].add(chunk.section)

            table = Table(show_header=True, header_style="bold")
            table.add_column("Источник", width=50)
            table.add_column("Страницы", width=15)
            table.add_column("Секции", width=20)

            for fname, info in sorted(seen.items()):
                pages = ", ".join(map(str, sorted(info["pages"])))
                sections = ", ".join(sorted(info["sections"]))
                table.add_row(fname, pages, sections)

            console.print(table)


# ============ CLI ============

if __name__ == "__main__":
    agent = LiteratureAgent(llm)

    # Пример: ответ на вопрос
    # agent.answer_question(
    #     "Какова средняя скорость глобального потепления?",
    #     n_results=5,
    # )
    # agent.answer_question(
    #     "Как изменяется среднегодовая температуры воздуха и среднегодовая температура горных пород на глубине 1 и 4 м?",
    #     n_results=5,
    # )
    agent.review_topic("Устойчивость зданий и инженерных сооружений")
