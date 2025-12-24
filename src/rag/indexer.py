"""Индексация PDF и DOCX файлов в RAG базу."""

import hashlib
import logging
import pathlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
import pymupdf4llm
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import (
    ARTICLES_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    MIN_CHUNK_LENGTH,
    PARSED_FILES_LOGS_DIR,
    SENTENCE_TRANSFORMER_MODEL,
    get_db_path,
    get_articles_subdir,
    list_available_dbs,
)

logger = logging.getLogger(__name__)

GARBAGE_PATTERNS = [
    r"^page\s*\d+\s*$",
    r"^\d+\s*$",
    r"^[\s\-_\.]+$",
    r"^copyright\s*©?",
    r"^all\s+rights\s+reserved",
    r"^doi:\s*10\.",
    r"^isbn",
    r"^issn",
    r"^https?://",
    r"^\[?\d+\]\.?\s*[A-ZА-Я][a-zа-я]+.*\d{4}",
]

# Глобальные компоненты
_model = None
_clients = {}  # Словарь для хранения клиентов для разных БД
_collections = {}  # Словарь для хранения коллекций для разных БД
_text_splitter = None
_docling_converter = None


def _get_docling_converter():
    """Получить Docling конвертер с поддержкой формул."""
    global _docling_converter
    if _docling_converter is None:
        # Настройка опций для PDF с поддержкой формул
        pdf_options = PdfPipelineOptions()
        pdf_options.do_formula_enrichment = True
        pdf_options.do_table_structure = True

        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                InputFormat.DOCX: WordFormatOption(),
            }
        )
    return _docling_converter


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _model


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


def _get_text_splitter():
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )
    return _text_splitter


@dataclass
class Chunk:
    text: str
    chunk_id: int
    page: int
    section: Optional[str] = None
    is_table: bool = False


def get_file_hash(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()[:8]


def is_garbage_chunk(text: str) -> bool:
    text_clean = text.strip()
    if len(text_clean) < MIN_CHUNK_LENGTH:
        return True
    for pattern in GARBAGE_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True
    alpha_ratio = sum(c.isalpha() for c in text_clean) / max(len(text_clean), 1)
    return alpha_ratio < 0.4


def is_references_section(text: str) -> bool:
    bib_patterns = [
        r"\[\d+\]\s*[A-ZА-Я]",
        r"\d+\.\s*[A-ZА-Я][a-zа-я]+\s*,?\s*[A-ZА-Я]",
        r"(?:pp?\.|с\.)\s*\d+[-–]\d+",
        r"\(\d{4}\)",
    ]
    matches = sum(1 for p in bib_patterns if re.search(p, text))
    return matches >= 2


@dataclass
class SectionBoundary:
    """Граница секции в документе."""

    section_name: str
    start_page: int
    start_position: int  # Позиция в полном тексте документа


def extract_bold_header(line: str) -> Optional[str]:
    """Извлекает текст жирного заголовка из строки.

    Args:
        line: Строка для проверки

    Returns:
        Текст заголовка без markdown форматирования, если найден, иначе None
    """
    # Паттерны для жирных заголовков с различным форматированием:
    # **Заголовок.** - обычный жирный
    # _**Заголовок**_ - курсив + жирный
    # _**Заголовок**_ _**продолжение**_ - разбитый заголовок
    bold_patterns = [
        # Курсив + жирный: _**Текст**_
        r"^_\*\*([А-ЯA-Z][^*]+?)\*\*_",
        # Обычный жирный: **Текст**
        r"^\*\*([А-ЯA-Z][^*]+?)\*\*\.?\s*$",  # Только заголовок
        r"^\*\*([А-ЯA-Z][^*]+?)\*\*\.?\s+",  # Заголовок + текст после
    ]

    line_stripped = line.strip()
    for pattern in bold_patterns:
        match = re.match(pattern, line_stripped)
        if match:
            header_text = match.group(1).strip()
            # Фильтруем слишком длинные (больше 100 символов - вероятно не заголовок)
            if len(header_text) < 100:
                return header_text

    return None


def find_section_headers(page_chunks: list[dict]) -> list[SectionBoundary]:
    """Находит все заголовки секций в документе и их позиции.

    Args:
        page_chunks: Список страниц от pymupdf4llm

    Returns:
        Список границ секций, отсортированный по позиции
    """
    # Основные секции научной статьи с регулярными выражениями для поиска заголовков
    # Поддерживаются форматы:
    # - Markdown заголовки: # Введение, ## Introduction
    # - Жирный текст: **Введение**, **Введение.**
    # - Курсив + жирный: _**Abstract**_
    # - Заглавные буквы: ВВЕДЕНИЕ, СПИСОК ЛИТЕРАТУРЫ
    # - Обычный текст на отдельной строке
    section_patterns = {
        "abstract": [
            r"^#{1,3}\s*abstract\s*\.?\s*$",
            r"^\*?\*?abstract\*?\*?\s*\.?\s*$",
            r"^_?\*{1,2}abstract\*{1,2}_?\s*\.?\s*$",
            r"^аннотация\s*\.?\s*$",
            r"^#{1,3}\s*аннотация\s*\.?\s*$",
            r"^\*?\*?аннотация\*?\*?\s*\.?\s*$",
            r"^_?\*{1,2}аннотация\*{1,2}_?\s*\.?\s*$",
            r"^АННОТАЦИЯ\s*\.?\s*$",
        ],
        "introduction": [
            r"^#{1,3}\s*\d*\.?\s*introduction\s*\.?\s*$",
            r"^\*?\*?introduction\*?\*?\s*\.?\s*$",
            r"^_?\*{1,2}introduction\*{1,2}_?\s*\.?\s*$",
            r"^введение\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*введение\s*\.?\s*$",
            r"^\*{1,2}введение\.?\*{1,2}\s*",  # **Введение.** или **Введение**
            r"^_?\*{1,2}введение\*{1,2}_?\s*\.?\s*$",
            r"^ВВЕДЕНИЕ\s*\.?\s*$",
        ],
        "methods": [
            r"^#{1,3}\s*\d*\.?\s*methods?\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*methodology\s*\.?\s*$",
            r"^\*?\*?methods?\*?\*?\s*\.?\s*$",
            r"^\*?\*?methodology\*?\*?\s*\.?\s*$",
            r"^материалы\s+и\s+методы\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*материалы\s+и\s+методы\s*\.?\s*$",
            r"^\*{1,2}материалы\s+и\s+методы\.?\*{1,2}\s*",
            r"^методология\s*\.?\s*$",
            r"^МАТЕРИАЛЫ\s+И\s+МЕТОДЫ\s*\.?\s*$",
            r"^МЕТОДОЛОГИЯ\s*\.?\s*$",
        ],
        "results": [
            r"^#{1,3}\s*\d*\.?\s*results?\s*\.?\s*$",
            r"^\*?\*?results?\*?\*?\s*\.?\s*$",
            r"^результаты\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*результаты\s*\.?\s*$",
            r"^\*{1,2}результаты\.?\*{1,2}\s*",
            r"^РЕЗУЛЬТАТЫ\s*\.?\s*$",
        ],
        "discussion": [
            r"^#{1,3}\s*\d*\.?\s*discussion\s*\.?\s*$",
            r"^\*?\*?discussion\*?\*?\s*\.?\s*$",
            r"^обсуждение\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*обсуждение\s*\.?\s*$",
            r"^\*{1,2}обсуждение\.?\*{1,2}\s*",
            r"^ОБСУЖДЕНИЕ\s*\.?\s*$",
        ],
        "conclusion": [
            r"^#{1,3}\s*\d*\.?\s*conclusions?\s*\.?\s*$",
            r"^\*?\*?conclusions?\*?\*?\s*\.?\s*$",
            r"^заключение\s*\.?\s*$",
            r"^выводы\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*заключение\s*\.?\s*$",
            r"^#{1,3}\s*\d*\.?\s*выводы\s*\.?\s*$",
            r"^\*{1,2}заключение\.?\*{1,2}\s*",
            r"^\*{1,2}выводы\.?\*{1,2}\s*",
            r"^ЗАКЛЮЧЕНИЕ\s*\.?\s*$",
            r"^ВЫВОДЫ\s*\.?\s*$",
        ],
        "references": [
            r"^#{1,3}\s*references?\s*\.?\s*$",
            r"^\*?\*?references?\*?\*?\s*\.?\s*$",
            r"^#{1,3}\s*bibliography\s*\.?\s*$",
            r"^\*?\*?bibliography\*?\*?\s*\.?\s*$",
            r"^литература\s*\.?\s*$",
            r"^список\s+литературы\s*\.?\s*$",
            r"^#{1,3}\s*литература\s*\.?\s*$",
            r"^#{1,3}\s*список\s+литературы\s*\.?\s*$",
            r"^\*{1,2}литература\.?\*{1,2}\s*",
            r"^\*{1,2}список\s+литературы\.?\*{1,2}\s*",
            r"^ЛИТЕРАТУРА\s*\.?\s*$",
            r"^СПИСОК\s+ЛИТЕРАТУРЫ\s*\.?\s*$",
        ],
    }

    boundaries = []
    current_position = 0

    for page_data in page_chunks:
        page_text = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0)

        # Ищем заголовки построчно
        lines = page_text.split("\n")
        line_position = current_position

        for line in lines:
            # Для поиска используем оригинальную строку (с markdown форматированием)
            line_original = line.strip()
            # Но также создаем версию без markdown для некоторых паттернов
            line_clean = re.sub(r"[_*]", "", line_original).strip().lower()

            # Также проверяем только начало строки (для заголовков с продолжением текста)
            # Например: "**Введение.** Явление глобального потепления..."
            line_start = line_original[:100].strip()  # Берем первые 100 символов

            # Сначала проверяем стандартные секции
            found_standard = False
            for section_name, patterns in section_patterns.items():
                for pattern in patterns:
                    # Проверяем полную строку
                    if re.match(pattern, line_clean, re.IGNORECASE):
                        boundaries.append(
                            SectionBoundary(
                                section_name=section_name,
                                start_page=page_num,
                                start_position=line_position,
                            )
                        )
                        found_standard = True
                        break
                    # Проверяем начало строки (для заголовков с продолжением)
                    if re.search(pattern, line_start.lower(), re.IGNORECASE):
                        boundaries.append(
                            SectionBoundary(
                                section_name=section_name,
                                start_page=page_num,
                                start_position=line_position,
                            )
                        )
                        found_standard = True
                        break
                if found_standard:
                    break

            # Если не нашли стандартную секцию, проверяем на любой жирный заголовок
            # Кастомные заголовки помечаются как "unknown" для сохранения веса 1.0
            if not found_standard:
                custom_header = extract_bold_header(line_original)
                if custom_header:
                    boundaries.append(
                        SectionBoundary(
                            section_name="unknown",
                            start_page=page_num,
                            start_position=line_position,
                        )
                    )

            line_position += len(line) + 1  # +1 для \n

        current_position += len(page_text) + 2  # +2 для \n\n между страницами

    return sorted(boundaries, key=lambda x: x.start_position)


def determine_chunk_section(
    chunk_start_pos: int,
    section_boundaries: list[SectionBoundary],
) -> Optional[str]:
    """Определяет секцию для чанка на основе его позиции.

    Args:
        chunk_start_pos: Позиция начала чанка в полном тексте
        section_boundaries: Список границ секций

    Returns:
        Название секции или None
    """
    if not section_boundaries:
        return None

    # Находим последнюю секцию, которая начинается до или в позиции чанка
    current_section = None
    for boundary in section_boundaries:
        if boundary.start_position <= chunk_start_pos:
            current_section = boundary.section_name
        else:
            break

    return current_section


@dataclass
class TableWithContext:
    """Таблица с контекстом (заголовком перед таблицей)."""

    table_text: str
    context: str  # Текст перед таблицей (заголовок)
    start_pos: int  # Позиция начала таблицы в тексте
    end_pos: int  # Позиция конца таблицы


def extract_tables_with_context(text: str) -> list[TableWithContext]:
    """Извлекает таблицы из Markdown текста с контекстом.

    Markdown таблицы имеют формат:
    | Header1 | Header2 |
    |---------|---------|
    | Cell1   | Cell2   |

    Args:
        text: Текст в Markdown формате

    Returns:
        Список таблиц с контекстом
    """
    tables = []
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Проверяем, является ли строка началом таблицы (содержит |)
        if "|" in line and line.strip().startswith("|"):
            table_start_line = i
            table_lines = [line]

            # Собираем контекст (предыдущие непустые строки до пустой строки)
            context_lines = []
            j = i - 1
            while j >= 0 and lines[j].strip():
                context_lines.insert(0, lines[j])
                j -= 1

            # Собираем все строки таблицы
            i += 1
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # Вычисляем позиции
            start_pos = sum(len(lines[k]) + 1 for k in range(table_start_line))
            end_pos = sum(len(lines[k]) + 1 for k in range(i))

            table_text = "\n".join(table_lines)
            context = "\n".join(context_lines)

            tables.append(
                TableWithContext(
                    table_text=table_text,
                    context=context,
                    start_pos=start_pos,
                    end_pos=end_pos,
                )
            )
        else:
            i += 1

    return tables


def split_text_preserving_tables(
    text: str, text_splitter, tables: list[TableWithContext]
) -> list[tuple[str, bool]]:
    """Разбивает текст на части, сохраняя таблицы целиком.

    Args:
        text: Исходный текст
        text_splitter: LangChain text splitter
        tables: Список таблиц с контекстом

    Returns:
        Список кортежей (текст_чанка, is_table)
    """
    if not tables:
        # Нет таблиц - стандартное разбиение
        return [(chunk, False) for chunk in text_splitter.split_text(text)]

    result = []
    current_pos = 0

    for table in sorted(tables, key=lambda t: t.start_pos):
        # Текст до таблицы
        text_before = text[current_pos : table.start_pos].strip()
        if text_before:
            # Разбиваем текст до таблицы стандартным способом
            for chunk in text_splitter.split_text(text_before):
                result.append((chunk, False))

        # Создаём чанк с таблицей + контекст
        table_chunk = ""
        if table.context:
            table_chunk = f"{table.context}\n\n{table.table_text}"
        else:
            table_chunk = table.table_text

        # Проверяем размер таблицы
        if len(table_chunk) > CHUNK_SIZE:
            logger.warning(
                f"[WARN] Таблица превышает CHUNK_SIZE ({len(table_chunk)} > {CHUNK_SIZE}). "
                f"Сохраняется целиком для сохранения целостности данных."
            )

        result.append((table_chunk, True))
        current_pos = table.end_pos

    # Текст после последней таблицы
    text_after = text[current_pos:].strip()
    if text_after:
        for chunk in text_splitter.split_text(text_after):
            result.append((chunk, False))

    return result


def create_chunks_with_metadata(page_chunks: list[dict]) -> list[Chunk]:
    """Создает чанки из данных pymupdf4llm с метаданными страниц и секций.

    Использует интеллектуальное определение секций:
    1. Находит все заголовки секций в документе
    2. Определяет границы каждой секции
    3. Относит каждый чанк к секции на основе его позиции

    Args:
        page_chunks: Список словарей от pymupdf4llm.to_markdown(page_chunks=True)
                     Каждый словарь содержит:
                     - 'text': текст страницы в Markdown
                     - 'metadata': {'page': номер_страницы, ...}

    Returns:
        Список чанков с метаданными
    """
    chunks = []
    chunk_id = 0
    text_splitter = _get_text_splitter()

    # Шаг 1: Находим все границы секций в документе
    section_boundaries = find_section_headers(page_chunks)

    # Проверяем, есть ли секция References
    references_boundary = next(
        (b for b in section_boundaries if b.section_name == "references"), None
    )

    # Шаг 2: Собираем полный текст и обрабатываем страницы
    current_position = 0

    for page_data in page_chunks:
        page_text = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0)

        # Пропускаем страницы после начала References
        if (
            references_boundary
            and current_position >= references_boundary.start_position
        ):
            current_position += len(page_text) + 2
            continue

        # Дополнительная проверка на секцию references
        if is_references_section(page_text):
            current_position += len(page_text) + 2
            continue

        # Шаг 3: Извлекаем таблицы и разбиваем текст с их сохранением
        tables = extract_tables_with_context(page_text)
        page_text_chunks = split_text_preserving_tables(
            page_text, text_splitter, tables
        )
        chunk_position = current_position

        for chunk_text, is_table in page_text_chunks:
            if is_garbage_chunk(chunk_text) and not is_table:
                # Не пропускаем таблицы даже если они маленькие
                chunk_position += len(chunk_text)
                continue

            # Определяем секцию чанка по его позиции
            chunk_section = determine_chunk_section(chunk_position, section_boundaries)

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    chunk_id=chunk_id,
                    page=page_num,
                    section=chunk_section,
                    is_table=is_table,
                )
            )
            chunk_id += 1
            chunk_position += len(chunk_text)

        # Переходим к следующей странице (+2 для \n\n между страницами)
        current_position += len(page_text) + 2

    return chunks


def save_parsed_file(file_path: str, md_text: str) -> None:
    """Сохраняет результат парсинга файла в parsed_files_logs."""
    # Создаем директорию parsed_logs, если она не существует
    PARSED_FILES_LOGS_DIR.mkdir(exist_ok=True)

    # Удаляем расширение (.pdf, .md или .docx)
    base_name = file_path.replace(".pdf", "").replace(".md", "").replace(".docx", "")
    log_file_name = f"{base_name}_parsed.md"
    log_file_path = PARSED_FILES_LOGS_DIR / log_file_name

    pathlib.Path(log_file_path).write_bytes(md_text.encode())

    print(f"     Парсинг сохранен: {log_file_path}")


def parse_markdown_file(file_path: Path) -> list[dict]:
    """Парсит Markdown файл в формат, аналогичный pymupdf4llm.

    Args:
        file_path: Путь к MD файлу

    Returns:
        Список словарей с текстом и метаданными
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Разбиваем по заголовкам первого уровня или по размеру (имитация страниц)
    # Для MD файлов "страница" = примерно 3000 символов или заголовок первого уровня
    chunks = []
    current_chunk = ""
    page_num = 1

    lines = content.split("\n")
    for line in lines:
        # Если встречаем заголовок первого уровня и уже есть текст, создаем новую "страницу"
        if line.startswith("# ") and current_chunk.strip():
            chunks.append(
                {"text": current_chunk.strip(), "metadata": {"page": page_num}}
            )
            page_num += 1
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

            # Если чанк стал слишком большим, разбиваем
            if len(current_chunk) > 3000:
                chunks.append(
                    {"text": current_chunk.strip(), "metadata": {"page": page_num}}
                )
                page_num += 1
                current_chunk = ""

    # Добавляем последний чанк
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "metadata": {"page": page_num}})

    return chunks if chunks else [{"text": content, "metadata": {"page": 1}}]


def parse_docx_file(file_path: Path) -> list[dict]:
    """Парсит DOCX файл с помощью Docling в формат, аналогичный pymupdf4llm.

    Использует Docling для корректного парсинга таблиц и формул.

    Args:
        file_path: Путь к DOCX файлу

    Returns:
        Список словарей с текстом и метаданными
    """
    converter = _get_docling_converter()
    result = converter.convert(str(file_path))

    # Экспортируем в Markdown для корректного отображения таблиц и формул
    md_content = result.document.export_to_markdown()

    # Разбиваем по заголовкам первого уровня или по размеру (имитация страниц)
    chunks = []
    current_chunk = ""
    page_num = 1

    lines = md_content.split("\n")
    for line in lines:
        # Если встречаем заголовок первого уровня и уже есть текст, создаем новую "страницу"
        if line.startswith("# ") and current_chunk.strip():
            chunks.append(
                {"text": current_chunk.strip(), "metadata": {"page": page_num}}
            )
            page_num += 1
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

            # Если чанк стал слишком большим, разбиваем
            if len(current_chunk) > 3000:
                chunks.append(
                    {"text": current_chunk.strip(), "metadata": {"page": page_num}}
                )
                page_num += 1
                current_chunk = ""

    # Добавляем последний чанк
    if current_chunk.strip():
        chunks.append({"text": current_chunk.strip(), "metadata": {"page": page_num}})

    return chunks if chunks else [{"text": md_content, "metadata": {"page": 1}}]


def index_file(file_path: Path, db_name: str) -> int:
    """Индексирует файл (PDF, MD или DOCX) в указанную БД.

    Args:
        file_path: Путь к файлу
        db_name: Имя базы данных

    Returns:
        Количество проиндексированных чанков
    """
    file_path = Path(file_path)
    file_hash = get_file_hash(file_path)
    file_name = file_path.name
    file_ext = file_path.suffix.lower()
    collection = _get_collection(db_name)

    existing = collection.get(where={"file_hash": file_hash})
    if existing["ids"]:
        print(f"[SKIP] {file_name} — уже в базе ({len(existing['ids'])} чанков)")
        return 0

    try:
        # Парсим файл в зависимости от типа
        if file_ext == ".pdf":
            page_chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        elif file_ext == ".md":
            page_chunks = parse_markdown_file(file_path)
        elif file_ext == ".docx":
            page_chunks = parse_docx_file(file_path)
        else:
            print(f"[WARN] {file_name} — неподдерживаемый формат: {file_ext}")
            return 0

        # Сохраняем полный текст для логирования
        full_md_text = "\n\n".join([page.get("text", "") for page in page_chunks])
        save_parsed_file(file_name, full_md_text)
    except Exception as e:
        print(f"[ERROR] {file_name} — ошибка чтения: {e}")
        return 0

    if not page_chunks:
        print(f"[WARN] {file_name} — пустой файл")
        return 0

    chunks = create_chunks_with_metadata(page_chunks)
    if not chunks:
        print(f"[WARN] {file_name} — нет валидных чанков после фильтрации")
        return 0

    model = _get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts)

    ids = [f"{file_hash}_{c.chunk_id}" for c in chunks]
    metadatas = [
        {
            "file_name": file_name,
            "file_hash": file_hash,
            "chunk_id": c.chunk_id,
            "page": c.page,
            "section": c.section or "unknown",
            "is_table": c.is_table,
        }
        for c in chunks
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    sections = {}
    table_count = 0
    for c in chunks:
        s = c.section or "unknown"
        sections[s] = sections.get(s, 0) + 1
        if c.is_table:
            table_count += 1

    # Определяем количество страниц
    total_pages = len(page_chunks)
    section_info = ", ".join(f"{k}: {v}" for k, v in sorted(sections.items()))
    file_type_map = {".pdf": "PDF", ".md": "MD", ".docx": "DOCX"}
    file_type = file_type_map.get(file_ext, file_ext.upper())
    print(f"[OK] {file_name} ({file_type})")
    print(
        f"     Чанков: {len(chunks)} | Страниц: {total_pages} | Таблиц: {table_count}"
    )
    print(f"     Секции: {section_info}")

    return len(chunks)


# Для обратной совместимости
def index_pdf(file_path: Path, db_name: str) -> int:
    """Индексирует PDF файл (устаревшая функция, используйте index_file)."""
    return index_file(file_path, db_name)


def index_all_pdfs(db_name: Optional[str] = None) -> None:
    """Индексирует все файлы (PDF, MD и DOCX) из папки articles.

    Args:
        db_name: Имя БД (поддиректория в articles/). Если None, индексирует все доступные БД.
    """
    if db_name:
        # Индексируем конкретную БД
        _index_db(db_name)
    else:
        # Индексируем все доступные БД
        available_dbs = list_available_dbs()
        if not available_dbs:
            print(f"Нет поддиректорий в папке {ARTICLES_DIR}")
            print("Создайте поддиректории и поместите в них PDF/MD/DOCX файлы")
            return

        print(f"{'=' * 60}")
        print(f"Найдено {len(available_dbs)} баз данных для индексации")
        print(f"{'=' * 60}\n")

        for db in available_dbs:
            _index_db(db)
            print()


def _index_db(db_name: str) -> None:
    """Индексирует одну БД (все PDF, MD и DOCX файлы из соответствующей директории)."""
    articles_subdir = get_articles_subdir(db_name)

    if not articles_subdir.exists():
        print(f"[ERROR] Директория {articles_subdir} не существует")
        return

    # Собираем все PDF, MD и DOCX файлы
    pdf_files = sorted(articles_subdir.glob("*.pdf"))
    md_files = sorted(articles_subdir.glob("*.md"))
    docx_files = sorted(articles_subdir.glob("*.docx"))
    all_files = pdf_files + md_files + docx_files

    if not all_files:
        print(f"[WARN] Нет PDF/MD/DOCX файлов в папке {articles_subdir}")
        return

    print(f"{'=' * 60}")
    print(f"База данных: {db_name}")
    print(f"Директория: {articles_subdir}")
    print(
        f"Найдено {len(pdf_files)} PDF, {len(md_files)} MD, {len(docx_files)} DOCX файлов"
    )
    print(f"{'=' * 60}\n")

    total_chunks = 0
    for file_path in all_files:
        total_chunks += index_file(file_path, db_name)
        print()

    collection = _get_collection(db_name)
    print(f"{'=' * 60}")
    print(f"База данных '{db_name}': {collection.count()} чанков")
    print(f"{'=' * 60}")


def clear_database(db_name: Optional[str] = None) -> None:
    """Очищает базу данных.

    Args:
        db_name: Имя БД для удаления. Если None, предлагает выбрать из списка.
    """
    global _clients, _collections

    if db_name is None:
        # Показываем список существующих БД для выбора
        from src.config import list_existing_dbs

        existing_dbs = list_existing_dbs()

        if not existing_dbs:
            print("Нет существующих баз данных")
            return

        print("\nСуществующие базы данных:")
        for i, db in enumerate(existing_dbs, 1):
            collection = _get_collection(db)
            count = collection.count()
            print(f"  {i}. {db} ({count} чанков)")

        print(f"  {len(existing_dbs) + 1}. Все базы данных")
        print("  0. Отмена")

        try:
            choice = int(input("\nВыберите номер БД для удаления: "))
            if choice == 0:
                print("Отменено")
                return
            elif choice == len(existing_dbs) + 1:
                # Удалить все БД
                for db in existing_dbs:
                    _clear_db(db)
                print("\nВсе базы данных очищены")
                return
            elif 1 <= choice <= len(existing_dbs):
                db_name = existing_dbs[choice - 1]
            else:
                print("Неверный выбор")
                return
        except (ValueError, EOFError):
            print("Неверный ввод")
            return

    _clear_db(db_name)
    print(f"База данных '{db_name}' очищена")


def _clear_db(db_name: str) -> None:
    """Очищает конкретную БД."""
    global _clients, _collections

    db_path = get_db_path(db_name)
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Коллекция может не существовать

    # Пересоздаем коллекцию
    if db_name in _collections:
        del _collections[db_name]
    if db_name in _clients:
        del _clients[db_name]

    _collections[db_name] = client.get_or_create_collection(name=COLLECTION_NAME)


def show_stats(db_name: Optional[str] = None) -> None:
    """Показывает статистику базы.

    Args:
        db_name: Имя БД. Если None, показывает статистику по всем БД.
    """
    from src.config import list_existing_dbs

    if db_name:
        _show_db_stats(db_name)
    else:
        # Показываем статистику по всем БД
        existing_dbs = list_existing_dbs()
        if not existing_dbs:
            print("Нет существующих баз данных")
            return

        print(f"\n{'=' * 60}")
        print("СТАТИСТИКА ВСЕХ БАЗ ДАННЫХ")
        print(f"{'=' * 60}\n")

        total_chunks = 0
        for db in existing_dbs:
            collection = _get_collection(db)
            count = collection.count()
            total_chunks += count
            print(f"База данных: {db}")
            print(f"  Всего чанков: {count}")
            print()

        print(f"{'=' * 60}")
        print(f"ВСЕГО ЧАНКОВ ВО ВСЕХ БАЗАХ: {total_chunks}")
        print(f"{'=' * 60}\n")


def _show_db_stats(db_name: str) -> None:
    """Показывает детальную статистику одной БД."""
    collection = _get_collection(db_name)
    total = collection.count()
    if total == 0:
        print(f"База данных '{db_name}' пуста")
        return

    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    files = {}
    sections = {}
    for m in metadatas:
        fname = m.get("file_name", "unknown")
        section = m.get("section", "unknown")
        files[fname] = files.get(fname, 0) + 1
        sections[section] = sections.get(section, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"СТАТИСТИКА БД: {db_name}")
    print(f"{'=' * 60}")
    print(f"Всего чанков: {total}")
    print("\nПо файлам:")
    for fname, count in sorted(files.items()):
        print(f"  • {fname}: {count} чанков")
    print("\nПо секциям:")
    for section, count in sorted(sections.items()):
        print(f"  • {section}: {count} чанков")
    print(f"{'=' * 60}\n")


def list_dbs() -> None:
    """Выводит список доступных и существующих баз данных."""
    from src.config import list_existing_dbs

    available = list_available_dbs()
    existing = list_existing_dbs()

    print(f"\n{'=' * 60}")
    print("ДОСТУПНЫЕ БАЗЫ ДАННЫХ")
    print(f"{'=' * 60}")

    if not available and not existing:
        print("Нет баз данных.")
        print(
            "Создайте поддиректории в папке articles/ и поместите в них PDF/MD/DOCX файлы"
        )
    else:
        if available:
            print("\nДиректории с файлами (готовы к индексации):")
            for db in available:
                articles_subdir = get_articles_subdir(db)
                pdf_count = len(list(articles_subdir.glob("*.pdf")))
                md_count = len(list(articles_subdir.glob("*.md")))
                docx_count = len(list(articles_subdir.glob("*.docx")))
                file_types = []
                if pdf_count:
                    file_types.append(f"{pdf_count} PDF")
                if md_count:
                    file_types.append(f"{md_count} MD")
                if docx_count:
                    file_types.append(f"{docx_count} DOCX")
                files_str = ", ".join(file_types) if file_types else "0 файлов"
                indexed = (
                    "✓ проиндексирована" if db in existing else "○ не проиндексирована"
                )
                print(f"  • {db} ({files_str}) — {indexed}")

        if existing:
            print("\nСуществующие БД:")
            for db in existing:
                collection = _get_collection(db)
                count = collection.count()
                in_articles = "✓" if db in available else "✗ (директория удалена)"
                print(f"  • {db} ({count} чанков) — {in_articles}")

    print(f"{'=' * 60}\n")
