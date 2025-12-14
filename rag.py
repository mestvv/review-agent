"""
RAG система для научных статей.

Особенности:
- Постраничное извлечение текста из PDF
- Автоматическое определение секций документа
- Фильтрация мусорных чанков
- Расширенные метаданные для трассируемости
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


# ============ Конфигурация ============

from config import (
    ARTICLES_DIR,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
)

# ============ Паттерны для определения секций ============

SECTION_PATTERNS = {
    "abstract": [
        r"^(?:abstract|аннотация|резюме)\b",
    ],
    "introduction": [
        r"^(?:\d+\.?\s*)?(?:introduction|введение|вступление)\b",
        r"^1\.\s*введение\b",
    ],
    "methods": [
        r"^(?:\d+\.?\s*)?(?:methods?|methodology|материалы?\s+и\s+методы|методы?\s+исследования?|методика)\b",
    ],
    "results": [
        r"^(?:\d+\.?\s*)?(?:results?|результаты?)\b",
    ],
    "discussion": [
        r"^(?:\d+\.?\s*)?(?:discussion|обсуждение)\b",
    ],
    "conclusion": [
        r"^(?:\d+\.?\s*)?(?:conclusions?|выводы?|заключение)\b",
    ],
    "references": [
        r"^(?:references?|литература|список\s+литературы|библиографи[яи]|источники)\b",
    ],
}

# Паттерны для фильтрации мусора
GARBAGE_PATTERNS = [
    r"^page\s*\d+\s*$",  # "Page 1"
    r"^\d+\s*$",  # Просто номер страницы
    r"^[\s\-_\.]+$",  # Пустые строки/разделители
    r"^copyright\s*©?",  # Копирайты
    r"^all\s+rights\s+reserved",
    r"^doi:\s*10\.",  # DOI в одиночку
    r"^isbn",
    r"^issn",
    r"^https?://",  # URL в одиночку
    r"^\[?\d+\]\.?\s*[A-ZА-Я][a-zа-я]+.*\d{4}",  # Библиографические записи
]


# ============ Инициализация компонентов ============

from config import SENTENCE_TRANSFORMER_MODEL

model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ============ Структуры данных ============


@dataclass
class PageContent:
    """Содержимое одной страницы PDF."""

    page_num: int
    text: str


@dataclass
class Chunk:
    """Чанк текста с метаданными."""

    text: str
    chunk_id: int
    page: int
    section: Optional[str] = None


# ============ Утилиты ============


def get_file_hash(file_path: Path) -> str:
    """Генерирует хеш файла для уникальной идентификации."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()[:8]


def detect_section(text: str) -> Optional[str]:
    """Определяет секцию документа по тексту."""
    # Берём первые 200 символов для анализа заголовка
    header = text[:200].lower().strip()

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, header, re.IGNORECASE | re.MULTILINE):
                return section
    return None


def is_garbage_chunk(text: str) -> bool:
    """Проверяет, является ли чанк мусорным."""
    text_clean = text.strip()

    # Слишком короткий
    if len(text_clean) < MIN_CHUNK_LENGTH:
        return True

    # Проверяем паттерны мусора
    for pattern in GARBAGE_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True

    # Слишком много цифр/спецсимволов (вероятно таблица или формула)
    alpha_ratio = sum(c.isalpha() for c in text_clean) / max(len(text_clean), 1)
    if alpha_ratio < 0.4:
        return True

    return False


def is_references_section(text: str) -> bool:
    """Проверяет, относится ли текст к разделу References."""
    # Проверяем наличие типичных паттернов библиографии
    bib_patterns = [
        r"\[\d+\]\s*[A-ZА-Я]",  # [1] Author
        r"\d+\.\s*[A-ZА-Я][a-zа-я]+\s*,?\s*[A-ZА-Я]",  # 1. Иванов И.
        r"(?:pp?\.|с\.)\s*\d+[-–]\d+",  # pp. 123-456
        r"\(\d{4}\)",  # (2024)
    ]

    matches = sum(1 for p in bib_patterns if re.search(p, text))
    return matches >= 2


# ============ Обработка PDF ============


def extract_pages_from_pdf(file_path: Path) -> list[PageContent]:
    """Извлекает текст из PDF постранично."""
    pages = []

    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(PageContent(page_num=page_num, text=text))

    return pages


def create_chunks_with_metadata(pages: list[PageContent]) -> list[Chunk]:
    """Создаёт чанки с метаданными из страниц."""
    chunks = []
    current_section = None
    in_references = False
    chunk_id = 0

    for page in pages:
        # Определяем секцию в начале страницы
        detected_section = detect_section(page.text)
        if detected_section:
            current_section = detected_section
            if detected_section == "references":
                in_references = True

        # Пропускаем раздел References
        if in_references or is_references_section(page.text):
            in_references = True
            continue

        # Разбиваем страницу на чанки
        page_chunks = text_splitter.split_text(page.text)

        for chunk_text in page_chunks:
            # Фильтруем мусор
            if is_garbage_chunk(chunk_text):
                continue

            # Проверяем секцию внутри чанка
            chunk_section = detect_section(chunk_text) or current_section

            chunks.append(
                Chunk(
                    text=chunk_text.strip(),
                    chunk_id=chunk_id,
                    page=page.page_num,
                    section=chunk_section,
                )
            )
            chunk_id += 1

    return chunks


# ============ Индексация ============


def index_pdf(file_path: Path) -> int:
    """Индексирует один PDF файл в базу данных."""
    file_path = Path(file_path)
    file_hash = get_file_hash(file_path)
    file_name = file_path.name

    # Проверяем, не проиндексирован ли уже файл
    existing = collection.get(where={"file_hash": file_hash})
    if existing["ids"]:
        print(f"[SKIP] {file_name} — уже в базе ({len(existing['ids'])} чанков)")
        return 0

    # Извлекаем страницы
    try:
        pages = extract_pages_from_pdf(file_path)
    except Exception as e:
        print(f"[ERROR] {file_name} — ошибка чтения: {e}")
        return 0

    if not pages:
        print(f"[WARN] {file_name} — пустой файл")
        return 0

    # Создаём чанки с метаданными
    chunks = create_chunks_with_metadata(pages)

    if not chunks:
        print(f"[WARN] {file_name} — нет валидных чанков после фильтрации")
        return 0

    # Создаем эмбеддинги
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts)

    # Формируем данные для ChromaDB
    ids = [f"{file_hash}_{c.chunk_id}" for c in chunks]
    metadatas = [
        {
            "file_name": file_name,
            "file_hash": file_hash,
            "chunk_id": c.chunk_id,
            "page": c.page,
            "section": c.section or "unknown",
        }
        for c in chunks
    ]

    # Добавляем в коллекцию
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    # Статистика по секциям
    sections = {}
    for c in chunks:
        s = c.section or "unknown"
        sections[s] = sections.get(s, 0) + 1

    section_info = ", ".join(f"{k}: {v}" for k, v in sorted(sections.items()))
    print(f"[OK] {file_name}")
    print(f"     Чанков: {len(chunks)} | Страниц: {pages[-1].page_num}")
    print(f"     Секции: {section_info}")

    return len(chunks)


def index_all_pdfs() -> None:
    """Индексирует все PDF файлы из папки articles."""
    pdf_files = sorted(ARTICLES_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Нет PDF файлов в папке {ARTICLES_DIR}")
        return

    print(f"{'='*50}")
    print(f"Найдено {len(pdf_files)} PDF файлов")
    print(f"{'='*50}\n")

    total_chunks = 0
    for pdf_file in pdf_files:
        total_chunks += index_pdf(pdf_file)
        print()

    print(f"{'='*50}")
    print(f"Всего в базе: {collection.count()} чанков")
    print(f"{'='*50}")


def clear_database() -> None:
    """Очищает базу данных."""
    global collection
    client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("База данных очищена")


def show_stats() -> None:
    """Показывает статистику базы данных."""
    total = collection.count()
    if total == 0:
        print("База данных пуста")
        return

    # Получаем все метаданные
    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]

    # Статистика по файлам
    files = {}
    sections = {}
    for m in metadatas:
        fname = m.get("file_name", "unknown")
        section = m.get("section", "unknown")
        files[fname] = files.get(fname, 0) + 1
        sections[section] = sections.get(section, 0) + 1

    print(f"\n{'='*50}")
    print(f"СТАТИСТИКА RAG БАЗЫ ДАННЫХ")
    print(f"{'='*50}")
    print(f"Всего чанков: {total}")
    print(f"\nПо файлам:")
    for fname, count in sorted(files.items()):
        print(f"  • {fname}: {count} чанков")
    print(f"\nПо секциям:")
    for section, count in sorted(sections.items()):
        print(f"  • {section}: {count} чанков")
    print(f"{'='*50}\n")


# ============ CLI ============

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "clear":
            clear_database()
        elif cmd == "stats":
            show_stats()
        elif cmd == "index":
            index_all_pdfs()
        else:
            print(f"Неизвестная команда: {cmd}")
            print("Доступные команды: index, clear, stats")
    else:
        index_all_pdfs()
