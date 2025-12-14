"""
Индексация всех PDF файлов из папки articles в RAG базу данных.
"""

from pathlib import Path
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import hashlib


ARTICLES_DIR = Path("articles")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "literature_review"

# Инициализация компонентов
md = MarkItDown()
model = SentenceTransformer("mlsa-iai-msu-lab/sci-rus-tiny")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def get_file_hash(file_path):
    """Генерирует хеш файла для уникальной идентификации."""
    return hashlib.md5(Path(file_path).read_bytes()).hexdigest()[:8]


def pdf_to_chunks(file_path):
    """Конвертирует PDF в чанки текста."""
    result = md.convert(str(file_path))
    chunks = text_splitter.split_text(result.text_content)
    return chunks


def index_pdf(file_path):
    """Индексирует один PDF файл в базу данных."""
    file_path = Path(file_path)
    file_hash = get_file_hash(file_path)
    file_name = file_path.name

    # Проверяем, не проиндексирован ли уже файл
    existing = collection.get(where={"file_hash": file_hash})
    if existing["ids"]:
        print(f"[SKIP] {file_name} - уже в базе")
        return 0

    # Обрабатываем PDF
    chunks = pdf_to_chunks(file_path)
    if not chunks:
        print(f"[WARN] {file_name} - пустой файл")
        return 0

    # Создаем эмбеддинги
    embeddings = model.encode(chunks)

    # Добавляем в коллекцию
    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{"document": file_name, "file_hash": file_hash} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids,
    )

    print(f"[OK] {file_name} - добавлено {len(chunks)} чанков")
    return len(chunks)


def index_all_pdfs():
    """Индексирует все PDF файлы из папки articles."""
    pdf_files = list(ARTICLES_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"Нет PDF файлов в папке {ARTICLES_DIR}")
        return

    print(f"Найдено {len(pdf_files)} PDF файлов")
    print("-" * 40)

    total_chunks = 0
    for pdf_file in pdf_files:
        total_chunks += index_pdf(pdf_file)

    print("-" * 40)
    print(f"Всего в базе: {collection.count()} чанков")


def clear_database():
    """Очищает базу данных."""
    client.delete_collection(COLLECTION_NAME)
    print("База данных очищена")


if __name__ == "__main__":
    index_all_pdfs()
