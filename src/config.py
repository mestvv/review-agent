"""Конфигурация приложения."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Директории
ARTICLES_DIR = Path("articles")
CHROMA_DB_BASE_PATH = "./chroma_db"  # Базовая директория для всех БД
COLLECTION_NAME = "literature_review"
CHUNKS_LOG_DIR = Path("./chunks_log")
RESPONSES_LOG_DIR = Path("./responses_log")
RESULTS_DIR = Path("./results")
PARSED_FILES_LOGS_DIR = Path("./parsed_files_logs")


def get_db_path(db_name: str) -> str:
    """Получить путь к БД для конкретной директории.

    Args:
        db_name: Имя директории внутри articles/

    Returns:
        Путь к БД ChromaDB
    """
    return f"{CHROMA_DB_BASE_PATH}/{db_name}"


def get_articles_subdir(db_name: str) -> Path:
    """Получить путь к поддиректории articles.

    Args:
        db_name: Имя директории внутри articles/

    Returns:
        Путь к директории
    """
    return ARTICLES_DIR / db_name


def list_available_dbs() -> list[str]:
    """Получить список доступных баз данных (поддиректорий в articles/).

    Returns:
        Список имён директорий
    """
    if not ARTICLES_DIR.exists():
        return []

    # Получаем все директории в articles/
    subdirs = [d.name for d in ARTICLES_DIR.iterdir() if d.is_dir()]
    return sorted(subdirs)


def list_existing_dbs() -> list[str]:
    """Получить список существующих баз данных.

    Returns:
        Список имён БД
    """
    base_path = Path(CHROMA_DB_BASE_PATH)
    if not base_path.exists():
        return []

    # Получаем все директории в chroma_db/
    dbs = [d.name for d in base_path.iterdir() if d.is_dir()]
    return sorted(dbs)


# Для обратной совместимости (если нет поддиректорий)
CHROMA_DB_PATH = CHROMA_DB_BASE_PATH

# Chunking параметры
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100

# Модели
SENTENCE_TRANSFORMER_MODEL = "intfloat/multilingual-e5-large"

# Reranker
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
)
RERANKER_TOP_K = 10
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# Retrieval параметры
INITIAL_FETCH_COUNT = int(
    os.getenv("INITIAL_FETCH_COUNT", "40")
)  # Увеличено с 10 до 40

# Параметры расширения контекста
EXPAND_WINDOW = int(
    os.getenv("EXPAND_WINDOW", "1")
)  # Размер окна для расширения контекста соседними чанками (количество чанков с каждой стороны)
EXPAND_TOP_N = int(
    os.getenv("EXPAND_TOP_N", "5")
)  # Количество топ-чанков, для которых добавляются соседи

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Section-aware Re-ranking веса
SECTION_WEIGHTS = {
    "results": {
        "results": 1.5,
        "conclusion": 1.4,
        "discussion": 1.3,
        "abstract": 1.2,
        "introduction": 0.8,
        "methods": 0.6,
        "unknown": 1.0,
    },
    "methods": {
        "methods": 1.5,
        "introduction": 1.1,
        "results": 0.9,
        "discussion": 0.8,
        "conclusion": 0.7,
        "abstract": 1.0,
        "unknown": 1.0,
    },
    "overview": {
        "abstract": 1.3,
        "introduction": 1.2,
        "conclusion": 1.2,
        "results": 1.1,
        "discussion": 1.1,
        "methods": 0.9,
        "unknown": 1.0,
    },
    "definitions": {
        "introduction": 1.4,
        "methods": 1.3,
        "abstract": 1.2,
        "discussion": 1.0,
        "results": 0.9,
        "conclusion": 0.9,
        "unknown": 1.0,
    },
    "default": {
        "results": 1.1,
        "conclusion": 1.1,
        "discussion": 1.0,
        "abstract": 1.1,
        "introduction": 1.0,
        "methods": 0.95,
        "unknown": 1.0,
    },
}

# Ключевые слова для типа запроса
QUERY_TYPE_KEYWORDS = {
    "results": [
        "результат",
        "показал",
        "обнаружил",
        "значение",
        "величина",
        "измерен",
        "получен",
        "скорость",
        "температура",
        "данные",
        "эффект",
        "влияние",
        "изменение",
        "динамика",
        "тренд",
    ],
    "methods": [
        "метод",
        "методика",
        "способ",
        "как измер",
        "как определ",
        "методология",
        "подход",
        "техника",
        "инструмент",
        "алгоритм",
    ],
    "definitions": [
        "что такое",
        "определение",
        "понятие",
        "термин",
        "означает",
        "называется",
        "представляет собой",
        "является",
    ],
    "overview": [
        "обзор",
        "состояние",
        "современн",
        "исследован",
        "литератур",
        "проблема",
        "актуальн",
        "тенденц",
        "перспектив",
    ],
}

# Confidence Score
CONFIDENCE_THRESHOLDS = {
    "high": 0.3,
    "medium": 0.5,
    "low": 0.7,
}
MIN_CHUNKS_FOR_CONFIDENT_ANSWER = 3
MAX_AVG_DISTANCE_FOR_CONFIDENT = 0.5
