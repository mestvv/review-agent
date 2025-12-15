"""Конфигурация приложения."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Директории
ARTICLES_DIR = Path("articles")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "literature_review"
CHUNKS_LOG_DIR = Path("./chunks_log")
RESPONSES_LOG_DIR = Path("./responses_log")
LATEX_OUTPUT_DIR = Path("./latex_output")
PARSED_FILES_LOGS_DIR = Path("./parsed_files_logs")

# Chunking параметры
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
MIN_CHUNK_LENGTH = 100

# Модели
SENTENCE_TRANSFORMER_MODEL = "mlsa-iai-msu-lab/sci-rus-tiny"

# Reranker
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
)
RERANKER_TOP_K = 10
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

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
