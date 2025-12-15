from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# RAG configuration
ARTICLES_DIR = Path("articles")
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "literature_review"

# Chunking параметры
CHUNK_SIZE = 600  # ~400-800 токенов
CHUNK_OVERLAP = 120  # ~20% overlap

# Фильтрация
MIN_CHUNK_LENGTH = 100  # минимум символов в чанке

SENTENCE_TRANSFORMER_MODEL = "mlsa-iai-msu-lab/sci-rus-tiny"

# ============ Reranker Configuration ============
# Варианты:
# - "jinaai/jina-reranker-v2-base-multilingual" (лучший multilingual, ~550MB)
# - "BAAI/bge-reranker-v2-m3" (хороший multilingual, ~560MB)
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" (лёгкий, но только English)
# - None — использовать эвристику с весами секций (без ML модели)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
)
RERANKER_TOP_K = 10  # сколько документов передавать в reranker
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"


# Agent configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
CHUNKS_LOG_DIR = Path("./chunks_log")
RESPONSES_LOG_DIR = Path("./responses_log")
LATEX_OUTPUT_DIR = Path("./latex_output")

# ============ Section-aware Re-ranking ============
# Веса секций для разных типов запросов
# Ключ - тип запроса, значение - словарь весов секций

SECTION_WEIGHTS = {
    # Для вопросов о результатах/выводах - предпочитаем results, conclusion, discussion
    "results": {
        "results": 1.5,
        "conclusion": 1.4,
        "discussion": 1.3,
        "abstract": 1.2,
        "introduction": 0.8,
        "methods": 0.6,
        "unknown": 1.0,
    },
    # Для вопросов о методах - предпочитаем methods
    "methods": {
        "methods": 1.5,
        "introduction": 1.1,
        "results": 0.9,
        "discussion": 0.8,
        "conclusion": 0.7,
        "abstract": 1.0,
        "unknown": 1.0,
    },
    # Для общих обзорных вопросов - сбалансированные веса
    "overview": {
        "abstract": 1.3,
        "introduction": 1.2,
        "conclusion": 1.2,
        "results": 1.1,
        "discussion": 1.1,
        "methods": 0.9,
        "unknown": 1.0,
    },
    # Для вопросов об определениях/терминах - введение и методы
    "definitions": {
        "introduction": 1.4,
        "methods": 1.3,
        "abstract": 1.2,
        "discussion": 1.0,
        "results": 0.9,
        "conclusion": 0.9,
        "unknown": 1.0,
    },
    # По умолчанию - нейтральные веса
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

# Ключевые слова для определения типа запроса
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

# ============ Confidence Score Configuration ============
# Пороги для оценки уверенности

CONFIDENCE_THRESHOLDS = {
    "high": 0.3,  # distance < 0.3 → высокая релевантность
    "medium": 0.5,  # 0.3 <= distance < 0.5 → средняя
    "low": 0.7,  # 0.5 <= distance < 0.7 → низкая
    # distance >= 0.7 → очень низкая / ненадёжно
}

# Минимальное количество чанков для достаточного контекста
MIN_CHUNKS_FOR_CONFIDENT_ANSWER = 3

# Максимально допустимое среднее расстояние
MAX_AVG_DISTANCE_FOR_CONFIDENT = 0.5
