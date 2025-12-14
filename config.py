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


# Agent configuration
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
CHUNKS_LOG_DIR = Path("./chunks_log")
RESPONSES_LOG_DIR = Path("./responses_log")
