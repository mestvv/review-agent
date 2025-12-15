"""Agent модуль для работы с научной литературой."""

from src.agent.literature import (
    answer_question,
    review_topic,
    search_chunks,
    create_llm,
)

__all__ = [
    "answer_question",
    "review_topic",
    "search_chunks",
    "create_llm",
]
