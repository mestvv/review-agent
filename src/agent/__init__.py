"""Agent модуль для работы с научной литературой."""

from src.agent.agent import (
    answer_question,
    review_topic,
    search_chunks,
    create_llm,
)
from src.agent.agent_with_tools import (
    run_agent,
    stream_agent,
    chat_with_agent,
    create_rag_agent,
)
from src.agent.tools import (
    search_vector_db,
    list_available_databases,
    search_by_section,
    ALL_TOOLS,
)

__all__ = [
    # Базовый агент
    "answer_question",
    "review_topic",
    "search_chunks",
    "create_llm",
    # Агент с инструментами
    "run_agent",
    "stream_agent",
    "chat_with_agent",
    "create_rag_agent",
    # Инструменты
    "search_vector_db",
    "list_available_databases",
    "search_by_section",
    "ALL_TOOLS",
]
