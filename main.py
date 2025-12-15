"""CLI для Review Agent - инструмент обзора научной литературы."""

import click
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.group()
def cli():
    """Review Agent - инструмент для работы с научной литературой."""
    pass


@cli.command("index")
def cmd_index():
    """Индексация PDF файлов в RAG базу."""
    from src.rag import index_all_pdfs

    index_all_pdfs()


@cli.command("clear")
def cmd_clear():
    """Очистка RAG базы данных."""
    from src.rag import clear_database

    clear_database()


@cli.command("stats")
def cmd_stats():
    """Статистика RAG базы данных."""
    from src.rag import show_stats

    show_stats()


@cli.command("ask")
@click.option("--question", "-q", prompt="Ваш вопрос", help="Вопрос для ответа")
@click.option("--n-results", "-n", default=5, help="Количество чанков для поиска")
def cmd_ask(question: str, n_results: int):
    """Ответ на вопрос по научной литературе."""
    from src.agent import answer_question

    answer_question(question, n_results=n_results)


@cli.command("review")
@click.option("--topic", "-t", prompt="Тема обзора", help="Тема для обзора литературы")
@click.option("--n-results", "-n", default=15, help="Количество чанков для поиска")
def cmd_review(topic: str, n_results: int):
    """Обзор литературы по теме."""
    from src.agent import review_topic

    review_topic(topic, n_results=n_results)


@cli.command("search")
@click.option("--query", "-q", prompt="Поисковый запрос", help="Запрос для поиска")
@click.option("--n-results", "-n", default=10, help="Количество результатов")
@click.option("--section", "-s", default=None, help="Фильтр по секции")
def cmd_search(query: str, n_results: int, section: str):
    """Поиск чанков в RAG базе (без LLM)."""
    from src.agent import search_chunks

    search_chunks(query, n_results=n_results, section=section)


if __name__ == "__main__":
    cli()
