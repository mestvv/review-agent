"""CLI для Review Agent - инструмент обзора научной литературы."""

import click
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def select_database(db_name: Optional[str] = None) -> Optional[str]:
    """Выбрать базу данных интерактивно или использовать указанную.

    Args:
        db_name: Имя БД (если указано) или None для интерактивного выбора

    Returns:
        Имя выбранной БД или None при ошибке
    """
    from src.config import list_existing_dbs

    if db_name:
        # Проверяем, существует ли указанная БД
        existing = list_existing_dbs()
        if db_name not in existing:
            click.echo(f"❌ База данных '{db_name}' не найдена.")
            click.echo(f"Доступные БД: {', '.join(existing)}")
            return None
        return db_name

    # Интерактивный выбор
    existing = list_existing_dbs()
    if not existing:
        click.echo("❌ Нет существующих баз данных. Сначала выполните индексацию.")
        return None

    if len(existing) == 1:
        # Если только одна БД, используем её
        click.echo(f"📚 Используется БД: {existing[0]}")
        return existing[0]

    # Выбор из списка
    click.echo("\n📚 Доступные базы данных:")
    for i, db in enumerate(existing, 1):
        click.echo(f"  {i}. {db}")

    try:
        choice = click.prompt("Выберите номер БД", type=int)
        if 1 <= choice <= len(existing):
            return existing[choice - 1]
        else:
            click.echo("❌ Неверный выбор")
            return None
    except (ValueError, click.Abort):
        click.echo("\n❌ Отменено")
        return None


@click.group()
def cli():
    """Review Agent - инструмент для работы с научной литературой."""
    pass


@cli.command("list-dbs")
def cmd_list_dbs():
    """Список доступных баз данных."""
    from src.rag import list_dbs

    list_dbs()


@cli.command("index")
@click.option("--db", "-d", default=None, help="Имя БД (поддиректория в articles/)")
def cmd_index(db: Optional[str]):
    """Индексация PDF файлов в RAG базу."""
    from src.rag import index_all_pdfs

    index_all_pdfs(db_name=db)


@cli.command("clear")
@click.option(
    "--db",
    "-d",
    default=None,
    help="Имя БД для удаления (если не указано, будет предложен выбор)",
)
def cmd_clear(db: Optional[str]):
    """Очистка RAG базы данных."""
    from src.rag import clear_database

    clear_database(db_name=db)


@cli.command("stats")
@click.option(
    "--db",
    "-d",
    default=None,
    help="Имя БД для статистики (если не указано, показывает все)",
)
def cmd_stats(db: Optional[str]):
    """Статистика RAG базы данных."""
    from src.rag import show_stats

    show_stats(db_name=db)


@cli.command("ask")
@click.option("--question", "-q", default=None, help="Вопрос для ответа")
@click.option("--n-results", "-n", default=5, help="Количество чанков для поиска")
@click.option("--db", "-d", default=None, help="Имя БД для поиска")
def cmd_ask(question: Optional[str], n_results: int, db: Optional[str]):
    """Ответ на вопрос по научной литературе."""
    from src.agent import answer_question

    # Выбираем БД
    db_name = select_database(db)
    if not db_name:
        return

    # Запрашиваем вопрос, если не указан
    if not question:
        question = click.prompt("Ваш вопрос")

    answer_question(question, db_name, n_results=n_results)


@cli.command("review")
@click.option("--topic", "-t", default=None, help="Тема для обзора литературы")
@click.option("--n-results", "-n", default=15, help="Количество чанков для поиска")
@click.option("--db", "-d", default=None, help="Имя БД для поиска")
def cmd_review(topic: Optional[str], n_results: int, db: Optional[str]):
    """Обзор литературы по теме."""
    from src.agent import review_topic

    # Выбираем БД
    db_name = select_database(db)
    if not db_name:
        return

    # Запрашиваем тему, если не указана
    if not topic:
        topic = click.prompt("Тема обзора")

    review_topic(topic, db_name, n_results=n_results)


@cli.command("search")
@click.option("--query", "-q", default=None, help="Запрос для поиска")
@click.option("--n-results", "-n", default=10, help="Количество результатов")
@click.option("--section", "-s", default=None, help="Фильтр по секции")
@click.option("--db", "-d", default=None, help="Имя БД для поиска")
def cmd_search(
    query: Optional[str], n_results: int, section: Optional[str], db: Optional[str]
):
    """Поиск чанков в RAG базе (без LLM)."""
    from src.agent import search_chunks

    # Выбираем БД
    db_name = select_database(db)
    if not db_name:
        return

    # Запрашиваем запрос, если не указан
    if not query:
        query = click.prompt("Поисковый запрос")

    search_chunks(query, db_name, n_results=n_results, section=section)


if __name__ == "__main__":
    cli()
