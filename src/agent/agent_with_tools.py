"""RAG-агент с инструментами для работы с векторной базой данных."""

import json
import re
import logging
from datetime import datetime
from typing import Optional

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from src.config import (
    RESPONSES_LOG_DIR,
    RESULTS_DIR,
    AGENT_LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_TEMPERATURE,
)
from src.agent.tools import ALL_TOOLS, reset_agent_session_dir

logger = logging.getLogger(__name__)
console = Console()


# Системный промпт для агента
AGENT_SYSTEM_PROMPT = """Ты эксперт-исследователь в области науки, работающий с базой научных статей.

ТВОИ ИНСТРУМЕНТЫ:
1. list_available_databases - узнать какие базы данных доступны
2. search_vector_db - искать информацию в векторной БД по запросу
3. search_by_section - искать в конкретных секциях статей (методы, результаты и т.д.)

ПРАВИЛА РАБОТЫ:
1. ВСЕГДА сначала проверяй доступные базы данных через list_available_databases, если не знаешь какую БД использовать
2. Формулируй поисковые запросы максимально конкретно
3. Если первый поиск не дал хороших результатов - переформулируй запрос или поищи в других секциях
4. Для полного ответа можешь сделать несколько поисковых запросов

ТРЕБОВАНИЯ К ОТВЕТАМ:
1. КАЖДОЕ фактическое утверждение ДОЛЖНО иметь цитату в формате [Файл, стр. X]
2. Используй научный стиль изложения  
3. Если информации недостаточно — честно сообщи об этом
4. Не выдумывай факты, которых нет в найденных фрагментах
5. При низкой уверенности используй осторожные формулировки

ФОРМАТ ФИНАЛЬНОГО ОТВЕТА:
После сбора информации дай структурированный ответ:
- Краткий ответ на вопрос
- Детальное объяснение с цитатами
- Список использованных источников
"""


def create_agent_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    """Создаёт LLM клиент для агента с инструментами.

    Args:
        temperature: Температура для генерации (если None, используется из конфига)

    Returns:
        Инициализированный ChatOpenAI клиент

    Note:
        Используем AGENT_LLM_MODEL (по умолчанию deepseek-chat) вместо
        deepseek-reasoner, так как reasoner требует передачи reasoning_content
        при tool calls, что несовместимо с LangGraph.
        См. https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
    """
    return ChatOpenAI(
        model=AGENT_LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
    )


def create_rag_agent(temperature: Optional[float] = None, new_session: bool = True):
    """Создаёт ReAct агента с инструментами для работы с RAG.

    Args:
        temperature: Температура для генерации (опционально)
        new_session: Создать новую сессию для логирования чанков (по умолчанию True)

    Returns:
        Скомпилированный LangGraph агент
    """
    # Сбрасываем директорию сессии для новой сессии агента
    if new_session:
        reset_agent_session_dir()

    llm = create_agent_llm(temperature)

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=AGENT_SYSTEM_PROMPT,
        name="rag_agent",
    )

    tool_names = ", ".join(t.name for t in ALL_TOOLS)
    logger.info("✅ RAG-агент создан с инструментами: %s", tool_names)
    return agent


def _save_agent_response(
    question: str,
    messages: list,
    response_content: str,
) -> None:
    """Сохраняет ответ агента в JSON."""
    RESPONSES_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r"[^\w\s-]", "", question[:50]).strip().replace(" ", "_")
    filepath = RESPONSES_LOG_DIR / f"agent_{timestamp}_{safe_query}.json"

    # Преобразуем сообщения в сериализуемый формат
    messages_data = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content if hasattr(msg, "content") else str(msg),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
            ]
        if hasattr(msg, "name"):
            msg_dict["name"] = msg.name
        messages_data.append(msg_dict)

    data = {
        "question": question,
        "messages": messages_data,
        "final_response": response_content,
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info("💾 Ответ агента сохранён в %s", filepath)


def _save_agent_result_to_markdown(
    question: str,
    response_content: str,
    tool_calls_count: int,
) -> None:
    """Сохраняет результат работы агента в Markdown."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r"[^\w\s-]", "", question[:50]).strip().replace(" ", "_")
    filepath = RESULTS_DIR / f"agent_{timestamp}_{safe_query}.md"

    md_content = f"""# {question}

**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Тип:** Ответ агента с инструментами  
**Количество вызовов инструментов:** {tool_calls_count}

---

## Ответ

{response_content}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("💾 Результат сохранён в %s", filepath)


def _serialize_messages(messages: list) -> list[dict]:
    """Преобразует список сообщений в сериализуемый формат."""
    messages_data = []
    for msg in messages:
        msg_dict = {
            "type": msg.__class__.__name__,
            "content": msg.content if hasattr(msg, "content") else str(msg),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls
            ]
        if hasattr(msg, "name"):
            msg_dict["name"] = msg.name
        messages_data.append(msg_dict)
    return messages_data


def _save_chat_session_to_json(
    chat_history: list[dict],
    db_name: Optional[str],
    session_start: datetime,
) -> None:
    """Сохраняет сессию чата в JSON."""
    RESPONSES_LOG_DIR.mkdir(exist_ok=True)
    timestamp = session_start.strftime("%Y%m%d_%H%M%S")
    filepath = RESPONSES_LOG_DIR / f"chat_session_{timestamp}.json"

    data = {
        "session_start": session_start.isoformat(),
        "session_end": datetime.now().isoformat(),
        "db_name": db_name,
        "total_exchanges": len(chat_history),
        "exchanges": chat_history,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    logger.info("💾 Сессия чата сохранена в %s", filepath)


def _save_chat_session_to_markdown(
    chat_history: list[dict],
    db_name: Optional[str],
    session_start: datetime,
) -> None:
    """Сохраняет сессию чата в Markdown."""
    if not chat_history:
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = session_start.strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"chat_session_{timestamp}.md"

    md_content = f"""# Сессия чата с RAG-агентом

**Начало сессии:** {session_start.strftime("%Y-%m-%d %H:%M:%S")}  
**Конец сессии:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**База данных:** {db_name or "не указана"}  
**Количество обменов:** {len(chat_history)}

---

"""

    for i, exchange in enumerate(chat_history, 1):
        md_content += f"""## Обмен {i}

**Вопрос:** {exchange["question"]}

**Вызовов инструментов:** {exchange["tool_calls_count"]}

### Ответ

{exchange["response"]}

---

"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("💾 Сессия чата сохранена в Markdown: %s", filepath)


def run_agent(
    question: str,
    db_name: Optional[str] = None,
    temperature: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Запускает агента для ответа на вопрос.

    Агент автоматически ищет информацию в векторной БД,
    может делать несколько запросов для полного ответа.

    Args:
        question: Вопрос пользователя
        db_name: Имя БД для поиска (опционально, агент сам определит)
        temperature: Температура генерации (опционально)
        verbose: Выводить информацию о работе агента

    Returns:
        Финальный ответ агента
    """
    if verbose:
        console.print(Rule("[bold blue]RAG-Агент[/bold blue]"))
        console.print(f"[bold]Вопрос:[/bold] {question}\n")

    # Создаём агента
    agent = create_rag_agent(temperature)

    # Формируем сообщение
    # Если указана БД, добавляем эту информацию
    user_message = question
    if db_name:
        user_message = f"[Используй базу данных: {db_name}]\n\n{question}"

    # Запускаем агента
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    messages = result["messages"]

    # Считаем вызовы инструментов
    tool_calls_count = sum(
        len(msg.tool_calls) if hasattr(msg, "tool_calls") and msg.tool_calls else 0
        for msg in messages
    )

    # Получаем финальный ответ (последнее сообщение от AI)
    final_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Пропускаем сообщения с tool_calls (это промежуточные)
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                final_response = msg.content
                break

    if verbose:
        # Выводим информацию о вызовах инструментов
        console.print(f"[dim]Вызовов инструментов: {tool_calls_count}[/dim]\n")

        # Выводим финальный ответ
        console.print(
            Panel(
                Markdown(final_response),
                title="[bold green]Ответ агента[/bold green]",
                border_style="green",
                expand=True,
            )
        )

    # Сохраняем результаты
    _save_agent_response(question, messages, final_response)
    _save_agent_result_to_markdown(question, final_response, tool_calls_count)

    return final_response


def stream_agent(
    question: str,
    db_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Запускает агента с потоковым выводом.

    Args:
        question: Вопрос пользователя
        db_name: Имя БД для поиска (опционально)
        temperature: Температура генерации (опционально)

    Yields:
        События от агента (сообщения, вызовы инструментов)
    """
    agent = create_rag_agent(temperature)

    user_message = question
    if db_name:
        user_message = f"[Используй базу данных: {db_name}]\n\n{question}"

    # Используем stream для потокового вывода
    for event in agent.stream(
        {"messages": [HumanMessage(content=user_message)]},
        stream_mode="values",
    ):
        yield event


def chat_with_agent(
    db_name: Optional[str] = None,
    temperature: Optional[float] = None,
    save_logs: bool = True,
) -> None:
    """Интерактивный чат с агентом с сохранением истории.

    Args:
        db_name: Имя БД по умолчанию (опционально)
        temperature: Температура генерации (опционально)
        save_logs: Сохранять логи сессии (по умолчанию True)

    Команды:
        exit, quit, выход - выход из чата
        clear, очистить - очистить историю диалога
    """
    console.print(Rule("[bold blue]RAG-Агент — Интерактивный режим[/bold blue]"))
    console.print(
        "[dim]Команды: 'exit'/'quit' - выход, 'clear' - очистить историю[/dim]\n"
    )

    if db_name:
        console.print(f"[dim]База данных по умолчанию: {db_name}[/dim]\n")

    # Создаём агента один раз для сохранения контекста
    agent = create_rag_agent(temperature)

    # История сообщений для агента
    messages: list = []

    # История обменов для логирования
    chat_history: list[dict] = []
    session_start = datetime.now()

    while True:
        try:
            question = console.input("[bold green]Вы:[/bold green] ").strip()

            if not question:
                continue

            # Команды управления
            if question.lower() in ("exit", "quit", "выход"):
                console.print("[dim]До свидания![/dim]")
                break

            if question.lower() in ("clear", "очистить"):
                messages = []
                console.print("[dim]🗑️ История очищена[/dim]\n")
                continue

            # Добавляем новый вопрос к истории
            user_message = question
            if db_name:
                user_message = f"[Используй базу данных: {db_name}]\n\n{question}"

            messages.append(HumanMessage(content=user_message))

            # Запускаем агента со всей историей
            console.print()
            result = agent.invoke({"messages": messages})

            # Обновляем историю из результата
            messages = result["messages"]

            # Считаем вызовы инструментов в этом ответе
            tool_calls_count = sum(
                (
                    len(msg.tool_calls)
                    if hasattr(msg, "tool_calls") and msg.tool_calls
                    else 0
                )
                for msg in messages
            )

            # Получаем последний ответ
            final_response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        final_response = msg.content
                        break

            # Сохраняем обмен в историю для логирования
            chat_history.append(
                {
                    "question": question,
                    "response": final_response,
                    "tool_calls_count": tool_calls_count,
                    "timestamp": datetime.now().isoformat(),
                    "messages_in_context": len(messages),
                }
            )

            # Выводим ответ
            console.print(
                f"[dim]Вызовов инструментов: {tool_calls_count} | Сообщений в истории: {len(messages)}[/dim]\n"
            )
            console.print(
                Panel(
                    Markdown(final_response),
                    title="[bold green]Ответ агента[/bold green]",
                    border_style="green",
                    expand=True,
                )
            )
            console.print()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Прервано пользователем[/dim]")
            break
        except RuntimeError as e:
            console.print(f"[red]Ошибка: {e}[/red]")
            logger.exception("Ошибка в chat_with_agent")

    # Сохраняем логи сессии при выходе
    if save_logs and chat_history:
        _save_chat_session_to_json(chat_history, db_name, session_start)
        _save_chat_session_to_markdown(chat_history, db_name, session_start)
        console.print(f"[dim]📝 Сессия сохранена ({len(chat_history)} обменов)[/dim]")
