"""Streamlit веб-интерфейс для Review Agent."""

import json
import logging
import re
from datetime import datetime
from typing import Optional

import streamlit as st

from src.agent.agent import (
    create_llm,
    stream_llm_answer_qa,
    stream_llm_answer_review,
    _save_response_to_json,
    _save_to_markdown,
)
from src.agent.agent_with_tools import (
    create_rag_agent,
    _save_chat_session_to_json,
    _save_chat_session_to_markdown,
)
from src.agent.prompts import QA_PROMPT, REVIEW_PROMPT
from langchain_core.messages import HumanMessage, AIMessage
from src.config import (
    list_existing_dbs,
    list_available_dbs,
    get_articles_subdir,
    CHUNKS_LOG_DIR,
    EXPAND_WINDOW,
    EXPAND_TOP_N,
    LLM_TEMPERATURE,
)
from src.rag import (
    index_all_pdfs,
    clear_database,
    retrieve_chunks,
    retrieve_with_reranking,
    get_neighbor_chunks,
    format_context_with_citations,
    calculate_confidence,
    RetrievedChunk,
    ConfidenceScore,
    ConfidenceLevel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Настройка страницы
st.set_page_config(
    page_title="Review Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Инициализация состояния сессии для чата с агентом
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # История для агента (LangChain messages)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # История для логирования
if "chat_session_start" not in st.session_state:
    st.session_state.chat_session_start = None
if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = None


def _save_chunks_to_json(
    chunks: list[RetrievedChunk],
    query: str,
    expanded_chunks: Optional[list[RetrievedChunk]] = None,
) -> str:
    """Сохраняет чанки в JSON и возвращает путь к файлу."""
    CHUNKS_LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r"[^\w\s-]", "", query[:50]).strip().replace(" ", "_")
    filepath = CHUNKS_LOG_DIR / f"chunks_{timestamp}_{safe_query}.json"

    expanded_ids = set()
    if expanded_chunks:
        expanded_ids = {f"{c.file_hash}_{c.chunk_id}" for c in expanded_chunks}

    chunks_data = []
    for chunk in chunks:
        chunk_key = f"{chunk.file_hash}_{chunk.chunk_id}"
        chunks_data.append(
            {
                "chunk_id": chunk.chunk_id,
                "file_name": chunk.file_name,
                "file_hash": chunk.file_hash,
                "page": chunk.page,
                "section": chunk.section,
                "distance": chunk.distance,
                "text": chunk.text,
                "is_expanded": chunk_key in expanded_ids,
            }
        )

    data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "expanded_chunks_count": len(expanded_chunks) if expanded_chunks else 0,
        "sources": sorted(set(c.file_name for c in chunks)),
        "chunks": chunks_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(filepath)


def _format_confidence_dict(confidence: ConfidenceScore) -> dict:
    """Преобразует ConfidenceScore в словарь."""
    level_text = {
        ConfidenceLevel.HIGH: "Высокая",
        ConfidenceLevel.MEDIUM: "Средняя",
        ConfidenceLevel.LOW: "Низкая",
        ConfidenceLevel.VERY_LOW: "Очень низкая",
    }

    return {
        "level": level_text.get(confidence.level, "Неизвестна"),
        "level_raw": (
            confidence.level.value
            if hasattr(confidence.level, "value")
            else str(confidence.level)
        ),
        "score": confidence.score,
        "num_sources": confidence.num_sources,
        "num_chunks": confidence.num_chunks,
        "avg_distance": confidence.avg_distance,
        "coverage_by_section": confidence.coverage_by_section,
        "warnings": confidence.warnings,
    }


def _create_response_object(content: str):
    """Создает объект-обертку для ответа LLM из строки.

    Args:
        content: Содержимое ответа

    Returns:
        Объект с методом dict() для совместимости с _save_response_to_json
    """

    class ResponseWrapper:
        def __init__(self, content: str):
            self.content = content

        def dict(self):
            return {"content": self.content}

    return ResponseWrapper(content)


def _format_confidence_for_prompt(confidence: ConfidenceScore, query_type: str) -> str:
    """Форматирует confidence для промпта."""
    level_text = {
        ConfidenceLevel.HIGH: "ВЫСОКАЯ — контекст релевантен, можно давать уверенные ответы",
        ConfidenceLevel.MEDIUM: "СРЕДНЯЯ — контекст частично релевантен, используй осторожные формулировки",
        ConfidenceLevel.LOW: "НИЗКАЯ — контекст слабо релевантен, указывай на ограничения",
        ConfidenceLevel.VERY_LOW: "ОЧЕНЬ НИЗКАЯ — контекст ненадёжен, явно сообщи о недостатке информации",
    }

    info = f"""Уровень релевантности контекста: {level_text.get(confidence.level, 'неизвестен')}
Тип запроса: {query_type}
Количество источников: {confidence.num_sources}
Средняя дистанция: {confidence.avg_distance:.3f}"""

    if confidence.warnings:
        info += f"\nПредупреждения: {'; '.join(confidence.warnings)}"
    return info


def answer_question_web(
    question: str,
    db_name: str,
    n_results: int = 5,
    expand_context: bool = True,
    temperature: Optional[float] = None,
    streaming: bool = False,
):
    """Отвечает на вопрос и возвращает результат для веб-интерфейса.

    Args:
        question: Вопрос
        db_name: Имя базы данных
        n_results: Количество результатов
        expand_context: Расширять контекст
        temperature: Температура генерации
        streaming: Использовать потоковый вывод (возвращает генератор)

    Returns:
        Если streaming=False: словарь с результатами
        Если streaming=True: генератор токенов и метаданные
    """
    initial_chunks, query_type, confidence = retrieve_with_reranking(
        question, db_name, n_results, fetch_multiplier=2
    )

    if not initial_chunks:
        if streaming:
            return None, {"success": False, "error": "Релевантный контекст не найден"}
        return {
            "success": False,
            "error": "Релевантный контекст не найден",
        }

    # Расширяем контекст соседями
    expanded_chunks = []
    if expand_context:
        seen_ids = {f"{c.file_hash}_{c.chunk_id}" for c in initial_chunks}
        top_n_for_expansion = min(EXPAND_TOP_N, len(initial_chunks))
        for chunk in initial_chunks[:top_n_for_expansion]:
            neighbors = get_neighbor_chunks(
                chunk, db_name, window=EXPAND_WINDOW, query=question
            )
            for n in neighbors:
                key = f"{n.file_hash}_{n.chunk_id}"
                if key not in seen_ids:
                    expanded_chunks.append(n)
                    seen_ids.add(key)
        chunks = (
            initial_chunks[:top_n_for_expansion]
            + expanded_chunks
            + initial_chunks[top_n_for_expansion:]
        )
    else:
        chunks = initial_chunks

    _save_chunks_to_json(chunks, question, expanded_chunks if expand_context else None)

    context = format_context_with_citations(chunks[:10])
    confidence_info = _format_confidence_for_prompt(confidence, query_type)

    # Форматируем источники
    sources = {}
    for chunk in chunks:
        if chunk.file_name not in sources:
            sources[chunk.file_name] = {
                "pages": set(),
                "sections": set(),
            }
        sources[chunk.file_name]["pages"].add(chunk.page)
        sources[chunk.file_name]["sections"].add(chunk.section)

    sources_list = []
    for source_name, info in sorted(sources.items()):
        sources_list.append(
            {
                "file_name": source_name,
                "pages": sorted(info["pages"]),
                "sections": sorted(info["sections"]),
            }
        )

    metadata = {
        "success": True,
        "confidence": _format_confidence_dict(confidence),
        "query_type": query_type,
        "sources": sources_list,
        "chunks_count": len(chunks),
    }

    if streaming:
        # Возвращаем генератор и метаданные
        # Используем список для хранения полного ответа (чтобы обновить метаданные)
        full_answer_container = [""]

        def answer_generator():
            for token in stream_llm_answer_qa(
                question, context, confidence_info, temperature
            ):
                full_answer_container[0] += token
                yield token
            # Сохраняем полный ответ после завершения
            full_answer = full_answer_container[0]
            metadata["answer"] = full_answer

            # Сохраняем логи после завершения генерации
            query_data = {
                "question": question,
                "context": context,
                "confidence": confidence.to_dict(),
            }
            response_obj = _create_response_object(full_answer)
            _save_response_to_json(query_data, response_obj)
            _save_to_markdown(
                query=question,
                response_content=full_answer,
                chunks=chunks,
                query_type="ask",
                confidence=confidence,
            )

        return answer_generator(), metadata
    else:
        # Обычный режим без streaming
        llm = create_llm(temperature=temperature)
        response = (QA_PROMPT | llm).invoke(
            {
                "question": question,
                "context": context,
                "confidence_info": confidence_info,
            }
        )
        metadata["answer"] = response.content

        # Сохраняем логи
        query_data = {
            "question": question,
            "context": context,
            "confidence": confidence.to_dict(),
        }
        _save_response_to_json(query_data, response)
        _save_to_markdown(
            query=question,
            response_content=response.content,
            chunks=chunks,
            query_type="ask",
            confidence=confidence,
        )

        return metadata


def review_topic_web(
    topic: str,
    db_name: str,
    n_results: int = 15,
    temperature: Optional[float] = None,
    streaming: bool = False,
):
    """Создает обзор литературы и возвращает результат для веб-интерфейса.

    Args:
        topic: Тема обзора
        db_name: Имя базы данных
        n_results: Количество результатов
        temperature: Температура генерации
        streaming: Использовать потоковый вывод (возвращает генератор)

    Returns:
        Если streaming=False: словарь с результатами
        Если streaming=True: генератор токенов и метаданные
    """
    from src.rag.retriever import detect_query_type

    query_type = detect_query_type(topic)
    all_chunks, query_type, confidence = retrieve_with_reranking(
        topic, db_name, n_results, fetch_multiplier=2
    )

    if not all_chunks:
        if streaming:
            return None, {"success": False, "error": "Нет данных для обзора"}
        return {
            "success": False,
            "error": "Нет данных для обзора",
        }

    confidence = calculate_confidence(all_chunks, query_type)
    _save_chunks_to_json(all_chunks, topic)

    context = format_context_with_citations(all_chunks)

    sources_detail = []
    seen = set()
    for chunk in all_chunks:
        if chunk.file_name not in seen:
            sources_detail.append(f"• {chunk.file_name}")
            seen.add(chunk.file_name)

    confidence_info = _format_confidence_for_prompt(confidence, query_type)

    # Форматируем источники
    sources = {}
    for chunk in all_chunks:
        if chunk.file_name not in sources:
            sources[chunk.file_name] = {
                "pages": set(),
                "sections": set(),
            }
        sources[chunk.file_name]["pages"].add(chunk.page)
        sources[chunk.file_name]["sections"].add(chunk.section)

    sources_list = []
    for source_name, info in sorted(sources.items()):
        sources_list.append(
            {
                "file_name": source_name,
                "pages": sorted(info["pages"]),
                "sections": sorted(info["sections"]),
            }
        )

    metadata = {
        "success": True,
        "confidence": _format_confidence_dict(confidence),
        "query_type": query_type,
        "sources": sources_list,
        "chunks_count": len(all_chunks),
    }

    if streaming:
        # Возвращаем генератор и метаданные
        # Используем список для хранения полного обзора (чтобы обновить метаданные)
        full_review_container = [""]

        def review_generator():
            for token in stream_llm_answer_review(
                topic,
                context[:8000],
                "\n".join(sources_detail),
                confidence_info,
                temperature,
            ):
                full_review_container[0] += token
                yield token
            # Сохраняем полный обзор после завершения
            full_review = full_review_container[0]
            metadata["review"] = full_review

            # Сохраняем логи после завершения генерации
            query_data = {
                "topic": topic,
                "context": context[:8000],
                "confidence": confidence.to_dict(),
            }
            response_obj = _create_response_object(full_review)
            _save_response_to_json(query_data, response_obj)
            _save_to_markdown(
                query=topic,
                response_content=full_review,
                chunks=all_chunks,
                query_type="review",
                confidence=confidence,
            )

        return review_generator(), metadata
    else:
        # Обычный режим без streaming
        llm = create_llm(temperature=temperature)
        response = (REVIEW_PROMPT | llm).invoke(
            {
                "topic": topic,
                "context": context[:8000],
                "sources": "\n".join(sources_detail),
                "confidence_info": confidence_info,
            }
        )
        metadata["review"] = response.content

        # Сохраняем логи
        query_data = {
            "topic": topic,
            "context": context[:8000],
            "confidence": confidence.to_dict(),
        }
        _save_response_to_json(query_data, response)
        _save_to_markdown(
            query=topic,
            response_content=response.content,
            chunks=all_chunks,
            query_type="review",
            confidence=confidence,
        )

        return metadata


def get_stats_dict(db_name: Optional[str] = None) -> dict:
    """Получает статистику БД в виде словаря."""
    from src.rag.indexer import _get_collection

    # Используем импортированную функцию из модуля
    import src.config as config

    if db_name:
        collection = _get_collection(db_name)
        total = collection.count()
        if total == 0:
            return {"db_name": db_name, "total": 0, "files": {}, "sections": {}}

        results = collection.get(include=["metadatas"])
        metadatas = results["metadatas"]

        files = {}
        sections = {}
        for metadata in metadatas:
            fname = metadata.get("file_name", "unknown")
            section = metadata.get("section", "unknown")
            files[fname] = files.get(fname, 0) + 1
            sections[section] = sections.get(section, 0) + 1

        return {
            "db_name": db_name,
            "total": total,
            "files": files,
            "sections": sections,
        }
    else:
        existing_dbs_list = config.list_existing_dbs()
        stats_dict = {}
        total_chunks = 0
        for db_item in existing_dbs_list:
            collection = _get_collection(db_item)
            count = collection.count()
            total_chunks += count
            stats_dict[db_item] = {"total": count}

        return {
            "all_dbs": stats_dict,
            "total_chunks": total_chunks,
        }


# Боковая панель для управления БД
with st.sidebar:
    st.title("📚 Review Agent")
    st.markdown("---")

    st.subheader("Базы данных")

    existing_dbs = list_existing_dbs()
    available_dbs = list_available_dbs()

    if existing_dbs:
        selected_db = st.selectbox(
            "Выберите базу данных",
            existing_dbs,
            key="selected_db",
        )
    else:
        st.warning("Нет индексированных баз данных")
        selected_db = None

    st.markdown("---")

    # Управление БД
    st.subheader("Управление")

    with st.expander("📊 Статистика"):
        if st.button("Показать статистику", use_container_width=True):
            if selected_db:
                stats = get_stats_dict(selected_db)
                st.json(stats)
            else:
                stats = get_stats_dict()
                st.json(stats)

    with st.expander("🔄 Индексация"):
        db_to_index = st.selectbox(
            "База для индексации",
            [None] + available_dbs,
            key="db_to_index",
        )
        if st.button("Индексировать", use_container_width=True):
            if db_to_index:
                with st.spinner(f"Индексация базы '{db_to_index}'..."):
                    try:
                        index_all_pdfs(db_name=db_to_index)
                        st.success(f"База '{db_to_index}' проиндексирована!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
            else:
                st.warning("Выберите базу для индексации")

    with st.expander("🗑️ Удаление"):
        db_to_delete = st.selectbox(
            "База для удаления",
            [None] + existing_dbs,
            key="db_to_delete",
        )
        if st.button("Удалить", use_container_width=True, type="primary"):
            if db_to_delete:
                try:
                    clear_database(db_name=db_to_delete)
                    st.success(f"База '{db_to_delete}' удалена!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")
            else:
                st.warning("Выберите базу для удаления")

# Главная область
st.title("📚 Review Agent")
st.markdown("RAG-система для анализа научных статей с генерацией обзоров литературы")

if not selected_db:
    st.info("👈 Выберите базу данных в боковой панели или создайте новую")
    st.markdown("---")
    st.subheader("Доступные директории для индексации")
    if available_dbs:
        for db in available_dbs:
            articles_subdir = get_articles_subdir(db)
            pdf_count = len(list(articles_subdir.glob("*.pdf")))
            md_count = len(list(articles_subdir.glob("*.md")))
            docx_count = len(list(articles_subdir.glob("*.docx")))
            st.write(f"**{db}**: {pdf_count} PDF, {md_count} MD, {docx_count} DOCX")
    else:
        st.warning(
            "Нет доступных директорий. Создайте поддиректории в папке `articles/`"
        )
else:
    # Вкладки для разных функций
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Задать вопрос", "📝 Обзор литературы", "🔍 Поиск чанков", "🤖 Чат с агентом"]
    )

    with tab1:
        st.subheader("Задать вопрос")

        question = st.text_area(
            "Ваш вопрос",
            placeholder="Например: С какой скоростью происходит глобальное потепление?",
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            n_results = st.slider("Количество чанков", 1, 20, 5)
        with col2:
            expand_context = st.checkbox("Расширять контекст", value=True)

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=LLM_TEMPERATURE,
            step=0.1,
            help="Температура генерации: низкие значения делают ответы более детерминированными, высокие - более креативными",
            key="ask_temperature",
        )

        if st.button("Отправить вопрос", type="primary", use_container_width=True):
            if question:
                # Получаем генератор и метаданные для потокового вывода
                answer_gen, metadata = answer_question_web(
                    question,
                    selected_db,
                    n_results=n_results,
                    expand_context=expand_context,
                    temperature=temperature,
                    streaming=True,
                )

                if metadata and metadata.get("success"):
                    # Отображение уверенности
                    confidence = metadata["confidence"]
                    level_colors = {
                        "Высокая": "🟢",
                        "Средняя": "🟡",
                        "Низкая": "🟠",
                        "Очень низкая": "🔴",
                    }
                    icon = level_colors.get(confidence["level"], "⚪")

                    st.markdown("---")
                    st.markdown(f"### {icon} Уверенность: {confidence['level']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Источников", confidence["num_sources"])
                    with col2:
                        st.metric("Чанков", confidence["num_chunks"])
                    with col3:
                        st.metric(
                            "Средняя дистанция", f"{confidence['avg_distance']:.3f}"
                        )

                    # Потоковый вывод ответа
                    st.markdown("---")
                    st.markdown("### Ответ")

                    # Используем st.write_stream для потокового вывода
                    # st.write_stream автоматически обновляет UI по мере поступления токенов
                    full_answer = st.write_stream(answer_gen)

                    # Полный ответ уже сохранен в metadata через генератор

                    # Источники
                    if metadata.get("sources"):
                        st.markdown("---")
                        st.markdown("### Использованные источники")
                        for source in metadata["sources"]:
                            pages_str = ", ".join(map(str, source["pages"]))
                            sections_str = (
                                ", ".join(source["sections"])
                                if source["sections"]
                                else "—"
                            )
                            with st.expander(f"📄 {source['file_name']}"):
                                st.write(f"**Страницы:** {pages_str}")
                                st.write(f"**Секции:** {sections_str}")
                else:
                    error_msg = (
                        metadata.get("error", "Произошла ошибка")
                        if metadata
                        else "Произошла ошибка"
                    )
                    st.error(error_msg)
            else:
                st.warning("Введите вопрос")

    with tab2:
        st.subheader("Обзор литературы")

        topic = st.text_area(
            "Тема обзора",
            placeholder="Например: Глобальное потепление и состояние вечной мерзлоты в России",
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            n_results = st.slider("Количество чанков", 5, 30, 15)
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=LLM_TEMPERATURE,
                step=0.1,
                help="Температура генерации: низкие значения делают ответы более детерминированными, высокие - более креативными",
                key="review_temperature",
            )

        if st.button("Создать обзор", type="primary", use_container_width=True):
            if topic:
                # Получаем генератор и метаданные для потокового вывода
                review_gen, metadata = review_topic_web(
                    topic,
                    selected_db,
                    n_results=n_results,
                    temperature=temperature,
                    streaming=True,
                )

                if metadata and metadata.get("success"):
                    # Отображение уверенности
                    confidence = metadata["confidence"]
                    level_colors = {
                        "Высокая": "🟢",
                        "Средняя": "🟡",
                        "Низкая": "🟠",
                        "Очень низкая": "🔴",
                    }
                    icon = level_colors.get(confidence["level"], "⚪")

                    st.markdown("---")
                    st.markdown(f"### {icon} Уверенность: {confidence['level']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Источников", confidence["num_sources"])
                    with col2:
                        st.metric("Чанков", confidence["num_chunks"])
                    with col3:
                        st.metric(
                            "Средняя дистанция", f"{confidence['avg_distance']:.3f}"
                        )

                    # Потоковый вывод обзора
                    st.markdown("---")
                    st.markdown("### Обзор литературы")

                    # Используем st.write_stream для потокового вывода
                    full_review = st.write_stream(review_gen)

                    # Источники
                    if metadata.get("sources"):
                        st.markdown("---")
                        st.markdown("### Использованные источники")
                        for source in metadata["sources"]:
                            pages_str = ", ".join(map(str, source["pages"]))
                            sections_str = (
                                ", ".join(source["sections"])
                                if source["sections"]
                                else "—"
                            )
                            with st.expander(f"📄 {source['file_name']}"):
                                st.write(f"**Страницы:** {pages_str}")
                                st.write(f"**Секции:** {sections_str}")
                else:
                    error_msg = (
                        metadata.get("error", "Произошла ошибка")
                        if metadata
                        else "Произошла ошибка"
                    )
                    st.error(error_msg)
            else:
                st.warning("Введите тему обзора")

    with tab3:
        st.subheader("Поиск чанков")

        query = st.text_input(
            "Поисковый запрос",
            placeholder="Введите запрос для поиска",
        )

        col1, col2 = st.columns(2)
        with col1:
            n_results = st.slider("Количество результатов", 1, 50, 10)
        with col2:
            section = st.selectbox(
                "Фильтр по секции",
                [
                    None,
                    "abstract",
                    "introduction",
                    "methods",
                    "results",
                    "discussion",
                    "conclusion",
                ],
            )

        if st.button("Поиск", type="primary", use_container_width=True):
            if query:
                with st.spinner("Поиск чанков..."):
                    chunks = retrieve_chunks(query, selected_db, n_results, section)

                    if chunks:
                        st.markdown(f"### Найдено чанков: {len(chunks)}")
                        st.markdown("---")

                        for i, chunk in enumerate(chunks, 1):
                            with st.expander(
                                f"Чанк {i}: {chunk.file_name} (стр. {chunk.page}, секция: {chunk.section}, dist: {chunk.distance:.3f})"
                            ):
                                st.text(chunk.text)
                    else:
                        st.warning("Ничего не найдено")
            else:
                st.warning("Введите поисковый запрос")

    with tab4:
        st.subheader("🤖 Чат с RAG-агентом")
        st.markdown(
            """
            Агент автоматически ищет информацию в базе данных и отвечает на вопросы.
            История диалога сохраняется между сообщениями.
            """
        )

        # Настройки агента
        with st.expander("⚙️ Настройки", expanded=False):
            agent_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=LLM_TEMPERATURE,
                step=0.1,
                help="Температура генерации",
                key="agent_temperature",
            )

        # Кнопки управления
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Очистить историю", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.chat_history = []
                st.session_state.chat_agent = None
                st.session_state.chat_session_start = None
                st.rerun()
        with col2:
            if st.button("💾 Сохранить сессию", use_container_width=True):
                if st.session_state.chat_history:
                    _save_chat_session_to_json(
                        st.session_state.chat_history,
                        selected_db,
                        st.session_state.chat_session_start or datetime.now(),
                    )
                    _save_chat_session_to_markdown(
                        st.session_state.chat_history,
                        selected_db,
                        st.session_state.chat_session_start or datetime.now(),
                    )
                    st.success(f"Сессия сохранена ({len(st.session_state.chat_history)} обменов)")
                else:
                    st.warning("История пуста")
        with col3:
            st.metric("Сообщений", len(st.session_state.chat_messages))

        st.markdown("---")

        # Отображение истории чата
        chat_container = st.container()
        with chat_container:
            for exchange in st.session_state.chat_history:
                # Сообщение пользователя
                with st.chat_message("user"):
                    st.markdown(exchange["question"])
                # Ответ агента
                with st.chat_message("assistant"):
                    st.markdown(exchange["response"])
                    st.caption(f"🔧 Вызовов инструментов: {exchange['tool_calls_count']}")

        # Поле ввода нового сообщения
        if prompt := st.chat_input("Задайте вопрос агенту..."):
            # Инициализируем агента если нужно
            if st.session_state.chat_agent is None:
                st.session_state.chat_agent = create_rag_agent(agent_temperature)
                st.session_state.chat_session_start = datetime.now()

            # Отображаем сообщение пользователя
            with st.chat_message("user"):
                st.markdown(prompt)

            # Формируем сообщение с указанием БД
            user_message = f"[Используй базу данных: {selected_db}]\n\n{prompt}"
            st.session_state.chat_messages.append(HumanMessage(content=user_message))

            # Получаем ответ от агента
            with st.chat_message("assistant"):
                with st.spinner("Агент думает..."):
                    try:
                        result = st.session_state.chat_agent.invoke(
                            {"messages": st.session_state.chat_messages}
                        )

                        # Обновляем историю сообщений
                        st.session_state.chat_messages = result["messages"]

                        # Считаем вызовы инструментов
                        tool_calls_count = sum(
                            len(msg.tool_calls)
                            if hasattr(msg, "tool_calls") and msg.tool_calls
                            else 0
                            for msg in st.session_state.chat_messages
                        )

                        # Получаем последний ответ
                        final_response = ""
                        for msg in reversed(st.session_state.chat_messages):
                            if isinstance(msg, AIMessage) and msg.content:
                                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                    final_response = msg.content
                                    break

                        # Отображаем ответ
                        st.markdown(final_response)
                        st.caption(f"🔧 Вызовов инструментов: {tool_calls_count}")

                        # Сохраняем в историю для логирования
                        st.session_state.chat_history.append({
                            "question": prompt,
                            "response": final_response,
                            "tool_calls_count": tool_calls_count,
                            "timestamp": datetime.now().isoformat(),
                            "messages_in_context": len(st.session_state.chat_messages),
                        })

                    except Exception as e:
                        st.error(f"Ошибка: {e}")
                        logging.exception("Ошибка в чате с агентом")
