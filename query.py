import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Инициализация
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "literature_review"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
model = SentenceTransformer("mlsa-iai-msu-lab/sci-rus-tiny")

llm = ChatOpenAI(
    model="deepseek/deepseek-r1-0528-qwen3-8b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.1,
)


def retrieve_context(question, n_results=5):
    """Получает релевантный контекст из базы данных."""
    query_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    return results["documents"][0], results["metadatas"][0]


def ask(question, n_results=5):
    """Отвечает на вопрос на основе литературы."""
    documents, metadatas = retrieve_context(question, n_results)
    context = "\n\n---\n\n".join(documents)

    prompt = f"""Ты эксперт-исследователь. На основе контекста ответь на вопрос.

КОНТЕКСТ:
{context[:3000]}

ВОПРОС: {question}

Ответь кратко, ссылаясь на источники."""

    response = llm.invoke(prompt)

    # Добавляем источники
    sources = set(m["document"] for m in metadatas)
    return f"{response.content}\n\nИсточники: {', '.join(sources)}"


# === Обзор литературы ===

REVIEW_PROMPT = PromptTemplate(
    input_variables=["topic", "context", "sources"],
    template="""Ты научный исследователь. Напиши обзор литературы на тему "{topic}".

ИЗВЛЕЧЕННЫЕ ФРАГМЕНТЫ ИЗ СТАТЕЙ:
{context}

ИСТОЧНИКИ: {sources}

ИНСТРУКЦИИ:
1. Проанализируй и синтезируй информацию из разных источников
2. Выдели ключевые темы и тенденции
3. Укажи противоречия или пробелы в исследованиях (если есть)
4. Ссылайся на конкретные источники
5. Пиши в научном стиле

СТРУКТУРА ОБЗОРА:
- Введение (актуальность темы)
- Основные результаты исследований
- Обсуждение и выводы
- Список источников""",
)


def generate_review(topic, n_results=15):
    """Генерирует обзор литературы по заданной теме."""
    # Получаем релевантные фрагменты
    documents, metadatas = retrieve_context(topic, n_results)

    if not documents:
        return "Нет данных по этой теме в базе."

    context = "\n\n---\n\n".join(documents)
    sources = list(set(m["document"] for m in metadatas))

    chain = REVIEW_PROMPT | llm
    response = chain.invoke(
        {
            "topic": topic,
            "context": context[:6000],
            "sources": ", ".join(sources),
        }
    )

    return response.content


def stream_review(topic, n_results=15):
    """Стримит обзор литературы."""
    documents, metadatas = retrieve_context(topic, n_results)

    if not documents:
        yield "Нет данных по этой теме в базе."
        return

    context = "\n\n---\n\n".join(documents)
    sources = list(set(m["document"] for m in metadatas))

    chain = REVIEW_PROMPT | llm

    for chunk in chain.stream(
        {
            "topic": topic,
            "context": context[:6000],
            "sources": ", ".join(sources),
        }
    ):
        yield getattr(chunk, "content", str(chunk))


# === Gradio интерфейс ===

import gradio as gr


def review_interface(topic):
    """Интерфейс для генерации обзора."""
    if not topic.strip():
        yield "Введите тему для обзора."
        return

    review_text = ""
    for token in stream_review(topic):
        review_text += token
        yield review_text


def qa_interface(question):
    """Интерфейс для Q&A."""
    if not question.strip():
        return "Введите вопрос."
    return ask(question)


with gr.Blocks(title="Literature Review Agent") as demo:
    gr.Markdown("# 📚 Агент обзора литературы")

    with gr.Tab("Обзор литературы"):
        topic_input = gr.Textbox(
            label="Тема обзора",
            placeholder="Влияние изменения климата на многолетнемерзлые грунты",
        )
        review_btn = gr.Button("Сгенерировать обзор", variant="primary")
        review_output = gr.Markdown(label="Обзор")
        review_btn.click(review_interface, topic_input, review_output)

    with gr.Tab("Вопрос-Ответ"):
        question_input = gr.Textbox(
            label="Вопрос",
            placeholder="Как изменяется температура вечной мерзлоты?",
        )
        qa_btn = gr.Button("Получить ответ", variant="primary")
        qa_output = gr.Markdown(label="Ответ")
        qa_btn.click(qa_interface, question_input, qa_output)


if __name__ == "__main__":
    demo.queue().launch()
