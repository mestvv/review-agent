# RAG для научных статей

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Индексация PDF

```bash
# Положить PDF файлы в папку articles/
python rag.py index   # индексировать
python rag.py stats   # статистика
python rag.py clear   # очистить базу
```

### Агент

```bash
python agent.py
```

В коде `agent.py`:

```python
agent = LiteratureAgent(llm)

# Ответ на вопрос
agent.answer_question("Ваш вопрос?")

# Обзор литературы
agent.review_topic("Тема обзора")

# Поиск чанков
agent.search_chunks("запрос", section="methods")
```
