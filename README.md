# Review Agent

RAG-система для анализа научных статей с генерацией обзоров литературы.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка

Создайте `.env` файл (см. `.env.example`):

```
LLM_MODEL=deepseek/deepseek-r1-0528-qwen3-8b
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
```

## Использование

```bash
# Индексация PDF файлов из папки articles/
python main.py index

# Ответ на вопрос
python main.py ask -q "Ваш вопрос"

# Обзор литературы
python main.py review -t "Тема обзора"

# Поиск чанков (без LLM)
python main.py search -q "запрос"

# Статистика базы
python main.py stats

# Очистка базы
python main.py clear
```

## Структура проекта

```
├── main.py              # CLI точка входа
├── src/
│   ├── config.py        # Конфигурация
│   ├── rag/             # RAG модуль
│   │   ├── indexer.py   # Индексация PDF
│   │   └── retriever.py # Поиск чанков
│   └── agent/           # Agent модуль
│       ├── prompts.py   # Промпты LLM
│       └── literature.py # Основная логика
├── articles/            # PDF файлы для индексации
└── chroma_db/           # Векторная база данных
```
