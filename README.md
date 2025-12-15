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

## Работа с множественными базами данных

Система поддерживает создание нескольких независимых баз данных. Для этого создайте поддиректории в папке `articles/` и поместите в них PDF файлы:

```
articles/
├── climate/           # БД для климатических статей
│   ├── paper1.pdf
│   └── paper2.pdf
├── biology/           # БД для биологических статей
│   ├── paper3.pdf
│   └── paper4.pdf
└── physics/           # БД для физических статей
    └── paper5.pdf
```

Каждая поддиректория будет индексирована в отдельную базу данных ChromaDB.

## Использование

### Просмотр доступных баз данных

```bash
python main.py list-dbs
```

### Индексация

```bash
# Индексация всех баз данных (все поддиректории в articles/)
python main.py index

# Индексация конкретной БД
python main.py index --db climate
```

### Работа с запросами

При выполнении запросов можно указать БД через опцию `--db` или выбрать интерактивно:

```bash
# Ответ на вопрос (с выбором БД)
python main.py ask -q "Ваш вопрос"

# Ответ на вопрос из конкретной БД
python main.py ask -q "Ваш вопрос" --db climate

# Обзор литературы
python main.py review -t "Тема обзора" --db biology

# Поиск чанков (без LLM)
python main.py search -q "запрос" --db physics
```

### Статистика и управление

```bash
# Статистика всех баз данных
python main.py stats

# Статистика конкретной БД
python main.py stats --db climate

# Очистка БД (интерактивный выбор)
python main.py clear

# Очистка конкретной БД
python main.py clear --db climate
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
│   ├── climate/         # Поддиректория для климатических статей
│   ├── biology/         # Поддиректория для биологических статей
│   └── physics/         # Поддиректория для физических статей
└── chroma_db/           # Векторные базы данных
    ├── climate/         # БД для climate
    ├── biology/         # БД для biology
    └── physics/         # БД для physics
```

## Примеры использования

### Создание новой базы данных

1. Создайте новую директорию в `articles/`:
   ```bash
   mkdir articles/my_topic
   ```

2. Поместите PDF файлы в эту директорию:
   ```bash
   cp *.pdf articles/my_topic/
   ```

3. Индексируйте базу данных:
   ```bash
   python main.py index --db my_topic
   ```

4. Используйте новую БД для запросов:
   ```bash
   python main.py ask -q "Ваш вопрос" --db my_topic
   ```

### Работа с несколькими базами данных

```bash
# Просмотр всех баз
python main.py list-dbs

# Статистика по всем базам
python main.py stats

# Удаление конкретной базы
python main.py clear --db old_topic
```
