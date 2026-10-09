# Task 2 — Telegram RAG Bot

## Цель

Telegram-бот отвечает на вопросы по предоставленным материалам электроэнергетики с использованием RAG.

## Возможности

- три проверенных PDF и локальный индекс;
- hybrid retrieval;
- опциональный перевод RU → EN для поиска;
- semantic search и optional reranker;
- генерация ответа через OpenAI-compatible API DeepSeek;
- ссылки на источники;
- сессионное хранение пользовательского API-ключа и `/reset`.

## Установка

```bash
python -m pip install -r task2_bot/requirements-semantic.txt
cp task2_bot/.env.example task2_bot/.env
```

В `.env` укажите новый `TELEGRAM_BOT_TOKEN`. Основные настройки уже заданы в `.env.example`:

```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-flash
RAG_SEARCH_MODE=hybrid
RAG_TRANSLATION=on
```

API-ключ DeepSeek пользователь вводит в личном чате бота; хранить его в репозитории не нужно.

## Документы и индекс

Если нужно скачать документы заново:

```bash
python -m task2_bot.rag.download fetch
```

Перестроить semantic-индекс:

```bash
python -m task2_bot.rag.build_index --backend semantic --device cpu
```

## Проверка и запуск

```bash
python -m task2_bot.bot.main --check
python -m task2_bot.bot.main
```

Быстрые офлайн-тесты:

```bash
python -m pytest -q tests/test_rag.py tests/test_bot_friendly.py tests/test_bot_reranking.py
```

`.env`, Telegram token и API keys не должны попадать в Git.
