# SBER Analyst Test

Репозиторий содержит три независимых задания аналитического теста:

1. прогнозирование месячной цены Brent;
2. Telegram RAG-бот по материалам электроэнергетики;
3. выгрузка данных СБР в Excel.

## Структура

```text
sber-analyst-test/
├── task1_brent/   # прогноз Brent
├── task2_bot/     # Telegram RAG-бот
├── task3_excel/   # Excel-выгрузка
├── tests/
└── docs/
```

## Установка

Рекомендуется Python 3.12.

```bash
conda create -n sber python=3.12 pip
conda activate sber
python -m pip install -r requirements-all.txt
```

Дополнительные зависимости:

```bash
# Neural-блок Task 1
python -m pip install -r task1_brent/requirements-neural.txt

# Semantic retrieval Task 2
python -m pip install -r task2_bot/requirements-semantic.txt
```

## Task 1 — Brent forecasting

Прогноз месячной цены Brent на горизонтах 1, 3, 6 и 12 месяцев. В проекте есть эконометрические, ML, вероятностные и компактные neural-модели.

Главный результат:

```text
task1_brent/Task1_Brent_Research.ipynb
```

```bash
jupyter notebook task1_brent/Task1_Brent_Research.ipynb
```

Подробнее: [task1_brent/README.md](task1_brent/README.md).

## Task 2 — Telegram RAG Bot

Бот отвечает на вопросы по трём энергетическим PDF, использует hybrid retrieval и OpenAI-compatible API DeepSeek.

```bash
python -m task2_bot.rag.download fetch
python -m task2_bot.bot.main --check
python -m task2_bot.bot.main
```

Подробнее: [task2_bot/README.md](task2_bot/README.md).

## Task 3 — Electricity Excel Export

Формирование Excel по 7 ОЭС: генерация, потребление и среднее ИБР.

```bash
python -m task3_excel.source_http \
  --month 2026-09 \
  --weight consumption_mwh \
  --allow-partial \
  --snapshot task3_excel/data/processed/check_2026-09.json \
  --output task3_excel/outputs/electricity_2026-09.xlsx
```

Подробнее: [task3_excel/README.md](task3_excel/README.md).

## Быстрые проверки

```bash
python -m pytest -q tests/test_stage2.py tests/test_stage3.py tests/test_stage4.py tests/test_extended_brent.py
python -m pytest -q tests/test_rag.py tests/test_bot_friendly.py tests/test_bot_reranking.py
python -m pytest -q tests/test_excel.py tests/test_calendar.py tests/test_source_http.py
```

Секреты (`.env`, API keys, Telegram token) в Git не добавляются.
