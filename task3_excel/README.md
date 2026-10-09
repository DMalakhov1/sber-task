# Task 3 — Electricity Excel Export

## Цель

Сформировать Excel за выбранный месяц по 7 ОЭС:

- Северо-Запад;
- Центр;
- Юг;
- Средняя Волга;
- Урал;
- Сибирь;
- Восток.

В книге три листа:

1. `Генерация (МВт·ч)`
2. `Потребление (МВт·ч)`
3. `Среднее ИБР (руб. за МВт·ч)`

Источник данных — публичный API `br.so-ups.ru`. Pipeline поддерживает raw cache, JSON snapshot и повторную сборку без новой загрузки.

## Установка

```bash
python -m pip install -r task3_excel/requirements.txt
```

## Запуск

Пример для сентября 2026:

```bash
python -m task3_excel.source_http \
  --month 2026-09 \
  --weight consumption_mwh \
  --allow-partial \
  --snapshot task3_excel/data/processed/check_2026-09.json \
  --output task3_excel/outputs/electricity_2026-09.xlsx
```

Если `--month` не указан, используется предыдущий календарный месяц по Москве.

Для новой загрузки:

```bash
python -m task3_excel.source_http \
  --month 2026-09 \
  --weight consumption_mwh \
  --allow-partial \
  --refresh \
  --insecure-br \
  --snapshot task3_excel/data/processed/live_check_2026-09.json \
  --output task3_excel/outputs/live_check_2026-09.xlsx
```

`--insecure-br` нужен только при проблеме TLS с источником. `--overwrite` разрешает заменить существующий результат с сохранением backup.

## Готовый результат

```text
task3_excel/outputs/electricity_2026-09.xlsx
```

Для сентября 2026 ожидается 5040 интервалов: `7 × 24 × 30`. Если часть цен ИБР отсутствует в API, `--allow-partial` сохраняет отчёт и отмечает неполноту; программа возвращает код 3.

## Проверка

```bash
python -m pytest -q \
  tests/test_excel.py \
  tests/test_calendar.py \
  tests/test_source_http.py \
  tests/test_excel_upgrade.py
```
