# Task 1 — Brent Forecasting

## Цель

Прогноз месячной цены Brent на горизонтах 1, 3, 6 и 12 месяцев.

Основной ряд: май 1987 — март 2026, 467 наблюдений, USD/баррель.

## Что реализовано

- EDA и проверка стационарности: ADF, KPSS, ACF/PACF;
- baselines и классические модели: naive, drift, ETS, mean reversion, AR/ARMA/ARIMA;
- ML: Ridge, ElasticNet, Random Forest, HistGradientBoosting;
- внешние признаки с лагами публикации;
- probabilistic forecasting: baseline, QAR, QR-X, Quantile Gradient Boosting;
- neural-блок: MLP, RNN, LSTM, GRU, CNN, QRNN.

Основной notebook:

```text
task1_brent/Task1_Brent_Research.ipynb
```

## Запуск

```bash
python -m pip install -r task1_brent/requirements.txt
jupyter notebook task1_brent/Task1_Brent_Research.ipynb
```

Для neural-моделей:

```bash
python -m pip install -r task1_brent/requirements-neural.txt
python -m task1_brent.extended_experiment --mode fast --neural --output task1_brent/outputs/my_fast
```

## Проверка

```bash
python -m pytest -q \
  tests/test_stage2.py \
  tests/test_stage3.py \
  tests/test_stage4.py \
  tests/test_extended_brent.py
```

`tests/test_brent.py` содержит более тяжёлые эконометрические проверки и выполняется заметно дольше.

## Валидация и ограничения

Модели оцениваются во временном порядке без random split и future leakage. Preprocessing обучается только на train; для макропризнаков используются лаги доступности.

Neural-блок использует фиксированный шаг origin 12 месяцев и только 5 held-out origins на горизонт, поэтому его результаты интерпретируются как exploratory. Метрики разных блоков не считаются напрямую сопоставимыми, если OOS-периоды различаются.
