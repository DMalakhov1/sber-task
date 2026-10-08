"""Загрузка и валидация данных задачи 1.

Ограничения происхождения данных:
- brent_monthly_1987-2026.csv: месячный ряд EIA RBRTE, май 1987 — март 2026.
  Последняя строка (2026-03) — последняя строка ПРИСЛАННОГО файла; мартовское значение сверено с таблицей EIA 07.10.2026. Дата первоначального скачивания
  неизвестна (original_download_date: null в metadata.json) и не восстанавливается.
- brent_macro_diploma_2011-2025.csv: таблица макропризнаков из диплома пользователя.
  За 2025-10 признаки отсутствуют. Временное соответствие лагов независимо
  не проверено; поэтому макропризнаки используются с дополнительным сдвигом
  macro_lag=1 месяц (см. backtest.build_feature_frame) как непроверенное
  допущение о задержке публикации.
"""
from pathlib import Path

import pandas as pd
import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BRENT_FILE = "brent_monthly_1987-2026.csv"
MACRO_FILE = "brent_macro_diploma_2011-2025.csv"
TARGET_COLUMN = "brent_usd_per_barrel"


def load_brent(path=None):
    """Загрузить месячный ряд Brent и проверить его целостность.

    Возвращает pd.Series с DatetimeIndex на первое число месяца.
    Проверки: сортировка, отсутствие дублей, непрерывность месяцев,
    положительные числовые значения. Пропуски не заполняются.
    """
    path = Path(path) if path else DEFAULT_DATA_DIR / BRENT_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    if list(df.columns) != ["date", TARGET_COLUMN]:
        raise ValueError(f"Неожиданные колонки {list(df.columns)} в {path.name}")
    if df.empty or df["date"].isna().any():
        raise ValueError("Пустой ряд или некорректные даты")
    if df["date"].duplicated().any():
        raise ValueError("Дубли дат в ряде Brent")
    df = df.sort_values("date").reset_index(drop=True)
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Пропуски в значениях Brent")
    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError("Нечисловые или бесконечные значения Brent")
    if (df[TARGET_COLUMN] <= 0).any():
        raise ValueError("Неположительные значения Brent")
    expected = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="MS")
    if len(expected) != len(df) or not (expected == df["date"]).all():
        raise ValueError("Ряд Brent не непрерывен по месяцам")
    series = df.set_index("date")[TARGET_COLUMN]
    series.index.freq = "MS"
    return series


def load_macro(path=None):
    """Загрузить таблицу макропризнаков как есть, без заполнения пропусков.

    Возвращает DataFrame с колонкой date. Пропуски (2011-01, 2025-10)
    сохраняются; обработка отложена на этап построения признаков, чтобы не
    подменять отсутствующие наблюдения выдуманными значениями.
    """
    path = Path(path) if path else DEFAULT_DATA_DIR / MACRO_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    if df.empty or df["date"].isna().any():
        raise ValueError("Пустой ряд или некорректные даты")
    if df["date"].duplicated().any():
        raise ValueError("Дубли дат в макро-таблице")
    df = df.sort_values("date").reset_index(drop=True)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Нет колонки {TARGET_COLUMN} в {path.name}")
    return df
