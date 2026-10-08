"""Временной бэктест без утечки будущего.

Принципы:
- расширяющееся окно, случайного train/test split нет;
- в каждом origin все преобразования (scaler, обучение) видят только данные
  не позднее этого origin;
- прямые ML-модели обучаются на парах (X(t), y(t+h)) с t+h не позднее origin,
  поэтому будущие значения макропризнаков не нужны;
- макропризнаки сдвинуты на macro_lag=1 месяц назад как непроверенное
  допущение о задержке публикации (CPI месяца t публикуется в месяце t+1);
- прогнозные интервалы — эмпирические, по квантилям ошибок бэктеста;
  покрытие измеряется на отложенных по времени origin'ах, не участвовавших
  в калибровке.
"""
import numpy as np
import pandas as pd

from .models import STATISTICAL_MODELS, DIRECT_MODELS, ENSEMBLE_MEMBERS

DEFAULT_HORIZONS = tuple(range(1, 22))
MIN_TRAIN_PAIRS = 60  # минимум обучающих пар для прямой ML-модели


def build_feature_frame(series, macro=None, macro_lag=1):
    """Признаки, доступные в момент выпуска прогноза (origin — последний
    известный месяц). Данные позже origin не используются нигде.

    Brent-признаки требуют 12 месяцев истории. Макропризнаки берутся
    из месяца origin - macro_lag; пропуски макро (например, 2025-10)
    сохраняются как NaN и отсекают соответствующие origin'ы у ML-моделей.
    """
    y = series.values.astype(float)
    idx = series.index
    log_ret = pd.Series(np.log(y), index=idx).diff()

    frame = pd.DataFrame(index=idx)
    frame["y0"] = y
    frame["y1"] = series.shift(1).values
    frame["y2"] = series.shift(2).values
    frame["y12"] = series.shift(12).values
    frame["mom1"] = series.pct_change(1).values
    frame["yoy"] = series.pct_change(12).values
    frame["ma3"] = series.rolling(3).mean().values
    frame["ma6"] = series.rolling(6).mean().values
    frame["ma12"] = series.rolling(12).mean().values
    frame["vol12"] = log_ret.rolling(12).std().values
    frame["y_over_ma12"] = (series / series.rolling(12).mean()).values

    if macro is not None:
        macro_cols = [c for c in macro.columns
                      if c not in ("date", "brent_usd_per_barrel")]
        m = macro.set_index("date")[macro_cols]
        m.index = m.index + pd.offsets.MonthBegin(macro_lag)  # доступность с задержкой
        frame = frame.join(m, how="left")
    return frame


def build_training_pairs(series, features, horizon, origin_pos):
    """Обучающие пары для прямой модели горизонта h в origin с позицией
    origin_pos: (X(t), y(t+h)) для всех t, где t+h не позже origin.
    Строки с NaN в признаках исключаются (макро-пропуски не заполняются).
    """
    stop = origin_pos - horizon + 1
    if stop <= 0:
        return None, None
    X = features.iloc[:stop].to_numpy(dtype=float)
    targets = series.iloc[horizon:origin_pos + 1].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(targets)
    if not mask.any():
        return None, None
    return X[mask], targets[mask]


def evaluate(series, horizons=DEFAULT_HORIZONS, initial_window=120, macro=None,
             macro_lag=1, min_train_pairs=MIN_TRAIN_PAIRS,
             direct_start=None, stat=True, direct_models=None):
    """Прогнать бэктест моделей. Возвращает DataFrame ошибок:
    model, horizon, origin, target, y_true, y_pred.

    initial_window — минимальная длина train для статистических моделей
    (в месяцах). direct_start — первая дата origin для прямых ML-моделей:
    им нужны min_train_pairs обучающих пар, поэтому честное сравнение со
    статистикой возможно только на общем окне origin'ов (metrics_table
    с common_origins=True выровняет состав автоматически).
    stat=False отключает статистические модели (для проверки устойчивости
    ML на макро-признаках без повторного прогона статистики).
    direct_models — словарь имя -> класс прямой модели; по умолчанию обе.
    """
    max_h = max(horizons)
    y = series.values.astype(float)
    idx = series.index
    features = build_feature_frame(series, macro=macro, macro_lag=macro_lag)

    stat_classes = {cls.name: cls for cls in STATISTICAL_MODELS} if stat else {}
    direct_classes = direct_models if direct_models is not None else {
        cls.name: cls for cls in DIRECT_MODELS}
    direct_start = pd.Timestamp(direct_start) if direct_start else None
    records = []

    for origin_pos in range(initial_window - 1, len(y) - 1):
        origin_date = idx[origin_pos]
        train = series.iloc[: origin_pos + 1]

        for name, cls in stat_classes.items():
            model = cls().fit(train)
            forecast = model.predict(max_h)
            for h in horizons:
                if origin_pos + h >= len(y):
                    continue
                records.append((name, h, origin_date, idx[origin_pos + h],
                                y[origin_pos + h], float(forecast[h - 1])))

        if not direct_classes:
            continue
        if direct_start is not None and origin_date < direct_start:
            continue
        feat_row = features.iloc[origin_pos]
        if feat_row.isna().any():
            continue  # например, origin вслед за макро-пропуском 2025-10
        for h in horizons:
            if origin_pos + h >= len(y):
                continue
            X_train, y_train = build_training_pairs(series, features, h, origin_pos)
            if X_train is None or len(y_train) < min_train_pairs:
                continue
            for name, cls in direct_classes.items():
                pred = cls().fit(X_train, y_train).predict(feat_row.values.reshape(1, -1))
                records.append((name, h, origin_date, idx[origin_pos + h],
                                y[origin_pos + h], pred))

    return pd.DataFrame.from_records(
        records, columns=["model", "horizon", "origin", "target", "y_true", "y_pred"])


def _metrics_frame(errors):
    """MAE / RMSE / MAPE и MAE относительно naive по тем же origin'ам."""
    rows = []
    for (model, h), g in errors.groupby(["model", "horizon"]):
        err = g["y_pred"] - g["y_true"]
        rows.append({
            "model": model, "horizon": h, "n": len(g),
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "MAPE_%": float((np.abs(err) / g["y_true"]).mean() * 100),
        })
    table = pd.DataFrame(rows)
    naive = table[table["model"] == "naive"].set_index("horizon")["MAE"]
    table["rel_MAE_vs_naive"] = table.apply(
        lambda r: r["MAE"] / naive.get(r["horizon"], np.nan), axis=1)
    return table.sort_values(["horizon", "MAE"]).reset_index(drop=True)


def metrics_table(errors, common_origins=True):
    """Сводная таблица метрик. common_origins=True оставляет только origin'ы,
    на которых отработали ВСЕ модели данного горизонта — честное сравнение
    статистических моделей с ML, вступающими позже."""
    if not common_origins:
        return _metrics_frame(errors)
    parts = []
    for h, g in errors.groupby("horizon"):
        counts = g.groupby("origin")["model"].nunique()
        full = counts[counts == g["model"].nunique()].index
        parts.append(g[g["origin"].isin(full)])
    return _metrics_frame(pd.concat(parts))


def interval_widths(errors, as_of, level=0.9, min_samples=20):
    """Квантиль абсолютной лог-ошибки; только уже наблюдаемые target."""
    available = errors[pd.to_datetime(errors['target']) <= pd.Timestamp(as_of)]
    widths = {}
    if not 0 < level < 1:
        raise ValueError('level must be between 0 and 1')
    for key, group in available.groupby(['model', 'horizon']):
        if (group[['y_true','y_pred']] <= 0).any().any():
            raise ValueError('Для лог-интервалов нужны положительные цены и прогнозы')
        residuals = np.abs(np.log(group.y_true / group.y_pred)).to_numpy()
        if len(residuals) >= min_samples:
            widths[key] = float(np.quantile(residuals, level, method='higher'))
    return widths


def interval_evaluation(errors, test_start='2022-01-01', level=0.9, min_samples=20):
    """Фиксируем ширину перед проверочным периодом; target < test_start."""
    cutoff = pd.Timestamp(test_start) - pd.Timedelta(days=1)
    widths = interval_widths(errors, cutoff, level, min_samples)
    records = []
    for (model, h), group in errors[errors.origin >= pd.Timestamp(test_start)].groupby(['model','horizon']):
        width = widths.get((model,h))
        if width is None:
            continue
        lower = group.y_pred.to_numpy() * np.exp(-width)
        upper = group.y_pred.to_numpy() * np.exp(width)
        actual = group.y_true.to_numpy()
        covered = int(((actual >= lower) & (actual <= upper)).sum())
        alpha = 1-level
        def pinball(pred, tau):
            residual = actual-pred
            return float(np.maximum(tau*residual,(tau-1)*residual).mean())
        interval_score = upper-lower + (2/alpha)*np.maximum(lower-actual,0) + (2/alpha)*np.maximum(actual-upper,0)
        records.append(dict(model=model,horizon=h,n=len(group),covered=covered,
                            coverage=covered/len(group),log_half_width=width,
                            mean_width=float((upper-lower).mean()),
                            pinball_lower=pinball(lower,alpha/2), pinball_upper=pinball(upper,1-alpha/2),
                            interval_score=float(interval_score.mean()),
                            calibration_n=int(((errors.model==model)&(errors.horizon==h)&(errors.target<=cutoff)).sum()),
                            calibration_target_cutoff=str(cutoff.date())))
    return pd.DataFrame(records)


def calibrate_intervals(errors, level=0.9, calib_share=0.7):
    """Совместимый интерфейс; граница origin с purge по target."""
    dates = sorted(errors.origin.unique())
    boundary = pd.Timestamp(dates[min(len(dates)-1, max(1,int(len(dates)*calib_share)))])
    widths = interval_widths(errors,boundary-pd.Timedelta(days=1),level,min_samples=1)
    coverage = interval_evaluation(errors,boundary,level,min_samples=1)
    return widths,{(r.model,r.horizon):r.coverage for r in coverage.itertuples()}


def detect_anomalies(series, recent=6):
    """Диагностика необычности последних месяцев (гипотеза сдвига 2026).

    Возвращает эмпирические перцентили месячных лог-изменений последних
    recent месяцев относительно всей истории. Это диагностика, а не тест
    смены режима: по трём наблюдениям 2026 года режим объявить нельзя.
    """
    log_ret = np.log(series).diff().dropna()
    rows = []
    for date, value in log_ret.iloc[-recent:].items():
        pct = float((log_ret.abs() <= abs(value)).mean() * 100)
        rows.append({"month": date.strftime("%Y-%m"),
                     "log_change": round(float(value), 4),
                     "change_percent": round(float(np.expm1(value)*100), 2),
                     "abs_change_percentile": round(pct, 1)})
    return pd.DataFrame(rows)


def add_ensemble(errors):
    """Равные веса, только для origin/horizon с прогнозами всех участников."""
    base = errors[errors.model.isin(ENSEMBLE_MEMBERS)]
    records = []
    for (h, origin, target), group in base.groupby(['horizon','origin','target']):
        if set(group.model) == set(ENSEMBLE_MEMBERS) and len(group) == len(ENSEMBLE_MEMBERS):
            records.append(dict(model='ensemble', horizon=h, origin=origin, target=target,
                                y_true=float(group.y_true.iloc[0]), y_pred=float(group.y_pred.mean())))
    return pd.concat([errors, pd.DataFrame(records)],ignore_index=True)
