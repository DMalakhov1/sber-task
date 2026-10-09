"""Econometric diagnostics and classical model helpers for Brent."""

from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss


def build_econometric_transforms(series: pd.Series) -> pd.DataFrame:
    """Return a compact feature frame used by diagnostics and baseline checks."""
    s = pd.Series(series, copy=True)
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.date_range(start="2000-01-01", periods=len(s), freq="MS")
    s = s.sort_index()
    log_s = np.log(s)
    frame = pd.DataFrame(index=s.index)
    frame["level"] = s.astype(float)
    frame["log_level"] = log_s.astype(float)
    frame["diff"] = s.diff().astype(float).fillna(0.0)
    frame["log_return"] = log_s.diff().astype(float).fillna(0.0)
    rolling_mean_12 = s.rolling(12, min_periods=12).mean().bfill()
    frame["rolling_mean_12"] = rolling_mean_12
    frame["roll_mean_12"] = rolling_mean_12
    rolling_vol_12 = frame["log_return"].rolling(12, min_periods=12).std().fillna(0.0)
    frame["rolling_vol_12"] = rolling_vol_12
    frame["rolling_std_12"] = rolling_vol_12
    return frame


def _stationarity_result(
    transform_name: str,
    test_name: str,
    statistic: float,
    p_value: float,
    conclusion: str,
) -> dict:
    return {
        "series": "brent",
        "transform": transform_name,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "conclusion": conclusion,
    }


def stationarity_summary(series: pd.Series) -> List[dict]:
    """Compact stationarity diagnostics across common Brent transforms."""
    transforms = {
        "level": series.astype(float),
        "log_level": np.log(series.astype(float)),
        "diff": series.astype(float).diff().dropna(),
        "log_return": np.log(series.astype(float)).diff().dropna(),
    }
    summary: List[dict] = []
    for name, values in transforms.items():
        values = pd.Series(values).dropna()
        if len(values) < 10:
            continue
        adf_stat, adf_p, *_ = adfuller(values, autolag="AIC")
        summary.append(
            _stationarity_result(
                name,
                "ADF",
                adf_stat,
                adf_p,
                "non-stationary" if adf_p > 0.05 else "stationary",
            )
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=InterpolationWarning,
                message="The test statistic is outside of the range of p-values available in the look-up table.*",
            )
            kpss_stat, kpss_p, *_ = kpss(values, regression="c", nlags="auto")
        summary.append(
            _stationarity_result(
                name,
                "KPSS",
                kpss_stat,
                kpss_p,
                "non-stationary" if kpss_p <= 0.05 else "stationary",
            )
        )
    return summary


def _safe_arima_fit(series: pd.Series, order: tuple[int, int, int]):
    try:
        model = ARIMA(
            series, order=order, enforce_stationarity=False, enforce_invertibility=False
        )
        fitted = model.fit()
        return fitted
    except Exception:
        return None


def _arima_validation_metrics(series: pd.Series, fit):
    n_test = max(6, min(24, len(series) // 5))
    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    if len(train) < 12 or len(test) < 3:
        return float("nan"), float("nan")
    fitted = _safe_arima_fit(train, fit.order)
    if fitted is None:
        return float("nan"), float("nan")
    pred = fitted.forecast(steps=len(test))
    err = pred.to_numpy() - test.to_numpy()
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def _candidate_metrics(y_true: np.ndarray, y_pred: np.ndarray, train: pd.Series):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = (
        float(np.mean(np.abs(np.diff(train.astype(float))))) if len(train) > 1 else 1.0
    )
    mase = float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan
    denom = (
        np.abs(np.asarray(y_true, dtype=float))
        + np.abs(np.asarray(y_pred, dtype=float))
    ) / 2.0
    smape = float(np.mean(200.0 * np.abs(err) / denom)) if np.any(denom > 0) else 0.0
    return {"MAE": mae, "RMSE": rmse, "MASE": mase, "sMAPE": smape}


def _autoreg_fit(train: pd.Series, lags: int):
    try:
        return AutoReg(train.astype(float), lags=lags, old_names=False).fit()
    except Exception:
        return None


def _last_forecast_value(forecast):
    """Return the last forecast point using explicit positional access."""
    if hasattr(forecast, "iloc"):
        return float(forecast.iloc[-1])
    arr = np.asarray(forecast).reshape(-1)
    return float(arr[-1]) if arr.size else np.nan


def evaluate_classic_autoregressive(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    max_lags: int = 2,
):
    """Fair expanding-window evaluation using the same origins as the main baseline models."""
    y = series.astype(float).to_numpy()
    idx = series.index
    records = []
    for origin_pos in range(initial_window - 1, len(y) - 1):
        origin_date = idx[origin_pos]
        train = series.iloc[: origin_pos + 1]
        for h in horizons:
            if origin_pos + h >= len(y):
                continue
            target = idx[origin_pos + h]
            for model_name, lags in (("AR1", 1), ("AR2", 2)):
                fit = _autoreg_fit(train, lags)
                if fit is None:
                    continue
                try:
                    pred = _last_forecast_value(
                        fit.forecast(steps=h, start=origin_date)
                    )
                except Exception:
                    pred = _last_forecast_value(fit.forecast(steps=h))
                records.append(
                    {
                        "model": model_name,
                        "horizon": h,
                        "origin": origin_date,
                        "target": target,
                        "y_true": float(y[origin_pos + h]),
                        "y_pred": pred,
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=["model", "horizon", "origin", "target", "y_true", "y_pred"]
    )


def evaluate_classic_arma(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    max_p: int = 2,
    max_q: int = 2,
):
    """Fair ARMA evaluation using the same origins and target dates as the baseline models."""
    y = series.astype(float).to_numpy()
    idx = series.index
    records = []
    specs = []
    for p in range(1, max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            specs.append((p, q))
    for origin_pos in range(initial_window - 1, len(y) - 1):
        origin_date = idx[origin_pos]
        train = series.iloc[: origin_pos + 1]
        for h in horizons:
            if origin_pos + h >= len(y):
                continue
            target = idx[origin_pos + h]
            for p, q in specs:
                fit = _safe_arima_fit(train, (p, 0, q))
                if fit is None:
                    continue
                pred = _last_forecast_value(fit.forecast(steps=h))
                records.append(
                    {
                        "model": f"ARMA({p},{q})",
                        "horizon": h,
                        "origin": origin_date,
                        "target": target,
                        "y_true": float(y[origin_pos + h]),
                        "y_pred": pred,
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=["model", "horizon", "origin", "target", "y_true", "y_pred"]
    )


def evaluate_classic_arima(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    d_values: Sequence[int] = (0, 1),
    max_p: int = 2,
    max_q: int = 2,
):
    """Fair ARIMA evaluation using identical expanding-window origins and horizons as the baseline models."""
    y = series.astype(float).to_numpy()
    idx = series.index
    records = []
    for origin_pos in range(initial_window - 1, len(y) - 1):
        origin_date = idx[origin_pos]
        train = series.iloc[: origin_pos + 1]
        for d in d_values:
            for p in range(max_p + 1):
                for q in range(max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    fit = _safe_arima_fit(train, (p, d, q))
                    if fit is None:
                        continue
                    for h in horizons:
                        if origin_pos + h >= len(y):
                            continue
                        target = idx[origin_pos + h]
                        pred = _last_forecast_value(fit.forecast(steps=h))
                        records.append(
                            {
                                "model": f"ARIMA({p},{d},{q})",
                                "horizon": h,
                                "origin": origin_date,
                                "target": target,
                                "y_true": float(y[origin_pos + h]),
                                "y_pred": pred,
                            }
                        )
    return pd.DataFrame.from_records(
        records, columns=["model", "horizon", "origin", "target", "y_true", "y_pred"]
    )


def point_leaderboard(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    direct_start=None,
):
    """Construct a common-origin point leaderboard with MAE/RMSE/MASE/sMAPE for classical models."""
    from task1_brent.brent_lib.backtest import evaluate  # local import to avoid cycle

    base = evaluate(
        series,
        horizons=horizons,
        initial_window=initial_window,
        direct_start=direct_start,
        stat=True,
    )
    classic = pd.concat(
        [
            base,
            evaluate_classic_autoregressive(
                series, horizons=horizons, initial_window=initial_window
            ),
            evaluate_classic_arma(
                series, horizons=horizons, initial_window=initial_window
            ),
            evaluate_classic_arima(
                series, horizons=horizons, initial_window=initial_window
            ),
        ],
        ignore_index=True,
    )
    rows = []
    for (model, h), g in classic.groupby(["model", "horizon"]):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        err = y_pred - y_true
        scale = (
            np.mean(
                np.abs(np.diff(series.astype(float).iloc[:initial_window].to_numpy()))
            )
            if len(series) > initial_window
            else 1.0
        )
        mase = float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan
        smape = float(np.mean(200.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))))
        rows.append(
            {
                "model": model,
                "horizon": int(h),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MASE": mase,
                "sMAPE": smape,
                "n_forecasts": int(len(g)),
            }
        )
    table = pd.DataFrame(rows)
    naive = (
        table[(table["model"] == "naive")]
        .set_index("horizon")[["MAE"]]
        .rename(columns={"MAE": "naive_MAE"})
    )
    table = table.merge(naive, left_on="horizon", right_index=True)
    table["rel_MAE_vs_naive"] = table["MAE"] / table["naive_MAE"]
    return table.sort_values(["horizon", "MAE"]).reset_index(drop=True)


def final_econometric_leaderboard(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    direct_start=None,
):
    """Canonical leaderboard for the Stage 1 classical candidate set."""
    from task1_brent.brent_lib.backtest import evaluate

    base = evaluate(
        series,
        horizons=horizons,
        initial_window=initial_window,
        direct_start=direct_start,
        stat=True,
    )
    family_map = {
        "naive": "baseline",
        "drift": "baseline",
        "historical_mean": "baseline",
        "ses": "classical",
        "holt": "classical",
        "ar_1": "classical",
        "selected_ar": "classical",
        "selected_arma": "classical",
        "selected_arima": "classical",
        "ets_log": "classical",
        "mean_reversion": "structural",
    }
    rows = []
    for (model, h), g in base.groupby(["model", "horizon"]):
        err = g["y_pred"].to_numpy(dtype=float) - g["y_true"].to_numpy(dtype=float)
        scale = (
            np.mean(
                np.abs(np.diff(series.astype(float).iloc[:initial_window].to_numpy()))
            )
            if len(series) > initial_window
            else 1.0
        )
        mase = float(np.mean(np.abs(err)) / scale) if scale > 0 else np.nan
        denom = (
            np.abs(g["y_true"].to_numpy(dtype=float))
            + np.abs(g["y_pred"].to_numpy(dtype=float))
        ) / 2.0
        smape = (
            float(np.mean(200.0 * np.abs(err) / denom)) if np.any(denom > 0) else 0.0
        )
        rows.append(
            {
                "model": model,
                "family": family_map.get(model, "classical"),
                "horizon": int(h),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MASE": mase,
                "sMAPE": smape,
                "n_forecasts": int(len(g)),
            }
        )
    table = pd.DataFrame(rows)
    naive = (
        table[(table["model"] == "naive")]
        .set_index("horizon")[["MAE"]]
        .rename(columns={"MAE": "naive_MAE"})
    )
    table = table.merge(naive, left_on="horizon", right_index=True)
    table["rel_MAE_vs_naive"] = table["MAE"] / table["naive_MAE"]
    return table.sort_values(["horizon", "MAE"]).reset_index(drop=True)


def fairness_table(
    series: pd.Series,
    horizons=(1, 3, 6, 12),
    initial_window: int = 120,
    direct_start=None,
):
    """Check that every model uses the same expanding-window origins at each horizon."""
    from task1_brent.brent_lib.backtest import evaluate

    errors = evaluate(
        series,
        horizons=horizons,
        initial_window=initial_window,
        direct_start=direct_start,
        stat=True,
    )
    rows = []
    for h in horizons:
        h_errors = errors[errors["horizon"] == h]
        origins_by_model = {
            model: set(group["origin"].unique())
            for model, group in h_errors.groupby("model")
        }
        if not origins_by_model:
            continue
        common = set.intersection(*[set(v) for v in origins_by_model.values()])
        all_same = all(
            set(v) == set(next(iter(origins_by_model.values())))
            for v in origins_by_model.values()
        )
        ordered = sorted(next(iter(origins_by_model.values())))
        rows.append(
            {
                "h": h,
                "n_origins": len(ordered),
                "first_origin": ordered[0] if ordered else None,
                "last_origin": ordered[-1] if ordered else None,
                "all_models_same_origins": bool(all_same),
                "common_origins": len(common),
            }
        )
    return pd.DataFrame(rows)


def seasonality_summary(series: pd.Series):
    """Short monthly seasonality check for Brent series."""
    s = pd.Series(series).astype(float)
    acf12 = s.autocorr(lag=12)
    by_month = s.groupby(s.index.month).mean()
    rel = by_month.std() / by_month.mean()
    return {
        "acf12": float(acf12) if pd.notna(acf12) else np.nan,
        "month_mean_std_over_mean": float(rel) if pd.notna(rel) else np.nan,
        "month_pattern": by_month.to_dict(),
        "decision": (
            "SARIMA rejected: no stable monthly seasonality"
            if abs(acf12) < 0.2 and rel < 0.15
            else "seasonality present"
        ),
    }


def residual_diagnostics_summary(fit, name: str):
    """Compact residual diagnostics for a fitted AR/ARMA/ARIMA model."""
    resid = pd.Series(fit.resid).dropna()
    if resid.empty:
        return {}
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[-1]) if "lb_pvalue" in lb.columns else np.nan
    jb = jarque_bera(resid)
    jb_p = float(jb[1]) if isinstance(jb, tuple) and len(jb) > 1 else float(jb)
    arch = het_arch(resid, nlags=12)
    arch_p = np.nan
    if isinstance(arch, tuple) and len(arch) >= 2:
        value = arch[1]
        arr = np.asarray(value)
        if arr.size:
            arch_p = float(arr.reshape(-1)[-1])
    elif isinstance(arch, np.ndarray):
        arr = np.asarray(arch).reshape(-1)
        if arr.size:
            arch_p = float(arr[-1])
    acf_lag1 = (
        float(np.corrcoef(resid.iloc[1:], resid.iloc[:-1])[0, 1])
        if len(resid) > 1
        else np.nan
    )
    roots = getattr(fit, "arroots", None)
    if roots is None:
        roots = getattr(fit.model, "arroots", None)
    ma_roots = getattr(fit, "maroots", None)
    if ma_roots is None:
        ma_roots = getattr(fit.model, "maroots", None)
    stable = bool(np.all(np.abs(np.asarray(roots)) < 1)) if roots is not None else True
    invertible = (
        bool(np.all(np.abs(np.asarray(ma_roots)) < 1)) if ma_roots is not None else True
    )
    return {
        "model": name,
        "lb_pvalue": lb_p,
        "jb_pvalue": jb_p,
        "arch_lm_pvalue": arch_p,
        "acf_lag1": acf_lag1,
        "stable": stable,
        "invertible": invertible,
    }


def arima_grid_search(
    series: pd.Series,
    max_p: int = 2,
    max_q: int = 2,
    d_values: Sequence[int] = (0, 1),
    max_candidates: int = 6,
):
    """Search a small ARIMA candidate grid and return concise diagnostics."""
    s = pd.Series(series).astype(float).dropna()
    if len(s) < 24:
        return []
    n_test = max(6, min(24, len(s) // 5))
    train = s.iloc[:-n_test]
    test = s.iloc[-n_test:]
    candidates = []

    for d in d_values:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                order = (p, d, q)
                fit = _safe_arima_fit(train, order)
                if fit is None:
                    continue
                pred = fit.forecast(steps=len(test))
                err = pred.to_numpy() - test.to_numpy()
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err**2)))

                residuals = pd.Series(fit.resid).dropna()
                if len(residuals) > 3:
                    lb = acorr_ljungbox(
                        residuals, lags=min(10, len(residuals) // 5), return_df=True
                    )
                    lb_p = (
                        float(lb["lb_pvalue"].iloc[-1])
                        if "lb_pvalue" in lb.columns
                        else 1.0
                    )
                else:
                    lb_p = 1.0
                candidates.append(
                    {
                        "model": "ARIMA",
                        "p": p,
                        "d": d,
                        "q": q,
                        "aic": float(fit.aic),
                        "bic": float(fit.bic),
                        "hqic": float(fit.hqic),
                        "lb_pvalue": lb_p,
                        "mae": mae,
                        "rmse": rmse,
                    }
                )

    if not candidates:
        return []
    candidates.sort(key=lambda row: (row["bic"], row["aic"]))
    return candidates[:max_candidates]
