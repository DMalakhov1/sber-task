"""Stage 2: compact external-predictor and ML forecasting setup.

Scope intentionally limited to:
- explicit predictor inventory with release-lag assumptions
- internal Brent feature pipeline
- external feature pipeline with monthly aggregation and lag treatment
- compact ML model set with small tuning
- direct multi-horizon evaluation using common origins and validation split

No probabilistic / quantile / neural models are included here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from task1_brent.brent_lib.models import ETSLogModel, NaiveModel, DriftModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXTERNAL_PREDICTOR_INVENTORY = [
    {
        "name": "usd_rub",
        "source_file": "usd_rub.xlsx",
        "frequency": "daily",
        "aggregation": "monthly_mean",
        "release_lag_months": 1,
        "classification": "reasonable",
    },
    {
        "name": "fx_volatility",
        "source_file": "usd_rub.xlsx",
        "frequency": "daily",
        "aggregation": "monthly_std_log_return",
        "release_lag_months": 1,
        "classification": "reasonable",
    },
    {
        "name": "us_cpi",
        "source_file": "us_cpi.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "reasonable",
    },
    {
        "name": "euro_area_cpi",
        "source_file": "euro_area_cpi.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "reasonable",
    },
    {
        "name": "ruonia",
        "source_file": "ruonia.xlsx",
        "frequency": "daily",
        "aggregation": "monthly_mean",
        "release_lag_months": 1,
        "classification": "reasonable",
    },
    {
        "name": "russia_key_rate",
        "source_file": "russia_cpi_key_rate.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_cpi",
        "source_file": "russia_cpi.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_m2",
        "source_file": "russia_m2.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_credit",
        "source_file": "russia_private_credit.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_retail",
        "source_file": "russia_retail_yoy.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_wages",
        "source_file": "russia_wages.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "unemployment",
        "source_file": "unemployment.csv",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_industrial_production",
        "source_file": "russia_industry_yoy.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "sensitivity",
    },
    {
        "name": "russia_public_debt",
        "source_file": "russia_public_debt.xlsx",
        "frequency": "monthly",
        "aggregation": "as_is",
        "release_lag_months": 1,
        "classification": "reject",
    },
]

FEATURE_SET_DEFINITIONS = {
    "internal": [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "return_lag_1",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "price_to_ma_12",
    ],
    "external_core": [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "return_lag_1",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "price_to_ma_12",
        "usd_rub",
        "fx_volatility",
        "us_cpi",
        "euro_area_cpi",
        "ruonia",
    ],
    "sensitivity": [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "return_lag_1",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",
        "momentum_3",
        "momentum_6",
        "momentum_12",
        "price_to_ma_12",
        "usd_rub",
        "fx_volatility",
        "us_cpi",
        "euro_area_cpi",
        "ruonia",
        "russia_key_rate",
        "russia_cpi",
        "russia_m2",
        "russia_credit",
        "russia_retail",
        "russia_wages",
        "unemployment",
        "russia_industrial_production",
    ],
}

MODEL_PRESETS = {
    "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
    "elastic_net": [
        {"alpha": 0.01, "l1_ratio": 0.2},
        {"alpha": 0.1, "l1_ratio": 0.2},
        {"alpha": 1.0, "l1_ratio": 0.8},
    ],
    "random_forest": [
        {"max_depth": 3, "min_samples_leaf": 2, "n_estimators": 200},
        {"max_depth": None, "min_samples_leaf": 5, "n_estimators": 200},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.03, "max_depth": 2},
        {"learning_rate": 0.1, "max_depth": 3},
    ],
}


def inventory_predictors() -> pd.DataFrame:
    """Return explicit Stage 2 metadata for all candidate external predictors."""
    inventory = pd.DataFrame(EXTERNAL_PREDICTOR_INVENTORY)
    inventory["source_file"] = inventory["source_file"].astype(str)
    inventory["release_lag_months"] = inventory["release_lag_months"].astype(int)
    inventory["classification"] = inventory["classification"].astype(str)
    return inventory


def _find_date_column(frame: pd.DataFrame) -> str:
    preferred = [
        "date",
        "Date",
        "DATE",
        "period",
        "Period",
        "month",
        "Month",
        "dt",
        "DT",
        "data",
        "Data",
        "дата",
        "Дата",
        "период",
        "Период",
    ]
    for candidate in preferred:
        if candidate in frame.columns:
            return candidate

    for candidate in frame.columns:
        label = str(candidate).lower()
        if (
            "date" in label
            or "data" in label
            or "period" in label
            or "month" in label
            or "dt" in label
            or "дата" in label
            or "период" in label
        ) and not pd.api.types.is_numeric_dtype(frame[candidate]):
            return candidate

    for candidate in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[candidate]):
            return candidate
    return frame.columns[0]


def _as_monthly_index(values: pd.Series) -> pd.DatetimeIndex:
    s = pd.Series(values)
    s = s.dropna()
    if s.empty:
        return pd.DatetimeIndex([])
    dt = pd.to_datetime(s)
    return pd.DatetimeIndex(dt.to_period("M").to_timestamp())


def _read_raw_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _coerce_datetime_series(values: pd.Series) -> pd.Series:
    """Parse mixed date formats common in macro source files."""
    out: List[pd.Timestamp | pd.NaT] = []
    for value in values:
        if pd.isna(value):
            out.append(pd.NaT)
            continue
        raw = str(value).strip()
        if not raw:
            out.append(pd.NaT)
            continue
        parsed = None
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%m.%Y",
            "%Y-%m",
            "%Y.%m",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ):
            try:
                parsed = pd.to_datetime(raw, format=fmt)
                break
            except (TypeError, ValueError):
                continue
        if parsed is None:
            try:
                parsed = pd.to_datetime(raw)
            except (TypeError, ValueError):
                parsed = pd.NaT
        out.append(parsed)
    return pd.to_datetime(out, errors="coerce")


def _select_numeric_value_column(
    frame: pd.DataFrame, feature_name: str, date_col: str
) -> str | None:
    good_tokens = {
        "usd_rub": ("curs", "rate", "exchange", "fx"),
        "fx_volatility": ("curs", "rate", "exchange", "fx"),
        "us_cpi": ("inflation", "cpi", "consumer"),
        "euro_area_cpi": ("inflation", "cpi", "consumer"),
        "ruonia": ("ruo", "rate", "minrate", "percentile", "maxrate"),
        "russia_key_rate": ("key", "rate", "ставка"),
        "russia_cpi": ("inflation", "cpi", "price", "index"),
        "russia_m2": ("m2", "money", "aggregate"),
        "russia_credit": ("credit", "loan"),
        "russia_retail": ("retail", "trade"),
        "russia_wages": ("wage", "salary", "income"),
        "unemployment": ("unemp", "employment", "job"),
        "russia_industrial_production": ("industry", "production", "industrial"),
        "russia_public_debt": ("debt", "public"),
    }
    banned_tokens = (
        "nominal",
        "statusxml",
        "vol",
        "code",
        "id",
        "count",
        "t",
        "c",
        "cdx",
        "percentile25",
        "percentile75",
        "minrate",
        "maxrate",
    )
    best_col = None
    best_score = -np.inf

    for candidate in frame.columns:
        if candidate in {date_col, "date"}:
            continue
        if not pd.api.types.is_numeric_dtype(frame[candidate]):
            continue
        label = str(candidate).lower()
        score = 0.0
        if any(token in label for token in banned_tokens):
            score -= 100.0
        if any(token in label for token in good_tokens.get(feature_name, ("",))):
            score += 50.0
        if "rate" in label or "curs" in label or "price" in label or "value" in label:
            score += 10.0
        if "inflation" in label or "cpi" in label:
            score += 12.0
        if feature_name == "ruonia" and "ruo" in label:
            score += 15.0
        if feature_name in {"usd_rub", "fx_volatility"} and "nominal" in label:
            score -= 500.0
        if feature_name == "russia_key_rate" and "ставка" in label:
            score += 30.0
        if score > best_score:
            best_score = score
            best_col = candidate
    if best_col is not None:
        return best_col

    for candidate in frame.columns:
        if candidate in {date_col, "date"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[candidate]):
            return candidate
    return None


def build_external_monthly_frame(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Read the raw macro files and aggregate them to a monthly frame.

    The function makes the actual lineage explicit and applies the release lag to the
    monthly values. Same-month macro observations are never used for forecasting.
    """
    base_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    monthly: Dict[str, pd.Series] = {}

    for row in EXTERNAL_PREDICTOR_INVENTORY:
        name = row["name"]
        path = base_dir / row["source_file"]
        if not path.exists():
            continue
        frame = _read_raw_table(path).copy()
        date_col = _find_date_column(frame)
        frame["date"] = _coerce_datetime_series(frame[date_col])
        value_col = _select_numeric_value_column(frame, name, date_col)
        if value_col is None:
            continue
        values = frame[["date", value_col]].rename(columns={value_col: "value"})
        values = values.dropna().sort_values("date").reset_index(drop=True)

        if row["frequency"] == "daily":
            s = values.set_index("date")["value"].sort_index()
            month = s.resample("MS").mean()
            if name == "fx_volatility":
                log_ret = np.log(s).diff().dropna()
                month = log_ret.groupby(
                    log_ret.index.to_period("M").to_timestamp()
                ).std()
            monthly[name] = month
        else:
            s = values.set_index("date")["value"].sort_index()
            monthly[name] = s.resample("MS").last()

    frame = pd.concat(monthly.values(), axis=1)
    frame.columns = list(monthly.keys())
    frame = frame.sort_index()
    frame = frame.groupby(level=0).last()
    return frame


def _apply_release_lag(frame: pd.DataFrame, lag_months: int = 1) -> pd.DataFrame:
    shifted = frame.copy()
    if lag_months == 0:
        return shifted
    shifted.index = shifted.index + pd.offsets.MonthBegin(lag_months)
    return shifted


def build_internal_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Compact internal Brent feature set using only information available at time t."""
    s = pd.Series(series, copy=True).sort_index()
    log_s = np.log(s.astype(float))
    log_ret = log_s.diff()

    frame = pd.DataFrame(index=s.index)
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = s.shift(lag)
    frame["return_lag_1"] = log_ret.shift(1)
    frame["rolling_mean_3"] = s.rolling(3).mean()
    frame["rolling_mean_6"] = s.rolling(6).mean()
    frame["rolling_mean_12"] = s.rolling(12).mean()
    frame["rolling_std_3"] = log_ret.rolling(3).std().fillna(0.0)
    frame["rolling_std_6"] = log_ret.rolling(6).std().fillna(0.0)
    frame["rolling_std_12"] = log_ret.rolling(12).std().fillna(0.0)
    for lag in (3, 6, 12):
        frame[f"momentum_{lag}"] = s / s.shift(lag) - 1.0
    frame["price_to_ma_12"] = s / s.rolling(12).mean()
    return frame


def build_stage2_feature_frame(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    feature_set: str = "internal",
    release_lag_months: int = 1,
) -> pd.DataFrame:
    """Combine internal features and release-lagged external features."""
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown feature_set={feature_set!r}")
    base = build_internal_feature_frame(series)
    if feature_set == "internal":
        return base[FEATURE_SET_DEFINITIONS["internal"]]

    if external is None:
        external = build_external_monthly_frame()
    external = external.copy()
    external = external[~external.index.duplicated(keep="last")]
    external = _apply_release_lag(external, lag_months=release_lag_months)
    aligned = base.join(external, how="left")
    selected = FEATURE_SET_DEFINITIONS[feature_set]
    available = [name for name in selected if name in aligned.columns]
    return aligned[available]


def build_stage2_targets(
    series: pd.Series,
    features: pd.DataFrame,
    horizons: Sequence[int] = (1, 3, 6, 12),
    min_origin_obs: int = 12,
) -> pd.DataFrame:
    """Direct forecasting records: X_t -> y_{t+h} for each horizon with no look-ahead."""
    records: List[dict] = []
    for origin_pos in range(min_origin_obs, len(series) - 1):
        origin = series.index[origin_pos]
        row = features.loc[origin]
        if pd.isna(row.to_numpy(dtype=float)).any():
            continue
        for h in horizons:
            target_pos = origin_pos + h
            if target_pos >= len(series):
                continue
            target = series.index[target_pos]
            records.append(
                {
                    "origin": origin,
                    "target": target,
                    "horizon": int(h),
                    "y_true": float(series.iloc[target_pos]),
                    **{str(k): float(v) for k, v in row.items()},
                }
            )
    return pd.DataFrame.from_records(records)


def _make_model(model_name: str, params: Dict[str, float | int]) -> object:
    if model_name == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(params.get("alpha", 1.0)))),
            ]
        )
    if model_name == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(params.get("alpha", 0.1)),
                        l1_ratio=float(params.get("l1_ratio", 0.5)),
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=42,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _score_metric(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_absolute_error(actual, pred))


def _split_stage2_records(
    records: pd.DataFrame,
    train_share: float = 0.7,
    valid_share: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    if records.empty:
        return records.copy(), records.copy(), records.copy(), []
    dates = sorted(records["origin"].drop_duplicates().tolist())
    if len(dates) < 3:
        return (
            records.iloc[:0].copy(),
            records.iloc[:0].copy(),
            records.iloc[:0].copy(),
            dates,
        )
    train_end_index = max(0, min(len(dates) - 1, int(len(dates) * train_share)))
    valid_end_index = max(
        train_end_index + 1,
        min(len(dates) - 1, int(len(dates) * (train_share + valid_share))),
    )
    train_end = dates[train_end_index]
    valid_end = dates[valid_end_index]
    train = records[records["origin"] <= train_end].copy()
    valid = records[
        (records["origin"] > train_end) & (records["origin"] <= valid_end)
    ].copy()
    oos = records[records["origin"] > valid_end].copy()
    return train, valid, oos, dates


def _naive_forecast_for_origin(
    series: pd.Series, origin: pd.Timestamp, horizon: int
) -> float:
    try:
        return float(series.loc[origin])
    except KeyError:
        return float(
            series.iloc[series.index.get_loc(origin) if origin in series.index else -1]
        )


def _compute_stage2_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, train_values: np.ndarray
) -> dict:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    if train_values.size > 1:
        scale = float(np.mean(np.abs(np.diff(train_values))))
    else:
        scale = 1.0
    mase = float(np.mean(abs_err) / scale) if scale > 0 else np.nan
    denom = np.where(
        np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8) > 0,
        np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8),
        1.0,
    )
    smape = float(np.mean(200.0 * abs_err / denom))
    return {"MAE": mae, "RMSE": rmse, "MASE": mase, "sMAPE": smape}


def select_stage2_config(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
):
    """Choose the best config from a tiny candidate set using validation MAE."""
    if model_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model_name={model_name!r}")
    best = None
    best_score = np.inf
    for params in MODEL_PRESETS[model_name]:
        model = _make_model(model_name, params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        score = _score_metric(y_valid, pred)
        if score < best_score - 1e-12:
            best = params
            best_score = score
        elif np.isclose(score, best_score) and best is not None:
            # Prefer simpler model in a tie.
            complexity = 1.0 if "alpha" in params and params["alpha"] <= 1.0 else 1.5
            prior = 1.0 if "alpha" in best and best["alpha"] <= 1.0 else 1.5
            if complexity < prior:
                best = params
    return best, float(best_score)


def evaluate_stage2_track(
    series: pd.Series,
    feature_set: str = "internal",
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_name: str = "ridge",
    train_share: float = 0.7,
    valid_share: float = 0.15,
    origin_filter_by_horizon: dict[int, set[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Train on split, tune on validation, score on final OOS with same-origin fairness."""
    if not 0.0 < train_share < 1.0 or not 0.0 < valid_share < 1.0:
        raise ValueError("train_share and valid_share must be between 0 and 1.")
    if train_share + valid_share >= 1.0:
        raise ValueError("train_share + valid_share must be < 1.0")

    features = build_stage2_feature_frame(
        series, external=external, feature_set=feature_set
    )
    records = build_stage2_targets(
        series, features, horizons=horizons, min_origin_obs=12
    )
    if origin_filter_by_horizon is not None:
        allowed = pd.Series(False, index=records.index)
        for horizon, origins in origin_filter_by_horizon.items():
            allowed |= (records["horizon"] == int(horizon)) & records["origin"].isin(
                origins
            )
        records = records[allowed].copy()
    if records.empty:
        return pd.DataFrame(
            columns=[
                "track",
                "model",
                "horizon",
                "MAE",
                "RMSE",
                "MASE",
                "sMAPE",
                "rel_MAE_vs_naive",
                "n_forecasts",
                "selected_params",
            ]
        )

    train, valid, oos, _ = _split_stage2_records(
        records, train_share=train_share, valid_share=valid_share
    )
    feature_cols = [
        c for c in records.columns if c not in {"origin", "target", "horizon", "y_true"}
    ]
    if train.empty or valid.empty or oos.empty:
        return pd.DataFrame(
            columns=[
                "track",
                "model",
                "horizon",
                "MAE",
                "RMSE",
                "MASE",
                "sMAPE",
                "rel_MAE_vs_naive",
                "n_forecasts",
                "selected_params",
            ]
        )

    params, _ = select_stage2_config(
        model_name,
        train[feature_cols].to_numpy(dtype=float),
        train["y_true"].to_numpy(dtype=float),
        valid[feature_cols].to_numpy(dtype=float),
        valid["y_true"].to_numpy(dtype=float),
    )
    model = _make_model(model_name, params)
    train_valid = pd.concat([train, valid], ignore_index=True)
    model.fit(
        train_valid[feature_cols].to_numpy(dtype=float),
        train_valid["y_true"].to_numpy(dtype=float),
    )

    rows = []
    for h in sorted(oos["horizon"].unique()):
        g = oos[oos["horizon"] == h].copy()
        if g.empty:
            continue
        pred = np.asarray(
            model.predict(g[feature_cols].to_numpy(dtype=float)), dtype=float
        )
        y_true = g["y_true"].to_numpy(dtype=float)
        naive_pred = np.asarray(
            [
                _naive_forecast_for_origin(series, origin, int(h))
                for origin in g["origin"]
            ],
            dtype=float,
        )
        naive_err = naive_pred - y_true
        metrics = _compute_stage2_metrics(
            y_true,
            pred,
            np.asarray(series.iloc[: max(12, len(series) - 12)], dtype=float),
        )
        rel_mae = (
            float(
                np.mean(np.abs(pred - y_true))
                / (
                    np.mean(np.abs(naive_err))
                    if np.mean(np.abs(naive_err)) > 0
                    else np.nan
                )
            )
            if np.mean(np.abs(naive_err)) > 0
            else np.nan
        )
        rows.append(
            {
                "track": feature_set,
                "model": model_name,
                "horizon": int(h),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MASE": metrics["MASE"],
                "sMAPE": metrics["sMAPE"],
                "rel_MAE_vs_naive": rel_mae,
                "n_forecasts": int(len(g)),
                "selected_params": params,
            }
        )
    return pd.DataFrame(rows)


def summarize_track_metrics(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    feature_sets: Sequence[str] = ("internal", "external_core", "sensitivity"),
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_names: Sequence[str] = (
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
    ),
) -> pd.DataFrame:
    """Compact summary across feature sets and models without heavy search."""
    records: List[dict] = []
    for track in feature_sets:
        for model_name in model_names:
            df = evaluate_stage2_track(
                series,
                feature_set=track,
                external=external,
                horizons=horizons,
                model_name=model_name,
            )
            if df.empty:
                continue
            records.extend(df.to_dict(orient="records"))
    return pd.DataFrame(records)


def build_stage2_comparison_table(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    feature_sets: Sequence[str] = ("internal", "external_core", "sensitivity"),
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Return a compact comparison table with baseline vs ML scores."""
    metrics = summarize_track_metrics(
        series, external=external, feature_sets=feature_sets, horizons=horizons
    )
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "model",
                "horizon",
                "MAE",
                "RMSE",
                "MASE",
                "sMAPE",
                "rel_MAE_vs_naive",
                "n_forecasts",
            ]
        )

    baseline_rows = []
    for h in horizons:
        for model_name, model in (
            ("naive", NaiveModel()),
            ("drift", DriftModel()),
            ("ets_log", ETSLogModel()),
            ("mean_reversion", MeanReversionModel()),
        ):
            train = series.iloc[: max(12, len(series) - 12)]
            model.fit(train)
            pred = np.asarray(model.predict(h), dtype=float)
            y_true = np.asarray(series.iloc[-max(1, h) :], dtype=float)
            err = (
                pred[: min(len(pred), len(y_true))]
                - y_true[: min(len(pred), len(y_true))]
            )
            baseline_rows.append(
                {
                    "feature_set": "baseline",
                    "model": model_name,
                    "horizon": int(h),
                    "MAE": float(np.mean(np.abs(err))) if err.size else np.nan,
                    "RMSE": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
                    "MASE": np.nan,
                    "sMAPE": np.nan,
                    "rel_MAE_vs_naive": 1.0 if model_name == "naive" else np.nan,
                    "n_forecasts": int(err.size),
                }
            )
    baseline = pd.DataFrame(baseline_rows)
    return pd.concat([metrics, baseline], ignore_index=True)


def build_stage2_inventory_summary() -> pd.DataFrame:
    """Convenient summary of the explicit Stage 2 metadata."""
    return inventory_predictors().copy()


def calculate_stage2_track_leaderboard(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    feature_set: str = "internal",
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_names: Sequence[str] = (
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
    ),
    train_share: float = 0.7,
    valid_share: float = 0.15,
) -> pd.DataFrame:
    """Return leaderboard for a single Stage 2 track (internal or external)."""
    rows = []
    for model_name in model_names:
        df = evaluate_stage2_track(
            series,
            feature_set=feature_set,
            external=external,
            horizons=horizons,
            model_name=model_name,
            train_share=train_share,
            valid_share=valid_share,
        )
        rows.extend(df.to_dict(orient="records"))
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    leaderboard = leaderboard[
        [
            "model",
            "horizon",
            "MAE",
            "RMSE",
            "MASE",
            "sMAPE",
            "rel_MAE_vs_naive",
            "n_forecasts",
        ]
    ].copy()
    return leaderboard.sort_values(["horizon", "MAE"]).reset_index(drop=True)


def _common_origin_filter_by_horizon(
    series: pd.Series,
    external: pd.DataFrame,
    horizons: Sequence[int],
    feature_sets: Sequence[str] = ("internal", "external_core", "sensitivity"),
) -> dict[int, set[pd.Timestamp]]:
    built: dict[str, pd.DataFrame] = {}
    for feature_set in feature_sets:
        features = build_stage2_feature_frame(
            series, external=external, feature_set=feature_set
        )
        built[feature_set] = build_stage2_targets(
            series, features, horizons=horizons, min_origin_obs=12
        )
    common: dict[int, set[pd.Timestamp]] = {}
    for h in horizons:
        sets = {
            feature_set: set(frame[frame["horizon"] == h]["origin"])
            for feature_set, frame in built.items()
        }
        common[int(h)] = set.intersection(*sets.values()) if sets else set()
    return common


def build_stage2_external_contribution(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_names: Sequence[str] = (
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
    ),
    train_share: float = 0.7,
    valid_share: float = 0.15,
) -> pd.DataFrame:
    """Compare internal vs external-core vs sensitivity forecasts on a common-origin basis."""
    if external is None:
        external = build_external_monthly_frame()
    common_by_horizon = _common_origin_filter_by_horizon(series, external, horizons)
    rows = []
    for model_name in model_names:
        common_filter = {
            h: origins for h, origins in common_by_horizon.items() if origins
        }
        internal_df = evaluate_stage2_track(
            series,
            feature_set="internal",
            external=external,
            horizons=horizons,
            model_name=model_name,
            train_share=train_share,
            valid_share=valid_share,
            origin_filter_by_horizon=common_filter,
        )
        ext_df = evaluate_stage2_track(
            series,
            feature_set="external_core",
            external=external,
            horizons=horizons,
            model_name=model_name,
            train_share=train_share,
            valid_share=valid_share,
            origin_filter_by_horizon=common_filter,
        )
        sens_df = evaluate_stage2_track(
            series,
            feature_set="sensitivity",
            external=external,
            horizons=horizons,
            model_name=model_name,
            train_share=train_share,
            valid_share=valid_share,
            origin_filter_by_horizon=common_filter,
        )
        for h in horizons:
            int_h = internal_df[internal_df["horizon"] == h]
            ext_h = ext_df[ext_df["horizon"] == h]
            sens_h = sens_df[sens_df["horizon"] == h]
            if int_h.empty or ext_h.empty or sens_h.empty:
                continue
            int_mae = float(int_h["MAE"].iloc[0])
            ext_mae = float(ext_h["MAE"].iloc[0])
            sens_mae = float(sens_h["MAE"].iloc[0])
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(h),
                    "internal_MAE": int_mae,
                    "external_core_MAE": ext_mae,
                    "sensitivity_MAE": sens_mae,
                    "delta_external_vs_internal": ext_mae - int_mae,
                    "delta_sensitivity_vs_internal": sens_mae - int_mae,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)


def build_stage2_report(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_names: Sequence[str] = (
        "ridge",
        "elastic_net",
        "random_forest",
        "hist_gradient_boosting",
    ),
    train_share: float = 0.7,
    valid_share: float = 0.15,
) -> dict:
    """Return a compact summary of the Stage 2 evaluation configuration and results."""
    if external is None:
        external = build_external_monthly_frame()
    track_a = calculate_stage2_track_leaderboard(
        series,
        external=external,
        feature_set="internal",
        horizons=horizons,
        model_names=model_names,
        train_share=train_share,
        valid_share=valid_share,
    )
    track_b = {
        "internal": calculate_stage2_track_leaderboard(
            series,
            external=external,
            feature_set="internal",
            horizons=horizons,
            model_names=model_names,
            train_share=train_share,
            valid_share=valid_share,
        ),
        "external_core": calculate_stage2_track_leaderboard(
            series,
            external=external,
            feature_set="external_core",
            horizons=horizons,
            model_names=model_names,
            train_share=train_share,
            valid_share=valid_share,
        ),
        "sensitivity": calculate_stage2_track_leaderboard(
            series,
            external=external,
            feature_set="sensitivity",
            horizons=horizons,
            model_names=model_names,
            train_share=train_share,
            valid_share=valid_share,
        ),
    }
    contribution = build_stage2_external_contribution(
        series,
        external=external,
        horizons=horizons,
        model_names=model_names,
        train_share=train_share,
        valid_share=valid_share,
    )
    return {
        "track_a": track_a,
        "track_b": track_b,
        "contribution": contribution,
        "sample_start": str(series.index.min().date()),
        "sample_end": str(series.index.max().date()),
        "oos_start": str(
            series.index[int(len(series) * (train_share + valid_share))].date()
        ),
        "oos_end": str(series.index[-1].date()),
        "n_origins": int(
            len(
                build_stage2_targets(
                    series,
                    build_stage2_feature_frame(
                        series, external=external, feature_set="internal"
                    ),
                    horizons=horizons,
                    min_origin_obs=12,
                )
            )
        ),
    }


def _apply_common_origin_mask(
    frame: pd.DataFrame,
    reference: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    origin_values = reference[reference["horizon"] == horizon]["origin"].tolist()
    return frame[frame["origin"].isin(origin_values)]


def build_stage2_common_origin_evidence(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    model_name: str = "ridge",
    train_share: float = 0.7,
    valid_share: float = 0.15,
) -> dict:
    """Return evidence that all feature sets are compared on a common set of rolling origins."""
    if external is None:
        external = build_external_monthly_frame()
    track_records = {}
    for feature_set in ("internal", "external_core", "sensitivity"):
        features = build_stage2_feature_frame(
            series, external=external, feature_set=feature_set
        )
        track_records[feature_set] = build_stage2_targets(
            series, features, horizons=horizons, min_origin_obs=12
        )

    result = {}
    for h in horizons:
        sets = {
            feature_set: set(frame[frame["horizon"] == int(h)]["origin"])
            for feature_set, frame in track_records.items()
        }
        result[int(h)] = {
            "internal": len(sets["internal"]),
            "external_core": len(sets["external_core"]),
            "sensitivity": len(sets["sensitivity"]),
            "intersection": len(set.intersection(*sets.values())),
        }
    return result


def build_stage2_inventory_summary() -> pd.DataFrame:
    """Convenient summary of the explicit Stage 2 metadata."""
    return inventory_predictors().copy()
