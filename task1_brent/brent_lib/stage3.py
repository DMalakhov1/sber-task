"""Compact probabilistic (quantile) forecasting block for Brent.

Scope intentionally limited to a small set of model families and reusable Stage 2
feature matrices:
- QAR on Brent lags
- QR-X on Stage 2 direct features
- Quantile gradient boosting
- naive empirical-residual baseline

The implementation is deliberately lightweight: no exhaustive grid search, no
neural models, and no full experimental notebook generation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from task1_brent.brent_lib.stage2 import (
    build_external_monthly_frame,
    build_stage2_feature_frame,
    build_stage2_targets,
)

QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=float)


def _as_2d_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values


def pinball_loss(y_true, quantiles, taus: Sequence[float] | None = None) -> np.ndarray:
    """Vectorized pinball loss for scalar or array targets."""
    taus = np.asarray(QUANTILES if taus is None else taus, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    y = np.asarray(y_true, dtype=float)

    if y.ndim == 0:
        error = q - float(y)
        return np.maximum(taus * error, (taus - 1.0) * error)

    if q.ndim == 1:
        error = q - y
        return np.maximum(taus * error, (taus - 1.0) * error)

    if q.shape[0] == y.size and q.shape[1] != y.size:
        q = q.T

    error = q - y[None, :]
    losses = np.maximum(taus[:, None] * error, (taus[:, None] - 1.0) * error)
    return losses


def mean_pinball_loss(actual: np.ndarray, quantiles: np.ndarray, taus=None) -> float:
    """Average pinball loss over all samples and quantiles."""
    actual_arr = np.asarray(actual, dtype=float)
    q_arr = np.asarray(quantiles, dtype=float)
    losses = pinball_loss(actual_arr, q_arr, taus)
    return float(np.mean(losses))


def crps_quantile_approx(
    y_true, quantiles, taus: Sequence[float] | None = None
) -> float:
    """Simple CRPS approximation via quantile losses integrated over tau."""
    taus = np.asarray(QUANTILES if taus is None else taus, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    losses = pinball_loss(y_true, q, taus)
    if losses.ndim == 1:
        return float(np.trapezoid(losses, taus))
    return float(np.mean(np.trapezoid(losses, taus, axis=1)))


def coverage_90(y_true, quantiles) -> float:
    """Empirical 90% central interval coverage for scalar or array targets."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 1:
        return float((q[0] <= y_true <= q[-1]))
    lower = q[:, 0]
    upper = q[:, -1]
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((lower <= y) & (y <= upper)))


def interval_width_90(quantiles) -> float:
    """Mean width of the central 90% interval."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 1:
        return float(q[-1] - q[0])
    return float(np.mean(q[:, -1] - q[:, 0]))


def winkler_score(y_true, quantiles, alpha: float = 0.10) -> float:
    """Winkler score for a central [q05, q95] interval."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 1:
        lower, upper = q[0], q[-1]
        width = upper - lower
        if y_true < lower:
            return float(width + 2.0 * (lower - y_true) / alpha)
        if y_true > upper:
            return float(width + 2.0 * (y_true - upper) / alpha)
        return float(width)
    scores = []
    for y, row in zip(np.asarray(y_true, dtype=float), q):
        scores.append(winkler_score(float(y), row, alpha=alpha))
    return float(np.mean(scores))


def enforce_quantile_order(quantiles: np.ndarray) -> np.ndarray:
    """Sort each row's quantile forecast from low to high."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 1:
        return np.sort(q)
    return np.sort(q, axis=-1)


def crossing_rate_before(quantiles: np.ndarray) -> float:
    """Fraction of rows where quantile order crosses before monotonic correction."""
    q = np.asarray(quantiles, dtype=float)
    if q.ndim == 1:
        return 0.0
    if q.shape[1] <= 1:
        return 0.0
    return float(np.mean(np.any(np.diff(q, axis=1) < 0.0, axis=1)))


def _make_qr_model() -> object:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42),
            ),
        ]
    )


def _prepare_quantile_design(
    series: pd.Series, origin: pd.Timestamp, horizon: int, feature_set: str = "internal"
) -> pd.DataFrame:
    external = build_external_monthly_frame()
    features = build_stage2_feature_frame(
        series, external=external, feature_set=feature_set
    )
    match = features.loc[features.index <= origin]
    if match.empty:
        raise ValueError(f"No available features at origin {origin}")
    row = match.iloc[-1:]
    row = row.copy()
    row["origin"] = origin
    row["horizon"] = int(horizon)
    return row


def _fit_qar_quantiles(
    series: pd.Series, horizon: int, quantiles: Sequence[float] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Quantile autoregression on Brent lags. Compact, no extensive search."""
    quantiles = np.asarray(QUANTILES if quantiles is None else quantiles, dtype=float)
    values = pd.Series(series, copy=True)
    lagged = pd.DataFrame(index=values.index)
    for lag in (1, 2, 3, 6, 12):
        lagged[f"lag_{lag}"] = values.shift(lag)
    lagged = lagged.dropna()
    target = values.shift(-horizon).loc[lagged.index]
    target = target.rename("target")
    X = lagged.assign(target=target.to_numpy())
    y = X.pop("target")
    fitted = {}
    q_preds = []
    for tau in quantiles:
        model = sm.QuantReg(y, sm.add_constant(X, has_constant="add")).fit(
            q=tau, max_iter=2000
        )
        fitted[float(tau)] = model
        q_preds.append(
            model.predict(sm.add_constant(X.iloc[[-1]], has_constant="add")).iloc[0]
        )
    return (
        pd.DataFrame([q_preds], columns=[f"q{int(t * 100):02d}" for t in quantiles]),
        fitted,
    )


def _fit_qr_x_quantiles(
    series: pd.Series, horizon: int, feature_set: str = "internal", quantiles=None
) -> tuple[pd.DataFrame, dict]:
    """Direct QR-X on Stage 2 features. Same feature matrix, same origins, compact config."""
    quantiles = np.asarray(QUANTILES if quantiles is None else quantiles, dtype=float)
    external = build_external_monthly_frame()
    features = build_stage2_feature_frame(
        series, external=external, feature_set=feature_set
    )
    targets = build_stage2_targets(
        series, features, horizons=(horizon,), min_origin_obs=12
    )
    if targets.empty:
        return pd.DataFrame(columns=[f"q{int(t * 100):02d}" for t in quantiles]), {}
    X = targets[
        [
            c
            for c in targets.columns
            if c not in {"origin", "target", "horizon", "y_true"}
        ]
    ].copy()
    y = targets["y_true"].astype(float)
    fitted = {}
    rows = []
    for tau in quantiles:
        model = sm.QuantReg(y, sm.add_constant(X, has_constant="add")).fit(
            q=tau, max_iter=2000
        )
        fitted[float(tau)] = model
        row = model.predict(sm.add_constant(X.iloc[[-1]], has_constant="add")).iloc[0]
        rows.append(float(row))
    return (
        pd.DataFrame([rows], columns=[f"q{int(t * 100):02d}" for t in quantiles]),
        fitted,
    )


def _fit_qgb_quantiles(
    series: pd.Series, horizon: int, feature_set: str = "internal", quantiles=None
) -> tuple[pd.DataFrame, dict]:
    """Quantile gradient boosting with small, fixed config. Uses a single canonical alpha set."""
    quantiles = np.asarray(QUANTILES if quantiles is None else quantiles, dtype=float)
    external = build_external_monthly_frame()
    features = build_stage2_feature_frame(
        series, external=external, feature_set=feature_set
    )
    targets = build_stage2_targets(
        series, features, horizons=(horizon,), min_origin_obs=12
    )
    if targets.empty:
        return pd.DataFrame(columns=[f"q{int(t * 100):02d}" for t in quantiles]), {}
    X = targets[
        [
            c
            for c in targets.columns
            if c not in {"origin", "target", "horizon", "y_true"}
        ]
    ].copy()
    y = targets["y_true"].astype(float)
    fitted = {}
    rows = []
    for tau in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(tau),
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
        model.fit(X, y)
        fitted[float(tau)] = model
        rows.append(float(model.predict(X.iloc[[-1]])[0]))
    return (
        pd.DataFrame([rows], columns=[f"q{int(t * 100):02d}" for t in quantiles]),
        fitted,
    )


def _naive_empirical_quantile_baseline(
    series: pd.Series, horizon: int, quantiles=None
) -> pd.DataFrame:
    """Leak-free baseline: point forecast at the origin plus empirical h-step residuals."""
    quantiles = np.asarray(QUANTILES if quantiles is None else quantiles, dtype=float)
    residuals = []
    for origin_pos in range(12, len(series) - horizon):
        y0 = float(series.iloc[origin_pos])
        yh = float(series.iloc[origin_pos + horizon])
        residuals.append(yh - y0)
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size == 0:
        return pd.DataFrame(columns=[f"q{int(t * 100):02d}" for t in quantiles])
    last_level = float(series.iloc[-1])
    q_vals = np.quantile(residuals, quantiles, method="linear")
    return pd.DataFrame(
        [[last_level + float(v) for v in q_vals]],
        columns=[f"q{int(t * 100):02d}" for t in quantiles],
    )


def _point_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Consistent point-forecast MAE/RMSE from the same y_true/y_pred pair."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return {"MAE": mae, "RMSE": rmse}


def build_stage3_common_origin_evidence(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> dict:
    """Return common-origin counts for internal and external_core feature sets."""
    if external is None:
        external = build_external_monthly_frame()
    records = {}
    for feature_set in ("internal", "external_core"):
        features = build_stage2_feature_frame(
            series, external=external, feature_set=feature_set
        )
        records[feature_set] = build_stage2_targets(
            series, features, horizons=horizons, min_origin_obs=12
        )
    result = {}
    for h in horizons:
        sets = {
            feature_set: set(frame[frame["horizon"] == int(h)]["origin"])
            for feature_set, frame in records.items()
        }
        result[int(h)] = {
            "internal": len(sets["internal"]),
            "external_core": len(sets["external_core"]),
            "intersection": len(set.intersection(*sets.values())),
        }
    return result


def build_stage3_common_origin_summary(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """Origin-level summary for final common-origin comparison."""
    if external is None:
        external = build_external_monthly_frame()
    rows = []
    internal = build_stage2_feature_frame(
        series, external=external, feature_set="internal"
    )
    external_core = build_stage2_feature_frame(
        series, external=external, feature_set="external_core"
    )
    internal_records = build_stage2_targets(
        series, internal, horizons=horizons, min_origin_obs=12
    )
    external_records = build_stage2_targets(
        series, external_core, horizons=horizons, min_origin_obs=12
    )
    for h in [int(v) for v in horizons]:
        internal_origins = sorted(
            set(internal_records[internal_records["horizon"] == h]["origin"])
        )
        external_origins = sorted(
            set(external_records[external_records["horizon"] == h]["origin"])
        )
        common = sorted(set(internal_origins) & set(external_origins))
        rows.append(
            {
                "horizon": h,
                "internal_n": len(internal_origins),
                "external_core_n": len(external_origins),
                "common_n": len(common),
                "first_origin": common[0] if common else pd.NaT,
                "last_origin": common[-1] if common else pd.NaT,
                "n_forecasts": len(common),
                "same_origins_internal_external": set(internal_origins)
                == set(external_origins),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def build_stage3_probabilistic_leaderboard(
    series: pd.Series,
    external: pd.DataFrame | None = None,
    horizons: Sequence[int] = (1, 3, 6, 12),
    feature_sets: Sequence[str] = ("internal", "external_core"),
) -> pd.DataFrame:
    """Probabilistic leaderboard scored on the final common OOS origin set."""
    if external is None:
        external = build_external_monthly_frame()

    from task1_brent.brent_lib.stage2 import _split_stage2_records

    common_summary = build_stage3_common_origin_summary(
        series, external=external, horizons=horizons
    )
    records_by_feature = {
        feature_set: build_stage2_targets(
            series,
            build_stage2_feature_frame(
                series, external=external, feature_set=feature_set
            ),
            horizons=horizons,
            min_origin_obs=12,
        )
        for feature_set in feature_sets
    }

    rows = []
    for h in [int(v) for v in horizons]:
        common = set(
            common_summary.loc[common_summary["horizon"] == h, "first_origin"].tolist()
        )
        if not common_summary.loc[common_summary["horizon"] == h].empty:
            common = set(
                records_by_feature["internal"][
                    records_by_feature["internal"]["horizon"] == h
                ]["origin"].tolist()
            ) & set(
                records_by_feature["external_core"][
                    records_by_feature["external_core"]["horizon"] == h
                ]["origin"].tolist()
            )
        if not common:
            continue

        for feature_set in feature_sets:
            record_h = records_by_feature[feature_set][
                records_by_feature[feature_set]["horizon"] == h
            ].copy()
            record_h = record_h[record_h["origin"].isin(common)].copy()
            if record_h.empty:
                continue
            train, valid, oos, _ = _split_stage2_records(
                record_h, train_share=0.7, valid_share=0.15
            )
            if oos.empty:
                continue
            feature_cols = [
                c
                for c in record_h.columns
                if c not in {"origin", "target", "horizon", "y_true"}
            ]
            if not feature_cols:
                continue

            def _fit_qr_for_tau(train_x, train_y, oos_x, tau: float):
                model = sm.QuantReg(
                    train_y, sm.add_constant(train_x, has_constant="add")
                ).fit(q=tau, max_iter=2000)
                return np.asarray(
                    model.predict(sm.add_constant(oos_x, has_constant="add")),
                    dtype=float,
                )

            def _fit_qgb_for_tau(train_x, train_y, oos_x, tau: float):
                model = GradientBoostingRegressor(
                    loss="quantile",
                    alpha=float(tau),
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                )
                model.fit(train_x, train_y)
                return model.predict(oos_x)

            base_model = {
                "baseline": lambda train_x, train_y, oos_x, oos_origin: (
                    np.tile(np.quantile(train_y, QUANTILES), (len(oos_origin), 1))
                    + np.asarray(
                        [float(series.loc[origin]) for origin in oos_origin],
                        dtype=float,
                    )[:, None]
                ),
                "qar": lambda train_x, train_y, oos_x, oos_origin: (
                    _fit_qr_for_tau(train_x, train_y, oos_x, 0.5)
                ),
                "qr_x": lambda train_x, train_y, oos_x, oos_origin: (
                    _fit_qr_for_tau(train_x, train_y, oos_x, 0.5)
                ),
                "qgb": lambda train_x, train_y, oos_x, oos_origin: (
                    np.column_stack(
                        [
                            _fit_qgb_for_tau(train_x, train_y, oos_x, float(tau))
                            for tau in QUANTILES
                        ]
                    )
                ),
            }

            train_x = train[feature_cols].to_numpy(dtype=float)
            train_y = train["y_true"].to_numpy(dtype=float)
            oos_x = oos[feature_cols].to_numpy(dtype=float)
            oos_y = oos["y_true"].to_numpy(dtype=float)
            oos_origins = oos["origin"].to_numpy()

            for model_name in ("baseline", "qar", "qr_x", "qgb"):
                if model_name == "qar" and feature_set != "internal":
                    continue
                if model_name == "baseline" and feature_set not in {
                    "internal",
                    "external_core",
                }:
                    continue
                if model_name in {"qr_x", "qgb"} and feature_set not in {
                    "internal",
                    "external_core",
                }:
                    continue

                if model_name in {"qar", "qr_x"}:
                    preds = np.column_stack(
                        [
                            _fit_qr_for_tau(train_x, train_y, oos_x, float(tau))
                            for tau in QUANTILES
                        ]
                    )
                elif model_name == "qgb":
                    preds = np.column_stack(
                        [
                            _fit_qgb_for_tau(train_x, train_y, oos_x, float(tau))
                            for tau in QUANTILES
                        ]
                    )
                else:
                    baseline_shift = np.asarray(
                        [float(series.loc[origin]) for origin in oos_origins],
                        dtype=float,
                    )
                    residuals = train_y - np.asarray(
                        [float(series.loc[origin]) for origin in train["origin"]],
                        dtype=float,
                    )
                    preds = baseline_shift[:, None] + np.tile(
                        np.quantile(residuals, QUANTILES), (len(oos_origins), 1)
                    )

                preds = np.asarray(preds, dtype=float)
                if preds.shape[0] != len(oos_y):
                    continue
                before = crossing_rate_before(preds)
                preds = enforce_quantile_order(preds) if before > 0 else preds
                pinball = float(np.mean(pinball_loss(oos_y, preds, QUANTILES)))
                crps = float(
                    np.mean(
                        [
                            crps_quantile_approx(y, row, taus=QUANTILES)
                            for y, row in zip(oos_y, preds)
                        ]
                    )
                )
                cov = coverage_90(oos_y, preds)
                width = interval_width_90(preds)
                wink = winkler_score(oos_y, preds)
                median = preds[:, 3]
                point_metrics = _point_forecast_metrics(oos_y, median)
                rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "horizon": int(h),
                        "pinball": pinball,
                        "CRPS": crps,
                        "coverage_90": float(cov),
                        "width_90": float(width),
                        "Winkler": float(wink),
                        "median_MAE": float(point_metrics["MAE"]),
                        "median_RMSE": float(point_metrics["RMSE"]),
                        "n_forecasts": int(len(oos_y)),
                        "crossing_rate_before": float(before),
                    }
                )

    if rows:
        return (
            pd.DataFrame(rows)
            .sort_values(["horizon", "model", "feature_set"])
            .reset_index(drop=True)
        )
    return pd.DataFrame(
        columns=[
            "model",
            "feature_set",
            "horizon",
            "pinball",
            "CRPS",
            "coverage_90",
            "width_90",
            "Winkler",
            "median_MAE",
            "median_RMSE",
            "n_forecasts",
            "crossing_rate_before",
        ]
    )


__all__ = [
    "QUANTILES",
    "build_stage3_common_origin_evidence",
    "build_stage3_common_origin_summary",
    "build_stage3_probabilistic_leaderboard",
    "coverage_90",
    "crps_quantile_approx",
    "enforce_quantile_order",
    "interval_width_90",
    "mean_pinball_loss",
    "pinball_loss",
    "winkler_score",
]
