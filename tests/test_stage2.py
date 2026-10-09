"""Stage 2 focused regression tests.

These are intentionally fast and deterministic: they cover release-lag logic,
internal/external feature construction, and direct multi-horizon alignment.
"""

import numpy as np
import pandas as pd
import pytest

from task1_brent.brent_lib.stage2 import (
    build_external_monthly_frame,
    build_internal_feature_frame,
    build_stage2_common_origin_evidence,
    build_stage2_feature_frame,
    build_stage2_targets,
    inventory_predictors,
)


def make_series(n=180, start="2000-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    values = 50 + np.cumsum(np.random.default_rng(1).normal(0, 0.05, n))
    return pd.Series(values, index=idx, name="brent_usd_per_barrel")


def test_stage2_inventory_has_required_predictors():
    inv = inventory_predictors()
    required = {
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
    }
    names = set(inv["name"])
    assert required.issubset(names)
    assert set(inv["classification"]).issubset({"reasonable", "sensitivity", "reject"})


def test_internal_features_do_not_leak_future():
    series = make_series()
    frame = build_internal_feature_frame(series)
    for lag in (1, 2, 3, 6, 12):
        assert pd.notna(frame[f"lag_{lag}"].iloc[lag])
    assert frame["rolling_mean_12"].iloc[:11].isna().all()
    assert frame["rolling_mean_12"].iloc[11:].notna().all()
    assert frame["price_to_ma_12"].iloc[:11].isna().all()
    assert frame["price_to_ma_12"].iloc[11:].notna().all()


def test_external_feature_frame_has_monthly_alignment():
    frame = build_external_monthly_frame()
    assert not frame.empty
    assert pd.api.types.is_datetime64_any_dtype(frame.index)
    assert frame.index.freq is not None or len(frame.index) >= 2


def test_external_feature_frame_uses_real_rate_columns():
    frame = build_external_monthly_frame()
    assert not frame.empty
    assert frame["usd_rub"].gt(1.0).any()
    assert frame["ruonia"].gt(0.0).any()


def test_direct_targets_are_horizon_aligned():
    series = make_series(n=120)
    features = build_stage2_feature_frame(series, feature_set="internal")
    targets = build_stage2_targets(
        series, features, horizons=(1, 3, 6, 12), min_origin_obs=12
    )
    assert not targets.empty
    assert set(targets["horizon"].unique()).issubset({1, 3, 6, 12})
    assert (targets["target"] > targets["origin"]).all()


def test_feature_set_has_expected_columns_for_internal_and_external_core():
    series = make_series(n=150)
    internal = build_stage2_feature_frame(series, feature_set="internal")
    external_core = build_stage2_feature_frame(series, feature_set="external_core")
    assert {"lag_1", "lag_12", "rolling_mean_12", "price_to_ma_12"}.issubset(
        internal.columns
    )
    assert {"usd_rub", "us_cpi", "euro_area_cpi", "ruonia"}.issubset(
        external_core.columns
    )


def test_common_origin_metrics_schema_is_stable():
    series = make_series(n=180)
    features = build_stage2_feature_frame(series, feature_set="internal")
    targets = build_stage2_targets(
        series, features, horizons=(1, 3, 6, 12), min_origin_obs=12
    )
    assert {"origin", "target", "horizon", "y_true"}.issubset(targets.columns)
    assert targets["y_true"].notna().all()


def test_common_origin_evidence_uses_real_origin_intersections():
    series = make_series(n=220)
    evidence = build_stage2_common_origin_evidence(series, horizons=(1, 3, 6, 12))
    for h in (1, 3, 6, 12):
        assert evidence[h]["intersection"] >= 0
        assert evidence[h]["intersection"] <= min(
            evidence[h]["internal"],
            evidence[h]["external_core"],
            evidence[h]["sensitivity"],
        )
        assert evidence[h]["intersection"] == len(
            set(
                build_stage2_targets(
                    series,
                    build_stage2_feature_frame(series, feature_set="internal"),
                    horizons=(h,),
                    min_origin_obs=12,
                )["origin"]
            )
            & set(
                build_stage2_targets(
                    series,
                    build_stage2_feature_frame(series, feature_set="external_core"),
                    horizons=(h,),
                    min_origin_obs=12,
                )["origin"]
            )
            & set(
                build_stage2_targets(
                    series,
                    build_stage2_feature_frame(series, feature_set="sensitivity"),
                    horizons=(h,),
                    min_origin_obs=12,
                )["origin"]
            )
        )
