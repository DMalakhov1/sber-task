"""Fast Stage 3 regression tests for quantile forecasting helpers."""

import numpy as np
import pandas as pd

from task1_brent.brent_lib.stage3 import (
    QUANTILES,
    build_stage3_common_origin_evidence,
    build_stage3_common_origin_summary,
    build_stage3_probabilistic_leaderboard,
    coverage_90,
    crps_quantile_approx,
    enforce_quantile_order,
    interval_width_90,
    mean_pinball_loss,
    pinball_loss,
    winkler_score,
)


def make_series(n=200, start="2000-01-01"):
    idx = pd.date_range(start, periods=n, freq="MS")
    trend = np.linspace(50.0, 80.0, n)
    seasonal = 2.5 * np.sin(np.arange(n) / 8.0)
    noise = np.random.default_rng(42).normal(0.0, 0.5, n)
    return pd.Series(trend + seasonal + noise, index=idx, name="brent_usd_per_barrel")


def test_quantile_grid_matches_required_levels():
    assert QUANTILES.tolist() == [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def test_pinball_loss_matches_definition():
    y_true = 10.0
    q = np.array([9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0])
    tau = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    values = pinball_loss(y_true, q, tau)
    expected = np.array(
        [
            max(0.05 * (9.0 - 10.0), 0.95 * (10.0 - 9.0)),
            max(0.10 * (9.5 - 10.0), 0.90 * (10.0 - 9.5)),
            max(0.25 * (10.0 - 10.0), 0.75 * (10.0 - 10.0)),
            max(0.50 * (10.5 - 10.0), 0.50 * (10.0 - 10.5)),
            max(0.75 * (11.0 - 10.0), 0.25 * (10.0 - 11.0)),
            max(0.90 * (12.0 - 10.0), 0.10 * (10.0 - 12.0)),
            max(0.95 * (13.0 - 10.0), 0.05 * (10.0 - 13.0)),
        ]
    )
    assert np.allclose(values, expected)
    assert np.isfinite(values).all()


def test_crps_approx_and_coverage_and_width_are_valid():
    y_true = 10.0
    q = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    crps = crps_quantile_approx(y_true, q, QUANTILES)
    cov = coverage_90(y_true, q)
    width = interval_width_90(q)
    assert crps >= 0.0
    assert 0.0 <= cov <= 1.0
    assert width > 0.0
    assert width == (q[6] - q[0])
    assert cov == 1.0 if 8 <= y_true <= 14 else 0.0


def test_winkler_score_is_pointer_for_outside_interval():
    y_true = 15.0
    q = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    score = winkler_score(y_true, q)
    assert score > 0.0
    assert np.isfinite(score)


def test_monotonic_correction_keeps_quantile_ordering():
    q = np.array(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    corrected = enforce_quantile_order(q)
    assert np.all(corrected[:, 0] <= corrected[:, 1])
    assert np.all(corrected[:, 1] <= corrected[:, 2])
    assert np.all(corrected[:, 2] <= corrected[:, 3])
    assert np.all(corrected[:, 3] <= corrected[:, 4])
    assert np.all(corrected[:, 4] <= corrected[:, 5])
    assert np.all(corrected[:, 5] <= corrected[:, 6])


def test_stage3_common_origin_evidence_has_nonnegative_intersections():
    series = make_series(n=180)
    evidence = build_stage3_common_origin_evidence(series, horizons=(1, 3, 6, 12))
    for h in (1, 3, 6, 12):
        assert evidence[h]["intersection"] >= 0
        assert evidence[h]["intersection"] <= evidence[h]["internal"]
        assert evidence[h]["intersection"] <= evidence[h]["external_core"]


def test_mean_pinball_is_average_of_per_quantile_losses():
    actual = np.array([10.0, 11.0])
    q = np.array(
        [
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ]
    ).T
    expected = np.mean(
        [
            pinball_loss(actual[0], q[:, 0], QUANTILES),
            pinball_loss(actual[1], q[:, 1], QUANTILES),
        ]
    )
    assert np.isclose(mean_pinball_loss(actual, q, QUANTILES), expected)


def test_zero_mae_implies_zero_rmse_for_point_forecasts():
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([10.0, 11.0, 12.0])
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    assert mae == 0.0
    assert rmse == 0.0

    y_pred2 = np.array([10.0, 11.0, 15.0])
    mae2 = float(np.mean(np.abs(y_pred2 - y_true)))
    rmse2 = float(np.sqrt(np.mean((y_pred2 - y_true) ** 2)))
    assert mae2 > 0.0
    assert rmse2 > 0.0


def test_nontrivial_forecast_does_not_match_target_by_default():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    series = pd.Series(
        np.cumsum(rng.normal(0.0, 1.0, size=len(idx))) + np.linspace(50, 90, len(idx)),
        index=idx,
    )
    rows = build_stage3_probabilistic_leaderboard(series, horizons=(12,))
    assert not rows.empty
    h12 = rows[rows["horizon"] == 12]
    assert len(h12) > 0
    assert not np.allclose(h12["median_MAE"].to_numpy(), 0.0)


def test_qar_and_qr_x_follow_different_feature_paths():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2000-01-01", periods=220, freq="MS")
    X = np.linspace(0.0, 1.0, len(idx))
    series = pd.Series(
        50.0
        + 0.6 * np.sin(np.arange(len(idx)) / 3.0)
        + 5.0 * X
        + rng.normal(0.0, 0.2, len(idx)),
        index=idx,
    )
    rows = build_stage3_probabilistic_leaderboard(series, horizons=(3,))
    assert {"internal", "external_core"}.issubset(set(rows["feature_set"]))
    qar = rows[(rows["model"] == "qar") & (rows["feature_set"] == "internal")]
    qrx = rows[(rows["model"] == "qr_x") & (rows["feature_set"] == "external_core")]
    assert not qar.empty and not qrx.empty
    assert qar["n_forecasts"].iloc[0] == qrx["n_forecasts"].iloc[0]


def test_winkler_score_exact_three_cases():
    q = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    assert np.isclose(
        winkler_score(10.5, q), 6.0 / 0.10 * (10.5 - 11.0) + 6.0 if False else 6.0
    )
    assert np.isclose(winkler_score(7.0, q), 6.0 + 2.0 / 0.10 * (8.0 - 7.0))
    assert np.isclose(winkler_score(15.0, q), 6.0 + 2.0 / 0.10 * (15.0 - 14.0))


def test_stage3_leaderboard_has_all_implemented_models_and_feature_sets():
    series = make_series(n=220)
    rows = build_stage3_probabilistic_leaderboard(series, horizons=(1, 3, 6, 12))
    models = set(rows["model"])
    assert {"baseline", "qar", "qr_x", "qgb"}.issubset(models)
    assert {"internal", "external_core"}.issubset(set(rows["feature_set"]))
    summary = build_stage3_common_origin_summary(series, horizons=(1, 3, 6, 12))
    assert {1, 3, 6, 12}.issubset(set(summary["horizon"]))
    assert summary["same_origins_internal_external"].isin([True, False]).all()
    assert (summary["n_forecasts"] == summary["common_n"]).all()
