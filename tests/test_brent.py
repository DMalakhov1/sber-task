"""Офлайн-тесты задачи 1: данные, отсутствие утечки будущего, бэктест,
интервалы, длина финального прогноза. Синтетические ряды, сеть не нужна.

Запуск из корня: python -m pytest -q tests/test_brent.py
"""
import numpy as np
import pandas as pd
import pytest

from task1_brent.brent_lib.backtest import (build_feature_frame,
                                            build_training_pairs,
                                            calibrate_intervals,
                                            detect_anomalies, evaluate,
                                            metrics_table)
from task1_brent.brent_lib.data import load_brent
from task1_brent.brent_lib.models import NaiveModel, DriftModel
from task1_brent.run import final_forecast
from task1_brent.brent_lib.backtest import add_ensemble


def make_series(n=120, start="2000-01-01", seed=1):
    rng = np.random.default_rng(seed)
    values = 50 * np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    return pd.Series(values, index=pd.date_range(start, periods=n, freq="MS"),
                     name="brent_usd_per_barrel")


# --- данные ---

def test_load_brent_real_file():
    series = load_brent()
    assert len(series) == 467
    assert series.index.is_monotonic_increasing
    assert (series > 0).all()
    full = pd.date_range(series.index[0], series.index[-1], freq="MS")
    assert len(full) == len(series)


def test_load_brent_rejects_gaps_and_duplicates(tmp_path):
    bad = pd.DataFrame({"date": ["2020-01-01", "2020-01-01", "2020-03-01"],
                        "brent_usd_per_barrel": [60, 61, 62]})
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_brent(path)


# --- утечка будущего ---

def test_features_do_not_depend_on_future():
    series = make_series()
    origin = series.index[60]
    before = build_feature_frame(series).loc[origin]
    altered = series.copy()
    altered.iloc[61:] = 999.0  # ломаем будущее после origin
    after = build_feature_frame(altered).loc[origin]
    pd.testing.assert_series_equal(before, after)


def test_training_pairs_never_exceed_origin():
    series = make_series()
    features = build_feature_frame(series)
    for origin_pos in (40, 80):
        for h in (1, 6, 12):
            X, y = build_training_pairs(series, features, h, origin_pos)
            # пар не больше, чем origin - h + 1; цель последней пары = origin
            assert len(y) <= origin_pos - h + 1
            last_target = series.iloc[: origin_pos + 1].iloc[-1]
            assert y[-1] == pytest.approx(last_target) or len(y) < origin_pos - h + 1


def test_backtest_targets_strictly_after_origin():
    series = make_series(n=150)
    errors = evaluate(series, horizons=(1, 6), initial_window=100,
                      direct_start=series.index[120])
    assert (errors["target"] > errors["origin"]).all()
    delta = (errors["target"].dt.year - errors["origin"].dt.year) * 12 + \
        (errors["target"].dt.month - errors["origin"].dt.month)
    assert (delta == errors["horizon"]).all()


# --- baseline и метрики ---

def test_naive_and_drift_are_correct():
    series = make_series(n=30)
    naive = NaiveModel().fit(series).predict(5)
    assert np.allclose(naive, series.iloc[-1])
    drift = DriftModel().fit(series).predict(3)
    expected_slope = np.mean(np.diff(series.values))
    assert drift[0] == pytest.approx(series.iloc[-1] + expected_slope)


def test_metrics_table_rel_mae_naive_is_one():
    series = make_series(n=150)
    errors = evaluate(series, horizons=(3,), initial_window=100,
                      direct_start=series.index[120])
    table = metrics_table(errors, common_origins=True)
    naive_row = table[table["model"] == "naive"].iloc[0]
    assert naive_row["rel_MAE_vs_naive"] == pytest.approx(1.0)
    assert (table["n"] > 0).all()


# --- интервалы ---

def test_interval_calibration_and_coverage():
    series = make_series(n=150)
    errors = evaluate(series, horizons=(3,), initial_window=100,
                      direct_models={}, stat=True)
    widths, coverage = calibrate_intervals(errors, level=0.9, calib_share=0.7)
    for (model, h), w in widths.items():
        assert w >= 0
        assert 0.0 <= coverage[(model, h)] <= 1.0


# --- финальный прогноз и диагностика ---

def test_final_forecast_length_and_columns():
    series = make_series(n=200, start="2008-01-01")
    errors = evaluate(series, horizons=tuple(range(1,7)), initial_window=100,
                      direct_start=series.index[120])
    end = series.index[-1] + pd.offsets.MonthBegin(6)
    forecast = final_forecast(series, add_ensemble(errors), forecast_end=end)
    assert len(forecast) == 6
    for model in ("naive", "drift", "ets_log", "ridge_direct", "hgb_direct"):
        assert model in forecast.columns
        assert forecast[model].notna().all()
        assert (forecast[f"{model}_upper"] >= forecast[model]).all()
        assert (forecast[f"{model}_lower"] <= forecast[model]).all()


def test_forecast_task_length_is_21_months():
    from task1_brent.run import FORECAST_END
    series=make_series(n=100,start='2017-12-01')  # through March 2026
    assert series.index[-1] == pd.Timestamp('2026-03-01')
    # Point models are real; interval fixture isolates shape/date contract cheaply.
    from task1_brent.brent_lib.models import STATISTICAL_MODELS, DIRECT_MODELS
    records=[]
    for name in [c.name for c in STATISTICAL_MODELS+DIRECT_MODELS]+['ensemble']:
        for h in range(1,22):
            for i in range(20):
                records.append(dict(model=name,horizon=h,origin=pd.Timestamp('2020-01-01'),target=pd.Timestamp('2021-12-01'),y_true=60+i,y_pred=60))
    result=final_forecast(series,pd.DataFrame(records),FORECAST_END)
    assert len(result)==21 and result.index[0]==pd.Timestamp('2026-04-01')
    assert result.index[-1]==pd.Timestamp('2027-12-01')
    assert (result.filter(like='_lower')>0).all().all()


def test_detect_anomalies_shape():
    series = make_series()
    anomalies = detect_anomalies(series, recent=4)
    assert len(anomalies) == 4
    assert anomalies["abs_change_percentile"].between(0, 100).all()


def test_calibration_rejects_unrealized_targets():
    from task1_brent.brent_lib.backtest import interval_widths, interval_evaluation
    rows = [dict(model='naive',horizon=21,origin=pd.Timestamp('2018-01-01'),
                 target=pd.Timestamp('2020-01-01'),y_true=110.,y_pred=100.),
            dict(model='naive',horizon=21,origin=pd.Timestamp('2020-08-01'),
                 target=pd.Timestamp('2022-05-01'),y_true=10000.,y_pred=100.),
            dict(model='naive',horizon=21,origin=pd.Timestamp('2022-01-01'),
                 target=pd.Timestamp('2023-10-01'),y_true=105.,y_pred=100.)]
    errors=pd.DataFrame(rows)
    widths=interval_widths(errors,'2021-12-31',min_samples=1)
    assert widths[('naive',21)] == pytest.approx(np.log(1.1))
    result=interval_evaluation(errors,min_samples=1).iloc[0]
    assert result.calibration_n == 1 and result.covered == 1
    altered=errors.copy(); altered.loc[1,'y_true']=99999999
    assert interval_widths(altered,'2021-12-31',min_samples=1)==widths


def test_training_pairs_unchanged_when_future_modified():
    series=make_series(n=150)
    origin=85
    for horizon in (1,12,21):
        a=build_training_pairs(series,build_feature_frame(series),horizon,origin)
        altered=series.copy();altered.iloc[origin+1:]=10000
        b=build_training_pairs(altered,build_feature_frame(altered),horizon,origin)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])


def test_mean_reversion_and_ensemble():
    from task1_brent.brent_lib.models import MeanReversionModel, ENSEMBLE_MEMBERS
    from task1_brent.brent_lib.backtest import add_ensemble
    series=pd.Series([50.]*119+[100.])
    model=MeanReversionModel().fit(series)
    forecast=model.predict(24)
    assert (np.diff(forecast)<0).all() and forecast[-1]>np.exp(model.mean_)
    rows=[dict(model=m,horizon=1,origin=pd.Timestamp('2020-01-01'),target=pd.Timestamp('2020-02-01'),y_true=90.,y_pred=10*(i+1)) for i,m in enumerate(ENSEMBLE_MEMBERS)]
    combined=add_ensemble(pd.DataFrame(rows))
    assert combined[combined.model=='ensemble'].iloc[0].y_pred == 25
    assert 'ensemble' not in set(add_ensemble(pd.DataFrame(rows[:-1])).model)


def test_loader_rejects_infinity(tmp_path):
    path=tmp_path/'infinite.csv'
    pd.DataFrame({'date':['2020-01-01'],'brent_usd_per_barrel':[np.inf]}).to_csv(path,index=False)
    with pytest.raises(ValueError): load_brent(path)


def test_intervals_require_each_horizon():
    from task1_brent.brent_lib.backtest import add_ensemble
    series=make_series(n=90)
    errors=evaluate(series,horizons=(1,),initial_window=60,direct_models={},stat=True)
    with pytest.raises(ValueError,match='Нет калибровки'):
        final_forecast(series,errors,forecast_end=series.index[-1]+pd.offsets.MonthBegin(2))
