"""Bounded, chronological comparison. Run from repository root; no API calls."""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from threadpoolctl import threadpool_limits
from task1_brent.brent_lib.data import load_brent, DEFAULT_DATA_DIR, BRENT_FILE
from task1_brent.brent_lib.models import (
    NaiveModel,
    DriftModel,
    ETSLogModel,
    DirectRidgeModel,
    DirectHGBModel,
)
from task1_brent.brent_lib.backtest import build_feature_frame, build_training_pairs
from task1_brent.brent_lib.neural import QUANTILES, fit_neural

HORIZON = 21


def split_plan(series):
    n = len(series)
    a = int(n * 0.70)
    b = int(n * 0.85)
    if min(a, b - a, n - b) <= HORIZON:
        raise ValueError("Periods too short")
    return (
        a,
        b,
        [
            dict(
                period=name,
                start=str(series.index[lo].date()),
                end=str(series.index[hi - 1].date()),
                n=hi - lo,
            )
            for name, lo, hi in [("train", 0, a), ("validation", a, b), ("test", b, n)]
        ],
    )


def origins_for(start, end, stride):
    # First prediction at the end of the preceding period; every target within this period.
    return list(range(start - 1, end - HORIZON, stride))


def point_forecasts(series, t, features):
    train = series.iloc[: t + 1]
    log = np.log(train.to_numpy())
    pred = {}
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always")
        for cls in (NaiveModel, DriftModel, ETSLogModel):
            pred[cls.name] = cls().fit(train).predict(HORIZON)
        for lag in (1, 3, 6, 12):
            fitted = AutoReg(log, lags=lag, trend="ct", old_names=False).fit()
            pred[f"ar_{lag}"] = np.exp(fitted.predict(len(log), len(log) + HORIZON - 1))
        for order in ((1, 1, 0), (0, 1, 1), (1, 1, 1)):
            fit = ARIMA(log, order=order, trend="t").fit()
            pred["arima_" + "".join(map(str, order))] = np.exp(fit.forecast(HORIZON))
        for window in (60, 120, 180):
            for half in (6, 12, 24):
                decay = np.exp(-np.log(2) * np.arange(1, HORIZON + 1) / half)
                pred[f"mr_{window}_{half}"] = np.exp(
                    log[-window:].mean() + (log[-1] - log[-window:].mean()) * decay
                )
        for cls in (DirectRidgeModel, DirectHGBModel):
            out = []
            for h in range(1, HORIZON + 1):
                x, y = build_training_pairs(series, features, h, t)
                out.append(cls().fit(x, y).predict(features.iloc[[t]].to_numpy()))
            pred[cls.name] = np.asarray(out)
    members = ["naive", "ets_log", "ridge_direct", "mr_120_12"]
    pred["ensemble_all"] = np.mean([pred[k] for k in members], axis=0)
    for member in members:
        pred["ensemble_without_" + member] = np.mean(
            [pred[k] for k in members if k != member], axis=0
        )
    if not all(np.isfinite(v).all() for v in pred.values()):
        raise ValueError("Nonfinite forecast")
    return pred, sorted(
        set(type(w.message).__name__ + ": " + str(w.message)[:180] for w in notes)
    )


def quantile_metrics(actual, forecast):
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    error = actual[:, None] - forecast
    loss = np.maximum(error * QUANTILES, error * (QUANTILES - 1))
    lo, hi = forecast[:, 0], forecast[:, -1]
    score = hi - lo + 20 * np.maximum(lo - actual, 0) + 20 * np.maximum(actual - hi, 0)
    # Integral only over [.05,.95]; no claims about unobserved tails.
    return dict(
        pinball=float(loss.mean()),
        crps_central_approx=float(2 * np.trapezoid(loss, QUANTILES, axis=1).mean()),
        coverage90=float(((actual >= lo) & (actual <= hi)).mean()),
        width90=float((hi - lo).mean()),
        interval_score90=float(score.mean()),
    )


def summarize(records):
    frame = pd.DataFrame(records)
    rows = []
    for (period, model, h), g in frame.groupby(["period", "model", "horizon"]):
        err = g.y_pred - g.y_true
        scale = (
            np.mean(np.abs(np.diff(g.y_true.to_numpy()))) if len(g.y_true) > 1 else 1.0
        )
        row = dict(
            period=period,
            model=model,
            horizon=h,
            n=len(g),
            MAE=float(abs(err).mean()),
            RMSE=float(np.sqrt(np.mean(err**2))),
            MASE=float(abs(err).mean() / max(scale, 1e-12)),
            sMAPE=float(
                np.mean(
                    200
                    * np.abs(err)
                    / (
                        np.abs(g.y_true.to_numpy())
                        + np.abs(g.y_pred.to_numpy())
                        + 1e-12
                    )
                )
            ),
        )
        if "q05" in g and g.q05.notna().all():
            row.update(
                quantile_metrics(
                    g.y_true.to_numpy(),
                    g[[f"q{int(q*100):02d}" for q in QUANTILES]].to_numpy(),
                )
            )
            row["median_MAE"] = float(np.abs(g.y_pred - g.y_true).mean())
            row["median_RMSE"] = float(np.sqrt(np.mean((g.y_pred - g.y_true) ** 2)))
        rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result.model == "naive"].set_index(["period", "horizon"]).MAE
    result["relative_MAE"] = result.apply(
        lambda r: r.MAE / base[(r.period, r.horizon)], axis=1
    )
    return result


def stage4_point_leaderboard(metrics):
    rows = []
    for (period, h), g in metrics[
        (metrics.model.isin({"mlp", "rnn", "lstm", "gru", "cnn"}))
        & (metrics.period == "test")
    ].groupby(["period", "horizon"]):
        rows.append(g.sort_values(["MAE", "RMSE"]).iloc[0].to_dict())
    return pd.DataFrame(rows)


def select_on_validation(metrics):
    val = metrics[metrics.period == "validation"]
    if val.empty:
        raise ValueError("No validation results")
    return {
        int(h): g.sort_values(["MAE", "model"]).iloc[0].model
        for h, g in val.groupby("horizon")
    }


def run(out, mode, neural):
    start = time.perf_counter()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty; choose a new --output")
    out.mkdir(parents=True, exist_ok=True)
    series = load_brent()
    a, b, split = split_plan(series)
    features = build_feature_frame(series)
    stride = 12 if mode == "fast" else 3
    epochs = 20 if mode == "fast" else 160
    seeds = (11,) if mode == "fast" else (11, 23, 37)
    qr_seeds = (11,) if mode == "fast" else (11, 23, 37, 53, 71)
    records = []
    seed_records = []
    training = []
    warning_log = []
    selection = None
    for period, begin, end in [("validation", a, b), ("test", b, len(series))]:
        if period == "test":
            selection = select_on_validation(summarize(records))
            (out / "selection.json").write_text(json.dumps(selection, indent=2))
        origins = origins_for(begin, end, stride)
        for i, t in enumerate(origins):
            pred, notes = point_forecasts(series, t, features)
            warning_log.extend(
                dict(period=period, origin=str(series.index[t].date()), warning=n)
                for n in notes
            )
            quantiles = {}
            # Historical h-step log-return distribution, known entirely at the origin.
            quantiles["naive_quantiles"] = np.stack(
                [
                    series.iloc[t]
                    * np.exp(
                        np.quantile(
                            np.log(
                                series.iloc[h : t + 1].to_numpy()
                                / series.iloc[: t + 1 - h].to_numpy()
                            ),
                            QUANTILES,
                        )
                    )
                    for h in range(1, HORIZON + 1)
                ]
            )
            pred["naive_quantiles"] = quantiles["naive_quantiles"][:, 9]
            if neural:
                for kind in ("mlp", "rnn", "lstm", "gru", "cnn", "qrnn"):
                    forecast, seed_preds, meta = fit_neural(
                        series.to_numpy(),
                        t,
                        kind,
                        epochs=epochs,
                        seeds=qr_seeds if kind == "qrnn" else seeds,
                    )
                    payload = dict(
                        period=period, origin=str(series.index[t].date()), model=kind
                    )
                    payload.update({k: v for k, v in meta.items() if k != "model"})
                    training.append(payload)
                    if kind == "qrnn":
                        quantiles[kind] = forecast
                        forecast = forecast[:, 9]
                    pred[kind] = forecast
                    for s, p in zip(meta["seeds"], seed_preds):
                        for h in range(1, HORIZON + 1):
                            row = dict(
                                period=period,
                                model=kind,
                                seed=s,
                                horizon=h,
                                origin=str(series.index[t].date()),
                                y_true=float(series.iloc[t + h]),
                                y_pred=float(
                                    p[h - 1, 9] if kind == "qrnn" else p[h - 1]
                                ),
                            )
                            if kind == "qrnn":
                                row.update(
                                    {
                                        f"q{int(q*100):02d}": float(v)
                                        for q, v in zip(QUANTILES, p[h - 1])
                                    }
                                )
                            seed_records.append(row)
            for name, p in pred.items():
                for h in range(1, HORIZON + 1):
                    row = dict(
                        period=period,
                        model=name,
                        horizon=h,
                        origin=str(series.index[t].date()),
                        target=str(series.index[t + h].date()),
                        y_true=float(series.iloc[t + h]),
                        y_pred=float(p[h - 1]),
                    )
                    if name in quantiles:
                        row.update(
                            {
                                f"q{int(q*100):02d}": float(v)
                                for q, v in zip(QUANTILES, quantiles[name][h - 1])
                            }
                        )
                    records.append(row)
            pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
            print(
                period,
                i + 1,
                "/",
                len(origins),
                str(series.index[t].date()),
                flush=True,
            )
    metrics = summarize(records)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(seed_records).to_csv(out / "seed_predictions.csv", index=False)
    seed_metrics = []
    if seed_records:
        for s, g in pd.DataFrame(seed_records).groupby("seed"):
            for (phase, model, h), sub in g.groupby(["period", "model", "horizon"]):
                err = sub.y_pred - sub.y_true
                row = dict(
                    period=phase,
                    model=model,
                    seed=int(s),
                    horizon=int(h),
                    MAE=float(abs(err).mean()),
                    RMSE=float(np.sqrt(np.mean(err**2))),
                )
                if model == "qrnn":
                    row.update(
                        quantile_metrics(
                            sub.y_true.to_numpy(),
                            sub[[f"q{int(q*100):02d}" for q in QUANTILES]].to_numpy(),
                        )
                    )
                seed_metrics.append(row)
        pd.DataFrame(seed_metrics).to_csv(out / "seed_metrics.csv", index=False)
    # Selection already frozen before test. Final refit cannot alter recorded scores.
    final, notes = point_forecasts(series, len(series) - 1, features)
    fan = None
    if neural:
        for kind in ("mlp", "rnn", "lstm", "gru", "cnn", "qrnn"):
            p, _, meta = fit_neural(
                series.to_numpy(),
                len(series) - 1,
                kind,
                epochs=epochs,
                seeds=qr_seeds if kind == "qrnn" else seeds,
            )
            payload = dict(
                period="final", origin=str(series.index[-1].date()), model=kind
            )
            payload.update({k: v for k, v in meta.items() if k != "model"})
            training.append(payload)
            if kind == "qrnn":
                fan = p
                p = p[:, 9]
            final[kind] = p
    final["naive_quantiles"] = np.asarray(
        [
            series.iloc[-1]
            * np.exp(
                np.quantile(
                    np.log(series.iloc[h:].to_numpy() / series.iloc[:-h].to_numpy()),
                    0.5,
                )
            )
            for h in range(1, HORIZON + 1)
        ]
    )
    future = pd.date_range(
        series.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS"
    )
    forecast = pd.DataFrame(
        [
            dict(
                date=str(d.date()),
                horizon=h,
                model=selection[h],
                forecast=float(final[selection[h]][h - 1]),
            )
            for h, d in enumerate(future, 1)
        ]
    )
    forecast.to_csv(out / "selected_forecast.csv", index=False)
    pd.DataFrame(final, index=future).rename_axis("date").to_csv(
        out / "all_forecasts.csv"
    )
    if fan is not None:
        pd.DataFrame(
            fan, index=future, columns=[f"q{int(q*100):02d}" for q in QUANTILES]
        ).rename_axis("date").to_csv(out / "qrnn_forecast.csv")
    (out / "training_audit.json").write_text(json.dumps(training, indent=2))
    (out / "warnings.json").write_text(
        json.dumps(warning_log, ensure_ascii=False, indent=2)
    )
    pd.DataFrame(split).to_csv(out / "split.csv", index=False)
    metadata = dict(
        mode=mode,
        neural=neural,
        epochs_max=epochs,
        seeds=list(seeds),
        qrnn_seeds=list(qr_seeds),
        stride_months=stride,
        elapsed_seconds=round(time.perf_counter() - start, 2),
        data_sha256=hashlib.sha256(
            (DEFAULT_DATA_DIR / BRENT_FILE).read_bytes()
        ).hexdigest(),
        selection_rule="Validation MAE separately per horizon; frozen before test",
        test_status="Retrospective: history previously inspected; not a pristine holdout",
        target="USD/barrel; neural standardized cumulative log-return, exp inverse; point exp is not bias-corrected mean",
        split=split,
    )
    import importlib.metadata

    metadata["versions"] = {
        k: importlib.metadata.version(k)
        for k in ["numpy", "pandas", "statsmodels", "scikit-learn"]
        + (["torch"] if neural else [])
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    make_report(
        out, series, pd.DataFrame(records), metrics, selection, forecast, fan, metadata
    )


def make_report(out, series, records, metrics, selection, forecast, fan, metadata):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), layout="constrained")
    for ax, h in zip(axes, [1, 12, 21]):
        g = records[(records.period == "test") & (records.horizon == h)]
        for name in ["naive", selection[h]] + (
            ["qrnn"] if "qrnn" in g.model.unique() and selection[h] != "qrnn" else []
        ):
            sub = g[g.model == name].sort_values("target")
            ax.plot(pd.to_datetime(sub.target), sub.y_pred, label=name)
        actual = g[g.model == "naive"].sort_values("target")
        ax.plot(
            pd.to_datetime(actual.target), actual.y_true, color="black", label="Факт"
        )
        ax.set_title(f"Отложенный период: h={h} мес.; выбор по validation")
        ax.set_ylabel("USD/барр.")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(out / "test_forecasts.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.plot(
        series.iloc[-72:].index,
        series.iloc[-72:],
        color="black",
        label="Факт до 2026-03",
    )
    dates = pd.to_datetime(forecast.date)
    ax.plot(dates, forecast.forecast, label="Выбор по validation")
    if fan is not None:
        ax.fill_between(
            dates,
            fan[:, 0],
            fan[:, -1],
            alpha=0.18,
            color="teal",
            label="QRNN: q05–q95 (номинально 90%)",
        )
        ax.fill_between(
            dates,
            fan[:, 4],
            fan[:, 14],
            alpha=0.25,
            color="teal",
            label="QRNN: q25–q75",
        )
        ax.plot(dates, fan[:, 9], color="teal", label="QRNN: медиана")
    ax.legend(fontsize=8)
    ax.set_ylabel("USD/барр.")
    ax.set_title("Исторический прогноз от марта 2026; без обновления данных")
    ax.grid(alpha=0.2)
    fig.savefig(out / "quantile_forecast.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Расширенный эксперимент Brent",
        "",
        f"Режим: {metadata['mode']}. Затрачено {metadata['elapsed_seconds']:.1f} с на этой среде; это не замер Mac.",
        "",
        "Выбор модели сделан по MAE validation до расчёта test. Период test уже обсуждался в предыдущих ревью: оценка ретроспективная, не новый независимый тест.",
        "",
        "| Горизонт | Выбрана на validation | MAE validation | MAE test | MAE naive test |",
        "|---|---|---:|---:|---:|",
    ]
    for h in (1, 3, 6, 12, 21):
        m = selection[h]
        v = metrics[(metrics.model == m) & (metrics.horizon == h)]
        test = float(v[v.period == "test"].MAE.iloc[0])
        val = float(v[v.period == "validation"].MAE.iloc[0])
        naive = float(
            metrics[
                (metrics.period == "test")
                & (metrics.model == "naive")
                & (metrics.horizon == h)
            ].MAE.iloc[0]
        )
        lines.append(f"| {h} | {m} | {val:.2f} | {test:.2f} | {naive:.2f} |")
    last = forecast.iloc[-1]
    lines += [
        "",
        f"Прогноз к декабрю 2027: **{last.forecast:.2f} USD/барр.**, модель `{last.model}`. Это результат модели от мартовского среза, а не актуальный прогноз от октября.",
        "",
        "Модели разных горизонтов выбираются отдельно, поэтому итоговая линия может иметь изломы. Экономические драйверы не оценивались причинно.",
        "",
        "Пересекающиеся горизонты зависимы: строки нельзя трактовать как независимые повторения. Статистическая значимость победы не заявляется. Сравнение всех моделей выполняется на одинаковых датах внутри каждого периода; последние месяцы используются только как цели, чтобы все 21 горизонт были наблюдаемы.",
    ]
    if fan is not None:
        q = metrics[
            (metrics.model == "qrnn")
            & (metrics.period == "test")
            & (metrics.horizon == 21)
        ].iloc[0]
        lines += [
            "",
            f"QRNN на h=21: измеренное покрытие {q.coverage90:.1%}, средняя ширина {q.width90:.2f} USD/барр.; номинал 90%. Интервал в декабре 2027: {fan[-1,0]:.2f}–{fan[-1,-1]:.2f}. Это интервал QRNN, а не выбранной точечной модели.",
            "Полная сетка и метрики есть в CSV. `crps_central_approx` — интеграл pinball только по [0.05,0.95], без хвостов; это не точный полный CRPS. Калибровка интервалов на test не выполнялась.",
        ]
    lines += [
        "",
        "Сопоставление с дипломом и схема обучения: ../../EXTENDED_METHOD.md. Ошибки оптимизации/сходимости: warnings.json.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["fast", "full"], default="fast")
    p.add_argument("--neural", action="store_true")
    p.add_argument("--output", default="task1_brent/outputs/extended_fast")
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.mode, args.neural)


if __name__ == "__main__":
    main()
