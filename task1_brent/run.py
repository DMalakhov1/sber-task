"""Конвейер задачи 1: загрузка → временной бэктест → сравнение → прогноз до 31.12.2027.

Запуск из корня репозитория: python -m task1_brent.run
Результаты в task1_brent/outputs/: метрики, прогноз, графики, диагностика.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .brent_lib.backtest import (DEFAULT_HORIZONS, build_feature_frame,
                                 build_training_pairs, calibrate_intervals,
                                 detect_anomalies, evaluate, metrics_table, interval_widths, interval_evaluation, add_ensemble)
from .brent_lib.data import load_brent, load_macro, DEFAULT_DATA_DIR, MACRO_FILE
from .brent_lib.models import (DIRECT_MODELS, STATISTICAL_MODELS,
                               DirectRidgeModel, ENSEMBLE_MEMBERS)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FINAL_ORIGIN = pd.Timestamp("2026-03-01")  # последняя строка файла; значение сверено с EIA 07.10.2026
FORECAST_END = pd.Timestamp("2027-12-01")
DIRECT_START = "2017-01-01"  # старт ML на бэктесте: ~5 лет обучающих пар
INTERVAL_LEVEL = 0.9


def final_forecast(series, errors, forecast_end=FORECAST_END):
    """Прогноз каждой модели до forecast_end + эмпирические интервалы.

    Статистические модели обучаются на всём ряде. Прямые ML — отдельная
    модель на каждый из 21 месяца, обучающие пары строго внутри ряда.
    Полуширины интервалов — квантили ошибок бэктеста (calibrate_intervals);
    для каждого горизонта требуется собственная калибровка.
    """
    origin = series.index[-1]
    horizon = (forecast_end.year - origin.year) * 12 + \
        (forecast_end.month - origin.month)
    dates = pd.date_range(origin + pd.offsets.MonthBegin(1),
                          periods=horizon, freq="MS")
    features = build_feature_frame(series)
    widths = interval_widths(errors, origin, level=INTERVAL_LEVEL)

    table = pd.DataFrame(index=dates)
    for cls in STATISTICAL_MODELS:
        table[cls.name] = cls().fit(series).predict(horizon)

    feat_last = features.iloc[-1]
    assert not feat_last.isna().any(), "В признаках последнего origin пропуски"
    for cls in DIRECT_MODELS:
        preds = []
        for h in range(1, horizon + 1):
            X_train, y_train = build_training_pairs(series, features, h, len(series) - 1)
            preds.append(cls().fit(X_train, y_train)
                         .predict(feat_last.values.reshape(1, -1)))
        table[cls.name] = preds

    table['ensemble'] = table[list(ENSEMBLE_MEMBERS)].mean(axis=1)
    for model in list(table.columns):
        missing = [h for h in range(1,horizon+1) if (model,h) not in widths]
        if missing:
            raise ValueError(f'Нет калибровки {model} для горизонтов {missing}')
        half_width = np.array([widths[(model,h)] for h in range(1,horizon+1)])
        table[f"{model}_lower"] = table[model] * np.exp(-half_width)
        table[f"{model}_upper"] = table[model] * np.exp(half_width)
    return table


def save_plots(series, forecast, metrics):
    recent = series.loc["2000":]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent.values, color="black", lw=1.2, label="факт")
    colors = {"naive": "gray", "drift": "brown", "ets_log": "tab:blue",
              "ridge_direct": "tab:green", "hgb_direct": "tab:orange",
              "mean_reversion": "tab:purple", "ensemble": "tab:red"}
    for model, color in colors.items():
        ax.plot(forecast.index, forecast[model], color=color, lw=1.2,
                label=f"прогноз: {model}")
    ax.fill_between(forecast.index, forecast["ensemble_lower"],
                    forecast["ensemble_upper"], color="tab:red", alpha=0.15,
                    label=f"интервал {INTERVAL_LEVEL:.0%} (ансамбль, эмпирический)")
    ax.set_title("Brent: история и прогноз до 31.12.2027 (origin 2026-03, "
                 "март сверён с EIA)")
    ax.set_ylabel("USD/барр.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "forecast_plot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, g in metrics.groupby("model"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["MAE"], marker="o", label=model)
    ax.set_title("MAE бэктеста по горизонтам (общие origin'ы всех моделей)")
    ax.set_xlabel("горизонт, мес.")
    ax.set_ylabel("MAE, USD/барр.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "backtest_mae.png", dpi=150)
    plt.close(fig)


def main():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return run_pipeline()


def run_pipeline():
    OUTPUT_DIR.mkdir(exist_ok=True)
    series = load_brent()
    print(f"Ряд Brent: {series.index[0]:%Y-%m} — {series.index[-1]:%Y-%m}, "
          f"{len(series)} месяцев. Последняя строка файла — последний origin; "
          f"мартовское значение сверено с EIA 07.10.2026.")

    print("\nБэктест: расширяющееся окно от 120 мес., горизонты "
          f"{DEFAULT_HORIZONS}, ML — с {DIRECT_START}...")
    errors = add_ensemble(evaluate(series, initial_window=120, direct_start=DIRECT_START))

    # В Git макрофайлы могут отсутствовать; основная задача работает без них.
    errors_all = errors.copy()
    if (DEFAULT_DATA_DIR / MACRO_FILE).exists():
        print('Дополнительный эксперимент с макро: задержки публикаций не подтверждены...', flush=True)
        errors_macro = evaluate(series, stat=False, macro=load_macro(), direct_start=DIRECT_START,
                                direct_models={'ridge_direct_macro': DirectRidgeModel})
        errors_all = pd.concat([errors,errors_macro],ignore_index=True)
    errors_all.to_csv(OUTPUT_DIR / "backtest_errors.csv", index=False)

    metrics = metrics_table(errors, common_origins=True)
    metrics_table(errors_all, common_origins=True).to_csv(OUTPUT_DIR/"macro_comparison.csv",index=False)
    metrics.to_csv(OUTPUT_DIR / "backtest_metrics.csv", index=False)
    print("\nМетрики на общих origin'ах (все модели горизонта присутствуют):")
    print(metrics[metrics.horizon.isin([1,3,6,12,21])].round(3).to_string(index=False))

    # Фиксированный ретроспективный период. Не называем его нетронутым:
    # отчёт Kimi уже показывал ошибки этих лет.
    test_start = pd.Timestamp('2022-01-01')
    development = errors[errors.target < test_start]
    validation = errors[errors.origin >= test_start]
    metrics_table(development).to_csv(OUTPUT_DIR/'development_metrics.csv',index=False)
    metrics_table(validation).to_csv(OUTPUT_DIR/'validation_metrics.csv',index=False)
    interval_evaluation(errors_all,test_start,INTERVAL_LEVEL).to_csv(OUTPUT_DIR/'interval_coverage.csv',index=False)
    chosen = metrics_table(development[development.model != 'ridge_direct_macro'])
    chosen = chosen.loc[chosen.groupby('horizon')['MAE'].idxmin(),['horizon','model']]
    chosen.to_csv(OUTPUT_DIR/'selection_development.csv',index=False)
    selected_validation = validation.merge(chosen, on=['model','horizon'])
    selected_validation.to_csv(OUTPUT_DIR/'selected_validation_errors.csv',index=False)

    anomalies = detect_anomalies(series)
    anomalies.to_csv(OUTPUT_DIR / "anomalies_2026.csv", index=False)
    print("\nДиагностика последних месяцев (перцентиль |лог-изменения| за всю историю):")
    print(anomalies.to_string(index=False))
    print("Три наблюдения 2026 года недостаточны, чтобы объявить смену режима; "
          "это диагностика необычности, а не тест.")

    forecast = final_forecast(series, errors)
    forecast.index.name = "date"
    forecast.round(3).to_csv(OUTPUT_DIR / "forecast_2026-04_2027-12.csv")
    save_plots(series, forecast, metrics)
    selected_forecast = []
    for h, date in enumerate(forecast.index,1):
        model = chosen.set_index('horizon').loc[h,'model']
        selected_forecast.append(dict(date=date, horizon=h, model=model, prediction=forecast.loc[date,model],
                                      lower=forecast.loc[date,model+'_lower'], upper=forecast.loc[date,model+'_upper']))
    pd.DataFrame(selected_forecast).to_csv(OUTPUT_DIR/'selected_forecast.csv',index=False)
    import json, hashlib
    import platform, importlib.metadata
    report = {"python":platform.python_version(),
              "versions":{p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","statsmodels","matplotlib")},
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *sorted((Path(__file__).parent/"brent_lib").glob("*.py"))]},
              "models":{"direct_target":"log_price", "ridge_alpha":1.0,"hgb_max_iter":200,"hgb_seed":42,"mean_reversion_window":120,"mean_reversion_half_life":12},"origin": str(series.index[-1].date()), "horizons":21, "validation_start":"2022-01-01",
              "validation_status":"retrospective; previously inspected in Kimi report",
              "macro_status":"experimental; common one-month publication lag not verified",
              "interval_method":"absolute log-ratio residual quantile per horizon; calibration target strictly before validation start",
              "data_sha256":hashlib.sha256((Path(__file__).parent/"data/raw/brent_monthly_1987-2026.csv").read_bytes()).hexdigest()}
    (OUTPUT_DIR/"run_metadata.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    from .brent_lib.report import write_report
    write_report(OUTPUT_DIR)
    print(f"\nФайлы сохранены в {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
