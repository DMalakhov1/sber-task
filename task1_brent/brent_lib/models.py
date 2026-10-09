"""Модели прогноза Brent. Единый интерфейс: fit(series_train) / predict(horizon).

Набор методов намеренно разнородный:
- NaiveModel, DriftModel — обязательные baseline;
- ETSLogModel — экспоненциальное сглаживание с затухающим трендом на лог-шкале
  (классический статистический метод, гарантирует положительный прогноз);
- DirectRidgeModel / DirectHGBModel — прямые модели «один горизонт — одна
  модель» на лагах Brent и макропризнаках. Прямая схема выбрана, потому что
  рекурсивный прогноз на 21 месяц потребовал бы неизвестных будущих значений
  макропоказателей; здесь используются только признаки, доступные в момент
  выпуска прогноза.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NaiveModel:
    """Прогноз = последнее наблюдение. Baseline для сравнения MAE."""

    name = "naive"

    def fit(self, train):
        self.last_ = float(train.iloc[-1])
        return self

    def predict(self, horizon):
        return np.full(horizon, self.last_)


class DriftModel:
    """Последнее наблюдение + наклон среднего прироста по train."""

    name = "drift"

    def fit(self, train):
        self.last_ = float(train.iloc[-1])
        self.drift_ = float(np.mean(np.diff(train.values))) if len(train) > 1 else 0.0
        return self

    def predict(self, horizon):
        steps = np.arange(1, horizon + 1)
        return self.last_ + steps * self.drift_


class HistoricalMeanModel:
    name = "historical_mean"

    def fit(self, train):
        self.mean_ = float(train.mean())
        return self

    def predict(self, horizon):
        return np.full(horizon, self.mean_)


class SESModel:
    name = "ses"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        self.model_ = ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True, use_brute=True)
        return self

    def predict(self, horizon):
        return np.asarray(self.model_.forecast(horizon), dtype=float)


class HoltModel:
    name = "holt"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        self.model_ = ExponentialSmoothing(
            y,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)
        return self

    def predict(self, horizon):
        return np.asarray(self.model_.forecast(horizon), dtype=float)


class ETSLogModel:
    """ETS(A,Ad,N) на логарифме ряда. Сезонность не задаётся: у месячных
    цен на нефть устойчивая годовая сезонность не установлена."""

    name = "ets_log"

    def fit(self, train):
        log_train = np.log(train.values.astype(float))
        self.fit_ = ExponentialSmoothing(
            log_train, trend="add", damped_trend=True, seasonal=None
        ).fit(optimized=True)
        return self

    def predict(self, horizon):
        return np.exp(self.fit_.forecast(horizon))


class AR1Model:
    name = "ar_1"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        self.model_ = AutoReg(y, lags=1, old_names=False).fit()
        return self

    def predict(self, horizon):
        return np.asarray(self.model_.forecast(steps=horizon), dtype=float)


class ARSelectModel:
    name = "selected_ar"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        best = None
        for p in range(1, 5):
            try:
                model = AutoReg(y, lags=p, old_names=False).fit()
                score = model.bic
                if best is None or score < best[0]:
                    best = (score, p, model)
            except Exception:
                pass
        if best is None:
            raise ValueError("AR selection failed")
        self.order_ = best[1]
        self.model_ = best[2]
        return self

    def predict(self, horizon):
        return np.asarray(self.model_.forecast(steps=horizon), dtype=float)


class ARMASelectModel:
    name = "selected_arma"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        stationary = y.diff().dropna()
        if len(stationary) < 12:
            stationary = y.pct_change().dropna()
        best = None
        for p in range(0, 3):
            for q in range(0, 3):
                if p == 0 and q == 0:
                    continue
                try:
                    fit = ARIMA(
                        stationary,
                        order=(p, 0, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    score = fit.bic
                    if best is None or score < best[0]:
                        best = (score, (p, q), fit)
                except Exception:
                    continue
        if best is None:
            raise ValueError("ARMA selection failed")
        self.order_ = best[1]
        self.model_ = best[2]
        self.level_last_ = float(y.iloc[-1])
        return self

    def predict(self, horizon):
        delta = np.asarray(self.model_.forecast(steps=horizon), dtype=float)
        return self.level_last_ + np.cumsum(delta)


class ARIMASelectModel:
    name = "selected_arima"

    def fit(self, train):
        y = pd.Series(train).astype(float)
        best = None
        for d in (0, 1):
            for p in range(0, 3):
                for q in range(0, 3):
                    if p == 0 and q == 0 and d == 0:
                        continue
                    try:
                        fit = ARIMA(
                            y,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                        score = fit.bic
                        if best is None or score < best[0]:
                            best = (score, (p, d, q), fit)
                    except Exception:
                        continue
        if best is None:
            raise ValueError("ARIMA selection failed")
        self.order_ = best[1]
        self.model_ = best[2]
        return self

    def predict(self, horizon):
        return np.asarray(self.model_.forecast(steps=horizon), dtype=float)


class _DirectBase:
    """Прямая модель: log(y(t+h)) = f(X(t)), X(t) — только данные, доступные в t.

    fit принимает матрицу признаков X и вектор цели y, уже выровненные
    по train-окну в backtest.build_training_frame; модель сама ряд не видит.
    Все преобразования (scaler) живут внутри sklearn Pipeline и обучаются
    только на train-парах.
    """

    regressor = None

    def fit(self, X, y):
        self.model_ = make_pipeline(StandardScaler(), self._make_regressor())
        self.model_.fit(X, np.log(y))
        return self

    def predict(self, X_row):
        return float(np.exp(self.model_.predict(X_row)[0]))


class DirectRidgeModel(_DirectBase):
    name = "ridge_direct"

    def _make_regressor(self):
        return Ridge(alpha=1.0)


class DirectHGBModel(_DirectBase):
    """Нелинейный градиентный бустинг; scaler ему безразличен, но единый
    конвейер оставлен ради одинакового интерфейса."""

    name = "hgb_direct"

    def _make_regressor(self):
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3, random_state=42
        )


class MeanReversionModel:
    """Возврат лог-цены к среднему последних 120 месяцев, half-life=12.

    Параметры фиксированы до пересчёта; это сценарная модель, не установленный
    закон цены нефти. Средний уровень оценивается только на train.
    """

    name = "mean_reversion"

    def fit(self, train):
        self.last_ = float(np.log(train.iloc[-1]))
        self.mean_ = float(np.log(train.iloc[-120:]).mean())
        return self

    def predict(self, horizon):
        decay = np.exp(-np.log(2) * np.arange(1, horizon + 1) / 12)
        return np.exp(self.mean_ + (self.last_ - self.mean_) * decay)


ENSEMBLE_MEMBERS = ("naive", "ets_log", "ridge_direct", "mean_reversion")
STATISTICAL_MODELS = [
    NaiveModel,
    DriftModel,
    HistoricalMeanModel,
    SESModel,
    HoltModel,
    AR1Model,
    ARSelectModel,
    ARMASelectModel,
    ARIMASelectModel,
    ETSLogModel,
    MeanReversionModel,
]
DIRECT_MODELS = [DirectRidgeModel, DirectHGBModel]


def fit_predict(train, horizon):
    return ETSLogModel().fit(train).predict(horizon)
