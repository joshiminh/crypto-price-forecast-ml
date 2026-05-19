from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any
import warnings

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
    sys.path.append(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit.runtime import exists as streamlit_runtime_exists
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_data
from src.feature_engineering import engineer_features

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")

RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_LOOKBACK = 30
DEFAULT_MAX_POINTS = 5000
TEST_RATIO = 0.2
DEFAULT_FORECAST_DAYS = 7
MAX_FORECAST_DAYS = 30
MODEL_OPTIONS = ["lstm", "gru", "arima", "prophet", "ensemble"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "ensemble": "Ensemble",
}
COLOR_MAP = {
    "lstm": "#0f766e",
    "gru": "#f97316",
    "arima": "#2563eb",
    "prophet": "#dc2626",
    "ensemble": "#111827",
}


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(249,115,22,0.12), transparent 28%),
                linear-gradient(180deg, #f6f7f4 0%, #eef2f7 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-shell {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 55%, #fef3e7 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.2rem;
        }
        .hero-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.10);
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: 0.85rem 0 0.35rem 0;
            font-size: 2.4rem;
            line-height: 1.05;
            color: #0f172a;
        }
        .hero-copy {
            margin: 0;
            max-width: 56rem;
            color: #475569;
            font-size: 1rem;
        }
        .metric-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }
        .metric-label {
            color: #64748b;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.05;
        }
        .metric-note {
            color: #475569;
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }
        .section-note {
            color: #475569;
            margin: 0.2rem 0 1rem 0;
        }
        .spotlight-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            min-height: 150px;
        }
        .spotlight-model {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
        }
        .spotlight-value {
            font-size: 1.9rem;
            color: #0f172a;
            font-weight: 700;
            margin: 0.4rem 0 0.35rem 0;
        }
        .spotlight-copy {
            color: #475569;
            font-size: 0.92rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.65rem 1rem;
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(15, 23, 42, 0.07);
        }
        .stTabs [aria-selected="true"] {
            background: #0f172a !important;
            color: #f8fafc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for index in range(len(series) - lookback):
        X.append(series[index : index + lookback])
        y.append(series[index + lookback])
    return np.asarray(X), np.asarray(y)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.ravel(y_true), np.ravel(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.ravel(np.asarray(y_true, dtype=float))
    y_pred = np.ravel(np.asarray(y_pred, dtype=float))
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@st.cache_data(show_spinner=False)
def load_prepared_frame(max_points: int = DEFAULT_MAX_POINTS) -> pd.DataFrame:
    df = load_data()
    if df.empty:
        return df

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df = engineer_features(df)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    if len(df) > max_points:
        df = df.iloc[-max_points:].reset_index(drop=True)

    return df


def prepare_split(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK, test_ratio: float = TEST_RATIO) -> dict[str, Any] | None:
    if df.empty or "close" not in df.columns:
        return None

    dates = pd.to_datetime(df["date"]).to_numpy() if "date" in df.columns else None
    close_series = df["close"].astype(float).to_numpy()

    split_idx = int(len(close_series) * (1 - test_ratio))
    split_idx = max(split_idx, lookback + 1)
    if split_idx >= len(close_series):
        return None

    train_series = close_series[:split_idx]
    test_series = close_series[split_idx:]
    train_dates = dates[:split_idx] if dates is not None else None
    test_dates = dates[split_idx:] if dates is not None else None

    if len(train_series) <= lookback or len(test_series) == 0:
        return None

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    if len(X_train) == 0:
        return None

    return {
        "train_series": train_series,
        "test_series": test_series,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "history": train_series.tolist(),
        "lookback": lookback,
        "full_dates": dates,
        "full_series": close_series,
    }


def prepare_forward_context(df: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> dict[str, Any] | None:
    if df.empty or "close" not in df.columns:
        return None

    close_series = df["close"].astype(float).to_numpy()
    if len(close_series) <= lookback:
        return None

    dates = pd.to_datetime(df["date"]).to_numpy() if "date" in df.columns else None
    scaler = MinMaxScaler()
    scaler.fit(close_series.reshape(-1, 1))
    return {
        "series": close_series,
        "dates": dates,
        "history": close_series.tolist(),
        "lookback": lookback,
        "scaler": scaler,
        "last_close": float(close_series[-1]),
    }


def resolve_sequence_artifact(model_name: str) -> Path:
    candidates = [
        RESULTS_DIR / f"{model_name}.keras",
        RESULTS_DIR / f"{model_name}_model.keras",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing model file for {MODEL_LABELS[model_name]} in {RESULTS_DIR}")


@st.cache_resource(show_spinner=False)
def load_sequence_model(model_name: str):
    from tensorflow.keras.models import load_model

    return load_model(resolve_sequence_artifact(model_name))


def forecast_sequence_model(model: Any, context: dict[str, Any], test_series: np.ndarray) -> np.ndarray:
    predictions = []
    history = list(context["history"])
    scaler = context["scaler"]
    lookback = context["lookback"]

    for actual_value in test_series:
        window = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)
        pred_scaled = model.predict(window_scaled, verbose=0)
        pred_value = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(float(pred_value))
        history.append(float(actual_value))

    return np.asarray(predictions, dtype=float)


def forecast_sequence_future(model: Any, context: dict[str, Any], forecast_days: int) -> np.ndarray:
    predictions = []
    history = list(context["history"])
    scaler = context["scaler"]
    lookback = context["lookback"]

    for _ in range(forecast_days):
        window = np.asarray(history[-lookback:], dtype=float).reshape(-1, 1)
        window_scaled = scaler.transform(window).reshape(1, lookback, 1)
        pred_scaled = model.predict(window_scaled, verbose=0)
        pred_value = float(scaler.inverse_transform(pred_scaled)[0, 0])
        predictions.append(pred_value)
        history.append(pred_value)

    return np.asarray(predictions, dtype=float)


def forecast_arima(train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(train_series, order=(5, 1, 0)).fit()
    forecast = fitted.forecast(steps=len(test_series))
    return np.asarray(forecast, dtype=float)


def forecast_arima_future(series: np.ndarray, forecast_days: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    fitted = ARIMA(series, order=(5, 1, 0)).fit()
    forecast = fitted.forecast(steps=forecast_days)
    return np.asarray(forecast, dtype=float)


def _resolve_future_step(dates: np.ndarray | None) -> pd.Timedelta:
    if dates is None or len(dates) < 2:
        return pd.Timedelta(days=1)

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    diffs = pd.Series(date_index).diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return pd.Timedelta(days=1)
    return positive_diffs.median()


def build_future_index(dates: np.ndarray | None, forecast_days: int):
    if dates is None or len(dates) == 0:
        return np.arange(1, forecast_days + 1)

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    step = _resolve_future_step(dates)
    return pd.date_range(start=date_index[-1] + step, periods=forecast_days, freq=step)


def forecast_prophet(train_series: np.ndarray, train_dates: np.ndarray | None, test_dates: np.ndarray | None) -> np.ndarray | None:
    if train_dates is None or test_dates is None:
        return None

    try:
        from prophet import Prophet
    except ImportError:
        return None

    train_df = pd.DataFrame({"ds": pd.to_datetime(train_dates), "y": train_series.astype(float)})
    forecast_frame = pd.DataFrame({"ds": pd.to_datetime(test_dates)})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(train_df)
    forecast = model.predict(forecast_frame)
    return forecast["yhat"].to_numpy()


def forecast_prophet_future(series: np.ndarray, dates: np.ndarray | None, forecast_days: int) -> tuple[np.ndarray, pd.Index] | tuple[None, None]:
    if dates is None:
        return None, None

    try:
        from prophet import Prophet
    except ImportError:
        return None, None

    train_df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": np.asarray(series, dtype=float)})
    future_index = build_future_index(dates, forecast_days)
    forecast_frame = pd.DataFrame({"ds": pd.to_datetime(future_index)})

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    model.fit(train_df)
    forecast = model.predict(forecast_frame)
    return forecast["yhat"].to_numpy(), pd.Index(future_index)


def forecast_ensemble(results: dict[str, dict[str, Any]]) -> np.ndarray | None:
    available_results = [result for result in results.values() if result.get("predictions") is not None]
    if len(available_results) < 2:
        return None

    min_length = min(len(result["predictions"]) for result in available_results)
    stacked_predictions = np.column_stack([result["predictions"][-min_length:] for result in available_results])
    return stacked_predictions.mean(axis=1)


def build_result(model_name: str, actual: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "predictions": predictions,
        "actual": actual,
        "rmse": rmse(actual, predictions),
        "mae": float(mean_absolute_error(actual, predictions)),
        "mape": mape(actual, predictions),
        "status": "ready",
    }


def build_future_result(model_name: str, predictions: np.ndarray, future_index, last_close: float) -> dict[str, Any]:
    final_value = float(predictions[-1])
    change_pct = 0.0 if last_close == 0 else float(((final_value - last_close) / last_close) * 100)
    return {
        "model_key": model_name,
        "model": MODEL_LABELS[model_name],
        "predictions": np.asarray(predictions, dtype=float),
        "future_index": future_index,
        "final_value": final_value,
        "change_pct": change_pct,
        "status": "ready",
    }


def run_predictions(selected_models: list[str], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    test_series = context["test_series"]

    for model_name in selected_models:
        if model_name in {"lstm", "gru"}:
            try:
                model = load_sequence_model(model_name)
                predictions = forecast_sequence_model(model, context, test_series)
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "arima":
            try:
                predictions = forecast_arima(context["train_series"], test_series)
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "prophet":
            try:
                predictions = forecast_prophet(context["train_series"], context["train_dates"], context["test_dates"])
                if predictions is None:
                    raise RuntimeError("Prophet requires the package to be installed and date values to be available.")
                results[model_name] = build_result(model_name, np.asarray(test_series, dtype=float), predictions)
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}

    if "ensemble" in selected_models:
        ensemble_predictions = forecast_ensemble(results)
        if ensemble_predictions is None:
            results["ensemble"] = {
                "model_key": "ensemble",
                "model": MODEL_LABELS["ensemble"],
                "status": "skipped",
                "message": "Ensemble requires at least two successful component predictions.",
            }
        else:
            actual = np.asarray(test_series, dtype=float)[-len(ensemble_predictions) :]
            results["ensemble"] = build_result("ensemble", actual, ensemble_predictions)

    return results


def run_future_forecasts(selected_models: list[str], context: dict[str, Any], forecast_days: int) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    future_index = build_future_index(context["dates"], forecast_days)

    for model_name in selected_models:
        if model_name in {"lstm", "gru"}:
            try:
                model = load_sequence_model(model_name)
                predictions = forecast_sequence_future(model, context, forecast_days)
                results[model_name] = build_future_result(model_name, predictions, future_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "arima":
            try:
                predictions = forecast_arima_future(context["series"], forecast_days)
                results[model_name] = build_future_result(model_name, predictions, future_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}
        elif model_name == "prophet":
            try:
                predictions, prophet_index = forecast_prophet_future(context["series"], context["dates"], forecast_days)
                if predictions is None or prophet_index is None:
                    raise RuntimeError("Prophet requires the package to be installed and date values to be available.")
                results[model_name] = build_future_result(model_name, predictions, prophet_index, context["last_close"])
            except Exception as exc:
                results[model_name] = {"model_key": model_name, "model": MODEL_LABELS[model_name], "status": "skipped", "message": str(exc)}

    if "ensemble" in selected_models:
        ensemble_predictions = forecast_ensemble(results)
        if ensemble_predictions is None:
            results["ensemble"] = {
                "model_key": "ensemble",
                "model": MODEL_LABELS["ensemble"],
                "status": "skipped",
                "message": "Ensemble requires at least two successful future forecasts.",
            }
        else:
            results["ensemble"] = build_future_result("ensemble", ensemble_predictions, future_index, context["last_close"])

    return results


def build_price_history_figure(df: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    x_values = pd.to_datetime(df["date"]) if "date" in df.columns else np.arange(len(df))
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=df["close"].astype(float),
            name="Close",
            line=dict(color="#0f766e", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(15,118,110,0.08)",
        )
    )
    figure.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        legend=dict(orientation="h"),
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return figure


def build_comparison_figure(results: dict[str, dict[str, Any]], test_dates: np.ndarray | None) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=("Predictions vs Actual", "Residuals"),
    )

    successful_results = {name: result for name, result in results.items() if result.get("status") == "ready"}
    if not successful_results:
        figure.add_annotation(text="No predictions available", x=0.5, y=0.5, showarrow=False, row=1, col=1)
        return figure

    first_result = next(iter(successful_results.values()))
    actual = np.asarray(first_result["actual"], dtype=float)
    x_values = pd.to_datetime(test_dates) if test_dates is not None else np.arange(len(actual))

    figure.add_trace(
        go.Scatter(x=x_values, y=actual, name="Actual", line=dict(color="#111827", width=2.4)),
        row=1,
        col=1,
    )

    for model_name, result in successful_results.items():
        figure.add_trace(
            go.Scatter(
                x=x_values[-len(result["predictions"]):],
                y=result["predictions"],
                name=result["model"],
                line=dict(color=COLOR_MAP.get(model_name, "#666666"), width=2),
            ),
            row=1,
            col=1,
        )

    best_name = min(successful_results, key=lambda name: successful_results[name]["rmse"])
    best_result = successful_results[best_name]
    residuals = np.asarray(best_result["actual"], dtype=float) - np.asarray(best_result["predictions"], dtype=float)
    figure.add_trace(
        go.Histogram(x=residuals, name=f"Residuals ({best_result['model']})", marker_color="rgba(15,23,42,0.55)"),
        row=2,
        col=1,
    )

    figure.update_layout(
        height=760,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        legend=dict(orientation="h"),
    )
    figure.update_xaxes(title_text="Date", row=1, col=1, showgrid=False)
    figure.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(148,163,184,0.18)")
    figure.update_xaxes(title_text="Residual", row=2, col=1, showgrid=False)
    figure.update_yaxes(title_text="Count", row=2, col=1, gridcolor="rgba(148,163,184,0.18)")
    return figure


def build_future_figure(results: dict[str, dict[str, Any]], forward_context: dict[str, Any]) -> go.Figure:
    figure = go.Figure()
    successful_results = {name: result for name, result in results.items() if result.get("status") == "ready"}
    if not successful_results:
        figure.add_annotation(text="No forward forecasts available", x=0.5, y=0.5, showarrow=False)
        return figure

    last_close = forward_context["last_close"]
    dates = forward_context["dates"]
    if dates is not None and len(dates) > 0:
        last_x = pd.to_datetime(dates[-1])
    else:
        last_x = 0

    for model_name, result in successful_results.items():
        future_index = result["future_index"]
        if isinstance(future_index, pd.Index):
            x_values = pd.to_datetime(future_index)
            anchor_x = [last_x]
        else:
            x_values = future_index
            anchor_x = [0]
        anchor_and_future_x = list(anchor_x) + list(x_values)
        anchor_and_future_y = [last_close] + list(result["predictions"])
        figure.add_trace(
            go.Scatter(
                x=anchor_and_future_x,
                y=anchor_and_future_y,
                name=result["model"],
                mode="lines+markers",
                line=dict(color=COLOR_MAP.get(model_name, "#666666"), width=2.5),
                marker=dict(size=5),
            )
        )

    figure.add_hline(y=last_close, line_dash="dot", line_color="rgba(15,23,42,0.35)")
    figure.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.78)",
        legend=dict(orientation="h"),
    )
    figure.update_xaxes(title_text="Forecast horizon", showgrid=False)
    figure.update_yaxes(title_text="Projected close", gridcolor="rgba(148,163,184,0.18)")
    return figure


def render_metric_cards(df: pd.DataFrame, evaluation_context: dict[str, Any] | None, forecast_days: int) -> None:
    cards = st.columns(4)
    last_close = float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else 0.0
    train_points = len(evaluation_context["train_series"]) if evaluation_context else 0
    test_points = len(evaluation_context["test_series"]) if evaluation_context else 0
    latest_date = pd.to_datetime(df["date"].iloc[-1]).date().isoformat() if "date" in df.columns and not df.empty else "n/a"

    card_specs = [
        ("Latest Close", f"{last_close:,.4f}", f"Most recent row: {latest_date}"),
        ("Rows Loaded", f"{len(df):,}", "Prepared dataset after feature engineering"),
        ("Backtest Split", f"{train_points:,}/{test_points:,}", "Train points / test points"),
        ("Forecast Horizon", f"{forecast_days} day(s)", "Forward projection control"),
    ]

    for column, (label, value, note) in zip(cards, card_specs):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_spotlight_cards(future_results: dict[str, dict[str, Any]], backtest_results: dict[str, dict[str, Any]]) -> None:
    successful_future = [result for result in future_results.values() if result.get("status") == "ready"]
    successful_backtest = [result for result in backtest_results.values() if result.get("status") == "ready"]
    if not successful_future and not successful_backtest:
        return

    best_future = max(successful_future, key=lambda result: result["change_pct"]) if successful_future else None
    most_defensive = min(successful_future, key=lambda result: result["change_pct"]) if successful_future else None
    best_backtest = min(successful_backtest, key=lambda result: result["rmse"]) if successful_backtest else None

    columns = st.columns(3)
    spotlight_specs = [
        (
            "Strongest Upside",
            best_future["model"] if best_future else "Unavailable",
            f"{best_future['change_pct']:+.2f}%" if best_future else "n/a",
            f"Projected final close: {best_future['final_value']:.4f}" if best_future else "No successful forecast available.",
        ),
        (
            "Best Backtest",
            best_backtest["model"] if best_backtest else "Unavailable",
            f"RMSE {best_backtest['rmse']:.4f}" if best_backtest else "n/a",
            f"MAE {best_backtest['mae']:.4f} | MAPE {best_backtest['mape']:.2f}%" if best_backtest else "No successful backtest available.",
        ),
        (
            "Most Defensive",
            most_defensive["model"] if most_defensive else "Unavailable",
            f"{most_defensive['change_pct']:+.2f}%" if most_defensive else "n/a",
            f"Projected final close: {most_defensive['final_value']:.4f}" if most_defensive else "No successful forecast available.",
        ),
    ]

    for column, (label, model_name, value, note) in zip(columns, spotlight_specs):
        column.markdown(
            f"""
            <div class="spotlight-card">
                <div class="spotlight-model">{label}</div>
                <div class="spotlight-value">{model_name}</div>
                <div class="spotlight-copy"><strong>{value}</strong></div>
                <div class="spotlight-copy">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metrics_table(results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for model_name in MODEL_OPTIONS:
        result = results.get(model_name)
        if not result:
            continue
        if result.get("status") != "ready":
            rows.append({"Model": result["model"], "Status": result.get("status", "skipped"), "Message": result.get("message", "")})
            continue
        rows.append(
            {
                "Model": result["model"],
                "Status": "ready",
                "RMSE": round(result["rmse"], 4),
                "MAE": round(result["mae"], 4),
                "MAPE %": round(result["mape"], 2),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_forward_table(results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for model_name in MODEL_OPTIONS:
        result = results.get(model_name)
        if not result:
            continue
        if result.get("status") != "ready":
            rows.append({"Model": result["model"], "Status": result.get("status", "skipped"), "Message": result.get("message", "")})
            continue
        rows.append(
            {
                "Model": result["model"],
                "Status": "ready",
                "Projected Close": round(result["final_value"], 4),
                "Change %": round(result["change_pct"], 2),
                "Horizon Points": len(result["predictions"]),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Crypto Forecast Studio", layout="wide")
    apply_app_styles()

    df = load_prepared_frame()
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Forecast Studio</div>
            <h1 class="hero-title">Crypto Price Forecast</h1>
            <p class="hero-copy">
                Compare saved model performance on the latest backtest split, then project the next 1–30 days from the
                most recent market history in the same workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error("No data loaded.")
        return

    evaluation_context = prepare_split(df)
    forward_context = prepare_forward_context(df)

    with st.sidebar:
        st.header("Forecast Controls")
        selected_models = st.multiselect(
            "Models",
            options=MODEL_OPTIONS,
            default=["lstm", "gru", "arima", "prophet", "ensemble"],
            format_func=lambda model_name: MODEL_LABELS[model_name],
        )
        forecast_days = st.slider("Days ahead", min_value=1, max_value=MAX_FORECAST_DAYS, value=DEFAULT_FORECAST_DAYS)
        st.caption(f"Lookback window: {DEFAULT_LOOKBACK} points")
        st.caption(f"Backtest split: {int(TEST_RATIO * 100)}%")
        submit = st.button("Run Forecast Studio", type="primary")

    render_metric_cards(df, evaluation_context, forecast_days)
    st.markdown('<p class="section-note">The controls drive both the backtest comparison and the forward-looking forecast view.</p>', unsafe_allow_html=True)

    if evaluation_context is None or forward_context is None:
        st.error("Not enough data to build the prediction views.")
        return

    if not submit:
        st.info("Choose one or more models, set the forecast horizon, then click Run Forecast Studio.")
        st.plotly_chart(build_price_history_figure(df), width="stretch")
        return

    if not selected_models:
        st.warning("Select at least one model.")
        return

    with st.spinner("Loading saved models, scoring the backtest split, and building the forward forecast..."):
        backtest_results = run_predictions(selected_models, evaluation_context)
        future_results = run_future_forecasts(selected_models, forward_context, forecast_days)

    skipped_messages = [
        f"{result['model']}: {result.get('message', 'Skipped')}"
        for result in list(backtest_results.values()) + list(future_results.values())
        if result.get("status") == "skipped"
    ]
    if skipped_messages:
        with st.expander("Skipped models", expanded=False):
            for message in skipped_messages:
                st.warning(message)

    render_spotlight_cards(future_results, backtest_results)
    tabs = st.tabs(["Forward Outlook", "Backtest", "Metrics"])

    with tabs[0]:
        st.subheader(f"{forecast_days}-Day Forward Outlook")
        st.plotly_chart(build_future_figure(future_results, forward_context), width="stretch")
        render_forward_table(future_results)

    with tabs[1]:
        st.subheader("Historical Backtest Comparison")
        st.plotly_chart(build_comparison_figure(backtest_results, evaluation_context["test_dates"]), width="stretch")

    with tabs[2]:
        st.subheader("Model Metrics")
        render_metrics_table(backtest_results)


if __name__ == "__main__":
    if not streamlit_runtime_exists():
        raise SystemExit("Unsupported launcher. Start the UI with: streamlit run src/streamlit.py")
    main()
