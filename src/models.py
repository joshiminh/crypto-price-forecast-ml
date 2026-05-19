import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense

MODEL_ORDER = ["lstm", "gru", "arima", "prophet", "ensemble"]
BASE_MODELS = ["lstm", "gru", "arima", "prophet"]

MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "arima": "ARIMA",
    "prophet": "Prophet",
    "ensemble": "Ensemble",
}

SEQUENCE_MODELS = ["lstm", "gru"]


def available_models(include_ensemble=True):
    return BASE_MODELS + ["ensemble"] if include_ensemble else list(BASE_MODELS)


def normalize_model_selection(raw_value):
    if not raw_value:
        return None

    if isinstance(raw_value, (list, tuple, set)):
        tokens = [str(item).strip().lower() for item in raw_value if str(item).strip()]
    else:
        tokens = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    if not tokens:
        return None

    if "all" in tokens:
        return available_models(include_ensemble=True)

    normalized = []
    wants_ensemble = "ensemble" in tokens
    for token in tokens:
        if token == "ensemble":
            continue
        if token in available_models(include_ensemble=False) and token not in normalized:
            normalized.append(token)

    if wants_ensemble:
        for base_model in BASE_MODELS:
            if base_model not in normalized:
                normalized.append(base_model)
        normalized.append("ensemble")

    return normalized or None


def menu_model_choices():
    return [
        ("1", "All models", None),
        ("2", "LSTM", ["lstm"]),
        ("3", "GRU", ["gru"]),
        ("4", "ARIMA", ["arima"]),
        ("5", "Prophet", ["prophet"]),
        ("6", "Ensemble", ["ensemble"]),
    ]


def menu_text():
    return "Models: " + "  ".join(f"{key}) {label}" for key, label, _ in menu_model_choices())


def resolve_menu_choice(choice):
    choice = choice.strip().lower()
    for key, _label, models in menu_model_choices():
        if choice == key:
            return models
    if choice == "all":
        return available_models(include_ensemble=True)
    if choice in MODEL_LABELS:
        if choice == "ensemble":
            return ["ensemble"]
        return [choice]
    return None

def moving_average_forecast(series, test_len, window=10):
    """Simple moving average forecast."""
    return np.full(test_len, np.mean(series[-window:]))

def gm11_forecast(series, n_steps):
    """Optimized GM(1,1) implementation."""
    x0 = np.array(series, dtype=float)
    if len(x0) > 500:
        x0 = x0[-500:]
    
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = 0.5 * (x1[:-1] + x1[1:])
    B = np.column_stack((-z1, np.ones_like(z1)))
    Y = x0[1:]
    
    try:
        params = np.linalg.lstsq(B, Y, rcond=None)[0]
        a, b = params
        
        if abs(a) < 1e-10:
            return np.full(n_steps, np.mean(x0[-10:]))
        
        def predict(k):
            return (x0[0] - b / a) * np.exp(-a * k) + b / a
        
        forecasts = [(predict(n + i) - predict(n + i - 1)) for i in range(1, n_steps + 1)]
        return np.array(forecasts)
    except Exception:
        return np.full(n_steps, np.mean(x0[-10:]))

def build_lstm_model(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        LSTM(32, activation='tanh', return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(lookback):
    model = Sequential([
        Input(shape=(lookback, 1)),
        GRU(32, activation='tanh', return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_prophet_model():
    if not HAS_PROPHET:
        raise ImportError("Prophet is not installed")
    
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0  # Disable uncertainty to avoid memory issues
    )
    return model
