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
from tensorflow.keras.layers import LSTM, GRU, Dense

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
        LSTM(32, input_shape=(lookback, 1), activation='tanh', return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(lookback):
    model = Sequential([
        GRU(32, input_shape=(lookback, 1), activation='tanh', return_sequences=False),
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
