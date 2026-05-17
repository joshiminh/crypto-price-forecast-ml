import numpy as np
import pandas as pd

def ema(series, span):
    """Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def ma(series, window):
    """Simple Moving Average"""
    return series.rolling(window=window).mean()

def compute_rsi(series, period=14):
    """Relative Strength Index with zero-division guard"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_rolling = pd.Series(gain, index=series.index).rolling(window=period).mean()
    loss_rolling = pd.Series(loss, index=series.index).rolling(window=period).mean()
    loss_rolling = loss_rolling.replace(0, 1e-9)  # avoid divide-by-zero
    rs = gain_rolling / loss_rolling
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

def engineer_features(df):
    """
    Compute features based on the close price.
    Assumes df has a 'close' column.
    """
    if 'close' not in df.columns:
        return df

    close = df['close']

    # Moving Averages & EMAs
    for span in [7, 12, 14, 21, 26]:
        df[f'EMA_{span}'] = ema(close, span)

    for w in [10, 20, 50]:
        df[f'MA_{w}'] = ma(close, w)

    # RSI (14-period)
    df['RSI_14'] = compute_rsi(close, period=14)

    # MACD (12, 26, 9)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = ema(df['MACD'], 9)

    # Bollinger Bands (20-period)
    window_bb = 20
    df['BB_MID'] = ma(close, window_bb)
    df['BB_STD'] = close.rolling(window=window_bb).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']

    # Returns & Volatility
    df['log_return'] = np.log(close / close.shift(1))
    df['return'] = close.pct_change()
    vol_window = 24  # ~1 day for hourly data
    df['volatility'] = df['log_return'].rolling(window=vol_window).std() * np.sqrt(vol_window)

    # Target variables
    df['next_close'] = df['close'].shift(-1)
    df['trend_up'] = (df['next_close'] > df['close']).astype(int)

    # Drop NaNs from rolling/shift operations
    df = df.dropna().reset_index(drop=True)
    return df
