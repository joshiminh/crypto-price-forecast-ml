import numpy as np
import pandas as pd

def load_data(path="data/crypto_statistics_data.csv"):
    """Load the main statistics dataset."""
    return pd.read_csv(path, parse_dates=["date"])

def load_raw_data(hourly_path="data/Crypto_Hourly_Refined.csv", stats_path="data/Crypto_Stats_Refined.csv"):
    """Load and merge optional raw hourly/stats datasets."""
    try:
        df_hourly = pd.read_csv(hourly_path)
        df_stats = pd.read_csv(stats_path)

        df_hourly["date"] = pd.to_datetime(df_hourly["date"])
        df_stats["date"] = pd.to_datetime(df_stats["date"])

        df_hourly = df_hourly.sort_values("date").reset_index(drop=True)
        df_stats = df_stats.sort_values("date").reset_index(drop=True)

        df_hourly["date_day"] = df_hourly["date"].dt.floor("D")
        df_stats["date_day"] = df_stats["date"]

        df = df_hourly.merge(
            df_stats[["date_day", "average", "range", "pct_change"]],
            on="date_day",
            how="left",
            suffixes=("", "_daily"),
        )
        return df.drop(columns=["date_day"])
    except FileNotFoundError as exc:
        print(f"File not found: {exc}")
        return pd.DataFrame()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def ma(series, window):
    return series.rolling(window=window).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_rolling = pd.Series(gain, index=series.index).rolling(window=period).mean()
    loss_rolling = pd.Series(loss, index=series.index).rolling(window=period).mean()
    loss_rolling = loss_rolling.replace(0, 1e-9)
    rs = gain_rolling / loss_rolling
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def engineer_features(df):
    if "close" not in df.columns:
        return df

    close = df["close"]

    for span in [7, 12, 14, 21, 26]:
        df[f"EMA_{span}"] = ema(close, span)

    for window in [10, 20, 50]:
        df[f"MA_{window}"] = ma(close, window)

    df["RSI_14"] = compute_rsi(close, period=14)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = ema(df["MACD"], 9)

    window_bb = 20
    df["BB_MID"] = ma(close, window_bb)
    df["BB_STD"] = close.rolling(window=window_bb).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]

    df["log_return"] = np.log(close / close.shift(1))
    df["return"] = close.pct_change()
    vol_window = 24
    df["volatility"] = df["log_return"].rolling(window=vol_window).std() * np.sqrt(vol_window)

    df["next_close"] = df["close"].shift(-1)
    df["trend_up"] = (df["next_close"] > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)
