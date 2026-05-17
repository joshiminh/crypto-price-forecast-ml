import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from .models import build_lstm_model, build_gru_model

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_sequences(series, lookback):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

def train_dl_models(df, max_dl_points=5000, lookback=30):
    mm_scaler = MinMaxScaler()
    
    df_dl_subset = df.iloc[-max_dl_points:] if len(df) > max_dl_points else df
    close_scaled = mm_scaler.fit_transform(df_dl_subset[['close']])
    
    X_all, y_all = create_sequences(close_scaled, lookback=lookback)
    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    
    print("Training LSTM (5 epochs)...")
    lstm_model = build_lstm_model(lookback)
    lstm_model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0, validation_split=0.1)
    
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0)
    lstm_pred = mm_scaler.inverse_transform(lstm_pred_scaled)
    y_test_inv = mm_scaler.inverse_transform(y_test)
    
    lstm_rmse = rmse(y_test_inv, lstm_pred)
    lstm_mae = mean_absolute_error(y_test_inv, lstm_pred)
    lstm_mape = mape(y_test_inv, lstm_pred)
    print(f'LSTM - RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, MAPE: {lstm_mape:.2f}%')
    
    # Save the model
    lstm_model.save('results/lstm_model.keras')
    print("LSTM model saved to results/lstm_model.keras")
    
    print("Training GRU (5 epochs)...")
    gru_model = build_gru_model(lookback)
    gru_model.fit(X_train, y_train, epochs=5, batch_size=128, verbose=0, validation_split=0.1)
    
    gru_pred_scaled = gru_model.predict(X_test, verbose=0)
    gru_pred = mm_scaler.inverse_transform(gru_pred_scaled)
    
    gru_rmse = rmse(y_test_inv, gru_pred)
    gru_mae = mean_absolute_error(y_test_inv, gru_pred)
    gru_mape = mape(y_test_inv, gru_pred)
    print(f'GRU - RMSE: {gru_rmse:.4f}, MAE: {gru_mae:.4f}, MAPE: {gru_mape:.2f}%')
    
    # Save the model
    gru_model.save('results/gru_model.keras')
    print("GRU model saved to results/gru_model.keras")
