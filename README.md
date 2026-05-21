# 🪙 Crypto Price Forecast ML

Lightweight ML project for cryptocurrency price forecasting with CLI + Streamlit.

## ✨ What This Project Does
- Loads historical crypto data from `data/crypto_statistics_data.csv`.
- Engineers technical indicators (EMA, MA, RSI, MACD, Bollinger Bands, volatility).
- Trains multiple forecasting models and saves artifacts to `results/`.
- Compares model performance with backtest metrics.

## 🧠 Models Trained
- `LSTM` (TensorFlow/Keras sequence model)
- `GRU` (TensorFlow/Keras sequence model)
- `ARIMA` (statsmodels)
- `Prophet` (optional, if installed)
- `Ensemble` (mean of available base-model predictions)

## ⚙️ Optimizers / Training Setup
- Supported optimizers for sequence models: `adam`, `rmsprop`
- LSTM loss: `MSE`
- GRU loss: `MSE`
- ARIMA: statistical fit via `ARIMA(...).fit()` (no NN optimizer)
- Prophet: model fit via `Prophet().fit()` (no Keras optimizer)
- Sequence training config in `src/train.py`:
  - epochs: `5`
  - batch size: `128`
  - validation split: `0.1`

## 📁 Project Structure
- `main.py` : CLI entrypoint
- `src/data.py` : data loading + feature engineering
- `src/models.py` : model definitions and builders
- `src/train.py` : training pipeline + evaluation + artifact writing
- `src/streamlit.py` : interactive dashboard
- `results/` : `.keras` and `.json` model artifacts

## 🚀 Quick Start
1. Create environment:
   - `python -m venv .venv`
2. Activate (PowerShell):
   - `.venv\Scripts\Activate.ps1`
3. Install deps:
   - `pip install -r requirements.txt`
4. Train all models:
   - `python main.py --run-pipeline --models all`

## 🖥️ Run App
- CLI menu:
  - `python main.py`
- Streamlit directly:
  - `streamlit run src/streamlit.py`

## 📊 Outputs
- Neural artifacts: `results/adam/lstm.keras`, `results/adam/gru.keras` (or `results/rmsprop/...`)
- Statistical/summary artifacts:
  - `results/<optimizer>/arima.json`
  - `results/<optimizer>/prophet.json` (if Prophet available)
  - `results/<optimizer>/ensemble.json`

## 🧪 Metrics
- RMSE, MAE, MAPE

## 🛠️ Notes
- Prophet is optional; training skips it when not installed.
- Training workflow in GitHub Actions retrains and commits `results/` on `main`.
