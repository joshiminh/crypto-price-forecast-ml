# 💰 Crypto Price Forecast ML

A concise machine learning project for forecasting cryptocurrency prices using LSTM and GRU models. 📈🤖

## ✨ Overview
This repository contains code, data, and trained models for building time-series models to predict crypto price movements.
- **Models:** LSTM and GRU implemented in TensorFlow / Keras.
- **Data:** historical crypto statistics in `data/`.
- **Notebook:** `crypto_price_forecast.ipynb` for exploration.

## 📁 Repository Structure
- `main.py` — entrypoint for running pipelines.
- `src/data_loader.py` — data ingestion and preprocessing.
- `src/feature_engineering.py` — feature generation and transformations.
- `src/models.py` — model architectures (LSTM, GRU).
- `src/training.py` — training and evaluation loops.
- `data/crypto_statistics_data.csv` — raw dataset.
- `results/` — trained models and outputs.

## ⚡ Quick Setup
1. Create a virtual environment:
   python -m venv .venv
2. Activate it (Windows PowerShell):
   .venv\Scripts\Activate.ps1
3. Install dependencies:
   pip install -r requirements.txt
4. Verify the environment:
   python main.py --help

## 🧭 Usage
- Run the full training pipeline:
   python main.py --run-pipeline --models all
- Launch the terminal menu:
   python main.py
- From the menu, choose option 3 to start the Streamlit UI in the current terminal.
- Stop the CLI-launched Streamlit server with `Ctrl+C`.
- Launch the Streamlit UI directly:
   streamlit run src/streamlit.py
- Open the exploratory notebook:
   Use Jupyter to open `crypto_price_forecast.ipynb`.

## 🌐 Streamlit UI
The Streamlit app compares the saved models on a backtest split and projects the next 1-30 days from the latest history.
- LSTM and GRU load from `results/lstm.keras` and `results/gru.keras`.
- ARIMA, Prophet, and Ensemble now generate named result artifacts in `results/` during training.
- The UI includes a forward-looking forecast horizon control and a separate backtest comparison view.

## 🧪 Data Notes
- `data/crypto_statistics_data.csv` contains historical price and indicator features.
- Ensure data is cleaned and aligned before training.
- Use `src/data_loader.py` to customize windows and resampling.

## 🔧 Configuration
- Hyperparameters are set in `src/training.py`; modify learning rate, batch size, epochs.
- Model save/load paths use `results/` by default.

## 🏗️ Model Details
- LSTM: stacked LSTM layers with dropout for regularization.
- GRU: gated recurrent units optimized for faster convergence.
- Both models output a regression forecast for the next price step.

## 📊 Evaluation
- Metrics: MSE, MAE, and directional accuracy.
- Evaluation utilities are available in `src/training.py`.

## 💾 Results
- Trained neural models are saved in `results/` as `.keras` files with the model key as the filename.
- Example files: `gru.keras`, `lstm.keras`.
- Additional generated artifacts include `arima.json`, `prophet.json`, `ensemble.json`, and `manifest.json`.

## 🛠️ Troubleshooting
- If GPU is not available, training falls back to CPU.
- On Windows, allow `Activate.ps1` via execution policy if blocked.
- For CUDA errors, verify drivers and TensorFlow compatibility.
- Do not start the Streamlit UI with `python src/streamlit.py`; use `streamlit run src/streamlit.py` instead.
 
