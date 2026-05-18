# Crypto Price Forecast ML

Simple CLI for loading crypto statistics data, engineering features, and training forecast models.

## Setup

Use a Python virtual environment (`venv`) for this project.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (cmd.exe):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the interactive menu with:

```bash
py main.py
```

Use the CLI directly if you want to skip the menu:

```bash
py main.py --run-pipeline --models all
py main.py --run-pipeline --models lstm
py main.py --run-pipeline --models lstm,arima
py main.py --run-pipeline --models ensemble
```

## Models

Available models:

- LSTM
- GRU
- ARIMA
- Prophet
- Ensemble

The ensemble uses the base model forecasts and averages their predictions on the shared test split.

## Layout

- `main.py`: CLI menu and optional arguments.
- `src/models.py`: model list, selection helpers, and model builders.
- `src/training.py`: metrics, algorithms, and training orchestration.
- `src/data_loader.py`: data loading.
- `src/feature_engineering.py`: feature engineering.

## Notes

- Use Python 3.8 or newer.
- The `.venv/` folder is ignored by Git via `.gitignore`.
