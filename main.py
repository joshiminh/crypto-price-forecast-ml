import argparse
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")
from src.data_loader import load_data
from src.models import menu_text, normalize_model_selection, resolve_menu_choice
from src.training import run_pipeline

def terminate_program(message="Terminating..."):
    print(message)
    raise SystemExit(0)

def _prompt_model_selection():
    while True:
        print(menu_text())
        choice = input("Select: ").strip().lower()
        if choice in {"1", "all"}:
            return normalize_model_selection("all")
        selected_models = resolve_menu_choice(choice)
        if selected_models is not None:
            return selected_models
        print("Invalid model choice.")

def interactive_menu():
    try:
        while True:
            print("\nCrypto Price Forecast")
            print("1) Train models")
            print("2) Load data")
            print("3) Exit")
            choice = input("Choice [1-3]: ").strip().lower()

            if choice == '1':
                run_pipeline(_prompt_model_selection())
            elif choice == '2':
                df = load_data()
                if not df.empty:
                    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
                    print(df.head(3).to_string(index=False))
                else:
                    print("No data loaded.")
            elif choice in {'3', 'q', 'quit', 'exit'}:
                terminate_program()
            else:
                print("Invalid choice.")
    except KeyboardInterrupt:
        terminate_program("\nInterrupted. Terminating...")

def main():
    parser = argparse.ArgumentParser(description="Crypto Price Forecast CLI")
    parser.add_argument('--run-pipeline', action='store_true', help="Run the full pipeline (Load -> Engineer -> Train)")
    parser.add_argument('--models', help="Comma-separated models to train: all,lstm,gru,arima,prophet,ensemble")
    
    args = parser.parse_args()
    
    if args.run_pipeline:
        selected_models = normalize_model_selection(args.models)
        if args.models and not selected_models:
            print("Invalid --models value.")
            raise SystemExit(1)
        run_pipeline(selected_models)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
