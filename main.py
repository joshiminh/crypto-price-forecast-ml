import argparse
import os
import subprocess
import sys
import warnings
import socket
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings("ignore", message="TensorFlow GPU support is not available.*")


def _find_available_port(start_port=8501, max_port=8510):
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found in range {start_port}-{max_port}.")

def terminate_program(message="Terminating..."):
    print(message)
    raise SystemExit(0)


def _launch_streamlit_app():
    project_root = Path(__file__).resolve().parent
    streamlit_script = project_root / "src" / "streamlit.py"

    if not streamlit_script.exists():
        print(f"Could not find Streamlit entrypoint: {streamlit_script}")
        return

    streamlit_port = _find_available_port()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_script),
        "--server.port",
        str(streamlit_port),
    ]

    print(f"Starting Streamlit UI at http://localhost:{streamlit_port}", flush=True)
    print("Streamlit will run in this terminal. Stop it with Ctrl+C.", flush=True)
    try:
        subprocess.run(command, cwd=str(project_root), check=True)
    except KeyboardInterrupt:
        print("\nStreamlit UI stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with status code {exc.returncode}.")
    except Exception as exc:
        print(f"Failed to launch Streamlit UI: {exc}")

def _prompt_model_selection():
    from src.models import menu_text, normalize_model_selection, resolve_menu_choice

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
            print("3) Launch Streamlit UI (runs here, stop with Ctrl+C)")
            print("4) Exit")
            choice = input("Choice [1-4]: ").strip().lower()

            if choice == '1':
                from src.training import run_pipeline

                run_pipeline(_prompt_model_selection())
            elif choice == '2':
                from src.data_loader import load_data

                df = load_data()
                if not df.empty:
                    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
                    print(df.head(3).to_string(index=False))
                else:
                    print("No data loaded.")
            elif choice == '3':
                _launch_streamlit_app()
            elif choice in {'4', 'q', 'quit', 'exit'}:
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
        from src.models import normalize_model_selection
        from src.training import run_pipeline

        selected_models = normalize_model_selection(args.models)
        if args.models and not selected_models:
            print("Invalid --models value.")
            raise SystemExit(1)
        run_pipeline(selected_models)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
