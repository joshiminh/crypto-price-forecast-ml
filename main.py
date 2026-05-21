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
    from src.models import MODEL_ORDER, MODEL_LABELS, normalize_model_selection

    while True:
        print("\nSelect model scope")
        print("1) All models")
        for index, model_name in enumerate(MODEL_ORDER, start=2):
            print(f"{index}) {MODEL_LABELS[model_name]}")

        choice = input("Choice: ").strip().lower()
        if choice in {"1", "all"}:
            return normalize_model_selection("all")

        if choice.isdigit():
            numeric_choice = int(choice)
            mapped_index = numeric_choice - 2
            if 0 <= mapped_index < len(MODEL_ORDER):
                return normalize_model_selection(MODEL_ORDER[mapped_index])

        normalized = normalize_model_selection(choice)
        if normalized and len(normalized) == 1:
            return normalized

        print("Invalid selection. Pick one model or 'all'.")

def interactive_menu():
    try:
        while True:
            print("\nCrypto Price Forecast")
            print("1) Train models")
            print("2) Run Streamlit")
            print("3) Exit")
            choice = input("Choice [1-3]: ").strip().lower()

            if choice == '1':
                from src.training import run_pipeline

                run_pipeline(_prompt_model_selection())
            elif choice == '2':
                _launch_streamlit_app()
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
        from src.models import normalize_model_selection
        from src.training import run_pipeline

        if args.models:
            selected_models = normalize_model_selection(args.models)
        else:
            selected_models = _prompt_model_selection()

        if args.models and not selected_models:
            print("Invalid --models value.")
            raise SystemExit(1)
        run_pipeline(selected_models)
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
