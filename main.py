import argparse
import sys
from src.data_loader import load_data, load_raw_data
from src.feature_engineering import engineer_features
from src.train import train_dl_models

def run_pipeline():
    print("Loading data...")
    df = load_data()
    if df.empty:
        print("Failed to load statistics data.")
        return
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Training models...")
    train_dl_models(df)
    print("Pipeline complete.")

def interactive_menu():
    while True:
        print("\n=== Crypto Price Forecast CLI ===")
        print("1. Run Full Pipeline (Load -> Engineer -> Train)")
        print("2. Load Data (Check if working)")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            run_pipeline()
        elif choice == '2':
            df = load_data()
            if not df.empty:
                print(f"Loaded successfully. Shape: {df.shape}")
                print(df.head())
            else:
                print("Failed to load data.")
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Crypto Price Forecast CLI")
    parser.add_argument('--run-pipeline', action='store_true', help="Run the full pipeline (Load -> Engineer -> Train)")
    
    args = parser.parse_args()
    
    if args.run_pipeline:
        run_pipeline()
    else:
        interactive_menu()

if __name__ == "__main__":
    main()
