import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from src.evaluate import run_evaluation
from src.model import load_model

def main():
    parser = argparse.ArgumentParser(
        description="House Price Prediction CLI - Linear Regression on Kaggle Dataset"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train model and evaluate performance")
    train_parser.add_argument("--data", default="data/train.csv", help="Path to train.csv")
    train_parser.add_argument("--model-output", default="model.joblib", help="Output path for trained model")
    train_parser.add_argument("--plot-output", default="actual_vs_predicted.png", help="Output path for evaluation scatter plot")
    
    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Predict house price from input features")
    predict_parser.add_argument("--sqft", type=float, required=True, help="Above ground living area square footage (GrLivArea)")
    predict_parser.add_argument("--bedrooms", type=int, required=True, help="Number of bedrooms above ground (BedroomAbvGr)")
    predict_parser.add_argument("--full-bath", type=int, default=1, help="Number of full bathrooms")
    predict_parser.add_argument("--half-bath", type=int, default=0, help="Number of half bathrooms")
    predict_parser.add_argument("--model", default="model.joblib", help="Path to trained model")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_evaluation(data_path=args.data, model_output_path=args.model_output, plot_output_path=args.plot_output)
    elif args.command == "predict":
        if not os.path.exists(args.model):
            print(f"Model file not found at {args.model}. Running automatic training first...")
            run_evaluation(model_output_path=args.model)
            
        model = load_model(args.model)
        
        # Calculate TotalBath
        total_bath = args.full_bath + 0.5 * args.half_bath
        
        # Prepare input dataframe matching model expected features
        input_data = pd.DataFrame([{
            'GrLivArea': args.sqft,
            'BedroomAbvGr': args.bedrooms,
            'FullBath': args.full_bath,
            'HalfBath': args.half_bath,
            'TotalBath': total_bath
        }])
        
        pred_price = model.predict(input_data)[0]
        
        print("\n" + "*"*50)
        print("HOUSE PRICE PREDICTION ESTIMATE")
        print("*"*50)
        print(f"Square Footage : {args.sqft:,.0f} sq.ft.")
        print(f"Bedrooms       : {args.bedrooms}")
        print(f"Bathrooms      : {args.full_bath} Full + {args.half_bath} Half (Total: {total_bath})")
        print("-" * 50)
        print(f"PREDICTED SALE PRICE: ${pred_price:,.2f}")
        print("*"*50 + "\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
