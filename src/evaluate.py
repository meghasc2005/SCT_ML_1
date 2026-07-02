import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_data, prepare_data, split_dataset
from src.model import build_pipeline, train_model, evaluate_model, save_model

def run_evaluation(data_path="data/train.csv", model_output_path="model.joblib", plot_output_path="actual_vs_predicted.png"):
    """
    Loads data, trains model, evaluates performance, prints metrics,
    and generates Actual vs. Predicted scatter plot.
    """
    print("Loading and preprocessing data...")
    df = load_data(data_path)
    X, y, feature_cols = prepare_data(df, target_col='SalePrice')
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training rows and {len(X_test)} testing rows.")
    print(f"Using features: {feature_cols}")
    
    print("Training Linear Regression model pipeline...")
    pipeline = build_pipeline(feature_cols)
    model = train_model(pipeline, X_train, y_train)
    
    # Save model
    save_model(model, model_output_path)
    
    # Evaluate
    metrics, preds = evaluate_model(model, X_test, y_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"R² Score : {metrics['r2']:.4f}")
    print(f"RMSE     : ${metrics['rmse']:,.2f}")
    print(f"MAE      : ${metrics['mae']:,.2f}")
    print("="*50 + "\n")
    
    # Generate Matplotlib plot
    plot_actual_vs_predicted(y_test, preds, r2_score_val=metrics['r2'], rmse_val=metrics['rmse'], save_path=plot_output_path)
    return metrics, model

def plot_actual_vs_predicted(y_true, y_pred, r2_score_val=None, rmse_val=None, save_path="actual_vs_predicted.png"):
    """
    Creates and saves a publication-grade scatter plot of Actual vs Predicted prices.
    """
    plt.figure(figsize=(9, 7), dpi=150)
    sns.set_style("whitegrid")
    
    # Scatter plot
    plt.scatter(y_true / 1000, y_pred / 1000, alpha=0.65, color='#4A90E2', edgecolors='k', linewidth=0.5, label='Test Predictions')
    
    # Ideal parity line
    min_val = min(y_true.min(), y_pred.min()) / 1000
    max_val = max(y_true.max(), y_pred.max()) / 1000
    plt.plot([min_val, max_val], [min_val, max_val], color='#E94E77', linestyle='--', linewidth=2, label='Ideal Parity (y=x)')
    
    plt.title("House Price Prediction using Linear Regression: Actual vs. Predicted Prices", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Sale Price ($ in Thousands)", fontsize=12, labelpad=10)
    plt.ylabel("Predicted Sale Price ($ in Thousands)", fontsize=12, labelpad=10)
    
    if r2_score_val is not None and rmse_val is not None:
        stats_text = f"R² Score: {r2_score_val:.3f}\nRMSE: ${rmse_val:,.0f}"
        plt.gca().text(0.05, 0.90, stats_text, transform=plt.gca().transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved successfully to {save_path}")
        
    plt.close()

if __name__ == "__main__":
    run_evaluation()
