import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from src.data_processing import get_preprocessor

def build_pipeline(feature_cols):
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining data preprocessing
    and a LinearRegression model.
    """
    preprocessor = get_preprocessor(feature_cols)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    return model_pipeline

def train_model(pipeline, X_train, y_train):
    """
    Fits the LinearRegression pipeline on training data.
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates model predictions against ground truth labels.
    
    Returns:
        dict: Containing R2 score, RMSE, and MAE.
        np.ndarray: Predicted values.
    """
    preds = pipeline.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }
    return metrics, preds

def save_model(model, filepath="model.joblib"):
    """
    Saves trained pipeline to disk using joblib.
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model successfully saved to {filepath}")

def load_model(filepath="model.joblib"):
    """
    Loads trained model pipeline from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at {filepath}. Run training first.")
    return joblib.load(filepath)
