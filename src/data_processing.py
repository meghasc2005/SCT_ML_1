import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

CORE_FEATURES = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']

def load_data(filepath="data/train.csv"):
    """
    Loads the house price dataset from CSV.
    If file doesn't exist, attempts to run downloader.
    """
    if not os.path.exists(filepath):
        from data.download_data import download_train_data
        filepath = download_train_data(filepath)
    df = pd.read_csv(filepath)
    return df

def clean_and_engineer(df, include_total_bath=True):
    """
    Cleans dataset, handles missing values, and engineers features.
    
    Parameters:
        df (pd.DataFrame): Raw input dataframe.
        include_total_bath (bool): Whether to create TotalBath feature (FullBath + 0.5*HalfBath).
        
    Returns:
        pd.DataFrame: Cleaned dataframe with engineered features.
    """
    df = df.copy()
    
    # Ensure numeric types for core columns
    for col in CORE_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
            
    # Feature Engineering: Total bathrooms
    if include_total_bath and 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
        
    # Drop rows where target variable is missing if present
    if 'SalePrice' in df.columns:
        df = df.dropna(subset=['SalePrice'])
        
    return df

def get_preprocessor(feature_cols):
    """
    Creates a Scikit-Learn ColumnTransformer to impute missing values
    and encode categorical variables.
    """
    numeric_features = [f for f in feature_cols if f not in ['Neighborhood', 'BldgType', 'HouseStyle']]
    categorical_features = [f for f in feature_cols if f in ['Neighborhood', 'BldgType', 'HouseStyle']]
    
    transformers = []
    
    if numeric_features:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('num', num_transformer, numeric_features))
        
    if categorical_features:
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_transformer, categorical_features))
        
    return ColumnTransformer(transformers=transformers, remainder='drop')

def prepare_data(df, target_col='SalePrice', feature_cols=None):
    """
    Prepares X (features) and y (target) for model training or inference.
    """
    cleaned_df = clean_and_engineer(df)
    
    if feature_cols is None:
        feature_cols = CORE_FEATURES.copy()
        if 'TotalBath' in cleaned_df.columns:
            feature_cols.append('TotalBath')
            
    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col] if target_col in cleaned_df.columns else None
    
    return X, y, feature_cols

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits features and target into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
