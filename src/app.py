import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import load_data, prepare_data
from src.model import load_model, build_pipeline, train_model, evaluate_model, save_model

# Page config
st.set_page_config(
    page_title="House Price Prediction using Linear Regression",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #0F172A;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    .price-box {
        background: #0F172A;
        color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_dataset():
    return load_data("data/train.csv")

@st.cache_resource
def get_trained_pipeline(_df):
    model_path = "model.joblib"
    X, y, cols = prepare_data(_df, target_col='SalePrice')
    if os.path.exists(model_path):
        try:
            return load_model(model_path), cols
        except Exception:
            pass
    pipeline = build_pipeline(cols)
    model = train_model(pipeline, X, y)
    save_model(model, model_path)
    return model, cols

def main():
    st.markdown('<div class="main-header">🏠 House Price Prediction using Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Linear Regression model trained on the Kaggle House Prices dataset</div>', unsafe_allow_html=True)

    df = get_dataset()
    model, feature_cols = get_trained_pipeline(df)

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Tab", ["Price Prediction", "Model Evaluation & EDA", "Dataset Overview"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Core Features Used:**")
    st.sidebar.markdown("• **GrLivArea**: Square footage above ground\n• **BedroomAbvGr**: Number of bedrooms\n• **FullBath / HalfBath**: Number of bathrooms")

    if app_mode == "Price Prediction":
        col1, col2 = st.columns([1.2, 1.8], gap="large")
        
        with col1:
            st.subheader("Property Input Parameters")
            
            sqft = st.slider("Square Footage (GrLivArea)", min_value=500, max_value=5000, value=1800, step=50)
            bedrooms = st.slider("Bedrooms Above Ground", min_value=1, max_value=8, value=3, step=1)
            
            bath_col1, bath_col2 = st.columns(2)
            with bath_col1:
                full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2, step=1)
            with bath_col2:
                half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=4, value=1, step=1)
                
            total_bath = full_bath + 0.5 * half_bath
            st.caption(f"Calculated Total Bathrooms: **{total_bath}**")

        with col2:
            st.subheader("Predicted Sale Price")
            
            # Prepare input data matching model features exactly
            input_df = pd.DataFrame([{
                'GrLivArea': sqft,
                'BedroomAbvGr': bedrooms,
                'FullBath': full_bath,
                'HalfBath': half_bath,
                'TotalBath': total_bath
            }])
            
            predicted_price = model.predict(input_df)[0]
            price_per_sqft = predicted_price / sqft if sqft > 0 else 0
            
            st.markdown(f"""
            <div class="price-box">
                <div style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8;">Estimated Sale Price</div>
                <div style="font-size: 3rem; font-weight: 700; margin: 0.6rem 0; color: #10B981;">${predicted_price:,.2f}</div>
                <div style="font-size: 1rem; color: #CBD5E1;">Approx. <b>${price_per_sqft:,.2f}</b> / sq. ft.</div>
            </div>
            """, unsafe_allow_html=True)

    elif app_mode == "Model Evaluation & EDA":
        st.subheader("Linear Regression Model Metrics")
        
        X, y, cols = prepare_data(df, target_col='SalePrice')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics, preds = evaluate_model(model, X_test, y_test)
        
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value">{metrics["r2"]:.4f}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Root Mean Squared Error</div><div class="metric-value">${metrics["rmse"]:,.2f}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Mean Absolute Error</div><div class="metric-value">${metrics["mae"]:,.2f}</div></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.markdown("#### Actual vs. Predicted Prices")
            fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
            sns.scatterplot(x=y_test/1000, y=preds/1000, alpha=0.6, color="#3B82F6", edgecolor="k", ax=ax)
            min_v = min(y_test.min(), preds.min()) / 1000
            max_v = max(y_test.max(), preds.max()) / 1000
            ax.plot([min_v, max_v], [min_v, max_v], color="#EF4444", linestyle="--", linewidth=2, label="Parity (y=x)")
            ax.set_xlabel("Actual Sale Price ($ in Thousands)")
            ax.set_ylabel("Predicted Sale Price ($ in Thousands)")
            ax.legend()
            st.pyplot(fig)
            
        with col_plot2:
            st.markdown("#### Residual Error Distribution")
            residuals = (y_test - preds) / 1000
            fig2, ax2 = plt.subplots(figsize=(7, 6), dpi=120)
            sns.histplot(residuals, kde=True, color="#10B981", bins=25, ax=ax2)
            ax2.axvline(0, color="#EF4444", linestyle="--")
            ax2.set_xlabel("Prediction Error ($ in Thousands)")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)

    elif app_mode == "Dataset Overview":
        st.subheader("Dataset Summary")
        st.write(f"Dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**.")
        
        st.markdown("#### Core Features Sample")
        preview_cols = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
        st.dataframe(df[preview_cols].head(10), use_container_width=True)
        
        st.markdown("#### Correlation with SalePrice")
        corrs = df[preview_cols].corr()['SalePrice'].sort_values(ascending=False)
        st.bar_chart(corrs.drop('SalePrice'))

if __name__ == "__main__":
    main()
