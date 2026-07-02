#  House Price Prediction using Linear Regression

A modular, end-to-end Python machine learning project implementing a **Scikit-Learn Linear Regression model** to predict housing market values using Kaggle's famous **"House Prices - Advanced Regression Techniques"** (`train.csv`) dataset.

---

##  Features & Capabilities

- **Automated Data Processing**: Robust pipeline handling missing values (`SimpleImputer`), categorical feature encoding (`OneHotEncoder`), and engineered combined features (`TotalBath = FullBath + 0.5 * HalfBath`).
- **Core Feature Modeling**: Leverages key physical dimensions including Above-Ground Square Footage (`GrLivArea`), Bedrooms (`BedroomAbvGr`), and Bathroom counts (`FullBath` & `HalfBath`).
- **Publication-Grade Evaluation**: Automatically evaluates test set performance computing **R² Score**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)**, while exporting high-resolution Matplotlib scatter plots (`actual_vs_predicted.png`).
- **Command-Line Interface (CLI)**: Intuitive CLI tool for instant model training, evaluation, and inference.
- **Interactive Streamlit Web GUI**: Clean, professional web application built with Streamlit featuring interactive prediction sliders and model residual exploration.
- **Jupyter Notebook Workspaces**: Includes a complete Exploratory Data Analysis (EDA) and step-by-step development notebook.

---

##  Repository Structure

```text
├── data/
│   ├── download_data.py               # Automated Kaggle train.csv dataset downloader
│   └── train.csv                      # Housing dataset (1,460 properties, 81 features)
├── notebooks/
│   └── eda_and_model_development.ipynb # Interactive EDA, correlation heatmaps, and regression walkthrough
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_processing.py             # Data loading, cleaning, imputation, and feature engineering
│   ├── model.py                       # Scikit-Learn pipeline construction, training, evaluation, & persistence
│   ├── evaluate.py                    # Evaluation workflow outputting metrics and Matplotlib scatter plots
│   ├── cli.py                         # Command-Line Interface for training and instant prediction
│   └── app.py                         # Rich Streamlit interactive web GUI application
├── requirements.txt                   # Python package dependency manifest
├── .gitignore                         # Version control exclusions
└── README.md                          # Project setup and usage documentation
```

---

##  Setup & Installation

### 1. Clone & Navigate to Repository
Ensure you are inside the project root folder:
```bash
cd /path/to/SCT_ML_1
```

### 2. Create & Activate Virtual Environment
We recommend using Python 3.10+ virtual environments:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset
Run the helper script to verify or automatically download `train.csv` into the `data/` directory:
```bash
python data/download_data.py
```

---

##  Usage Instructions

### Command-Line Interface (CLI)

#### 1. Train & Evaluate the Model
Run the CLI training command to fit the regression model, print evaluation metrics, and generate the `actual_vs_predicted.png` scatter plot:
```bash
python -m src.cli train
```
*Output Metrics Example:*
```text
==================================================
MODEL EVALUATION RESULTS
==================================================
R² Score : 0.63+
RMSE     : $53,000+
==================================================
Plot saved successfully to actual_vs_predicted.png
```

#### 2. Predict House Price from CLI
Run instant predictions by passing custom property parameters:
```bash
python -m src.cli predict --sqft 2200 --bedrooms 4 --full-bath 2 --half-bath 1
```

---

###  Interactive Streamlit Web App

Launch the dynamic web UI locally:
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501` to access:
- **Interactive Valuator**: Adjust square footage and room sliders to view estimated market valuation in real-time.
- **Model Performance & EDA**: View visual parity charts, error residuals, and key accuracy metrics.
- **Dataset Overview**: Inspect raw property data and correlation profiles.

---

##  Running Jupyter Notebooks

To run the interactive exploratory analysis notebook:
```bash
jupyter notebook notebooks/eda_and_model_development.ipynb
```
