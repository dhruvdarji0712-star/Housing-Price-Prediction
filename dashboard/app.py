import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from pathlib import Path

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# Global Dashboard Theme: Professional Blue, White & Navy
st.markdown("""
<style>
    :root {
        --primary: #2563EB;
        --navy: #142B4A;
        --muted: #64748B;
        --surface: #FFFFFF;
        --soft-blue: #EFF6FF;
    }

    .stApp {
        background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 42%);
        color: #1E293B;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 250, 255, 0.88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #142B4A 0%, #1E3A5F 100%);
        border-right: 1px solid rgba(255,255,255,0.10);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px 12px;
        margin: 4px 0;
        transition: background 0.2s ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,0.16);
    }

    h1, h2, h3 {
        color: #142B4A;
        letter-spacing: -0.4px;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE8F8;
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 5px 18px rgba(20, 43, 74, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #1D4ED8 !important;
        font-weight: 750;
    }

    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #2563EB, #1D4ED8);
        color: white;
        border: 0;
        border-radius: 10px;
        padding: 0.55rem 1.15rem;
        font-weight: 650;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.18);
    }

    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #1E40AF;
        color: white;
        border: 0;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #DCE8F8;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #DCE8F8;
    }

    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
</style>
""", unsafe_allow_html=True)

# Consistent Plotly appearance across dashboard charts
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#2563EB", "#14B8A6", "#F59E0B", "#64748B"]

# Sidebar Navigation
st.sidebar.title("🏠 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Model Comparison",
        "Feature Importance",
        "House Price Prediction",
        "Synthetic Data Analysis",
        "Project Findings"
    ]
)

if page == "Home":

    # Custom CSS
    st.markdown("""
    <style>

        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #2563EB;
            letter-spacing: -1px;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 17px;
            color: #64748B;
            margin-top: 0px;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(37, 99, 235, 0.06);
            padding: 22px;
            border-radius: 14px;
            border: 1px solid rgba(37, 99, 235, 0.20);
            box-shadow: 0px 3px 8px rgba(0,0,0,0.04);
        }
    
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }
    
        div[data-testid="stMetricValue"] {
            font-size: 27px;
            font-weight: 700;
        }

    </style>
    """, unsafe_allow_html=True)
    
    # Dashboard Header
    st.markdown(
        '<p class="main-title">🏠 House Price Prediction</p>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<p class="subtitle">Machine Learning & Synthetic Data Analysis Dashboard</p>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Model Performance Metrics
    st.subheader("📊 Model Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("R² Score", "89.10%")
    col2.metric("MAE", "$17,447")
    col3.metric("RMSE", "$28,909")
    col4.metric("Test Houses", "292")
    
    st.divider()
    
    # Project Introduction
    st.subheader("📌 About This Project")
    
    st.write("""
    This project focuses on predicting residential house prices
    using Machine Learning.
    
    It also explores manual synthetic data generation and
    compares model performance using original and modified data.
    """)
    
    st.success("Final Tuned Random Forest Model Successfully Validated!")

if page == "Model Comparison":

    # Model Comparison Section

    st.divider()
    
    st.subheader("📈 Model Performance Comparison")
    
    model_data = pd.DataFrame({
        "Model": [
            "Original Model",
            "Augmented Model",
            "Log Transformation",
            "Tuned Model"
        ],
        "MAE": [
            17505.21,
            17844.76,
            17382.09,
            17447.31
        ],
        "RMSE": [
            29089.23,
            29481.31,
            29830.57,
            28909.14
        ],
        "R2 Score": [
            0.8897,
            0.8867,
            0.8840,
            0.8910
        ]
    })
    
    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Interactive Charts
    
    col1, col2 = st.columns(2)
    
    with col1:
        mae_chart = px.bar(
            model_data,
            x="Model",
            y="MAE",
            title="MAE Comparison",
            text_auto=".0f"
        )
    
        st.plotly_chart(mae_chart, use_container_width=True)
    
    with col2:
        rmse_chart = px.bar(
            model_data,
            x="Model",
            y="RMSE",
            title="RMSE Comparison",
            text_auto=".0f"
        )
    
        st.plotly_chart(rmse_chart, use_container_width=True)
    
    # R2 Score Chart
    
    r2_chart = px.bar(
        model_data,
        x="Model",
        y="R2 Score",
        title="R² Score Comparison",
        text_auto=".3f"
    )
    
    st.plotly_chart(r2_chart, use_container_width=True)

if page == "Feature Importance":

    # Feature Importance Analysis
    
    st.divider()
    
    st.subheader("🎯 Feature Importance Analysis")
    
    feature_data = pd.DataFrame({
        "Feature": [
            "OverallQual",
            "GrLivArea",
            "TotalBsmtSF",
            "2ndFlrSF",
            "BsmtFinSF1",
            "1stFlrSF",
            "LotArea",
            "GarageArea",
            "GarageCars",
            "YearBuilt"
        ],
        "Original Model": [
            0.556444, 0.122748, 0.034290, 0.030186,
            0.027349, 0.026791, 0.017700, 0.014785,
            0.013979, 0.012846
        ],
        "Augmented Model": [
            0.572866, 0.116463, 0.040276, 0.019779,
            0.032342, 0.023939, 0.012657, 0.016216,
            0.012684, 0.009738
        ]
    })
    
    feature_long = feature_data.melt(
        id_vars="Feature",
        var_name="Dataset",
        value_name="Importance"
    )
    
    feature_chart = px.bar(
        feature_long,
        x="Importance",
        y="Feature",
        color="Dataset",
        barmode="group",
        orientation="h",
        title="Top 10 Feature Importance Comparison"
    )
    
    feature_chart.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )
    
    st.plotly_chart(
        feature_chart,
        use_container_width=True
    )
    
    st.caption(
        "Feature importance represents relative model importance, "
        "not causal impact on house prices."
    )

if page == "House Price Prediction":

    # Interactive House Price Prediction
    
    st.divider()
    
    st.subheader("🏠 Predict House Price")
    
    # Load Dataset and Saved Model
    df = pd.read_csv(BASE_DIR / "cleaned_housing_data.csv", keep_default_na=False)
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    
    loaded_model = joblib.load(BASE_DIR / "best_housing_model.pkl")
    
    st.write("Enter house details to get a predicted price.")
    
    with st.form("prediction_form"):
    
        col1, col2 = st.columns(2)
    
        with col1:
            overall_qual = st.slider(
                "Overall Quality",
                min_value=1,
                max_value=10,
                value=7
            )
    
            living_area = st.number_input(
                "Living Area (sq ft)",
                min_value=334,
                max_value=5642,
                value=1500
            )
    
            basement_area = st.number_input(
                "Basement Area (sq ft)",
                min_value=0,
                max_value=6110,
                value=1000
            )
    
        with col2:
            garage_cars = st.selectbox(
                "Garage Capacity (Cars)",
                [0, 1, 2, 3, 4]
            )
    
            year_built = st.number_input(
                "Year Built",
                min_value=int(df["YearBuilt"].min()),
                max_value=int(df["YearBuilt"].max()),
                value=2000
            )
    
            neighborhood = st.selectbox(
                "Neighborhood",
                sorted(df["Neighborhood"].unique())
            )
    
        predict_button = st.form_submit_button(
            "Predict House Price"
        )
    
    if predict_button:
    
        input_house = df.drop(
            columns=["SalePrice", "Id"]
        ).iloc[[0]].copy()
    
        input_house["OverallQual"] = overall_qual
        input_house["GrLivArea"] = living_area
        input_house["TotalBsmtSF"] = basement_area
        input_house["GarageCars"] = garage_cars
        input_house["YearBuilt"] = year_built
        input_house["Neighborhood"] = neighborhood
    
        predicted_price = loaded_model.predict(input_house)[0]
    
        st.success(
            f"Predicted House Price: ${predicted_price:,.2f}"
        )
    
        st.caption(
            "Demo estimate: features not exposed in this form "
            "are held at a reference house's values."
        )

        # Prepare Prediction Result for Download
        prediction_result = pd.DataFrame({
            "OverallQual": [overall_qual],
            "GrLivArea": [living_area],
            "TotalBsmtSF": [basement_area],
            "GarageCars": [garage_cars],
            "YearBuilt": [year_built],
            "Neighborhood": [neighborhood],
            "Predicted_Price": [round(predicted_price, 2)]
        })

        # Download Prediction as CSV
        csv_data = prediction_result.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Result",
            data=csv_data,
            file_name="house_price_prediction.csv",
            mime="text/csv"
        )
if page == "Synthetic Data Analysis":

    # Synthetic Data Analysis
    
    st.divider()
    
    st.subheader("🧪 Synthetic Data Generation Overview")
    
    # Load Project Datasets
    cleaned_data = pd.read_csv(
        BASE_DIR / "cleaned_housing_data.csv",
        keep_default_na=False
    )
    
    synthetic_data = pd.read_csv(
        BASE_DIR / "manual_synthetic_data.csv",
        keep_default_na=False
    )
    
    # Calculate Dataset Records
    original_records = 1168
    synthetic_records = len(synthetic_data)
    augmented_records = original_records + synthetic_records
    
    # Display Dataset Metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Original Training Records",
        f"{original_records:,}"
    )
    
    col2.metric(
        "Synthetic Records",
        f"{synthetic_records:,}"
    )
    
    col3.metric(
        "Augmented Training Records",
        f"{augmented_records:,}"
    )
    
    # Dataset Comparison Chart
    dataset_comparison = pd.DataFrame({
        "Dataset": [
            "Original Training Data",
            "Synthetic Data",
            "Augmented Training Data"
        ],
        "Records": [
            original_records,
            synthetic_records,
            augmented_records
        ]
    })
    
    dataset_chart = px.bar(
        dataset_comparison,
        x="Dataset",
        y="Records",
        color="Dataset",
        text="Records",
        title="Original vs Synthetic vs Augmented Data"
    )
    
    st.plotly_chart(
        dataset_chart,
        use_container_width=True
    )
    
    st.info(
        "Synthetic records were generated using a manual "
        "bootstrap-based modification approach."
    )

if page == "Project Findings":

    # Key Project Findings
    
    st.divider()
    
    st.subheader("💡 Key Project Findings")
    
    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """**Best Overall Model**

Tuned Random Forest achieved:
- MAE: $17,447.31
- RMSE: $28,909.14
- R² Score: 0.891"""
        )

    with col2:
        st.info(
            """**Synthetic Data Experiment**

- 500 modified synthetic records generated.
- Augmented model did not improve overall test performance.
- Synthetic data increased training records to 1,668."""
        )

    st.markdown("### Project Limitations")
    
    st.write("""
    - Synthetic records are based on bootstrap sampling and modifications.
    - SalePrice was retained while selected features were modified.
    - Only 7 expensive houses were available in the test set.
    - Only Random Forest models were explored.
    """)
    
    st.caption(
        "Housing Price Prediction | Machine Learning Portfolio Project"
    )
