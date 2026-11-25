# Housing Price Prediction & Manual Synthetic Data Generation

## Project Overview

This project focuses on predicting house prices using Machine Learning and analyzing the impact of manually generated synthetic data on model performance.

Dataset: Kaggle House Prices - Advanced Regression Techniques.

## Project Objectives

- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess housing data.
- Generate modified bootstrap-based synthetic records.
- Compare original and synthetic data distributions.
- Train and evaluate Random Forest Regression models.
- Build an interactive Streamlit dashboard.

## Dataset

- Original records: 1,460
- Original columns: 81
- Model features: 79 after excluding Id and SalePrice
- Target variable: SalePrice

Source: [Kaggle House Prices Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

## Data Cleaning

The original dataset contained 7,829 missing values.

Cleaning included:
- Handling missing numerical and categorical values.
- Representing absent house features appropriately.
- Checking constant columns and outliers.

After cleaning, missing values were reduced to zero.

## Manual Synthetic Data Generation

A bootstrap-based approach using NumPy and Pandas was used.

- Generated 500 modified synthetic records.
- Applied controlled modifications to selected numerical features.
- Preserved categorical values and SalePrice.
- Validated missing values, feature distributions, and logical relationships.

Note: These are modified bootstrap-based experimental records, not independent real-world houses.

## Model Development

Four Random Forest Regression experiments:

1. Original Model
2. Augmented Model
3. Log Transformation Model
4. Tuned Random Forest Model

A preprocessing pipeline handled numerical and categorical features. OneHotEncoder was used for categorical features.

GridSearchCV with 5-fold cross-validation was used for hyperparameter tuning.

## Model Evaluation

Evaluation metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Final Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Original Model | 17,505.21 | 29,089.23 | 0.88968 |
| Augmented Model | 17,844.76 | 29,481.31 | 0.88669 |
| Log Transformation Model | 17,382.09 | 29,830.57 | 0.88399 |
| Tuned Random Forest | 17,447.31 | 28,909.14 | 0.89104 |

### Key Findings

- Tuned Random Forest achieved the lowest RMSE and highest R² Score.
- Log Transformation Model achieved the lowest MAE.
- Synthetic data augmentation did not improve overall test performance.
- The augmented model performed slightly better on the expensive-house subset, which contained only 7 records.
- OverallQual, GrLivArea, and TotalBsmtSF were among the important predictive features.

## Interactive Dashboard

An interactive dashboard was developed using Streamlit and Plotly.

Dashboard sections:
1. Home
2. Model Comparison
3. Feature Importance
4. House Price Prediction
5. Synthetic Data Analysis
6. Project Findings

The prediction section allows users to enter house characteristics and obtain an estimated price using the saved tuned model.

## Project Workflow

1. Data exploration and cleaning.
2. Manual synthetic data generation.
3. Feature engineering and model training.
4. Model evaluation and hyperparameter tuning.
5. Saving the final model.
6. Building the interactive Streamlit dashboard.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## How to Run the Dashboard

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard from the project root directory:

```bash
streamlit run dashboard/app.py
```

## Project Limitations

- The dataset is relatively small.
- Synthetic records are based on bootstrap sampling.
- Retaining SalePrice while modifying selected features may affect feature-target relationships.
- Expensive-house evaluation is based on only 7 test records.
- The project focuses on Random Forest Regression.

## Final Conclusion

The Tuned Random Forest Model was selected as the final model because it achieved the lowest RMSE and highest R² Score among the evaluated experiments.

This project demonstrates an end-to-end Machine Learning workflow, from data cleaning and synthetic data experimentation to model evaluation and interactive dashboard development.
