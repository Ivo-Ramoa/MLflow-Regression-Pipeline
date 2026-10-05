# MLflow Regression Pipeline

End-to-end machine learning regression project combining data preprocessing, model comparison, MLflow experiment tracking, cross-validation and an interactive Streamlit dashboard.

The project predicts the `indicador_kpi` target variable and compares multiple regression approaches using consistent preprocessing pipelines and evaluation metrics.

## Project Files

- [View Project Presentation](presentation.pdf)
- [View Streamlit Application](app.py)
- [View Training Pipeline](src/train.py)

## Project Overview

The project implements a complete machine learning workflow, including:

- Data loading and preprocessing
- Feature engineering from date information
- Numerical and categorical preprocessing
- Comparison of multiple regression models
- Holdout evaluation
- 5-fold cross-validation
- MLflow experiment tracking
- Model persistence
- Feature importance analysis
- Error analysis
- Interactive predictions
- Multi-page Streamlit dashboard

## Models

Five regression approaches are compared:

- Dummy Regressor
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Neural Network (MLP)

The Dummy Regressor is used as a baseline for comparison.

## Data Preprocessing

The preprocessing pipeline handles both numerical and categorical features.

Numerical features use:

- Median imputation
- Standard scaling

Categorical features use:

- Most-frequent-value imputation
- One-hot encoding

Additional features are extracted from the registration date:

- Year
- Month
- Day
- Quarter
- Day of the week

## Model Evaluation

The models are evaluated using:

- MAE
- RMSE
- R²

The project uses an 80/20 train-test split and 5-fold cross-validation to provide a more robust comparison between models.

The model with the lowest RMSE on the holdout set is selected as the final model.

## MLflow

MLflow is used to track the machine learning experiments.

For each model, the project records:

- Model parameters
- Holdout metrics
- Cross-validation metrics
- Trained model artifacts

## Streamlit Dashboard

The project includes a multi-page Streamlit application for exploring the machine learning workflow and results.

The dashboard includes:

- General project overview
- Exploratory data analysis
- Model comparison
- Error analysis
- Feature importance
- New predictions
- Decision tree visualization

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Streamlit
- Plotly
- Joblib

## Repository Structure

    MLflow-Regression-Pipeline/
    ├── data/
    │   └── raw/
    │       └── Dataset-iGov.csv
    ├── pages/
    │   ├── 1_Visao_Geral.py
    │   ├── 2_Analise_Exploratoria.py
    │   ├── 3_Comparacao_Modelos.py
    │   ├── 4_Analise_de_Erros.py
    │   ├── 5_Importancia_das_Variaveis.py
    │   ├── 6_Nova_Previsao.py
    │   └── 7_Arvore_de_Decisao.py
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data_utils.py
    │   ├── ml_utils.py
    │   └── train.py
    ├── app.py
    ├── presentation.pdf
    ├── README.md
    ├── requirements.txt
    └── .gitignore

## Running the Project

Install the dependencies:

    pip install -r requirements.txt

Train and evaluate the models:

    python -m src.train

Start the MLflow interface:

    mlflow ui --backend-store-uri ./mlruns --port 5001

Start the Streamlit dashboard:

    streamlit run app.py

## Authors

Developed as part of an Artificial Intelligence / Machine Learning academic project at **IPVC - Escola Superior de Tecnologia e Gestão**.

- Ivo Ramoa
