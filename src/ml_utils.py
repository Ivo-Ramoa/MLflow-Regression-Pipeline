import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def get_models():
    return {
        "Dummy Regressor": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
            max_depth=6,
            min_samples_leaf=3
        ),
        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2
        ),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="lbfgs",
            alpha=0.001,
            max_iter=2000,
            random_state=42
        )
    }


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


def get_model_params_for_logging(model):
    params = model.get_params()

    keys_of_interest = [
        "strategy",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "random_state",
        "hidden_layer_sizes",
        "activation",
        "solver",
        "alpha",
        "learning_rate_init",
        "max_iter"
    ]

    filtered = {}
    for key in keys_of_interest:
        if key in params and params[key] is not None:
            filtered[key] = params[key]

    return filtered


def build_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def extract_feature_importance(trained_pipeline):
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]

    feature_names_raw = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefficients = model.coef_
        if np.ndim(coefficients) > 1:
            coefficients = coefficients.ravel()
        importances = np.abs(coefficients)
    else:
        return pd.DataFrame(columns=["feature_raw", "feature", "importance"])

    cleaned_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names_raw
    ]

    importance_df = pd.DataFrame({
        "feature_raw": feature_names_raw,
        "feature": cleaned_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return importance_df