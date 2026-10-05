import json
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import KFold, cross_validate, train_test_split

from src.config import (
    BEST_MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
    CROSS_VALIDATION_FOLDS,
    CV_METRICS_PATH,
    DECISION_TREE_MODEL_PATH,
    EXPERIMENT_NAME,
    FEATURE_IMPORTANCE_PATH,
    MODEL_METRICS_PATH,
    MLRUNS_DIR,
    PREPARED_DATASET_PATH,
    RAW_DATASET_PATH,
    RANDOM_STATE,
    TEST_PREDICTIONS_PATH,
    TEST_SIZE,
    ensure_directories,
)
from src.data_utils import (
    get_feature_groups,
    load_dataset,
    prepare_dataset,
    split_features_target,
)
from src.ml_utils import (
    build_pipeline,
    evaluate_regression,
    extract_feature_importance,
    get_model_params_for_logging,
    get_models,
)


def main():
    ensure_directories()

    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {RAW_DATASET_PATH}\n"
            "Coloca o ficheiro Dataset-iGov.csv dentro de data/raw/."
        )

    raw_df = load_dataset(RAW_DATASET_PATH)
    prepared_df = prepare_dataset(raw_df)
    prepared_df.to_csv(PREPARED_DATASET_PATH, index=False)

    x, y = split_features_target(prepared_df)
    numeric_cols, categorical_cols = get_feature_groups(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics_rows = []
    cv_rows = []
    predictions_rows = []

    models = get_models()

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }

    cv = KFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for model_name, model in models.items():
        pipeline = build_pipeline(model, numeric_cols, categorical_cols)

        cv_pipeline = build_pipeline(
            get_models()[model_name],
            numeric_cols,
            categorical_cols
        )

        cv_results = cross_validate(
            cv_pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=None
        )

        cv_mae_values = -cv_results["test_MAE"]
        cv_rmse_values = -cv_results["test_RMSE"]
        cv_r2_values = cv_results["test_R2"]

        cv_summary = {
            "model": model_name,
            "CV_MAE_mean": float(cv_mae_values.mean()),
            "CV_MAE_std": float(cv_mae_values.std()),
            "CV_RMSE_mean": float(cv_rmse_values.mean()),
            "CV_RMSE_std": float(cv_rmse_values.std()),
            "CV_R2_mean": float(cv_r2_values.mean()),
            "CV_R2_std": float(cv_r2_values.std())
        }

        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)

            metrics = evaluate_regression(y_test, predictions)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_folds", CROSS_VALIDATION_FOLDS)

            model_specific_params = get_model_params_for_logging(model)
            if model_specific_params:
                mlflow.log_params(model_specific_params)

            mlflow.log_metrics(metrics)
            mlflow.log_metric("CV_MAE_mean", cv_summary["CV_MAE_mean"])
            mlflow.log_metric("CV_MAE_std", cv_summary["CV_MAE_std"])
            mlflow.log_metric("CV_RMSE_mean", cv_summary["CV_RMSE_mean"])
            mlflow.log_metric("CV_RMSE_std", cv_summary["CV_RMSE_std"])
            mlflow.log_metric("CV_R2_mean", cv_summary["CV_R2_mean"])
            mlflow.log_metric("CV_R2_std", cv_summary["CV_R2_std"])

            mlflow.sklearn.log_model(pipeline, name="model")

            metrics_rows.append({
                "run_id": run.info.run_id,
                "model": model_name,
                **metrics
            })

            cv_rows.append(cv_summary)

            for row_index, predicted_value in zip(x_test.index, predictions):
                actual_value = y_test.loc[row_index]
                residual = actual_value - predicted_value

                predictions_rows.append({
                    "row_id": int(row_index),
                    "model": model_name,
                    "actual": float(actual_value),
                    "predicted": float(predicted_value),
                    "residual": float(residual),
                    "abs_error": float(abs(residual))
                })

    metrics_df = pd.DataFrame(metrics_rows).sort_values("RMSE", ascending=True)
    cv_metrics_df = pd.DataFrame(cv_rows).sort_values("CV_RMSE_mean", ascending=True)
    predictions_df = pd.DataFrame(predictions_rows)

    metrics_df.to_csv(MODEL_METRICS_PATH, index=False)
    cv_metrics_df.to_csv(CV_METRICS_PATH, index=False)
    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    final_model = build_pipeline(
        get_models()[best_model_name],
        numeric_cols,
        categorical_cols
    )
    final_model.fit(x, y)

    joblib.dump(final_model, BEST_MODEL_PATH)

    importance_model_name = "Random Forest"
    importance_pipeline = build_pipeline(
        get_models()[importance_model_name],
        numeric_cols,
        categorical_cols
    )
    importance_pipeline.fit(x, y)

    feature_importance_df = extract_feature_importance(importance_pipeline)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    decision_tree_name = "Decision Tree"
    decision_tree_pipeline = build_pipeline(
        get_models()[decision_tree_name],
        numeric_cols,
        categorical_cols
    )
    decision_tree_pipeline.fit(x, y)
    joblib.dump(decision_tree_pipeline, DECISION_TREE_MODEL_PATH)

    metadata = {
        "best_model": best_model_name,
        "feature_importance_model": importance_model_name,
        "decision_tree_model": decision_tree_name,
        "baseline_model": "Dummy Regressor",
        "cv_folds": CROSS_VALIDATION_FOLDS,
        "trained_at": datetime.now().isoformat(),
        "rows_used": int(len(prepared_df)),
        "columns_used": int(len(x.columns))
    }

    with open(BEST_MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Treino concluído com sucesso.")
    print(f"Dataset preparado: {PREPARED_DATASET_PATH}")
    print(f"Métricas holdout dos modelos: {MODEL_METRICS_PATH}")
    print(f"Métricas de validação cruzada: {CV_METRICS_PATH}")
    print(f"Predições de teste: {TEST_PREDICTIONS_PATH}")
    print(f"Importância das variáveis: {FEATURE_IMPORTANCE_PATH}")
    print(f"Melhor modelo guardado em: {BEST_MODEL_PATH}")
    print(f"Modelo da Árvore de Decisão guardado em: {DECISION_TREE_MODEL_PATH}")
    print(f"Melhor modelo identificado: {best_model_name}")
    print(f"Modelo usado para importância das variáveis: {importance_model_name}")


if __name__ == "__main__":
    main()