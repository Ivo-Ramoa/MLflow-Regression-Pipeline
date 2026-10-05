from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MLRUNS_DIR = BASE_DIR / "mlruns"

RAW_DATASET_PATH = DATA_RAW_DIR / "Dataset-iGov.csv"
PREPARED_DATASET_PATH = DATA_PROCESSED_DIR / "dataset_prepared.csv"

MODEL_METRICS_PATH = OUTPUTS_DIR / "model_metrics.csv"
CV_METRICS_PATH = OUTPUTS_DIR / "cv_metrics.csv"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "predictions_test.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
DECISION_TREE_MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"

TARGET_COLUMN = "indicador_kpi"
DATE_COLUMN = "data_registo"
DROP_COLUMNS = ["id_registo"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5
EXPERIMENT_NAME = "TP1_D2_iGov"


def ensure_directories() -> None:
    for folder in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        MLRUNS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)