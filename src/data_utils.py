import pandas as pd

from src.config import DATE_COLUMN, DROP_COLUMNS, TARGET_COLUMN


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().any():
        raise ValueError("Existem datas inválidas em 'data_registo'.")

    df["ano"] = df[DATE_COLUMN].dt.year
    df["mes"] = df[DATE_COLUMN].dt.month
    df["dia"] = df[DATE_COLUMN].dt.day
    df["trimestre"] = df[DATE_COLUMN].dt.quarter
    df["dia_semana"] = df[DATE_COLUMN].dt.dayofweek

    return df


def split_features_target(prepared_df: pd.DataFrame):
    x = prepared_df.drop(
        columns=[TARGET_COLUMN, DATE_COLUMN, *DROP_COLUMNS],
        errors="ignore"
    )
    y = prepared_df[TARGET_COLUMN].copy()
    return x, y


def get_feature_groups(x: pd.DataFrame):
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = x.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_scoring_dataframe(form_values: dict) -> pd.DataFrame:
    row = pd.DataFrame([form_values])

    row[DATE_COLUMN] = pd.to_datetime(row[DATE_COLUMN], errors="coerce")
    if row[DATE_COLUMN].isna().any():
        raise ValueError("A data introduzida é inválida.")

    row["ano"] = row[DATE_COLUMN].dt.year
    row["mes"] = row[DATE_COLUMN].dt.month
    row["dia"] = row[DATE_COLUMN].dt.day
    row["trimestre"] = row[DATE_COLUMN].dt.quarter
    row["dia_semana"] = row[DATE_COLUMN].dt.dayofweek

    row = row.drop(columns=[DATE_COLUMN, *DROP_COLUMNS], errors="ignore")
    return row  