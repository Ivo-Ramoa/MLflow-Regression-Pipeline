import json

import pandas as pd
import streamlit as st

from src.config import BEST_MODEL_METADATA_PATH, CV_METRICS_PATH, MODEL_METRICS_PATH

st.set_page_config(
    page_title="TP1-D2 | iGov MLflow Dashboard",
    layout="wide"
)

st.title("TP1-D2 | Dashboard de Modelos Preditivos iGov")
st.markdown(
    """
    Esta aplicação organiza o trabalho em várias páginas e compara cinco abordagens:
    - Dummy Regressor (baseline)
    - Regressão Linear
    - Árvore de Decisão
    - Random Forest
    - Rede Neuronal (MLP)

    O objetivo é analisar o desempenho dos modelos, comparar métricas holdout,
    reforçar a análise com validação cruzada e disponibilizar um formulário para novas previsões.
    """
)

st.subheader("Fluxo do trabalho")
st.code(
    "1) python -m src.train\n"
    "2) python -m mlflow ui --backend-store-uri .\\mlruns --port 5001\n"
    "3) python -m streamlit run app.py",
    language="bash"
)

if MODEL_METRICS_PATH.exists():
    metrics_df = pd.read_csv(MODEL_METRICS_PATH).sort_values("RMSE")
    best_row = metrics_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Melhor modelo (holdout)", best_row["model"])
    col2.metric("RMSE", f"{best_row['RMSE']:.4f}")
    col3.metric("MAE", f"{best_row['MAE']:.4f}")
    col4.metric("Modelos comparados", len(metrics_df))

    st.subheader("Resumo holdout")
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning(
        "Ainda não existem métricas holdout geradas. "
        "Corre primeiro: python -m src.train"
    )

if CV_METRICS_PATH.exists():
    cv_df = pd.read_csv(CV_METRICS_PATH).sort_values("CV_RMSE_mean")
    best_cv_row = cv_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Melhor modelo (CV)", best_cv_row["model"])
    col2.metric("CV RMSE médio", f"{best_cv_row['CV_RMSE_mean']:.4f}")
    col3.metric("CV R² médio", f"{best_cv_row['CV_R2_mean']:.4f}")

    st.subheader("Resumo de validação cruzada")
    st.dataframe(cv_df, use_container_width=True)
else:
    st.info("As métricas de validação cruzada ainda não foram geradas.")

st.subheader("Leitura geral")
st.markdown(
    """
    - O **Dummy Regressor** funciona como baseline estatístico.
    - O **holdout test** mostra o desempenho numa divisão treino/teste fixa.
    - A **validação cruzada** reforça a robustez da comparação entre modelos.
    - O modelo com menor RMSE é tratado como melhor modelo global no contexto deste trabalho.
    """
)

if BEST_MODEL_METADATA_PATH.exists():
    st.subheader("Metadados do modelo final")
    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    st.json(metadata)