import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CV_METRICS_PATH, MODEL_METRICS_PATH

st.set_page_config(page_title="Comparação de Modelos", layout="wide")
st.title("Comparação de Modelos")

if not MODEL_METRICS_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()

if not CV_METRICS_PATH.exists():
    st.warning("Falta o ficheiro de validação cruzada. Corre novamente: python -m src.train")
    st.stop()


@st.cache_data
def load_holdout_metrics():
    return pd.read_csv(MODEL_METRICS_PATH).sort_values("RMSE", ascending=True)


@st.cache_data
def load_cv_metrics():
    return pd.read_csv(CV_METRICS_PATH).sort_values("CV_RMSE_mean", ascending=True)


metrics_df = load_holdout_metrics()
cv_df = load_cv_metrics()

best_model_holdout = metrics_df.iloc[0]["model"]
best_model_cv = cv_df.iloc[0]["model"]

col1, col2 = st.columns(2)
col1.success(f"Melhor modelo segundo holdout RMSE: {best_model_holdout}")
col2.info(f"Melhor modelo segundo CV RMSE médio: {best_model_cv}")

st.subheader("Métricas holdout")
st.dataframe(metrics_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_rmse = px.bar(
        metrics_df,
        x="model",
        y="RMSE",
        title="RMSE por modelo (holdout)"
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col2:
    fig_mae = px.bar(
        metrics_df,
        x="model",
        y="MAE",
        title="MAE por modelo (holdout)"
    )
    st.plotly_chart(fig_mae, use_container_width=True)

st.subheader("R² por modelo (holdout)")
fig_r2 = px.bar(
    metrics_df,
    x="model",
    y="R2",
    title="R² por modelo (holdout)"
)
st.plotly_chart(fig_r2, use_container_width=True)

st.subheader("Métricas de validação cruzada (5 folds)")
st.dataframe(cv_df, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_cv_rmse = px.bar(
        cv_df,
        x="model",
        y="CV_RMSE_mean",
        error_y="CV_RMSE_std",
        title="CV RMSE médio por modelo"
    )
    st.plotly_chart(fig_cv_rmse, use_container_width=True)

with col2:
    fig_cv_mae = px.bar(
        cv_df,
        x="model",
        y="CV_MAE_mean",
        error_y="CV_MAE_std",
        title="CV MAE médio por modelo"
    )
    st.plotly_chart(fig_cv_mae, use_container_width=True)

st.subheader("CV R² médio por modelo")
fig_cv_r2 = px.bar(
    cv_df,
    x="model",
    y="CV_R2_mean",
    error_y="CV_R2_std",
    title="CV R² médio por modelo"
)
st.plotly_chart(fig_cv_r2, use_container_width=True)

st.subheader("Leitura correta dos resultados")
st.markdown(
    """
    - O **Dummy Regressor** é a baseline estatística. Serve para verificar se os outros modelos
      acrescentam valor face a uma previsão trivial pela média.
    - O **holdout test** mostra o desempenho numa divisão treino/teste fixa.
    - A **validação cruzada** mostra o comportamento médio dos modelos ao longo de várias divisões.
    - O **Random Forest** continua a ser o modelo mais forte neste trabalho.
    - A **Árvore de Decisão** é útil pela interpretabilidade, mas tem desempenho inferior ao Random Forest.
    - A **Rede Neuronal (MLP)** foi incluída para alargar a comparação experimental, mas neste dataset não superou os melhores modelos baseados em árvores.
    - Valores negativos de **R²** indicam capacidade explicativa limitada neste problema.
    """
)

st.subheader("Interpretação metodológica")
st.markdown(
    """
    Neste projeto são comparadas cinco abordagens diferentes:

    - **Dummy Regressor**: baseline estatística simples, baseada na média.
    - **Regressão Linear**: modelo linear, útil como referência base.
    - **Árvore de Decisão**: modelo baseado em regras, fácil de interpretar.
    - **Random Forest**: conjunto de árvores, normalmente mais robusto.
    - **Rede Neuronal (MLP)**: abordagem baseada em redes neuronais, incluída para comparação adicional.

    A combinação entre métricas holdout e validação cruzada permite uma avaliação mais sólida
    e mais defensável do que uma comparação baseada apenas num único split treino/teste.
    """
)