import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TEST_PREDICTIONS_PATH

st.set_page_config(page_title="Análise de Erros", layout="wide")
st.title("Análise de Erros")

if not TEST_PREDICTIONS_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()


@st.cache_data
def load_predictions():
    return pd.read_csv(TEST_PREDICTIONS_PATH)


pred_df = load_predictions()

model_options = pred_df["model"].unique().tolist()
selected_model = st.selectbox("Seleciona o modelo", model_options)

filtered_df = pred_df[pred_df["model"] == selected_model].copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Erro absoluto médio", f"{filtered_df['abs_error'].mean():.4f}")
col2.metric("Erro máximo", f"{filtered_df['abs_error'].max():.4f}")
col3.metric("Média dos resíduos", f"{filtered_df['residual'].mean():.4f}")
col4.metric("Desvio padrão dos resíduos", f"{filtered_df['residual'].std():.4f}")

st.subheader("Valores reais vs previstos")
fig_scatter = px.scatter(
    filtered_df,
    x="actual",
    y="predicted",
    title=f"Actual vs Predicted | {selected_model}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Resíduos vs valores previstos")
fig_residual_scatter = px.scatter(
    filtered_df,
    x="predicted",
    y="residual",
    title=f"Resíduos vs valores previstos | {selected_model}"
)
st.plotly_chart(fig_residual_scatter, use_container_width=True)

st.subheader("Distribuição dos resíduos")
fig_residual = px.histogram(
    filtered_df,
    x="residual",
    nbins=20,
    title=f"Distribuição dos resíduos | {selected_model}"
)
st.plotly_chart(fig_residual, use_container_width=True)

st.subheader("Casos com maior erro")
top_errors = filtered_df.sort_values("abs_error", ascending=False).head(15)
st.dataframe(top_errors, use_container_width=True)

st.subheader("Leitura dos resíduos")
st.markdown(
    """
    - Resíduos próximos de zero indicam previsões mais ajustadas.
    - Uma média dos resíduos próxima de zero sugere menor enviesamento médio.
    - A dispersão dos resíduos ajuda a perceber se o erro é estável ou irregular.
    """
)