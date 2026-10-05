import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PREPARED_DATASET_PATH

st.set_page_config(page_title="Análise Exploratória", layout="wide")
st.title("Análise Exploratória")

if not PREPARED_DATASET_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(PREPARED_DATASET_PATH, parse_dates=["data_registo"])


df = load_data()

st.subheader("Distribuição do indicador_kpi")
fig_kpi_hist = px.histogram(
    df,
    x="indicador_kpi",
    nbins=20,
    title="Distribuição do indicador_kpi"
)
st.plotly_chart(fig_kpi_hist, use_container_width=True)

st.subheader("Distribuição do tempo de resposta")
fig_tempo_hist = px.histogram(
    df,
    x="tempo_resposta",
    nbins=20,
    title="Distribuição do tempo de resposta"
)
st.plotly_chart(fig_tempo_hist, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_box_service = px.box(
        df,
        x="tipo_servico",
        y="indicador_kpi",
        title="KPI por tipo de serviço"
    )
    st.plotly_chart(fig_box_service, use_container_width=True)

with col2:
    fig_box_channel = px.box(
        df,
        x="canal_utilizado",
        y="indicador_kpi",
        title="KPI por canal utilizado"
    )
    st.plotly_chart(fig_box_channel, use_container_width=True)

st.subheader("Relação entre volume de interações e KPI")
fig_scatter = px.scatter(
    df,
    x="volume_interacoes",
    y="indicador_kpi",
    color="transparencia",
    title="Volume de interações vs KPI"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Resumo estatístico")
numeric_columns = [
    "indicador_si",
    "taxa_resolucao",
    "tempo_resposta",
    "satisfacao_cidadao",
    "volume_interacoes",
    "taxa_abandono",
    "erros_tecnicos",
    "indicador_kpi"
]
st.dataframe(df[numeric_columns].describe().T, use_container_width=True)