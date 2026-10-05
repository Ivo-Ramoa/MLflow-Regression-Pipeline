import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PREPARED_DATASET_PATH

st.set_page_config(page_title="Visão Geral", layout="wide")
st.title("Visão Geral do Dataset")

if not PREPARED_DATASET_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(PREPARED_DATASET_PATH, parse_dates=["data_registo"])


df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de registos", len(df))
col2.metric("KPI médio", f"{df['indicador_kpi'].mean():.2f}")
col3.metric("Taxa de resolução média", f"{df['taxa_resolucao'].mean():.2f}")
col4.metric("Tempo de resposta médio", f"{df['tempo_resposta'].mean():.2f}")

st.subheader("Distribuição por área temática")
area_df = (
    df.groupby("area_tematica", as_index=False)["indicador_kpi"]
    .mean()
    .sort_values("indicador_kpi", ascending=False)
)

fig_area = px.bar(
    area_df,
    x="area_tematica",
    y="indicador_kpi",
    title="KPI médio por área temática"
)
st.plotly_chart(fig_area, use_container_width=True)

st.subheader("Distribuição por tipo de serviço")
service_df = (
    df.groupby("tipo_servico", as_index=False)["indicador_kpi"]
    .mean()
    .sort_values("indicador_kpi", ascending=False)
)

fig_service = px.bar(
    service_df,
    x="tipo_servico",
    y="indicador_kpi",
    title="KPI médio por tipo de serviço"
)
st.plotly_chart(fig_service, use_container_width=True)

st.subheader("Comparação por transparência")
transparency_df = (
    df.groupby("transparencia", as_index=False)["indicador_kpi"]
    .mean()
    .sort_values("indicador_kpi", ascending=False)
)

fig_transparency = px.bar(
    transparency_df,
    x="transparencia",
    y="indicador_kpi",
    title="KPI médio por transparência"
)
st.plotly_chart(fig_transparency, use_container_width=True)

st.subheader("Evolução temporal mensal do KPI")
monthly_df = (
    df.groupby("mes", as_index=False)["indicador_kpi"]
    .mean()
    .sort_values("mes")
)

fig_monthly = px.line(
    monthly_df,
    x="mes",
    y="indicador_kpi",
    markers=True,
    title="KPI médio por mês"
)
st.plotly_chart(fig_monthly, use_container_width=True)