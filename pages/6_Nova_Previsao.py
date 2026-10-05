import json

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    BEST_MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
    PREPARED_DATASET_PATH,
    TARGET_COLUMN,
)
from src.data_utils import build_scoring_dataframe

st.set_page_config(page_title="Nova Previsão", layout="wide")
st.title("Formulário para Nova Previsão")

if not PREPARED_DATASET_PATH.exists() or not BEST_MODEL_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()


@st.cache_data
def load_reference_data():
    return pd.read_csv(PREPARED_DATASET_PATH, parse_dates=["data_registo"])


@st.cache_resource
def load_model():
    return joblib.load(BEST_MODEL_PATH)


df = load_reference_data()
model = load_model()

metadata = {}
if BEST_MODEL_METADATA_PATH.exists():
    with open(BEST_MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

if metadata:
    st.info(f"Modelo final usado no formulário: {metadata['best_model']}")

kpi_q1 = df[TARGET_COLUMN].quantile(0.25)
kpi_q3 = df[TARGET_COLUMN].quantile(0.75)

with st.form("prediction_form"):
    st.subheader("Dados de entrada")

    col1, col2 = st.columns(2)

    with col1:
        data_registo = st.date_input(
            "Data de registo",
            value=df["data_registo"].min().date()
        )
        unidade_organizacional = st.selectbox(
            "Unidade organizacional",
            sorted(df["unidade_organizacional"].unique().tolist())
        )
        tipo_servico = st.selectbox(
            "Tipo de serviço",
            sorted(df["tipo_servico"].unique().tolist())
        )
        indicador_si = st.number_input(
            "Indicador SI",
            min_value=float(df["indicador_si"].min()),
            max_value=float(df["indicador_si"].max()),
            value=float(df["indicador_si"].mean()),
            step=0.01
        )
        taxa_resolucao = st.number_input(
            "Taxa de resolução",
            min_value=float(df["taxa_resolucao"].min()),
            max_value=float(df["taxa_resolucao"].max()),
            value=float(df["taxa_resolucao"].mean()),
            step=0.01
        )
        tempo_resposta = st.number_input(
            "Tempo de resposta",
            min_value=int(df["tempo_resposta"].min()),
            max_value=int(df["tempo_resposta"].max()),
            value=int(df["tempo_resposta"].median()),
            step=1
        )
        satisfacao_cidadao = st.number_input(
            "Satisfação do cidadão",
            min_value=float(df["satisfacao_cidadao"].min()),
            max_value=float(df["satisfacao_cidadao"].max()),
            value=float(df["satisfacao_cidadao"].mean()),
            step=0.1
        )

    with col2:
        volume_interacoes = st.number_input(
            "Volume de interações",
            min_value=int(df["volume_interacoes"].min()),
            max_value=int(df["volume_interacoes"].max()),
            value=int(df["volume_interacoes"].median()),
            step=1
        )
        canal_utilizado = st.selectbox(
            "Canal utilizado",
            sorted(df["canal_utilizado"].unique().tolist())
        )
        taxa_abandono = st.number_input(
            "Taxa de abandono",
            min_value=float(df["taxa_abandono"].min()),
            max_value=float(df["taxa_abandono"].max()),
            value=float(df["taxa_abandono"].mean()),
            step=0.01
        )
        erros_tecnicos = st.number_input(
            "Erros técnicos",
            min_value=int(df["erros_tecnicos"].min()),
            max_value=int(df["erros_tecnicos"].max()),
            value=int(df["erros_tecnicos"].median()),
            step=1
        )
        transparencia = st.selectbox(
            "Transparência",
            sorted(df["transparencia"].unique().tolist())
        )
        feedback_cidadao = st.selectbox(
            "Feedback do cidadão",
            sorted(df["feedback_cidadao"].unique().tolist())
        )
        segmentacao_utilizador = st.selectbox(
            "Segmentação do utilizador",
            sorted(df["segmentacao_utilizador"].unique().tolist())
        )
        area_tematica = st.selectbox(
            "Área temática",
            sorted(df["area_tematica"].unique().tolist())
        )

    submitted = st.form_submit_button("Gerar previsão")

if submitted:
    form_data = {
        "data_registo": data_registo,
        "unidade_organizacional": unidade_organizacional,
        "tipo_servico": tipo_servico,
        "indicador_si": indicador_si,
        "taxa_resolucao": taxa_resolucao,
        "tempo_resposta": tempo_resposta,
        "satisfacao_cidadao": satisfacao_cidadao,
        "volume_interacoes": volume_interacoes,
        "canal_utilizado": canal_utilizado,
        "taxa_abandono": taxa_abandono,
        "erros_tecnicos": erros_tecnicos,
        "transparencia": transparencia,
        "feedback_cidadao": feedback_cidadao,
        "segmentacao_utilizador": segmentacao_utilizador,
        "area_tematica": area_tematica
    }

    scoring_df = build_scoring_dataframe(form_data)
    prediction = float(model.predict(scoring_df)[0])

    if prediction < kpi_q1:
        interpretation = "Previsão baixa face ao histórico."
    elif prediction > kpi_q3:
        interpretation = "Previsão alta face ao histórico."
    else:
        interpretation = "Previsão intermédia face ao histórico."

    st.subheader("Resultado da previsão")
    st.metric("Indicador KPI previsto", f"{prediction:.2f}")
    st.write(interpretation)

    st.subheader("Dados usados na previsão")
    st.dataframe(pd.DataFrame([form_data]), use_container_width=True)