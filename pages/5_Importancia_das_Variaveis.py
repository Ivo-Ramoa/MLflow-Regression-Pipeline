import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import FEATURE_IMPORTANCE_PATH

st.set_page_config(page_title="Importância das Variáveis", layout="wide")
st.title("Importância das Variáveis")

st.info(
    "As importâncias apresentadas nesta página são calculadas com base no modelo "
    "Random Forest, para manter a interpretabilidade, mesmo quando o melhor modelo "
    "global é outro."
)

if not FEATURE_IMPORTANCE_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()


@st.cache_data
def load_importance():
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


importance_df = load_importance()

if importance_df.empty:
    st.warning("O modelo selecionado não devolveu importância de variáveis.")
    st.stop()

top_n = st.slider("Número de variáveis a mostrar", 5, 30, 15)
top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

fig_importance = px.bar(
    top_df,
    x="importance",
    y="feature",
    orientation="h",
    title=f"Top {top_n} variáveis mais importantes"
)
st.plotly_chart(fig_importance, use_container_width=True)

st.subheader("Tabela completa")
st.dataframe(importance_df, use_container_width=True)

st.subheader("Interpretação")
st.markdown(
    """
    Esta página mostra quais as variáveis que mais influenciaram o modelo usado
    para calcular importância das variáveis.

    Atenção:
    - importância não significa causalidade;
    - significa apenas que o modelo utilizou mais essas variáveis para construir a previsão;
    - neste projeto, esta análise é feita com base no Random Forest.
    """
)