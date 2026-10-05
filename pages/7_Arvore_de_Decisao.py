import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import export_text, plot_tree

from src.config import PREPARED_DATASET_PATH

DECISION_TREE_MODEL_PATH = "models/decision_tree_model.joblib"

st.set_page_config(page_title="Árvore de Decisão", layout="wide")
st.title("Árvore de Decisão")

st.markdown(
    """
    Esta página mostra explicitamente o modelo de Árvore de Decisão usado no trabalho.
    Aqui podes observar:
    - a estrutura visual da árvore;
    - a profundidade e número de folhas;
    - a importância das variáveis na árvore;
    - as regras geradas pelo modelo.
    """
)

if not PREPARED_DATASET_PATH.exists():
    st.warning("Corre primeiro: python -m src.train")
    st.stop()

try:
    decision_tree_pipeline = joblib.load(DECISION_TREE_MODEL_PATH)
except FileNotFoundError:
    st.warning(
        "Não foi encontrado o modelo da Árvore de Decisão. "
        "Corre primeiro: python -m src.train"
    )
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv(PREPARED_DATASET_PATH)


df = load_data()

preprocessor = decision_tree_pipeline.named_steps["preprocessor"]
tree_model = decision_tree_pipeline.named_steps["model"]

feature_names_raw = preprocessor.get_feature_names_out()
feature_names = [
    name.replace("num__", "").replace("cat__", "")
    for name in feature_names_raw
]

st.subheader("Resumo estrutural")
col1, col2, col3 = st.columns(3)
col1.metric("Profundidade da árvore", int(tree_model.get_depth()))
col2.metric("Número de folhas", int(tree_model.get_n_leaves()))
col3.metric("Número de features usadas", int(len(feature_names)))

st.subheader("Importância das variáveis na Árvore de Decisão")
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": tree_model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df_nonzero = importance_df[importance_df["importance"] > 0].copy()

if importance_df_nonzero.empty:
    st.info("Esta árvore não devolveu importâncias positivas para as variáveis.")
else:
    top_n = st.slider("Número de variáveis a mostrar na árvore", 5, 30, 15)
    top_df = importance_df_nonzero.head(top_n)

    st.dataframe(top_df, use_container_width=True)

st.subheader("Visualização da Árvore de Decisão")
max_depth_to_plot = st.slider(
    "Profundidade máxima a desenhar",
    min_value=1,
    max_value=max(1, int(tree_model.get_depth())),
    value=min(3, max(1, int(tree_model.get_depth())))
)

fig, ax = plt.subplots(figsize=(22, 12))
plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=max_depth_to_plot,
    ax=ax
)
st.pyplot(fig)

st.subheader("Regras da Árvore")
max_depth_rules = st.slider(
    "Profundidade máxima das regras em texto",
    min_value=1,
    max_value=max(1, int(tree_model.get_depth())),
    value=min(4, max(1, int(tree_model.get_depth()))),
    key="rules_depth"
)

rules_text = export_text(
    tree_model,
    feature_names=feature_names,
    max_depth=max_depth_rules
)

st.code(rules_text, language="text")

st.subheader("Interpretação")
st.markdown(
    """
    - A Árvore de Decisão divide os dados em regras sequenciais.
    - Cada nó representa uma condição sobre uma variável.
    - Cada folha representa um valor previsto para o indicador KPI.
    - É um modelo mais interpretável do que Random Forest ou Rede Neuronal.
    - No entanto, no teu trabalho, o Random Forest teve melhor desempenho global.
    """
)