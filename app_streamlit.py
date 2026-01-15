"""Dashboard Streamlit para detecção de fraudes.

Fluxo básico:
1. Carrega o modelo e os artefatos salvos durante o treino.
2. Permite que o usuário faça upload de um CSV preparado com as mesmas colunas do treino.
3. Possibilita ajustar o limiar de fraude em tempo real via slider.
4. Exibe resultados resumidos e permite download.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from catboost import CatBoostClassifier

from src.models.predict import validate_columns


@st.cache_resource(show_spinner=True)
def load_config_artifacts():
    with open("src/config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(config["paths"]["artifacts_file"], "r", encoding="utf-8") as f:
        artifacts = json.load(f)
    model = CatBoostClassifier()
    model.load_model(config["paths"]["model_file"])
    return config, artifacts, model


def main() -> None:
    st.set_page_config(page_title="Detecção de Fraudes IEEE-CIS", layout="wide")
    st.title("🕵️ Detecção de Fraudes - Dashboard Interativo")
    st.write(
        "Bem-vindo! Aqui você pode testar o modelo de detecção de fraudes com seus próprios dados"
        " (mesmas colunas do treino). Ajuste o limiar conforme o apetite de risco."
    )

    config, artifacts, model = load_config_artifacts()

    st.sidebar.header("Configurações")
    default_threshold = float(artifacts["threshold"]["value"])
    threshold = st.sidebar.slider(
        "Threshold de fraude",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.01,
        help="Valores menores aumentam recall (mais alertas), valores maiores priorizam precisão.",
    )

    uploaded = st.file_uploader(
        "Envie um CSV com as colunas do pred_template.csv",
        type="csv",
        help="Use o arquivo pred_template.csv como referência de cabeçalhos.",
    )

    if uploaded is None:
        st.info("🚀 Faça upload de um CSV para começar. Nenhum dado é enviado para servidores externos.")
        return

    df = pd.read_csv(uploaded)
    try:
        validate_columns(df, artifacts["features"])
    except ValueError as err:
        st.error(f"Colunas inválidas: {err}")
        st.stop()

    X = df[artifacts["features"]]
    probs = model.predict_proba(X)[:, 1]
    flags = (probs >= threshold).astype(int)

    df_result = df.copy()
    df_result["fraud_proba"] = probs
    df_result["fraud_flag"] = flags

    st.subheader("📊 Resumo das Previsões")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de transações", len(df_result))
    with col2:
        st.metric("Alertas de fraude", int(flags.sum()))
    with col3:
        st.metric("Threshold atual", f"{threshold:.2f}")

    st.write("Visualização das primeiras linhas:")
    st.dataframe(df_result.head(20))

    st.download_button(
        label="💾 Baixar CSV com fraudes",
        data=df_result.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Métricas do treino:")
    for key, value in artifacts["metrics"].items():
        st.sidebar.write(f"**{key}**: {value:.4f}")

    shap_path = Path(config["explainability"].get("output_summary_png", ""))
    if shap_path.exists():
        st.subheader("🧠 Interpretação (SHAP)")
        st.image(str(shap_path), caption="Principais variáveis que empurram a probabilidade de fraude")
    else:
        st.info("Execute o treino novamente para gerar gráfico SHAP e visualizá-lo aqui.")


if __name__ == "__main__":
    main()
