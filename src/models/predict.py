"""Script de predição para novos arquivos CSV com o mesmo schema do treino.

Uso:
    python -m src.models.predict --input caminho/para/arquivo.csv --output saida.csv

O script garante que todas as colunas esperadas estão presentes, aplica o modelo
CatBoost salvo e gera duas colunas extras: fraud_proba e fraud_flag.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier


LOGGER = logging.getLogger("predict")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_artifacts(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            "Artefatos não encontrados. Treine o modelo para gerar models/artifacts.json."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = set(expected_columns) - set(df.columns)
    extras = set(df.columns) - set(expected_columns)
    if missing:
        raise ValueError(
            "O arquivo de entrada está faltando colunas obrigatórias: " + ", ".join(sorted(missing))
        )
    if extras:
        LOGGER.warning(
            "Arquivo possui colunas extras que serão ignoradas: %s", ", ".join(sorted(extras))
        )


def predict(
    config_path: str,
    input_path: str,
    output_path: str | None = None,
) -> Path:
    config = load_config(config_path)
    artifacts = load_artifacts(Path(config["paths"]["artifacts_file"]))

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada {input_path} não encontrado")

    LOGGER.info("Carregando dados de entrada: %s", input_path)
    df = pd.read_csv(input_path)

    validate_columns(df, artifacts["features"])
    X = df[artifacts["features"]]

    model = CatBoostClassifier()
    model.load_model(Path(config["paths"]["model_file"]))

    probs = model.predict_proba(X)[:, 1]
    threshold = artifacts["threshold"]["value"]
    flags = (probs >= threshold).astype(int)

    df_output = df.copy()
    df_output["fraud_proba"] = probs
    df_output["fraud_flag"] = flags

    output_path = Path(output_path) if output_path else input_path.with_name(
        input_path.stem + "_predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    LOGGER.info("Predições salvas em %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predição com modelo CatBoost")
    parser.add_argument("--config", type=str, default="src/config/config.yaml")
    parser.add_argument("--input", type=str, required=True, help="CSV com dados a predizer")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho opcional para salvar a saída",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.config, args.input, args.output)
