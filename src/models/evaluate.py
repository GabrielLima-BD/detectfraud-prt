"""Avaliação de modelo CatBoost usando artefatos salvos.

Este módulo permite recalcular métricas em um conjunto de validação ou teste
qualificado. Ele espera que o modelo (.cbm) e artifacts.json já tenham sido
gerados pelo script de treino supervisionado.
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

from src.utils.metrics import compute_classification_metrics


LOGGER = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de avaliação {path} não encontrado. Rode o preprocessamento.")
    LOGGER.info("Carregando dataset de avaliação: %s", path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_artifacts(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError("Artefatos não encontrados. Treine o modelo antes de avaliar.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(config_path: str, dataset_path: str | None = None) -> Dict[str, float]:
    config = load_config(config_path)
    artifacts = load_artifacts(Path(config["paths"]["artifacts_file"]))

    dataset_path = dataset_path or config["paths"].get("merged_test")
    df = load_dataframe(Path(dataset_path))

    target_column = config["ieee_dataset"].get("target_column")
    if target_column not in df.columns:
        raise KeyError(
            "Dataset de avaliação precisa conter coluna alvo. Caso seja o conjunto de teste oficial do Kaggle," \
            " utilize os scripts de predição ao invés deste."
        )

    X = df[artifacts["features"]]
    y = df[target_column].values

    model = CatBoostClassifier()
    model.load_model(Path(config["paths"]["model_file"]))

    proba = model.predict_proba(X)[:, 1]
    threshold = artifacts["threshold"]["value"]
    preds = (proba >= threshold).astype(int)
    metrics_dict = compute_classification_metrics(y, preds, proba)

    output_path = Path(config["paths"]["reports_dir"]) / "evaluation_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    LOGGER.info("Resultados salvos em %s", output_path)
    return metrics_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação do modelo CatBoost")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Caminho para arquivo de configuração",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Caminho opcional para dataset específico (Parquet ou CSV) com coluna alvo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.dataset)
