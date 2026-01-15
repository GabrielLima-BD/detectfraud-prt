"""Treino opcional de modelo não supervisionado (IsolationForest) para fraudes.

Este arquivo é útil quando você deseja comparar o CatBoost supervisionado com
um detector não supervisionado simples. Ele salva um modelo sklearn em disco
usando joblib e gera métricas básicas se a coluna alvo estiver disponível.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


LOGGER = logging.getLogger("train_unsupervised")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo {path} não encontrado. Rode src/data/load_merge_ieee.py antes."
        )
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def train_isolation_forest(config_path: str) -> None:
    config = load_config(config_path)
    data_path = Path(config["paths"]["merged_train"])
    df = load_dataframe(data_path)

    target_column = config["ieee_dataset"].get("target_column")
    X = df.drop(columns=[target_column]) if target_column in df.columns else df

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        max_samples=0.7,
        random_state=42,
    )
    model.fit(X)
    LOGGER.info("IsolationForest treinado com sucesso.")

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "isolation_forest.joblib")

    if target_column in df.columns:
        # IsolationForest retorna -1 para anomalias, +1 para normais
        scores = -model.score_samples(X)
        preds = (scores > 0).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(df[target_column], scores),
            "pr_auc": average_precision_score(df[target_column], scores),
        }
        report_path = Path(config["paths"]["reports_dir"]) / "isolation_forest_metrics.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        LOGGER.info("Métricas salvas em %s", report_path)
        LOGGER.info("Resumo:\n%s", classification_report(df[target_column], preds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino IsolationForest")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Caminho do arquivo de configuração",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_isolation_forest(args.config)
