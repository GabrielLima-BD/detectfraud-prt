"""Pipeline de ingestão e preprocessamento do dataset IEEE-CIS Fraud Detection.

Este script foi escrito para ser amigável e resiliente: ele consulta as configurações
em src/config/config.yaml, baixa os dados locais da pasta data/raw, faz o merge das
bases de transações e identidades e salva o resultado como Parquet (ou CSV, se
Parquet não estiver disponível).

A ideia é que você rode este arquivo como módulo ou script sempre que precisar
preparar os dados de treino e teste antes dos experimentos.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger("load_merge_ieee")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_config(config_path: str) -> Dict:
    """Carrega o arquivo YAML de configuração."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_directory(path: Path) -> None:
    """Cria diretórios de forma segura."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Lê um CSV e fornece mensagem amigável em caso de falha."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado em {path}. Garanta que os CSVs foram baixados do Kaggle."
        )
    LOGGER.info("Carregando %s", path)
    df = pd.read_csv(path)

    # Alguns exports/espelhos do dataset podem trazer colunas do tipo id-01 em vez de id_01.
    # Padronizamos aqui para garantir que treino e teste fiquem com o MESMO schema.
    df.columns = [str(c).replace("-", "_") for c in df.columns]
    return df


def merge_datasets(
    transaction: pd.DataFrame,
    identity: pd.DataFrame,
    index_column: str,
) -> pd.DataFrame:
    """Realiza merge left join usando TransactionID como chave principal."""
    if index_column not in transaction.columns:
        raise KeyError(f"Coluna de índice {index_column} não existe em train_transaction.csv")
    if index_column not in identity.columns:
        LOGGER.warning(
            "Coluna %s ausente em identity; retornando apenas dados de transação.", index_column
        )
        return transaction
    merged = transaction.merge(identity, how="left", on=index_column)
    LOGGER.info("Merge final possui %d linhas e %d colunas", merged.shape[0], merged.shape[1])
    return merged


def detect_column_types(df: pd.DataFrame, manual_categoricals: Optional[list[str]] = None) -> Tuple[list[str], list[str]]:
    """Detecta colunas numéricas e categóricas com fallback manual.

    CatBoost aceita colunas categóricas por nome sem one-hot, então priorizamos
    dtypes object e category. Colunas manuais podem ser adicionadas via config.
    """
    manual_categoricals = manual_categoricals or []

    categorical_cols = [
        col for col in df.columns if df[col].dtype == "object" or col in manual_categoricals
    ]
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: float,
    categorical_strategy: str,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Preenche valores ausentes com estratégias simples e previsíveis."""
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].fillna(categorical_strategy)
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    df[numeric_cols] = df[numeric_cols].fillna(numeric_strategy)
    return df


def downcast_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reduz uso de memória convertendo floats/ints para tipos menores."""
    df = df.copy()
    for col in columns:
        col_data = df[col]
        if pd.api.types.is_float_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="float")
        elif pd.api.types.is_integer_dtype(col_data):
            df[col] = pd.to_numeric(col_data, downcast="integer")
    return df


def save_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    parquet_engine: str,
    allow_csv_fallback: bool,
) -> Path:
    """Salva DataFrame em Parquet com fallback para CSV caso PyArrow não esteja disponível."""
    ensure_directory(output_path)
    try:
        df.to_parquet(output_path, engine=parquet_engine, index=False)
        LOGGER.info("Arquivo salvo em formato Parquet em %s", output_path)
        return output_path
    except Exception as parquet_error:  # noqa: BLE001
        if not allow_csv_fallback:
            raise parquet_error
        LOGGER.warning(
            "Falha ao salvar Parquet (%s). Salvando como CSV em fallback...", parquet_error
        )
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    """Extrai estatísticas simples para ajudar o usuário."""
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        "fraud_rate": float(df[df.columns[-1]].mean()) if df.columns[-1].startswith("is") else np.nan,
    }


def process_split(
    config: Dict,
    split: str,
    transaction_file: str,
    identity_file: str,
    output_path: Path,
    target_column: Optional[str] = None,
) -> Dict:
    """Executa pipeline para treino ou teste e retorna metadados."""
    raw_dir = Path(config["paths"]["raw_dir"])
    transaction_path = raw_dir / transaction_file
    identity_path = raw_dir / identity_file

    transaction = read_csv_safe(transaction_path)
    identity = read_csv_safe(identity_path)

    merged = merge_datasets(transaction, identity, config["ieee_dataset"]["index_column"])

    drop_cols = config["preprocessing"].get("drop_columns", [])
    if drop_cols:
        merged = merged.drop(columns=drop_cols, errors="ignore")

    numeric_cols, categorical_cols = detect_column_types(
        merged,
        manual_categoricals=config["preprocessing"].get("categorical_cols_manual", []),
    )

    merged = fill_missing_values(
        merged,
        numeric_strategy=config["preprocessing"]["missing_fill_strategy"]["numeric"],
        categorical_strategy=config["preprocessing"]["missing_fill_strategy"]["categorical"],
        categorical_cols=categorical_cols,
    )

    if config["preprocessing"].get("enable_downcast", True):
        merged = downcast_numeric(merged, numeric_cols)

    saved_path = save_dataframe(
        merged,
        output_path=output_path,
        parquet_engine=config["preprocessing"].get("parquet_engine", "pyarrow"),
        allow_csv_fallback=config["preprocessing"].get("parquet_fallback_to_csv", True),
    )

    summary = summarize_dataframe(merged)
    summary.update(
        {
            "split": split,
            "output_path": str(saved_path),
            "categorical_columns": categorical_cols,
            "numeric_columns": numeric_cols,
        }
    )

    # Guardamos um resumo json para consulta rápida
    # pathlib.Path.with_suffix() substitui a extensão (ex: .parquet), não serve para "anexar" texto.
    # Aqui queremos um arquivo "irmão": train_merged_summary.json
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    ensure_directory(summary_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("Resumo salvo em %s", summary_path)
    return summary


def run_pipeline(config_path: str) -> Dict[str, Dict]:
    """Executa pipeline completo para treino e teste."""
    config = load_config(config_path)
    outputs = {}
    outputs["train"] = process_split(
        config,
        split="train",
        transaction_file=config["ieee_dataset"]["transaction_file"],
        identity_file=config["ieee_dataset"]["identity_file"],
        output_path=Path(config["paths"]["merged_train"]),
        target_column=config["ieee_dataset"]["target_column"],
    )

    # Teste pode não possuir coluna alvo, portanto usamos opção separada
    outputs["test"] = process_split(
        config,
        split="test",
        transaction_file=config["ieee_dataset"].get("test_transaction_file", "test_transaction.csv"),
        identity_file=config["ieee_dataset"].get("test_identity_file", "test_identity.csv"),
        output_path=Path(config["paths"].get("merged_test", "data/processed/test_merged.parquet")),
        target_column=None,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de merge IEEE-CIS")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Caminho para o arquivo de configuração",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata = run_pipeline(args.config)
    LOGGER.info("Pipeline finalizado com sucesso:\n%s", json.dumps(metadata, indent=2, ensure_ascii=False))
