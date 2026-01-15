"""Treino supervisionado com CatBoost para detecção de fraudes.

Este script carrega os dados pré-processados, treina um CatBoostClassifier com
class weights automáticos para lidar com desbalanceamento e seleciona o melhor
threshold com base na métrica configurada (default: F1). Também salva artefatos
importantes como o modelo, métricas, gráficos e um arquivo JSON com metadados.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import shap
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src.utils.metrics import ThresholdResult, compute_classification_metrics, select_best_threshold


LOGGER = logging.getLogger("train_supervised")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de treino não encontrado em {path}. Rode load_merge antes.")
    LOGGER.info("Carregando dados de %s", path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_summary(path: Path) -> Dict:
    if not path.exists():
        LOGGER.warning("Resumo %s não encontrado; detectando categorias automaticamente.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(f"Coluna alvo {target_column} não encontrada no dataset. Confirme o preprocessamento.")
    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def get_categorical_columns(summary: Dict, df: pd.DataFrame) -> Tuple[list[str], list[int]]:
    categorical_cols = summary.get("categorical_columns")
    if not categorical_cols:
        categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
    cat_indices = [df.columns.get_loc(col) for col in categorical_cols if col in df.columns]
    return categorical_cols, cat_indices


def compute_class_weights(y: pd.Series, strategy: str, manual_weights: list | None = None) -> list[float] | None:
    if strategy == "manual" and manual_weights:
        return manual_weights
    if strategy == "auto":
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return weights.tolist()
    LOGGER.info("Sem class weights personalizados.")
    return None


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_indices: list[int],
    config: Dict,
) -> Tuple[CatBoostClassifier, ThresholdResult, Dict[str, float]]:
    params = config["model"]["params"].copy()

    class_weights = compute_class_weights(
        y_train,
        strategy=config["model"].get("class_weight_strategy", "auto"),
        manual_weights=config["model"].get("manual_class_weights"),
    )
    if class_weights:
        params["class_weights"] = class_weights
        LOGGER.info("Class weights usados: %s", class_weights)

    train_pool = Pool(X_train, y_train, cat_features=categorical_indices)
    val_pool = Pool(X_val, y_val, cat_features=categorical_indices)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=params.get("verbose", 100))

    val_proba = model.predict_proba(X_val)[:, 1]
    best_threshold = select_best_threshold(
        y_val.values,
        val_proba,
        scoring=config["thresholding"].get("metric", "f1"),
    )
    y_val_pred = (val_proba >= best_threshold.threshold).astype(int)
    metrics_dict = compute_classification_metrics(y_val.values, y_val_pred, val_proba)
    return model, best_threshold, metrics_dict


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info("Matriz de confusão salva em %s", output_path)


def save_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay

    disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    disp.ax_.set_title("Curva Precisão-Recall")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    disp.figure_.tight_layout()
    disp.figure_.savefig(output_path)
    LOGGER.info("Curva Precisão-Recall salva em %s", output_path)
    plt.close(disp.figure_)


def save_metrics(metrics_dict: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    LOGGER.info("Métricas salvas em %s", output_path)


def save_artifacts(
    output_path: Path,
    features: list[str],
    categorical_columns: list[str],
    threshold: ThresholdResult,
    metrics_dict: Dict[str, float],
    config: Dict,
) -> None:
    artifacts = {
        "features": features,
        "categorical_columns": categorical_columns,
        "threshold": {
            "value": threshold.threshold,
            "precision": threshold.precision,
            "recall": threshold.recall,
            "f1": threshold.f1,
        },
        "metrics": metrics_dict,
        "model_params": config["model"]["params"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2, ensure_ascii=False)
    LOGGER.info("Artefatos salvos em %s", output_path)


def run_shap_analysis(
    model: CatBoostClassifier,
    X_val: pd.DataFrame,
    categorical_indices: list[int],
    output_path: Path,
    sample_size: int,
) -> None:
    if sample_size <= 0:
        LOGGER.info("SHAP desabilitado por sample_size <= 0")
        return
    sample = X_val.sample(min(sample_size, len(X_val)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, sample, show=False)
    plt.title("Resumo SHAP - Impacto nas previsões")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info("Gráfico SHAP salvo em %s", output_path)


def train_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    data_path = Path(config["paths"]["merged_train"])
    df = load_dataframe(data_path)

    summary = load_summary(data_path.with_name(data_path.stem + "_summary.json"))
    target_column = config["ieee_dataset"]["target_column"]

    X, y = split_features_target(df, target_column)
    categorical_columns, categorical_indices = get_categorical_columns(summary, X)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["preprocessing"].get("test_size", 0.2),
        random_state=config["preprocessing"].get("random_state", 42),
        stratify=y,
    )

    model, best_threshold, metrics_dict = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        categorical_indices,
        config,
    )

    model_path = Path(config["paths"]["model_file"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    LOGGER.info("Modelo salvo em %s", model_path)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= best_threshold.threshold).astype(int)

    save_confusion_matrix(y_val.values, val_pred, Path(config["paths"]["confusion_matrix_png"]))
    save_precision_recall_curve(y_val.values, val_proba, Path(config["paths"]["pr_curve_png"]))
    save_metrics(metrics_dict, Path(config["paths"]["metrics_report"]))
    save_artifacts(
        Path(config["paths"]["artifacts_file"]),
        features=X.columns.tolist(),
        categorical_columns=categorical_columns,
        threshold=best_threshold,
        metrics_dict=metrics_dict,
        config=config,
    )

    shap_output = Path(config["explainability"]["output_summary_png"])
    run_shap_analysis(
        model,
        X_val,
        categorical_indices,
        shap_output,
        sample_size=config["explainability"].get("sample_size", 0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino supervisionado CatBoost")
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Caminho para o arquivo de configuração",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(args.config)
