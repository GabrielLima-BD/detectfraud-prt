"""Funções utilitárias de métricas e seleção de limiar.

Este módulo concentra as métricas usadas no pipeline. A ideia é manter tudo
centralizado para facilitar ajustes rápidos sem quebrar outros módulos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Literal, Tuple

import numpy as np
from sklearn import metrics


MetricName = Literal["precision", "recall", "f1", "roc_auc", "pr_auc"]


@dataclass
class ThresholdResult:
    threshold: float
    metric_value: float
    precision: float
    recall: float
    f1: float


def _metric_functions() -> Dict[MetricName, Callable[[Iterable[float], Iterable[float]], float]]:
    return {
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "roc_auc": metrics.roc_auc_score,
        "pr_auc": metrics.average_precision_score,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Calcula métricas principais e retorna em dicionário simples."""
    results = {
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_proba),
        "pr_auc": metrics.average_precision_score(y_true, y_proba),
    }
    return results


def select_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    scoring: MetricName = "f1",
) -> ThresholdResult:
    """Seleciona threshold com base na métrica desejada usando curva Precisão-Recall."""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_proba)

    # A curva retorna len(thresholds) = len(precision) - 1, então ajustamos arrays
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    metric_map = {
        "precision": precision[:-1],
        "recall": recall[:-1],
        "f1": f1_scores,
        "pr_auc": f1_scores,  # usa F1 como substituto quando métrica não depende de threshold
        "roc_auc": f1_scores,
    }

    if scoring not in metric_map:
        raise ValueError(f"Métrica {scoring} não suportada. Veja opções em MetricName.")

    best_idx = int(np.argmax(metric_map[scoring]))
    best_threshold = float(thresholds[best_idx])

    return ThresholdResult(
        threshold=best_threshold,
        metric_value=float(metric_map[scoring][best_idx]),
        precision=float(precision[best_idx]),
        recall=float(recall[best_idx]),
        f1=float(f1_scores[best_idx]),
    )
