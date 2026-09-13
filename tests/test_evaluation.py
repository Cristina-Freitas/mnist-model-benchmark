import numpy as np
from src.evaluation import (
    calcular_metricas,
    calcular_matriz_confusao,
    identificar_maior_confusao
)

from src.evaluation import calcular_metricas, calcular_matriz_confusao


def test_calcular_metricas():
    """Verifica se as métricas de classificação são calculadas corretamente."""

    y_real = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])

    metricas = calcular_metricas(y_real, y_pred)

    assert set(metricas.keys()) == {
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted"
    }

    for valor in metricas.values():
        assert 0.0 <= valor <= 1.0


def test_calcular_matriz_confusao():
    """Verifica as dimensões e o total de amostras da matriz de confusão."""

    y_real = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])

    matriz = calcular_matriz_confusao(y_real, y_pred)

    assert matriz.shape == (3, 3)
    assert matriz.sum() == len(y_real)


def test_identificar_maior_confusao():
    """Verifica se a maior taxa de confusão entre classes é identificada corretamente."""

    matriz = np.array([
        [8, 2, 0],
        [1, 9, 0],
        [0, 3, 7]
    ])

    classe_real, classe_prevista, taxa = identificar_maior_confusao(matriz)

    assert classe_real == 2
    assert classe_prevista == 1
    assert np.isclose(taxa, 0.30)