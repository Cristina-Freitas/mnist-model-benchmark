import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calcular_metricas(y_real, y_pred):
    """Calcula as principais métricas de classificação exigidas no projeto."""

    metricas = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision_weighted": precision_score(y_real, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_real, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_real, y_pred, average="weighted")
    }

    return metricas

def calcular_matriz_confusao(y_real, y_pred):
    """Calcula a matriz de confusão multiclasse."""

    return confusion_matrix(y_real, y_pred)

def identificar_maior_confusao(matriz):
    """Identifica a maior taxa de confusão entre a classe real e a classe prevista."""

    # Normaliza cada linha pela quantidade de exemplos da respectiva classe real
    matriz_normalizada = matriz / matriz.sum(axis=1, keepdims=True)

    # Remove a diagonal principal, que representa os acertos
    matriz_erros = matriz_normalizada.copy()
    np.fill_diagonal(matriz_erros, 0)

    # Localiza o maior erro proporcional fora da diagonal
    classe_real, classe_prevista = np.unravel_index(
        np.argmax(matriz_erros),
        matriz_erros.shape
    )

    taxa_confusao = matriz_erros[classe_real, classe_prevista]

    return classe_real, classe_prevista, taxa_confusao


def calcular_matriz_ood(y_real, y_pred, classes_reais, classes_previstas):
    """Constrói uma matriz para analisar classes reais ocultadas contra classes conhecidas previstas."""

    matriz = np.zeros(
        (len(classes_reais), len(classes_previstas)),
        dtype=int
    )

    for i, classe_real in enumerate(classes_reais):
        for j, classe_prevista in enumerate(classes_previstas):
            matriz[i, j] = np.sum(
                (y_real == classe_real) & (y_pred == classe_prevista)
            )

    return matriz


