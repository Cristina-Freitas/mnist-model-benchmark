import numpy as np
from sklearn.datasets import make_classification

from src.modeling.train import (
    treinar_random_forest,
    treinar_svm,
    treinar_mlp,
    avaliar_modelo
)


# Cria um pequeno conjunto artificial apenas para validar as funções de treinamento
X_teste_modelos, y_teste_modelos = make_classification(
    n_samples=100,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)


def test_treinar_random_forest():
    """Verifica se o Random Forest é treinado e retorna tempo válido."""

    modelo, tempo = treinar_random_forest(
        X_teste_modelos,
        y_teste_modelos,
        n_estimators=10,
        max_depth=5
    )

    assert hasattr(modelo, "predict")
    assert tempo >= 0


def test_treinar_svm():
    """Verifica se o SVM é treinado e retorna tempo válido."""

    modelo, tempo = treinar_svm(
        X_teste_modelos,
        y_teste_modelos,
        C=1.0,
        kernel="linear"
    )

    assert hasattr(modelo, "predict")
    assert tempo >= 0


def test_treinar_mlp():
    """Verifica se a MLP é treinada e retorna tempo válido."""

    modelo, tempo = treinar_mlp(
        X_teste_modelos,
        y_teste_modelos,
        hidden_layer_sizes=(10,),
        learning_rate_init=0.001
    )

    assert hasattr(modelo, "predict")
    assert tempo >= 0


def test_avaliar_modelo():
    """Verifica se a função de avaliação retorna uma acurácia válida."""

    modelo, _ = treinar_random_forest(
        X_teste_modelos,
        y_teste_modelos,
        n_estimators=10,
        max_depth=5
    )

    acuracia = avaliar_modelo(
        modelo,
        X_teste_modelos,
        y_teste_modelos
    )

    assert isinstance(acuracia, (float, np.floating))
    assert 0.0 <= acuracia <= 1.0