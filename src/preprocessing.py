import numpy as np
from sklearn.model_selection import train_test_split  # Importa a função usada para dividir os dados de forma estratificada


def dividir_dados(X, y, random_state=42):
    """Divide os dados em treino, validação e teste mantendo a proporção das classes."""

    X_treino, X_temp, y_treino, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,          # Reserva 30% dos dados para validação e teste
        stratify=y,              # Mantém a proporção das classes de 0 a 9
        random_state=random_state  # Garante reprodutibilidade da divisão
    )

    X_validacao, X_teste, y_validacao, y_teste = train_test_split(
        X_temp,
        y_temp,
        test_size=2/3,             # Dos 30% restantes, separa 20% do total para teste e 10% para validação
        stratify=y_temp,           # Mantém a distribuição das classes também nesta segunda divisão
        random_state=random_state  # Mantém a divisão reproduzível
    )

    return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste

def normalizar_pixels(X):
    """Normaliza os valores dos pixels do intervalo [0, 255] para [0, 1]."""

    X_normalizado = X.astype("float32") / 255.0  # Converte para ponto flutuante e redimensiona os pixels para o intervalo [0, 1]

    return X_normalizado  # Retorna as imagens normalizadas


def achatar_imagens(X):
    """Transforma imagens 28 x 28 em vetores de 784 atributos."""

    X_achatado = X.reshape(X.shape[0], -1)  # Mantém a quantidade de imagens e transforma cada matriz 28 x 28 em um vetor de 784 pixels

    return X_achatado  # Retorna os dados no formato esperado pelos modelos clássicos

def ocultar_classes(X, y, classes_ocultas):
    """Remove do conjunto as amostras pertencentes às classes informadas."""

    mascara = ~np.isin(y, classes_ocultas)  # Mantém apenas exemplos que não pertencem às classes ocultadas

    X_filtrado = X[mascara]
    y_filtrado = y[mascara]

    return X_filtrado, y_filtrado