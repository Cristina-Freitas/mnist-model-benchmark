import numpy as np  # Importa o NumPy para manipulação dos arrays
from keras.datasets import mnist  # Importa o dataset MNIST disponibilizado pelo Keras


def carregar_mnist():
    """Carrega o dataset MNIST completo e retorna imagens e rótulos."""

    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # Carrega as divisões padrão do MNIST

    X = np.concatenate((X_train, X_test), axis=0)  # Junta as imagens de treino e teste em um único conjunto
    y = np.concatenate((y_train, y_test), axis=0)  # Junta os rótulos de treino e teste em um único conjunto

    return X, y  # Retorna as 70.000 imagens e seus respectivos rótulos

