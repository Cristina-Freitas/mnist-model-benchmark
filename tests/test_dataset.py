import numpy as np  # Importa o NumPy para validar o tipo e o conteúdo dos dados
from src.dataset import carregar_mnist  # Importa a função responsável por carregar o MNIST


def test_carregamento_mnist():
    X, y = carregar_mnist()  # Carrega as imagens e os rótulos

    assert X.shape == (70000, 28, 28)  # Confirma que existem 70.000 imagens de 28 x 28 pixels
    assert y.shape == (70000,)  # Confirma que existe um rótulo para cada imagem
    assert set(np.unique(y)) == set(range(10))  # Confirma que os rótulos representam os dígitos de 0 a 9