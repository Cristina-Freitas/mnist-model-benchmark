import numpy as np  # Importa o NumPy para criar dados simulados para o teste

from src.preprocessing import dividir_dados, normalizar_pixels, achatar_imagens  # Importa a função de divisão que será testada


def test_proporcoes_divisao():
    """Verifica se a divisão dos dados respeita 70% treino, 10% validação e 20% teste."""

    X = np.arange(1000).reshape(100, 10)  # Cria 100 amostras fictícias com 10 atributos cada
    y = np.repeat(np.arange(10), 10)  # Cria 10 classes com 10 amostras cada

    X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = dividir_dados(X, y)  # Executa a divisão estratificada

    assert len(X_treino) == 70  # Confirma que 70% das amostras foram destinadas ao treino
    assert len(X_validacao) == 10  # Confirma que 10% das amostras foram destinadas à validação
    assert len(X_teste) == 20  # Confirma que 20% das amostras foram destinadas ao teste

    assert len(y_treino) == 70  # Confirma a correspondência dos rótulos de treino
    assert len(y_validacao) == 10  # Confirma a correspondência dos rótulos de validação
    assert len(y_teste) == 20  # Confirma a correspondência dos rótulos de teste


def test_estratificacao_divisao():
    """Verifica se todas as classes permanecem representadas após a divisão estratificada."""

    X = np.arange(1000).reshape(100, 10)  # Cria 100 amostras fictícias com 10 atributos cada
    y = np.repeat(np.arange(10), 10)  # Cria 10 classes balanceadas com 10 amostras cada

    _, _, _, y_treino, y_validacao, y_teste = dividir_dados(X, y)  # Executa a divisão e utiliza apenas os rótulos

    assert np.array_equal(np.unique(y_treino), np.arange(10))  # Confirma as 10 classes no conjunto de treino
    assert np.array_equal(np.unique(y_validacao), np.arange(10))  # Confirma as 10 classes no conjunto de validação
    assert np.array_equal(np.unique(y_teste), np.arange(10))  # Confirma as 10 classes no conjunto de teste

def test_normalizacao_pixels():
    """Verifica se os pixels são convertidos corretamente para o intervalo [0, 1]."""

    X = np.array([0, 127, 255], dtype=np.uint8)  # Cria valores representativos da escala original dos pixels

    X_normalizado = normalizar_pixels(X)  # Aplica a normalização definida no pré-processamento

    assert X_normalizado.min() >= 0.0  # Confirma que nenhum valor ficou abaixo de 0
    assert X_normalizado.max() <= 1.0  # Confirma que nenhum valor ficou acima de 1
    assert np.isclose(X_normalizado[0], 0.0)  # Confirma que 0 permanece 0
    assert np.isclose(X_normalizado[2], 1.0)  # Confirma que 255 é convertido para 1


def test_normalizacao_preserva_formato():
    """Verifica se a normalização altera apenas a escala dos pixels, preservando o formato das imagens."""

    X = np.array([
        [[0, 127], [255, 64]],
        [[255, 0], [128, 32]]
    ], dtype=np.uint8)  # Cria duas imagens fictícias 2 x 2

    X_normalizado = normalizar_pixels(X)  # Aplica a normalização dos pixels

    assert X_normalizado.shape == X.shape  # Confirma que a normalização não alterou as dimensões das imagens
    assert X_normalizado.dtype == np.float32  # Confirma a conversão dos pixels para float32


def test_achatar_imagens():
    """Verifica se imagens 28 x 28 são transformadas corretamente em vetores de 784 atributos."""

    X = np.arange(2 * 28 * 28).reshape(2, 28, 28)  # Cria duas imagens artificiais de 28 x 28 pixels

    X_achatado = achatar_imagens(X)  # Aplica a transformação para o formato vetorial

    assert X_achatado.shape == (2, 784)  # Verifica se cada imagem passou a possuir 784 atributos
    assert np.array_equal(X_achatado[0], X[0].reshape(-1))  # Confirma que os valores e sua ordem foram preservados