import numpy as np
from PIL import Image

from src.inference import preprocessar_imagem_propria


def test_preprocessamento_fundo_claro(tmp_path):
    """Verifica o pré-processamento de uma imagem com fundo branco e traço preto."""

    imagem = np.full((100, 100), 255, dtype=np.uint8)

    # Cria um traço artificial semelhante a um dígito
    imagem[20:80, 45:55] = 0
    imagem[50:60, 25:75] = 0

    caminho = tmp_path / "digito_fundo_claro.png"
    Image.fromarray(imagem).save(caminho)

    resultado = preprocessar_imagem_propria(caminho)

    assert resultado.shape == (28, 28)
    assert resultado.dtype == np.float32
    assert resultado.min() >= 0.0
    assert resultado.max() <= 1.0
    assert resultado.max() > 0.0


def test_preprocessamento_fundo_escuro(tmp_path):
    """Verifica o pré-processamento de uma imagem com fundo preto e traço branco."""

    imagem = np.zeros((100, 100), dtype=np.uint8)

    # Cria o mesmo padrão com a polaridade invertida
    imagem[20:80, 45:55] = 255
    imagem[50:60, 25:75] = 255

    caminho = tmp_path / "digito_fundo_escuro.png"
    Image.fromarray(imagem).save(caminho)

    resultado = preprocessar_imagem_propria(caminho)

    assert resultado.shape == (28, 28)
    assert resultado.dtype == np.float32
    assert resultado.min() >= 0.0
    assert resultado.max() <= 1.0
    assert resultado.max() > 0.0