from PIL import Image
import numpy as np

from scipy.ndimage import (
    gaussian_filter,
    binary_closing,
    label,
    center_of_mass,
    shift
)


def preprocessar_imagem_propria(caminho_imagem):
    """
    Prepara uma imagem externa para classificação pelo modelo MNIST.

    Etapas:
    1. Conversão para escala de cinza.
    2. Detecção automática da polaridade do fundo.
    3. Correção de iluminação por contraste local.
    4. Segmentação do dígito.
    5. Remoção de pequenos ruídos.
    6. Bounding box.
    7. Redimensionamento proporcional.
    8. Centralização pelo centro de massa.
    9. Normalização para [0, 1].
    """

    # ---------------------------------------------------------
    # 1. Carregamento em escala de cinza
    # ---------------------------------------------------------

    imagem = Image.open(caminho_imagem).convert("L")
    imagem_np = np.asarray(imagem, dtype=np.float32)

    altura, largura = imagem_np.shape

    # ---------------------------------------------------------
    # 2. Detecta se o fundo predominante é claro ou escuro
    #    utilizando as bordas da imagem
    # ---------------------------------------------------------

    borda = np.concatenate([
        imagem_np[0, :],
        imagem_np[-1, :],
        imagem_np[:, 0],
        imagem_np[:, -1]
    ])

    fundo_claro = np.median(borda) > 127

    # ---------------------------------------------------------
    # 3. Estima localmente o fundo/iluminação
    # ---------------------------------------------------------

    sigma = max(5, min(altura, largura) * 0.03)

    fundo_estimado = gaussian_filter(
        imagem_np,
        sigma=sigma
    )

    # Fundo claro + traço escuro
    if fundo_claro:
        contraste = fundo_estimado - imagem_np

    # Fundo escuro + traço claro
    else:
        contraste = imagem_np - fundo_estimado

    contraste = np.clip(
        contraste,
        0,
        None
    )

    # ---------------------------------------------------------
    # 4. Normaliza o contraste sem depender de um único
    #    pixel extremamente claro
    # ---------------------------------------------------------

    referencia = np.percentile(
        contraste,
        99
    )

    if referencia <= 0:
        raise ValueError(
            "Não foi possível identificar o dígito na imagem."
        )

    contraste = np.clip(
        contraste / referencia * 255,
        0,
        255
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # 5. Segmentação do traço
    # ---------------------------------------------------------

    mascara = contraste > 25

    # Fecha pequenas falhas existentes dentro dos traços
    mascara = binary_closing(
        mascara,
        structure=np.ones((3, 3)),
        iterations=2
    )

    if not np.any(mascara):
        raise ValueError(
            "Nenhum dígito foi detectado na imagem."
        )

    # ---------------------------------------------------------
    # 6. Remove somente componentes muito pequenos
    # ---------------------------------------------------------

    componentes, quantidade = label(mascara)

    if quantidade == 0:
        raise ValueError(
            "Nenhum componente válido foi detectado."
        )

    tamanhos = np.bincount(
        componentes.ravel()
    )

    tamanhos[0] = 0

    maior_tamanho = tamanhos.max()

    # Componentes relevantes do desenho
    limite = max(
        10,
        maior_tamanho * 0.02
    )

    componentes_validos = np.where(
        tamanhos >= limite
    )[0]

    mascara_final = np.isin(
        componentes,
        componentes_validos
    )

    # ---------------------------------------------------------
    # 7. Calcula o bounding box envolvendo o dígito
    # ---------------------------------------------------------

    linhas, colunas = np.where(
        mascara_final
    )

    topo = linhas.min()
    base = linhas.max() + 1
    esquerda = colunas.min()
    direita = colunas.max() + 1

    # Pequena margem ao redor do dígito
    margem_y = max(
        2,
        int((base - topo) * 0.05)
    )

    margem_x = max(
        2,
        int((direita - esquerda) * 0.05)
    )

    topo = max(
        0,
        topo - margem_y
    )

    base = min(
        altura,
        base + margem_y
    )

    esquerda = max(
        0,
        esquerda - margem_x
    )

    direita = min(
        largura,
        direita + margem_x
    )

    # ---------------------------------------------------------
    # 8. Mantém a intensidade dos traços e remove o fundo
    # ---------------------------------------------------------

    imagem_limpa = np.zeros_like(
        contraste
    )

    imagem_limpa[mascara_final] = (
        contraste[mascara_final]
    )

    recorte = imagem_limpa[
        topo:base,
        esquerda:direita
    ]

    # ---------------------------------------------------------
    # 9. Redimensiona mantendo a proporção
    #    O MNIST utiliza o dígito dentro de aproximadamente
    #    20x20 pixels em uma tela 28x28
    # ---------------------------------------------------------

    imagem_recortada = Image.fromarray(
        recorte
    )

    imagem_recortada.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    deslocamento_x = (
        28 - imagem_recortada.width
    ) // 2

    deslocamento_y = (
        28 - imagem_recortada.height
    ) // 2

    canvas.paste(
        imagem_recortada,
        (
            deslocamento_x,
            deslocamento_y
        )
    )

    # ---------------------------------------------------------
    # 10. Centralização pelo centro de massa
    # ---------------------------------------------------------

    canvas_np = np.asarray(
        canvas,
        dtype=np.float32
    )

    centro_y, centro_x = center_of_mass(
        canvas_np
    )

    if np.isfinite(centro_x) and np.isfinite(centro_y):

        ajuste_x = 13.5 - centro_x
        ajuste_y = 13.5 - centro_y

        canvas_np = shift(
            canvas_np,
            shift=(
                ajuste_y,
                ajuste_x
            ),
            order=1,
            mode="constant",
            cval=0
        )

    # ---------------------------------------------------------
    # 11. Normalização para [0, 1]
    # ---------------------------------------------------------

    imagem_final = np.clip(
        canvas_np,
        0,
        255
    ).astype("float32") / 255.0

    return imagem_final