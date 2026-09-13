import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plotar_matriz_confusao(matriz, titulo):
    """Exibe a matriz de confusão em formato de heatmap."""

    fig, ax = plt.subplots(figsize=(9, 7))

    # Cria o mapa de calor
    imagem = ax.imshow(
        matriz,
        cmap="Blues",
        interpolation="nearest"
    )

    # Barra lateral com a escala de valores
    fig.colorbar(imagem, ax=ax)

    # Classes de 0 a 9 nos dois eixos
    classes = range(10)

    ax.set_xticks(classes)
    ax.set_yticks(classes)

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Identificação dos eixos e título
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)

    # Linhas brancas para separar visualmente as células
    ax.set_xticks([x - 0.5 for x in range(1, 10)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 10)], minor=True)

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1
    )

    ax.tick_params(which="minor", bottom=False, left=False)

    # Insere o valor numérico dentro de cada célula
    limite_cor = matriz.max() / 2

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
                color="white" if matriz[i, j] > limite_cor else "black"
            )

    plt.tight_layout()
    plt.show()


def plotar_matriz_ood(matriz, classes_reais, classes_previstas, titulo):
    """Exibe a matriz OOD em formato de heatmap."""

    fig, ax = plt.subplots(figsize=(10, 4))

    imagem = ax.imshow(
        matriz,
        cmap="Blues",
        interpolation="nearest"
    )

    fig.colorbar(imagem, ax=ax)

    ax.set_xticks(range(len(classes_previstas)))
    ax.set_yticks(range(len(classes_reais)))

    ax.set_xticklabels(classes_previstas)
    ax.set_yticklabels(classes_reais)

    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo)

    limite_cor = matriz.max() / 2

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(
                j,
                i,
                matriz[i, j],
                ha="center",
                va="center",
                color="white" if matriz[i, j] > limite_cor else "black"
            )

    plt.tight_layout()
    plt.show()