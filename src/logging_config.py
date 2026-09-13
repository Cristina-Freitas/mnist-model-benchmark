import logging
from pathlib import Path


def configurar_logger():
    """
    Configura o logger principal do projeto.

    Os registros são gravados em:
    logs/mnist_benchmark.log
    """

    logger = logging.getLogger("mnist_benchmark")

    # Evita adicionar handlers duplicados ao executar o notebook várias vezes
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Localiza a raiz do projeto a partir deste arquivo
    raiz_projeto = Path(__file__).resolve().parent.parent

    pasta_logs = raiz_projeto / "logs"
    pasta_logs.mkdir(exist_ok=True)

    caminho_log = pasta_logs / "mnist_benchmark.log"

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Salva os registros em arquivo
    arquivo_handler = logging.FileHandler(
        caminho_log,
        encoding="utf-8"
    )
    arquivo_handler.setFormatter(formato)

    # Também mostra os registros no terminal/notebook
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formato)

    logger.addHandler(arquivo_handler)
    logger.addHandler(console_handler)

    return logger