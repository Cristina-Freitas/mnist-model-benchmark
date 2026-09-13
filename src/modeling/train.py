from time import perf_counter  # Permite medir o tempo gasto no treinamento do modelo

from sklearn.ensemble import RandomForestClassifier  # Implementa o classificador Random Forest
from sklearn.metrics import accuracy_score  # Calcula a acurácia no conjunto de validação
from sklearn.svm import SVC  # Implementa o classificador Support Vector Machine
from sklearn.neural_network import MLPClassifier  # Implementa a rede neural Perceptron Multicamadas

def treinar_random_forest(X_treino, y_treino, n_estimators, max_depth):
    """Treina um modelo Random Forest e registra o tempo de treinamento."""

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,  # Quantidade de árvores da floresta
        max_depth=max_depth,        # Profundidade máxima das árvores
        random_state=42,            # Garante reprodutibilidade
        n_jobs=-1                   # Utiliza os núcleos disponíveis da CPU
    )

    inicio = perf_counter()  # Registra o instante anterior ao treinamento

    modelo.fit(X_treino, y_treino)  # Treina o modelo com os dados fornecidos

    tempo_treinamento = perf_counter() - inicio  # Calcula o tempo total de treinamento

    return modelo, tempo_treinamento  # Retorna o modelo treinado e o tempo gasto


def avaliar_modelo(modelo, X_validacao, y_validacao):
    """Calcula a acurácia do modelo no conjunto de validação."""

    previsoes = modelo.predict(X_validacao)  # Gera as previsões para os dados de validação
    acuracia = accuracy_score(y_validacao, previsoes)  # Compara as previsões com os rótulos reais

    return acuracia  # Retorna a acurácia obtida


def treinar_svm(X_treino, y_treino, C, kernel):
    """Treina um modelo SVM e registra o tempo de treinamento."""

    modelo = SVC(
        C=C,                      # Controla a penalização dos erros de classificação
        kernel=kernel,            # Define o tipo de fronteira de decisão utilizada
        random_state=42           # Mantém a configuração reproduzível
    )

    inicio = perf_counter()  # Registra o instante anterior ao treinamento
    modelo.fit(X_treino, y_treino)  # Treina o SVM
    tempo_treinamento = perf_counter() - inicio  # Calcula o tempo total de treinamento

    return modelo, tempo_treinamento  # Retorna o modelo treinado e o tempo gasto


def treinar_mlp(X_treino, y_treino, hidden_layer_sizes, learning_rate_init):
    """Treina uma rede neural MLP e registra o tempo de treinamento."""

    modelo = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,  # Define a quantidade de neurônios das camadas ocultas
        learning_rate_init=learning_rate_init,  # Define a taxa de aprendizado inicial
        activation="relu",                      # Utiliza ReLU como função de ativação
        max_iter=30,                            # Limita a quantidade máxima de épocas de treinamento
        random_state=42                         # Garante reprodutibilidade
    )

    inicio = perf_counter()  # Registra o instante anterior ao treinamento
    modelo.fit(X_treino, y_treino)  # Treina a rede neural
    tempo_treinamento = perf_counter() - inicio  # Calcula o tempo total de treinamento

    return modelo, tempo_treinamento  # Retorna o modelo treinado e o tempo gasto