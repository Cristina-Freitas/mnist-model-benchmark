# Documentação Técnica — MNIST Model Benchmark

Esta documentação complementa o `README.md` do projeto **MNIST Model Benchmark** e apresenta, em maior profundidade, as decisões metodológicas, os experimentos realizados e a interpretação técnica dos resultados.

O objetivo não é repetir as instruções de execução ou a apresentação geral do projeto, mas registrar as escolhas que orientaram o desenvolvimento do pipeline de Machine Learning e analisar seus resultados, limitações e comportamento diante de situações diferentes da classificação convencional do MNIST.

---

## 1. Metodologia Experimental

O projeto foi estruturado como um problema de **classificação multiclasse supervisionada**, utilizando o dataset MNIST, composto por 70.000 imagens em escala de cinza de dígitos manuscritos pertencentes às classes de 0 a 9.

Cada imagem possui dimensão original de:

```text
28 × 28 pixels
```

Para utilização pelos modelos empregados no projeto, cada imagem foi posteriormente representada por um vetor de:

```text
28 × 28 = 784 atributos
```

A metodologia foi organizada em três níveis de avaliação:

1. **Validação:** comparação de configurações e hiperparâmetros;
2. **Teste independente:** comparação final dos modelos selecionados;
3. **Testes de robustez:** análise com classes ausentes do treinamento e imagens externas ao MNIST.

Essa separação permitiu avaliar não apenas o desempenho preditivo convencional, mas também algumas limitações do comportamento dos classificadores.

---

## 2. Preparação e Separação dos Dados

O conjunto completo de 70.000 amostras foi dividido em:

| Conjunto | Percentual | Quantidade |
|---|---:|---:|
| Treinamento | 70% | 49.000 |
| Validação | 10% | 7.000 |
| Teste | 20% | 14.000 |
| **Total** | **100%** | **70.000** |

A divisão foi realizada de forma **estratificada**, preservando aproximadamente a distribuição das dez classes em cada subconjunto.

Essa decisão é importante porque cada conjunto possui uma finalidade distinta:

- o treinamento ajusta os parâmetros internos dos modelos;
- a validação permite comparar configurações e hiperparâmetros;
- o teste permanece independente para a avaliação final.

A separação evita utilizar o conjunto de teste para selecionar o modelo, reduzindo o risco de uma avaliação excessivamente otimista.

### Normalização

Os valores originais dos pixels pertencem ao intervalo:

```text
[0, 255]
```

Foi aplicada a transformação:

```text
pixel_normalizado = pixel / 255
```

resultando no intervalo:

```text
[0, 1]
```

A normalização padroniza a escala numérica das entradas e é especialmente relevante para algoritmos sensíveis à magnitude dos atributos, como SVM e redes neurais.

### Transformação das imagens

As imagens com formato `(28, 28)` foram achatadas para vetores de 784 atributos.

Essa transformação modifica apenas a representação dos dados, sem alterar os valores dos pixels, permitindo sua utilização pelos classificadores empregados neste projeto.

---

## 3. Estratégia de Modelagem

Foram avaliadas três famílias distintas de modelos:

- **Random Forest**, representando métodos ensemble baseados em árvores;
- **Support Vector Machine (SVM)**, representando métodos baseados em margens e funções kernel;
- **Multilayer Perceptron (MLP)**, representando redes neurais artificiais feedforward.

A escolha de algoritmos com princípios de funcionamento diferentes permitiu comparar não apenas métricas finais, mas também comportamento, custo computacional e sensibilidade às configurações adotadas.

Para cada algoritmo foram avaliadas diferentes configurações de hiperparâmetros utilizando o conjunto de validação.

---

## 4. Random Forest

Foram avaliadas duas configurações principais.

### Configuração 1

- `n_estimators = 100`
- `max_depth = None`

Resultado de validação:

- Accuracy: **96,49%**
- Tempo de treinamento: **8,75 s**

### Configuração 2

- `n_estimators = 200`
- `max_depth = 20`

Resultado de validação:

- Accuracy: **96,61%**
- Tempo de treinamento: **16,76 s**

A segunda configuração foi selecionada.

O aumento do número de árvores elevou o custo computacional e produziu uma melhora pequena na Accuracy de validação. A limitação de profundidade também fornece maior controle sobre a complexidade individual das árvores.

O experimento mostrou que aumentar a complexidade computacional não implica necessariamente ganho proporcional de desempenho.

---

## 5. Support Vector Machine — SVM

O SVM apresentou uma característica importante durante o desenvolvimento: **custo computacional significativamente superior no ambiente local**.

Uma primeira execução utilizando aproximadamente 49.000 amostras e kernel linear apresentou:

- Accuracy de validação: **93,34%**
- Tempo de treinamento: **155,91 s**

Também foi observado elevado consumo de memória.

Para viabilizar a comparação de hiperparâmetros sem comprometer a execução local, o ajuste posterior foi realizado utilizando uma **amostra estratificada de 20.000 registros do conjunto de treinamento**.

A estratificação foi mantida para preservar a representação das dez classes.

### Configuração linear

- `kernel = linear`
- `C = 1`

Resultado:

- Accuracy: **92,19%**
- Tempo: **27,52 s**

### Configuração RBF

- `kernel = rbf`
- `C = 10`

Resultado:

- Accuracy: **97,21%**
- Tempo: **32,14 s**

A configuração com kernel RBF foi selecionada.

O resultado indica que, entre as configurações avaliadas, a fronteira não linear produzida pelo kernel RBF foi mais adequada ao problema que a configuração linear.

### Consideração metodológica

Os tempos obtidos durante o ajuste do SVM com 20.000 registros **não constituem uma comparação computacional perfeitamente equivalente** aos modelos treinados com conjuntos maiores.

A redução foi uma decisão operacional decorrente dos recursos computacionais disponíveis e não uma tentativa de favorecer o algoritmo.

Por isso, a diferença no volume de treinamento deve ser considerada ao interpretar principalmente os tempos de execução.

---

## 6. Multilayer Perceptron — MLP

Foram avaliadas duas arquiteturas de rede neural.

### Configuração 1

- camadas ocultas: `(100,)`
- learning rate inicial: `0.001`

Resultado:

- Accuracy: **97,56%**
- Tempo: **17,91 s**

### Configuração 2

- camadas ocultas: `(128, 64)`
- learning rate inicial: `0.0005`

Resultado:

- Accuracy: **97,59%**
- Tempo: **26,26 s**

A segunda configuração foi selecionada.

A arquitetura com duas camadas ocultas oferece maior capacidade de representação de relações não lineares entre os atributos.

Durante o treinamento foi emitido um `ConvergenceWarning`, informando que o limite definido de 30 iterações foi atingido antes da convergência completa do otimizador.

Esse aviso não representa falha na execução. Ele indica que o processo de otimização ainda poderia evoluir caso fosse permitido um número maior de iterações.

Como o modelo já apresentou desempenho competitivo dentro do escopo experimental estabelecido, o limite foi mantido e a ocorrência documentada.

---

## 7. Avaliação Comparativa no Conjunto de Teste

Após a seleção das configurações, os modelos foram avaliados no conjunto de teste independente.

Foram utilizadas:

- Accuracy;
- Precision ponderada;
- Recall ponderado;
- F1-score ponderado.

Os resultados foram:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 96,50% | 96,50% | 96,50% | 96,50% |
| SVM | 97,44% | 97,44% | 97,44% | 97,43% |
| **MLP** | **97,66%** | **97,66%** | **97,66%** | **97,66%** |

A **MLP apresentou o melhor desempenho global** entre as configurações avaliadas.

Entretanto, a diferença entre MLP e SVM é relativamente pequena. Portanto, o resultado não deve ser interpretado como superioridade universal da MLP sobre o SVM, mas como o melhor resultado obtido dentro das condições e configurações deste experimento.

### Desempenho versus custo computacional

Os experimentos também mostraram que aumento de custo computacional não necessariamente produz ganho proporcional de desempenho.

No Random Forest, por exemplo, o aumento de 100 para 200 árvores praticamente dobrou o tempo de treinamento, enquanto o ganho de Accuracy na validação foi de apenas 0,12 ponto percentual.

A seleção de um modelo, portanto, não deve considerar exclusivamente a maior métrica obtida, mas também recursos computacionais, tempo de processamento e contexto de utilização.

---

## 8. Análise das Matrizes de Confusão

Além das métricas consolidadas, foram construídas matrizes de confusão 10 × 10 para analisar os erros entre as classes.

A maior confusão normalizada identificada em cada modelo foi:

| Modelo | Confusão predominante | Taxa |
|---|---|---:|
| Random Forest | 4 → 9 | 2,34% |
| SVM | 4 → 9 | 1,61% |
| MLP | 4 → 9 | 1,90% |

Um resultado relevante é que os três modelos apresentaram a mesma direção predominante de erro: **imagens reais do dígito 4 classificadas como 9**.

Isso sugere que determinadas formas de escrita dos dois dígitos apresentam características visuais suficientemente semelhantes para produzir confusão mesmo em algoritmos com princípios de funcionamento diferentes.

A análise demonstra também por que Accuracy isoladamente não descreve completamente o comportamento de um classificador. Dois modelos com métricas globais semelhantes podem apresentar padrões de erro diferentes entre as classes.

---

## 9. Experimento com Classes Ocultadas

Após a avaliação convencional, foi realizado um experimento específico para analisar o comportamento do classificador diante de **classes ausentes do treinamento**.

Foram selecionados os dígitos:

```text
4 e 7
```

Essas classes foram completamente removidas do conjunto utilizado para treinar uma nova MLP.

O conjunto original de treinamento possuía:

```text
49.000 amostras
```

Após a remoção:

```text
39.118 amostras
```

A nova MLP foi treinada apenas com:

```text
0, 1, 2, 3, 5, 6, 8 e 9
```

A arquitetura selecionada anteriormente foi preservada para que o experimento se concentrasse no efeito da ausência das classes, sem introduzir uma nova mudança de arquitetura.

---

## 10. Avaliação Out-of-Distribution — OOD

A MLP restrita foi testada exclusivamente com imagens reais dos dígitos 4 e 7, ou seja, entradas pertencentes a classes que nunca participaram de seu treinamento.

O conjunto OOD continha:

- 1.365 imagens do dígito 4;
- 1.459 imagens do dígito 7;
- **2.824 imagens no total**.

### Comportamento do dígito 4

Das 1.365 imagens reais do dígito 4:

- 1.194 foram classificadas como 9;
- aproximadamente **87,47%**.

A concentração demonstra que, diante de uma classe desconhecida, o modelo encontrou no dígito 9 a classe conhecida mais compatível com grande parte das representações do dígito 4.

### Comportamento do dígito 7

Das 1.459 imagens reais do dígito 7, as principais classificações foram:

| Classe prevista | Quantidade | Percentual |
|---:|---:|---:|
| 9 | 708 | 48,53% |
| 3 | 463 | 31,73% |
| 2 | 213 | 14,60% |

O comportamento foi menos concentrado do que para o dígito 4, mas novamente o modelo distribuiu as entradas desconhecidas entre as classes que conhecia.

---

## 11. Closed-Set e Falsa Certeza

A MLP utilizada é um **classificador de conjunto fechado (*closed-set classifier*)**.

Isso significa que o modelo parte da premissa de que uma entrada pertence a uma das classes aprendidas durante o treinamento.

Como não existe uma classe “desconhecida” nem um mecanismo explícito de rejeição, o modelo necessariamente atribui cada entrada a uma das classes conhecidas.

Esse comportamento tornou possível investigar a confiança produzida pelo classificador diante de dados OOD.

Os resultados foram:

| Indicador | Resultado |
|---|---:|
| Confiança média | 91,30% |
| Confiança mínima | 29,56% |
| Confiança máxima | 100% |
| Previsões com confiança ≥ 90% | 73,97% |
| Previsões com confiança ≥ 99% | 54,36% |

O resultado mais significativo é que **mais da metade das imagens de classes desconhecidas recebeu uma previsão com confiança igual ou superior a 99%**.

Isso caracteriza um comportamento de **overconfidence**.

Uma probabilidade elevada produzida pelo classificador não significa, por si só, que a entrada pertence ao domínio conhecido pelo modelo.

Em uma aplicação que necessitasse reconhecer entradas desconhecidas, seria necessário acrescentar mecanismos específicos, como limiares de rejeição, calibração de confiança ou técnicas de detecção OOD/open-set.

Esses mecanismos não foram implementados nesta etapa porque o objetivo experimental era justamente observar a resposta natural do classificador quando confrontado com classes que nunca aprendeu.

---

## 12. Inferência com Imagens Externas

O modelo MLP original, treinado com as dez classes, também foi avaliado utilizando imagens externas ao MNIST.

Foram utilizadas cinco representações do dígito 4 produzidas em condições diferentes:

- caligrafia digital;
- desenho realizado no Paint;
- manuscrito fotografado;
- manuscrito digitalizado;
- outra caligrafia manuscrita.

Esse experimento buscou avaliar qualitativamente a capacidade de generalização diante de entradas que não possuem exatamente as mesmas características das imagens originais do MNIST.

### Pré-processamento

Imagens externas podem apresentar diferenças de fundo, iluminação, contraste, escala, enquadramento e posicionamento.

Foi necessário, portanto, construir um fluxo específico de preparação que inclui:

1. conversão para escala de cinza;
2. correção do fundo;
3. identificação da polaridade entre fundo e dígito;
4. identificação da região relevante;
5. recorte do dígito;
6. redimensionamento preservando a proporção;
7. posicionamento em uma imagem 28 × 28;
8. centralização;
9. normalização para `[0,1]`;
10. transformação para 784 atributos.

O objetivo foi aproximar as imagens externas da representação esperada pelo modelo sem alterar artificialmente a identidade visual do dígito.

### Resultados

| Imagem | Classe real | Predição | Confiança | Resultado |
|---|---:|---:|---:|---|
| Caligrafia digital | 4 | 4 | 67,59% | Correto |
| Desenho no Paint | 4 | 4 | 99,28% | Correto |
| Manuscrito fotografado | 4 | 5 | 37,95% | Incorreto |
| Manuscrito digitalizado | 4 | 4 | 96,02% | Correto |
| Outra caligrafia | 4 | 4 | 40,11% | Correto |

O modelo classificou corretamente **4 das 5 imagens**.

Isso corresponde a 80% dentro deste pequeno experimento, mas esse percentual **não deve ser interpretado como uma nova estimativa formal de Accuracy**.

Cinco imagens não constituem uma amostra estatisticamente suficiente para estimar o desempenho geral do modelo em dados externos. O resultado deve ser interpretado como um experimento qualitativo de generalização.

---

## 13. Domain Shift e Generalização

O erro observado na imagem manuscrita fotografada evidencia um fenômeno importante em Machine Learning: **domain shift**.

As imagens do MNIST são altamente padronizadas. Imagens obtidas fora desse ambiente podem apresentar diferenças relacionadas a:

- iluminação;
- fundo;
- contraste;
- espessura do traço;
- escala;
- enquadramento;
- ruído;
- estilo individual de escrita.

Mesmo após o pré-processamento, a distribuição das entradas externas pode permanecer diferente daquela observada durante o treinamento.

Isso explica por que um modelo com **97,66% de Accuracy no conjunto de teste MNIST** pode apresentar comportamento diferente quando recebe imagens produzidas em condições reais.

O experimento demonstra que desempenho elevado no conjunto de teste é uma evidência importante de generalização dentro daquele domínio, mas não garante automaticamente o mesmo comportamento quando a distribuição dos dados muda.

---

## 14. Limitações e Considerações Metodológicas

Os resultados deste projeto devem ser interpretados considerando algumas limitações.

### Recursos computacionais

O ajuste de hiperparâmetros do SVM utilizou uma amostra estratificada de 20.000 registros devido ao custo de processamento observado no ambiente local.

Essa diferença deve ser considerada principalmente nas comparações de tempo de treinamento.

### Convergência da MLP

O limite `max_iter=30` produziu um `ConvergenceWarning`, indicando que o processo de otimização poderia continuar com um número maior de iterações.

O aviso foi mantido e documentado em vez de ser tratado como erro de execução.

### Quantidade de imagens externas

O experimento externo utilizou cinco imagens. Ele permite observar comportamentos relevantes, mas não constitui uma avaliação estatística abrangente de desempenho em dados reais.

### Classificador de conjunto fechado

A MLP não possui mecanismo nativo para declarar uma entrada como desconhecida. O experimento OOD foi desenvolvido justamente para evidenciar as consequências dessa característica.

### Características do MNIST

O MNIST é um dataset controlado e padronizado. Sistemas destinados a operar com imagens reais exigiriam validação adicional utilizando dados mais representativos do ambiente de produção.

Essas limitações não invalidam os resultados obtidos. Elas estabelecem o contexto dentro do qual as conclusões podem ser interpretadas corretamente.

---

## 15. Conclusão Técnica

O projeto permitiu comparar três abordagens distintas de classificação e analisar seus resultados para além de uma única métrica de desempenho.

Entre as configurações avaliadas, a **MLP apresentou o melhor resultado global**, atingindo aproximadamente **97,66% de Accuracy e F1-score ponderado no conjunto de teste**.

A análise das matrizes de confusão mostrou que os três algoritmos apresentaram como principal direção de erro a classificação do dígito **4 como 9**, evidenciando que a análise por classe complementa as métricas consolidadas.

Os experimentos de robustez revelaram uma limitação ainda mais relevante. Quando as classes 4 e 7 foram completamente retiradas do treinamento, a MLP restrita continuou classificando essas entradas como classes conhecidas e frequentemente com confiança elevada. **54,36% das previsões OOD apresentaram confiança igual ou superior a 99%**, apesar de todas pertencerem a classes desconhecidas pelo modelo.

O experimento com imagens externas acrescentou outra perspectiva. Quatro das cinco imagens foram classificadas corretamente, enquanto uma imagem manuscrita fotografada foi classificada incorretamente, demonstrando na prática o impacto que diferenças entre o domínio de treinamento e o domínio de utilização podem exercer sobre a inferência.

Dessa forma, os resultados evidenciam três dimensões complementares na avaliação de um sistema de Machine Learning:

1. **desempenho preditivo em dados conhecidos;**
2. **comportamento diante de dados desconhecidos;**
3. **capacidade de generalização diante de mudanças na distribuição dos dados.**

A principal conclusão técnica é que uma métrica elevada no conjunto de teste, embora necessária para avaliar o modelo, **não é suficiente para caracterizar isoladamente sua robustez**. Matrizes de confusão, análise de erros, confiança das previsões, testes OOD e experimentos com dados externos fornecem informações adicionais essenciais para compreender tanto as capacidades quanto as limitações do classificador.


### Autora

**Cristina Freitas**