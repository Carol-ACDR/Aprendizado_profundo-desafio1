# Detecção de Falhas no Sistema APS com Rede Neural

Este projeto utiliza uma rede neural para identificar falhas no sistema de pressão de ar (APS) de veículos pesados. O problema é tratado como uma classificação binária e altamente desbalanceada, em que deixar de detectar uma falha tem custo muito superior ao de gerar um alarme falso.

## Notebook

- `deteccao_falhas_aps.ipynb`

## Objetivo

Classificar cada amostra como:

- `neg`: ausência de falha relacionada ao APS;
- `pos`: presença de falha relacionada ao APS.

Além de precisão, recall e F1-score, o notebook utiliza a seguinte função de custo:

```text
custo = 10 × falsos positivos + 500 × falsos negativos
```

Essa função reflete a maior gravidade de uma falha real não detectada.

## Dataset

O notebook espera os seguintes arquivos no mesmo diretório:

```text
aps_failure_training_set.csv
aps_failure_test_set.csv
```

Os valores ausentes são representados pela string `na`. O conjunto de teste registrado possui 16.000 amostras, sendo 375 positivas e 15.625 negativas.

> Os datasets não estão incluídos neste repositório. Antes de distribuí-los, verifique sua origem, licença e condições de uso.

## Pipeline

O fluxo implementado contém as seguintes etapas:

1. leitura dos conjuntos de treino e teste;
2. conversão das variáveis para formato numérico;
3. remoção de atributos com grande quantidade de valores ausentes;
4. separação estratificada de 20% do treino para validação;
5. imputação de valores ausentes com `KNNImputer`;
6. padronização com `StandardScaler`;
7. balanceamento apenas do treino com SMOTE;
8. treinamento de uma rede neural com early stopping;
9. avaliação no conjunto de teste;
10. gravação da matriz de confusão em `resultados/cm_RedeNeural1.csv`.

O imputador e o normalizador são ajustados exclusivamente sobre os dados de treinamento, reduzindo o risco de vazamento de informação.

## Arquitetura da rede

A rede neural utiliza:

- camada densa com 64 neurônios;
- batch normalization e ativação ReLU;
- dropout de 20%;
- camada densa com 16 neurônios;
- batch normalization e ativação ReLU;
- saída sigmoide para classificação binária.

O treinamento usa Adam, binary cross-entropy, lotes de 64 amostras e até 50 épocas. O early stopping monitora o recall de validação com paciência de sete épocas.

## Resultados registrados

Os resultados abaixo foram extraídos da última execução salva no notebook:

| Métrica | Classe negativa | Classe positiva |
|---|---:|---:|
| Precisão | 1,00 | 0,43 |
| Recall | 0,97 | 0,93 |
| F1-score | 0,98 | 0,59 |

- Acurácia global: aproximadamente `0,97`;
- custo calculado: `17.140`.

O recall elevado da classe positiva indica que o modelo detectou a maior parte das falhas. A precisão positiva mais baixa mostra que ainda há uma quantidade relevante de alarmes falsos, comportamento esperado em parte pela prioridade dada à redução de falsos negativos.

Esses números documentam a execução já armazenada no notebook e não foram recalculados durante a preparação deste repositório.

## Como executar

1. Instale o Python 3.10 ou superior.
2. Coloque os dois arquivos CSV no mesmo diretório do notebook.
3. Instale as dependências:

```bash
pip install jupyter pandas numpy scikit-learn imbalanced-learn tensorflow xgboost lightgbm
```

4. Execute:

```bash
jupyter notebook deteccao_falhas_aps.ipynb
```

O treinamento pode ser executado em CPU, mas uma GPU compatível com TensorFlow reduz o tempo necessário.

## Arquivos gerados

Ao ser executado, o notebook cria:

```text
resultados/
└── cm_RedeNeural1.csv
```

O CSV contém a matriz de confusão obtida no conjunto de teste.


## Tecnologias

- Python;
- pandas e NumPy;
- scikit-learn;
- imbalanced-learn;
- TensorFlow/Keras;
- XGBoost e LightGBM.

## Licença

Nenhuma licença foi definida para este projeto. Adicione um arquivo `LICENSE` antes de permitir reutilização ou redistribuição do código.
