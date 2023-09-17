# Classificadores de culinária 1

Nesta lição, você usará o _dataset_ balanceado e tratado que salvou da última lição cheio de dados sobre cozinhas diferentes.

Você usará este _dataset_ com uma variedade de classificadores para _prever uma determinada culinária nacional com base em um grupo de ingredientes_. Enquanto isso, você aprenderá mais sobre algumas das maneiras como os algoritmos podem ser aproveitados para tarefas de classificação.

## [Questionário inicial](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/21?loc=ptbr)

# Preparação

Assumindo que você completou [lição 1](../../1-Introduction/translations/README.pt-br.md), certifique-se de que existe o arquivo _cleaned_cuisines.csv_ na pasta `/data`, pois será usado para todas as lições de classificação.

## Exercício - prevendo uma culinária nacional

1. Usando o _notebook.ipynb_ que está na pasta desta lição, importe a biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Os dados estarão mais ou menos assim:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Agora importe algumas bibliotecas:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divida as coordenadas X e y em dois _dataframes_ para treinamento. `cuisine` pode ser o rótulo do _dataframe_:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    O resultado será mais ou menos assim:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Remova as colunas `Unnamed: 0` e `cuisine`, chamando o método `drop()`. Salve o restante dos dados como características treináveis:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    As características serão mais ou menos assim:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Já podemos treinar nosso modelo!

## Escolhendo um classificador

Agora que seus dados estão tratados e prontos para treinamento, você deve decidir qual algoritmo usar para o trabalho.

O Scikit-learn agrupa classificação em apredizagem supervisionada e, nessa categoria, você encontrará muitas maneiras de classificar. [A variedade](https://scikit-learn.org/stable/supervised_learning.html) é bastante desconcertante à primeira vista. Todos os métodos a seguir incluem técnicas de classificação:

- Modelos Lineares
- Máquinas de vetor de suporte (SVM)
- Descida do gradiente estocástico (SGD)
- Vizinhos mais próximos
- Processos Gaussianos
- Árvores de decisão
- Métodos de conjunto (classificador de votação, _ensemble_)
- Algoritmos de multiclasse e saída múltipla (classificação multiclasse e multilabel, classificação multiclasse-saída múltipla)

> Você também pode usar [redes neurais para classificar dados](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), mas isso está fora do escopo desta lição.

### Qual classificador escolher?

Então, qual classificador você deve escolher? Freqüentemente, percorrer vários e procurar um bom resultado é uma forma de testar. Scikit-learn oferece uma [comparação lado a lado](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) em um _dataset_ criado, comparando com KNeighbors, SVC de duas maneiras, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB e QuadraticDiscrinationAnalysis, mostrando os resultados visualizados:

![comparação de classificadores](../images/comparison.png)
> Plots gerados na documentação do Scikit-learn

> O AutoML resolve esse problema perfeitamente executando essas comparações na nuvem, permitindo que você escolha o melhor algoritmo para seus dados. Teste-o [aqui](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott).

### Uma abordagem melhor

Melhor do que adivinhar, é seguir as ideias nesta [planilha com dicas de ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Aqui, descobrimos que para o nosso problema multiclasse, temos algumas opções:

![planilha com dicas para problemas multiclasse](../images/cheatsheet.png)
> Uma planilha com dicas de algoritmo da Microsoft, detalhando as opções de classificação multiclasse.

✅ Baixe esta planilha, imprima e pendure na parede!

### Raciocinando

Vamos ver se podemos raciocinar através de diferentes abordagens, dadas as restrições que temos:

- **Redes neurais são muito pesadas**. Dado que nosso _dataset_ está tratado mas é pequeno, e o fato de estarmos executando o treinamento localmente por meio de _notebooks_, redes neurais são muito pesadas para essa tarefa.
- **Nenhum classificador de duas classes**. Não usamos um classificador de duas classes, isso exclui o esquema um versus todos (one-vs-all).
- **Árvore de decisão ou regressão logística podem funcionar**. Árvore de decisão pode funcionar, ou regressão logística para dados multiclasse.
- **Árvores de decisão impulsionadas à multiclasse resolvem um problema diferente**. Árvore de decisão impulsionada à multiclasse é mais adequada para tarefas não paramétricas, por exemplo, tarefas destinadas a construir _rankings_, por isso não são úteis para nós.

### Usando Scikit-learn 

Usaremos o Scikit-learn para analisar nossos dados. No entanto, existem muitas maneiras de usar a regressão logística no Scikit-learn. Dê uma olhada nos [possíveis parâmetros](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression). 

Essencialmente, existem dois parâmetros importantes - `multi_class` e `solver` -, que precisamos especificar quando pedimos ao Scikit-learn para realizar uma regressão logística. O valor de `multi_class` aplica um certo comportamento. O valor de `solver` é o algoritmo a ser usado. Nem todos os valores de `solver` podem ser combinados com os valores de `multi_class`.

De acordo com a documentação, no caso de multiclasse, o algoritmo de treinamento:

- **Usa o esquema one-vs-rest (OvR, ou "um contra o resto")**, se a opção `multi_class` estiver definida como `ovr`.
- **Usa a perda de entropia cruzada**, se a opção `multi_class` estiver definida como `multinomial` (atualmente, a opção `multinomial` é compatível apenas com os "_solvers_" (ou solucionadores): `lbfgs`, `sag`, `saga` e `newton-cg`).

> 🎓 O 'esquema' aqui pode ser 'ovr' ou 'multinomial'. Uma vez que a regressão logística é realmente projetada para oferecer suporte à classificação binária, esses esquemas permitem um melhor tratamento das tarefas de classificação multiclasse. [Fonte](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/).

> 🎓 O 'solucionador' é definido como "o algoritmo a ser usado no problema de otimização". [Fonte](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

O Scikit-learn oferece esta tabela para explicar como os solucionadores lidam com diferentes desafios apresentados por diferentes tipos de estruturas de dados:

![Solucionadores](../images/solvers.png)

## Exercício - dividindo os dados

Podemos nos concentrar na regressão logística para nosso primeiro teste de treinamento, uma vez que você aprendeu recentemente sobre o último em uma lição anterior.
Divida seus dados em grupos de treinamento e teste chamando o método `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercício - aplicando regressão logística

Já que estamos usando um caso multiclasse, você precisa escolher qual _scheme_ usar e qual _solver_ definir. Use LogisticRegression com uma configuração multiclasse e o solucionador **liblinear** para treinar.

1. Crie uma regressão logística com multi_class definido como `ovr` e solver definido como `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Teste diferentes _solvers_ como o `lbfgs`, que já é definido como padrão.

    > Use a função Pandas chamada [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) para nivelar seus dados quando necessário.

    A acurácia está boa quando é maior que **80%**!

1. Você pode ver este modelo em ação testando dessa forma (linha #50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    O resultado será:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Tente um número de linha diferente e verifique os resultados.

1. Indo mais fundo, você pode verificar a precisão desta previsão:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    A culinária indiana é seu melhor palpite, com boa probabilidade:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Você pode explicar por que a modelo tem certeza de que se trata de uma culinária indiana?

1. Obtenha mais detalhes imprimindo um relatório de classificação, como você fez nas aulas de regressão:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Desafio

Nesta lição, você usou seus dados para construir um modelo de aprendizado de máquina que pode prever uma culinária nacional com base em uma série de ingredientes. Reserve algum tempo para ler as opções que o Scikit-learn oferece para classificar dados. Aprofunde-se no conceito de 'solucionador' para entender o que acontece nos bastidores.

## [Questionário para fixação](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/22?loc=ptbr)

## Revisão e Auto Aprendizagem

Aprofunde-se um pouco mais na matemática por trás da regressão logística [nesta lição](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf).

## Tarefa

[Estudando solucionadores](assignment.pt-br.md).
