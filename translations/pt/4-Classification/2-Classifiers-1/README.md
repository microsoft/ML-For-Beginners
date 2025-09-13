<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T08:46:23+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "pt"
}
-->
# Classificadores de culinária 1

Nesta lição, vais utilizar o conjunto de dados que guardaste na última lição, cheio de dados equilibrados e limpos sobre culinárias.

Vais usar este conjunto de dados com uma variedade de classificadores para _prever uma culinária nacional com base num grupo de ingredientes_. Enquanto fazes isso, vais aprender mais sobre algumas das formas como os algoritmos podem ser utilizados para tarefas de classificação.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)
# Preparação

Assumindo que completaste [Lição 1](../1-Introduction/README.md), certifica-te de que existe um ficheiro _cleaned_cuisines.csv_ na pasta raiz `/data` para estas quatro lições.

## Exercício - prever uma culinária nacional

1. Trabalhando na pasta _notebook.ipynb_ desta lição, importa esse ficheiro juntamente com a biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Os dados têm este aspeto:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Agora, importa mais algumas bibliotecas:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divide as coordenadas X e y em dois dataframes para treino. `cuisine` pode ser o dataframe de etiquetas:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Vai ter este aspeto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Remove a coluna `Unnamed: 0` e a coluna `cuisine`, utilizando `drop()`. Guarda o resto dos dados como características treináveis:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    As tuas características têm este aspeto:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Agora estás pronto para treinar o teu modelo!

## Escolher o classificador

Agora que os teus dados estão limpos e prontos para treino, tens de decidir qual o algoritmo a usar para o trabalho.

O Scikit-learn agrupa a classificação sob Aprendizagem Supervisionada, e nessa categoria vais encontrar muitas formas de classificar. [A variedade](https://scikit-learn.org/stable/supervised_learning.html) pode parecer confusa à primeira vista. Os seguintes métodos incluem técnicas de classificação:

- Modelos Lineares
- Máquinas de Vetores de Suporte
- Descida de Gradiente Estocástica
- Vizinhos Mais Próximos
- Processos Gaussianos
- Árvores de Decisão
- Métodos de Ensemble (classificador por votação)
- Algoritmos multiclasses e multioutput (classificação multiclasses e multilabel, classificação multiclasses-multioutput)

> Também podes usar [redes neuronais para classificar dados](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), mas isso está fora do âmbito desta lição.

### Qual classificador escolher?

Então, qual classificador deves escolher? Muitas vezes, testar vários e procurar um bom resultado é uma forma de experimentar. O Scikit-learn oferece uma [comparação lado a lado](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) num conjunto de dados criado, comparando KNeighbors, SVC de duas formas, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB e QuadraticDiscriminationAnalysis, mostrando os resultados visualizados:

![comparação de classificadores](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Gráficos gerados na documentação do Scikit-learn

> O AutoML resolve este problema de forma prática ao realizar estas comparações na nuvem, permitindo-te escolher o melhor algoritmo para os teus dados. Experimenta [aqui](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Uma abordagem melhor

Uma forma melhor do que adivinhar aleatoriamente é seguir as ideias deste [guia de consulta de ML](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) que podes descarregar. Aqui, descobrimos que, para o nosso problema multiclasses, temos algumas opções:

![guia para problemas multiclasses](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Uma secção do Guia de Algoritmos da Microsoft, detalhando opções de classificação multiclasses

✅ Descarrega este guia, imprime-o e pendura-o na tua parede!

### Raciocínio

Vamos ver se conseguimos raciocinar sobre diferentes abordagens dadas as restrições que temos:

- **Redes neuronais são demasiado pesadas**. Dado o nosso conjunto de dados limpo, mas minimalista, e o facto de estarmos a realizar o treino localmente via notebooks, redes neuronais são demasiado pesadas para esta tarefa.
- **Sem classificadores de duas classes**. Não usamos um classificador de duas classes, o que exclui o one-vs-all.
- **Árvore de decisão ou regressão logística podem funcionar**. Uma árvore de decisão pode funcionar, ou regressão logística para dados multiclasses.
- **Árvores de decisão impulsionadas multiclasses resolvem outro problema**. A árvore de decisão impulsionada multiclasses é mais adequada para tarefas não paramétricas, como tarefas projetadas para criar rankings, por isso não é útil para nós.

### Usar Scikit-learn 

Vamos usar o Scikit-learn para analisar os nossos dados. No entanto, existem muitas formas de usar regressão logística no Scikit-learn. Dá uma olhada nos [parâmetros a passar](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Basicamente, há dois parâmetros importantes - `multi_class` e `solver` - que precisamos especificar quando pedimos ao Scikit-learn para realizar uma regressão logística. O valor de `multi_class` aplica um certo comportamento. O valor do solver é o algoritmo a usar. Nem todos os solvers podem ser combinados com todos os valores de `multi_class`.

De acordo com a documentação, no caso multiclasses, o algoritmo de treino:

- **Usa o esquema one-vs-rest (OvR)**, se a opção `multi_class` estiver definida como `ovr`
- **Usa a perda de entropia cruzada**, se a opção `multi_class` estiver definida como `multinomial`. (Atualmente, a opção `multinomial` é suportada apenas pelos solvers ‘lbfgs’, ‘sag’, ‘saga’ e ‘newton-cg’.)

> 🎓 O 'esquema' aqui pode ser 'ovr' (one-vs-rest) ou 'multinomial'. Como a regressão logística foi projetada para suportar classificação binária, esses esquemas permitem que ela lide melhor com tarefas de classificação multiclasses. [fonte](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 O 'solver' é definido como "o algoritmo a usar no problema de otimização". [fonte](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

O Scikit-learn oferece esta tabela para explicar como os solvers lidam com diferentes desafios apresentados por diferentes tipos de estruturas de dados:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Exercício - dividir os dados

Podemos focar-nos na regressão logística para o nosso primeiro teste de treino, já que aprendeste sobre ela recentemente numa lição anterior.
Divide os teus dados em grupos de treino e teste chamando `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercício - aplicar regressão logística

Como estás a usar o caso multiclasses, precisas de escolher que _esquema_ usar e que _solver_ definir. Usa LogisticRegression com uma configuração multiclasses e o solver **liblinear** para treinar.

1. Cria uma regressão logística com multi_class definida como `ovr` e o solver definido como `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Experimenta um solver diferente como `lbfgs`, que muitas vezes é definido como padrão
> Nota, utilize a função Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) para achatar os seus dados quando necessário.
A precisão é boa, acima de **80%**!

1. Pode ver este modelo em ação ao testar uma linha de dados (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    O resultado é impresso:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Experimente um número de linha diferente e verifique os resultados.

1. Explorando mais a fundo, pode verificar a precisão desta previsão:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    O resultado é impresso - cozinha indiana é a melhor estimativa, com boa probabilidade:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Consegue explicar por que o modelo tem tanta certeza de que se trata de uma cozinha indiana?

1. Obtenha mais detalhes imprimindo um relatório de classificação, como fez nas lições de regressão:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precisão | recall | f1-score | suporte |
    | ------------ | -------- | ------ | -------- | ------- |
    | chinese      | 0.73     | 0.71   | 0.72     | 229     |
    | indian       | 0.91     | 0.93   | 0.92     | 254     |
    | japanese     | 0.70     | 0.75   | 0.72     | 220     |
    | korean       | 0.86     | 0.76   | 0.81     | 242     |
    | thai         | 0.79     | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80     | 1199   |          |         |
    | macro avg    | 0.80     | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80     | 0.80   | 0.80     | 1199    |

## 🚀Desafio

Nesta lição, utilizou os seus dados limpos para construir um modelo de aprendizagem automática que pode prever uma cozinha nacional com base numa série de ingredientes. Dedique algum tempo a explorar as muitas opções que o Scikit-learn oferece para classificar dados. Aprofunde o conceito de 'solver' para entender o que acontece nos bastidores.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Individual

Explore um pouco mais a matemática por trás da regressão logística nesta [lição](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf).
## Tarefa 

[Estude os solvers](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.