# Classificadores de culinária 2

Nesta segunda lição de classificação, você explorará outras maneiras de classificar dados numéricos. Você também aprenderá sobre as ramificações para escolher um classificador em vez de outro.

## [Questionário inicial](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23?loc=ptbr)

### Pré-requisito

Presumimos que você tenha concluído as lições anteriores e tenha um arquivo com o _dataset_ em sua pasta `data` chamado _cleaned_cuisines.csv_.

### Preparação

Carregando o arquivo _notebook.ipynb_ com o _dataset_ e o dividimos em dataframes X e y, estamos prontos para o processo de construção do modelo.

## Um mapa de classificação

Anteriormente, você aprendeu sobre as várias opções para classificar dados usando a planilha da Microsoft. O Scikit-learn oferece uma planilha semelhante, com mais informações, que pode ajudar ainda mais a restringir seus estimadores (outro termo para classificadores):

![Mapa da ML do Scikit-learn](../images/map.png)
> Dica: [visite este site](https://scikit-learn.org/stable/tutorial/machine_learning_map/) para ler a documentação.

### O plano

Este mapa é muito útil, uma vez que você tenha uma compreensão clara de seus dados, pois você pode 'andar' no mapa ao longo dos caminhos para então, tomar uma decisão:

- Temos mais que 50 amostras
- Queremos prever uma categoria
- Nós rotulamos os dados
- Temos menos de 100 mil amostras
- ✨ Podemos escolher um SVC linear
- Se isso não funcionar, já que temos dados numéricos:
     - Podemos tentar um classificador KNeighbors ✨
       - Se não funcionar, tente o ✨ SVC e ✨ Classificadores de conjunto (ensemble)

Esta é uma trilha muito útil a seguir.

## Exercício - dividindo os dados

Seguindo este caminho, devemos começar importando algumas bibliotecas.

1. Importe essas bibliotecas:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Divida os dados em dados de treinamento e teste:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Classificador linear SVC

O Clustering de Vetores de Suporte (SVC, ou no inglês, Support-Vector clustering) é um filho da família de máquinas de vetores de suporte de técnicas de ML. Neste método, você pode escolher um 'kernel' para decidir como agrupar os rótulos. O parâmetro 'C' refere-se à 'regularização' que regula a influência dos parâmetros. O kernel pode ser um de [vários](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); aqui, nós o definimos como 'linear' para garantir impulsionar o classificador. O padrão de probabilidade é 'false'; aqui, nós o definimos como 'true' para reunir estimativas de probabilidade. Definimos random_state como '0' para embaralhar os dados e obter probabilidades.

### Exercício - aplicando um SVC linear

Comece criando um array de classificadores. Você adicionará itens progressivamente a este array enquanto testamos.

1. Comece com um SVC linear:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Treine seu modelo usando o SVC e imprima um relatório:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    O resultado é bom:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Classificador K-Neighbors

K-Neighbors faz parte da família "neighbors" de métodos de ML, que podem ser usados para aprendizado supervisionado e não supervisionado. Neste método, um número predefinido de pontos é criado e os dados são reunidos em torno desses pontos de modo que rótulos generalizados podem ser previstos para os dados.

### Exercício - aplicando o classificador K-Neighbors

O classificador anterior era bom e funcionou bem com os dados, mas talvez possamos obter uma melhor acurácia. Experimente um classificador K-Neighbors.

1. Adicione uma linha ao seu array de classificadores (adicione uma vírgula após o item do SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    O resultado é um pouco pior:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Aprenda mais sobre [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Classificador de Vetores de Suporte

Os Classificadores de Vetores de Suporte (SVM, ou no inglês, Support-Vector Machine) fazem parte da família [Classificadores de Vetores de Suporte](https://wikipedia.org/wiki/Support-vector_machine) de métodos de ML que são usados para tarefas de classificação e regressão. Os SVMs "mapeiam exemplos de treinamento para pontos no espaço" para maximizar a distância entre duas categorias. Os dados subsequentes são mapeados neste espaço para que sua categoria possa ser prevista.

###  Exercício - aplicando o Classificador de Vetores de Suporte

Vamos tentar aumentar a acurácia com um Classificador de Vetores de Suporte.

1. Adicione uma vírgula após o item K-Neighbors e, em seguida, adicione esta linha:

    ```python
    'SVC': SVC(),
    ```

    O resultado é muito bom!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Aprenda mais sobre [Vetores de Suporte](https://scikit-learn.org/stable/modules/svm.html#svm)

## Classificadores de conjunto (ensemble)

Vamos seguir o caminho até o fim, embora o teste anterior tenha sido muito bom. Vamos tentar alguns 'Classificadores de conjunto, especificamente Random Forest (Árvores Aleatórias) e AdaBoost:

```python
'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

O resultado é muito bom, especialmente para Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Aprenda mais sobre [Classificadores de conjunto](https://scikit-learn.org/stable/modules/ensemble.html)

Este método de arendizado de máquina "combina as previsões de vários estimadores de base" para melhorar a qualidade do modelo. Em nosso exemplo, usamos Random Forest e AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), um método de média, constrói uma 'floresta' de 'árvores de decisão' infundidas com aleatoriedade para evitar _overfitting_. O parâmetro `n_estimators` define a quantidade de árvores.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajusta um classificador a um _dataset_ e, em seguida, ajusta cópias desse classificador ao mesmo _dataset_. Ele se concentra nos pesos dos itens classificados incorretamente e corrige o ajuste para o próximo classificador.

---

## 🚀Desafio

Cada uma dessas técnicas possui um grande número de parâmetros. Pesquise os parâmetros padrão de cada um e pense no que o ajuste desses parâmetros significaria para a qualidade do modelo.

## [Questionário para fixação](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24?loc=ptbr)

## Revisão e Auto Aprendizagem

Há muitos termos nessas lições, então reserve um minuto para revisar [esta lista útil](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) sobre terminologias!

## Tarefa

[Brincando com parâmetros](assignment.pt-br.md).
