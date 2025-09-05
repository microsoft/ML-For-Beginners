<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T08:47:46+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "pt"
}
-->
# Classificadores de culinária 2

Nesta segunda lição sobre classificação, vais explorar mais formas de classificar dados numéricos. Também vais aprender sobre as implicações de escolher um classificador em vez de outro.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

### Pré-requisitos

Assumimos que completaste as lições anteriores e tens um conjunto de dados limpo na tua pasta `data`, chamado _cleaned_cuisines.csv_, na raiz desta pasta de 4 lições.

### Preparação

Carregámos o teu ficheiro _notebook.ipynb_ com o conjunto de dados limpo e dividimo-lo em dataframes X e y, prontos para o processo de construção do modelo.

## Um mapa de classificação

Anteriormente, aprendeste sobre as várias opções disponíveis para classificar dados utilizando o guia da Microsoft. O Scikit-learn oferece um guia semelhante, mas mais detalhado, que pode ajudar ainda mais a restringir os estimadores (outro termo para classificadores):

![Mapa de ML do Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Dica: [visita este mapa online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) e explora os caminhos para ler a documentação.

### O plano

Este mapa é muito útil quando tens uma compreensão clara dos teus dados, pois podes 'percorrer' os seus caminhos até uma decisão:

- Temos >50 amostras
- Queremos prever uma categoria
- Temos dados rotulados
- Temos menos de 100K amostras
- ✨ Podemos escolher um Linear SVC
- Se isso não funcionar, como temos dados numéricos:
    - Podemos tentar um ✨ KNeighbors Classifier 
      - Se isso não funcionar, tentar ✨ SVC e ✨ Ensemble Classifiers

Este é um caminho muito útil a seguir.

## Exercício - dividir os dados

Seguindo este caminho, devemos começar por importar algumas bibliotecas para usar.

1. Importa as bibliotecas necessárias:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Divide os teus dados de treino e teste:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Classificador Linear SVC

O clustering por Support-Vector (SVC) é um membro da família de técnicas de ML chamadas Support-Vector Machines (aprende mais sobre estas abaixo). Neste método, podes escolher um 'kernel' para decidir como agrupar os rótulos. O parâmetro 'C' refere-se à 'regularização', que regula a influência dos parâmetros. O kernel pode ser um de [vários](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); aqui definimos como 'linear' para garantir que utilizamos o Linear SVC. A probabilidade por padrão é 'falsa'; aqui definimos como 'verdadeira' para obter estimativas de probabilidade. Definimos o estado aleatório como '0' para embaralhar os dados e obter probabilidades.

### Exercício - aplicar um Linear SVC

Começa por criar um array de classificadores. Vais adicionar progressivamente a este array à medida que testamos.

1. Começa com um Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Treina o teu modelo utilizando o Linear SVC e imprime um relatório:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    O resultado é bastante bom:

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

K-Neighbors faz parte da família de métodos de ML "neighbors", que podem ser usados tanto para aprendizagem supervisionada como não supervisionada. Neste método, é criado um número pré-definido de pontos e os dados são agrupados em torno desses pontos, de forma que rótulos generalizados possam ser previstos para os dados.

### Exercício - aplicar o classificador K-Neighbors

O classificador anterior foi bom e funcionou bem com os dados, mas talvez possamos obter uma melhor precisão. Experimenta um classificador K-Neighbors.

1. Adiciona uma linha ao teu array de classificadores (adiciona uma vírgula após o item Linear SVC):

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

    ✅ Aprende sobre [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Classificador Support Vector

Os classificadores Support-Vector fazem parte da família [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) de métodos de ML que são usados para tarefas de classificação e regressão. Os SVMs "mapeiam exemplos de treino para pontos no espaço" para maximizar a distância entre duas categorias. Dados subsequentes são mapeados neste espaço para que a sua categoria possa ser prevista.

### Exercício - aplicar um classificador Support Vector

Vamos tentar obter uma precisão um pouco melhor com um classificador Support Vector.

1. Adiciona uma vírgula após o item K-Neighbors e, em seguida, adiciona esta linha:

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

    ✅ Aprende sobre [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Classificadores Ensemble

Vamos seguir o caminho até ao fim, mesmo que o teste anterior tenha sido muito bom. Vamos experimentar alguns 'Classificadores Ensemble', especificamente Random Forest e AdaBoost:

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

✅ Aprende sobre [Classificadores Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Este método de Machine Learning "combina as previsões de vários estimadores base" para melhorar a qualidade do modelo. No nosso exemplo, utilizámos Random Trees e AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), um método de média, constrói uma 'floresta' de 'árvores de decisão' com infusão de aleatoriedade para evitar overfitting. O parâmetro n_estimators é definido como o número de árvores.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajusta um classificador a um conjunto de dados e, em seguida, ajusta cópias desse classificador ao mesmo conjunto de dados. Foca-se nos pesos dos itens classificados incorretamente e ajusta o ajuste para o próximo classificador corrigir.

---

## 🚀Desafio

Cada uma destas técnicas tem um grande número de parâmetros que podes ajustar. Pesquisa os parâmetros padrão de cada uma e pensa no que ajustar esses parâmetros significaria para a qualidade do modelo.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Autoestudo

Há muito jargão nestas lições, por isso tira um momento para rever [esta lista](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) de terminologia útil!

## Tarefa 

[Exploração de parâmetros](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.