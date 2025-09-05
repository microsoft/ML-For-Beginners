<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T16:19:16+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "ro"
}
-->
# Clasificatori de bucătărie 1

În această lecție, vei folosi setul de date salvat din lecția anterioară, plin de date echilibrate și curate despre bucătării.

Vei utiliza acest set de date cu o varietate de clasificatori pentru a _prezice o bucătărie națională dată pe baza unui grup de ingrediente_. În timp ce faci acest lucru, vei învăța mai multe despre unele dintre modurile în care algoritmii pot fi utilizați pentru sarcini de clasificare.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)
# Pregătire

Presupunând că ai finalizat [Lecția 1](../1-Introduction/README.md), asigură-te că un fișier _cleaned_cuisines.csv_ există în folderul rădăcină `/data` pentru aceste patru lecții.

## Exercițiu - prezicerea unei bucătării naționale

1. Lucrând în folderul _notebook.ipynb_ al acestei lecții, importă acel fișier împreună cu biblioteca Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Datele arată astfel:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Acum, importă mai multe biblioteci:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Împarte coordonatele X și y în două cadre de date pentru antrenament. `cuisine` poate fi cadrul de date pentru etichete:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Va arăta astfel:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Elimină coloana `Unnamed: 0` și coloana `cuisine`, folosind `drop()`. Salvează restul datelor ca caracteristici antrenabile:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Caracteristicile tale arată astfel:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Acum ești gata să îți antrenezi modelul!

## Alegerea clasificatorului

Acum că datele tale sunt curate și pregătite pentru antrenament, trebuie să decizi ce algoritm să folosești pentru sarcină. 

Scikit-learn grupează clasificarea sub Învățare Supervizată, iar în această categorie vei găsi multe moduri de a clasifica. [Varietatea](https://scikit-learn.org/stable/supervised_learning.html) poate fi copleșitoare la prima vedere. Metodele următoare includ tehnici de clasificare:

- Modele liniare
- Mașini cu vectori de suport
- Gradient descendent stochastic
- Vecini cei mai apropiați
- Procese Gaussiene
- Arbori de decizie
- Metode de ansamblu (clasificator prin vot)
- Algoritmi multiclasă și multioutput (clasificare multiclasă și multilabel, clasificare multiclasă-multioutput)

> Poți folosi și [rețele neuronale pentru a clasifica date](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), dar acest subiect este în afara scopului acestei lecții.

### Ce clasificator să alegi?

Deci, ce clasificator ar trebui să alegi? Deseori, testarea mai multor clasificatori și căutarea unui rezultat bun este o metodă de testare. Scikit-learn oferă o [comparație alăturată](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pe un set de date creat, comparând KNeighbors, SVC în două moduri, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB și QuadraticDiscriminationAnalysis, arătând rezultatele vizualizate: 

![comparație clasificatori](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafice generate din documentația Scikit-learn

> AutoML rezolvă această problemă elegant prin rularea acestor comparații în cloud, permițându-ți să alegi cel mai bun algoritm pentru datele tale. Încearcă [aici](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### O abordare mai bună

O metodă mai bună decât ghicitul aleatoriu este să urmezi ideile din acest [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) descărcabil. Aici descoperim că, pentru problema noastră multiclasă, avem câteva opțiuni:

![cheatsheet pentru probleme multiclasă](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> O secțiune din Algorithm Cheat Sheet de la Microsoft, detaliind opțiuni de clasificare multiclasă

✅ Descarcă acest cheat sheet, printează-l și agață-l pe perete!

### Raționament

Să vedem dacă putem raționa prin diferite abordări, având în vedere constrângerile pe care le avem:

- **Rețelele neuronale sunt prea grele**. Având în vedere setul nostru de date curat, dar minimal, și faptul că rulăm antrenamentul local prin notebook-uri, rețelele neuronale sunt prea complexe pentru această sarcină.
- **Nu folosim clasificatori cu două clase**. Nu utilizăm un clasificator cu două clase, deci excludem metoda one-vs-all. 
- **Arborele de decizie sau regresia logistică ar putea funcționa**. Un arbore de decizie ar putea funcționa, sau regresia logistică pentru date multiclasă. 
- **Arborii de decizie boostați multiclasă rezolvă o altă problemă**. Arborele de decizie boostat multiclasă este cel mai potrivit pentru sarcini nonparametrice, de exemplu sarcini concepute pentru a construi clasamente, deci nu este util pentru noi.

### Utilizarea Scikit-learn 

Vom folosi Scikit-learn pentru a analiza datele noastre. Totuși, există multe moduri de a utiliza regresia logistică în Scikit-learn. Consultă [parametrii de transmis](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Practic, există doi parametri importanți - `multi_class` și `solver` - pe care trebuie să îi specificăm atunci când cerem Scikit-learn să efectueze o regresie logistică. Valoarea `multi_class` aplică un anumit comportament. Valoarea solverului indică ce algoritm să se folosească. Nu toți solverii pot fi combinați cu toate valorile `multi_class`.

Conform documentației, în cazul multiclasă, algoritmul de antrenament:

- **Folosește schema one-vs-rest (OvR)**, dacă opțiunea `multi_class` este setată la `ovr`
- **Folosește pierderea cross-entropy**, dacă opțiunea `multi_class` este setată la `multinomial`. (În prezent, opțiunea `multinomial` este suportată doar de solverii ‘lbfgs’, ‘sag’, ‘saga’ și ‘newton-cg’.)"

> 🎓 'Schema' aici poate fi 'ovr' (one-vs-rest) sau 'multinomial'. Deoarece regresia logistică este concepută pentru a susține clasificarea binară, aceste scheme îi permit să gestioneze mai bine sarcinile de clasificare multiclasă. [sursa](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solverul' este definit ca "algoritmul utilizat în problema de optimizare". [sursa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn oferă acest tabel pentru a explica modul în care solverii gestionează diferite provocări prezentate de diferite structuri de date:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Exercițiu - împarte datele

Ne putem concentra pe regresia logistică pentru primul nostru test de antrenament, deoarece ai învățat recent despre aceasta într-o lecție anterioară.
Împarte datele tale în grupuri de antrenament și testare, apelând `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercițiu - aplică regresia logistică

Deoarece folosești cazul multiclasă, trebuie să alegi ce _schemă_ să folosești și ce _solver_ să setezi. Folosește LogisticRegression cu o setare multiclasă și solverul **liblinear** pentru antrenament.

1. Creează o regresie logistică cu `multi_class` setat la `ovr` și solverul setat la `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Încearcă un alt solver, cum ar fi `lbfgs`, care este adesea setat ca implicit
> Notă, folosește funcția Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) pentru a aplatiza datele tale atunci când este necesar.
Acuratețea este bună la peste **80%**!

1. Poți vedea acest model în acțiune testând un rând de date (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultatul este afișat:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Încearcă un alt număr de rând și verifică rezultatele

1. Explorând mai profund, poți verifica acuratețea acestei predicții:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultatul este afișat - bucătăria indiană este cea mai probabilă, cu o bună probabilitate:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Poți explica de ce modelul este destul de sigur că aceasta este o bucătărie indiană?

1. Obține mai multe detalii prin afișarea unui raport de clasificare, așa cum ai făcut în lecțiile despre regresie:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precizie | recall | f1-score | suport |
    | ------------ | -------- | ------ | -------- | ------ |
    | chinese      | 0.73     | 0.71   | 0.72     | 229    |
    | indian       | 0.91     | 0.93   | 0.92     | 254    |
    | japanese     | 0.70     | 0.75   | 0.72     | 220    |
    | korean       | 0.86     | 0.76   | 0.81     | 242    |
    | thai         | 0.79     | 0.85   | 0.82     | 254    |
    | acuratețe    | 0.80     | 1199   |          |        |
    | macro avg    | 0.80     | 0.80   | 0.80     | 1199   |
    | weighted avg | 0.80     | 0.80   | 0.80     | 1199   |

## 🚀Provocare

În această lecție, ai folosit datele curățate pentru a construi un model de învățare automată care poate prezice o bucătărie națională pe baza unei serii de ingrediente. Ia-ți timp să explorezi numeroasele opțiuni pe care Scikit-learn le oferă pentru clasificarea datelor. Explorează mai profund conceptul de 'solver' pentru a înțelege ce se întâmplă în culise.

## [Chestionar post-lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare & Studiu Individual

Explorează puțin mai mult matematica din spatele regresiei logistice în [această lecție](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Temă 

[Studiază solvers](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.