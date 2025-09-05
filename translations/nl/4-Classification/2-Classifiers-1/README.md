<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T19:48:53+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "nl"
}
-->
# Categorieën van keukens 1

In deze les gebruik je de dataset die je in de vorige les hebt opgeslagen, vol met gebalanceerde, schone gegevens over keukens.

Je zult deze dataset gebruiken met verschillende classificatiemethoden om _een nationale keuken te voorspellen op basis van een groep ingrediënten_. Terwijl je dit doet, leer je meer over enkele manieren waarop algoritmen kunnen worden ingezet voor classificatietaken.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)
# Voorbereiding

Als je [Les 1](../1-Introduction/README.md) hebt voltooid, zorg er dan voor dat een bestand genaamd _cleaned_cuisines.csv_ zich bevindt in de root `/data` map voor deze vier lessen.

## Oefening - voorspel een nationale keuken

1. Werk in de map _notebook.ipynb_ van deze les en importeer dat bestand samen met de Pandas-bibliotheek:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    De gegevens zien er als volgt uit:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importeer nu nog enkele andere bibliotheken:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Verdeel de X- en y-coördinaten in twee dataframes voor training. `cuisine` kan de labels-dataset zijn:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Het ziet er als volgt uit:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Laat die `Unnamed: 0`-kolom en de `cuisine`-kolom vallen door `drop()` aan te roepen. Sla de rest van de gegevens op als trainbare kenmerken:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Je kenmerken zien er als volgt uit:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nu ben je klaar om je model te trainen!

## Het kiezen van je classifier

Nu je gegevens schoon en klaar zijn voor training, moet je beslissen welk algoritme je voor de taak wilt gebruiken.

Scikit-learn groepeert classificatie onder Gecontroleerd Leren, en in die categorie vind je veel manieren om te classificeren. [De variëteit](https://scikit-learn.org/stable/supervised_learning.html) kan in eerste instantie overweldigend lijken. De volgende methoden bevatten allemaal technieken voor classificatie:

- Lineaire modellen
- Support Vector Machines
- Stochastic Gradient Descent
- Nabijheid van buren
- Gaussiaanse processen
- Beslissingsbomen
- Ensemble-methoden (stemmen Classifier)
- Multiclass- en multioutput-algoritmen (multiclass- en multilabel-classificatie, multiclass-multioutput-classificatie)

> Je kunt ook [neurale netwerken gebruiken om gegevens te classificeren](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), maar dat valt buiten de scope van deze les.

### Welke classifier moet je kiezen?

Dus, welke classifier moet je kiezen? Vaak is het testen van verschillende classifiers en kijken naar een goed resultaat een manier om te testen. Scikit-learn biedt een [vergelijking naast elkaar](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) op een gecreëerde dataset, waarbij KNeighbors, SVC op twee manieren, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB en QuadraticDiscriminationAnalysis worden vergeleken, en de resultaten worden gevisualiseerd:

![vergelijking van classifiers](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Plots gegenereerd in de documentatie van Scikit-learn

> AutoML lost dit probleem handig op door deze vergelijkingen in de cloud uit te voeren, zodat je het beste algoritme voor je gegevens kunt kiezen. Probeer het [hier](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Een betere aanpak

Een betere manier dan willekeurig gokken is echter om de ideeën te volgen op dit downloadbare [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Hier ontdekken we dat we voor ons multiclass-probleem enkele keuzes hebben:

![cheatsheet voor multiclass-problemen](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Een sectie van Microsoft's Algorithm Cheat Sheet, met opties voor multiclass-classificatie

✅ Download dit cheat sheet, print het uit en hang het aan je muur!

### Redenering

Laten we kijken of we verschillende benaderingen kunnen beredeneren, gegeven de beperkingen die we hebben:

- **Neurale netwerken zijn te zwaar**. Gezien onze schone, maar minimale dataset, en het feit dat we lokaal trainen via notebooks, zijn neurale netwerken te zwaar voor deze taak.
- **Geen tweeklassen-classifier**. We gebruiken geen tweeklassen-classifier, dus dat sluit one-vs-all uit.
- **Beslissingsboom of logistische regressie zou kunnen werken**. Een beslissingsboom zou kunnen werken, of logistische regressie voor multiclass-gegevens.
- **Multiclass Boosted Decision Trees lossen een ander probleem op**. De multiclass boosted decision tree is het meest geschikt voor niet-parametrische taken, bijvoorbeeld taken die zijn ontworpen om ranglijsten te maken, dus het is niet nuttig voor ons.

### Gebruik van Scikit-learn 

We zullen Scikit-learn gebruiken om onze gegevens te analyseren. Er zijn echter veel manieren om logistische regressie te gebruiken in Scikit-learn. Bekijk de [parameters om door te geven](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

In wezen zijn er twee belangrijke parameters - `multi_class` en `solver` - die we moeten specificeren wanneer we Scikit-learn vragen om een logistische regressie uit te voeren. De waarde van `multi_class` past een bepaald gedrag toe. De waarde van de solver bepaalt welk algoritme wordt gebruikt. Niet alle solvers kunnen worden gecombineerd met alle `multi_class`-waarden.

Volgens de documentatie, in het multiclass-geval, gebruikt het trainingsalgoritme:

- **Het one-vs-rest (OvR) schema**, als de `multi_class` optie is ingesteld op `ovr`
- **De cross-entropy loss**, als de `multi_class` optie is ingesteld op `multinomial`. (Momenteel wordt de `multinomial` optie alleen ondersteund door de ‘lbfgs’, ‘sag’, ‘saga’ en ‘newton-cg’ solvers.)"

> 🎓 Het 'schema' hier kan 'ovr' (one-vs-rest) of 'multinomial' zijn. Aangezien logistische regressie eigenlijk is ontworpen om binaire classificatie te ondersteunen, stellen deze schema's het in staat om beter om te gaan met multiclass-classificatietaken. [bron](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 De 'solver' wordt gedefinieerd als "het algoritme dat wordt gebruikt in het optimalisatieprobleem". [bron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn biedt deze tabel om uit te leggen hoe solvers omgaan met verschillende uitdagingen die worden gepresenteerd door verschillende soorten datastructuren:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Oefening - splits de gegevens

We kunnen ons richten op logistische regressie voor onze eerste trainingspoging, aangezien je onlangs over de laatste hebt geleerd in een vorige les.
Splits je gegevens in trainings- en testgroepen door `train_test_split()` aan te roepen:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Oefening - pas logistische regressie toe

Aangezien je het multiclass-geval gebruikt, moet je kiezen welk _schema_ je wilt gebruiken en welke _solver_ je wilt instellen. Gebruik LogisticRegression met een multiclass-instelling en de **liblinear** solver om te trainen.

1. Maak een logistische regressie met multi_class ingesteld op `ovr` en de solver ingesteld op `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Probeer een andere solver zoals `lbfgs`, die vaak als standaard wordt ingesteld
> Let op, gebruik de Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) functie om je data te flatten wanneer nodig.
De nauwkeurigheid is goed bij meer dan **80%**!

1. Je kunt dit model in actie zien door één rij gegevens te testen (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Het resultaat wordt afgedrukt:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Probeer een ander rijnummer en controleer de resultaten.

1. Door dieper te graven, kun je de nauwkeurigheid van deze voorspelling controleren:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Het resultaat wordt afgedrukt - Indiase keuken is de beste gok, met een goede waarschijnlijkheid:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kun je uitleggen waarom het model er vrij zeker van is dat dit een Indiase keuken is?

1. Krijg meer details door een classificatierapport af te drukken, zoals je deed in de lessen over regressie:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precisie | recall | f1-score | support |
    | ------------ | -------- | ------ | -------- | ------- |
    | chinese      | 0.73     | 0.71   | 0.72     | 229     |
    | indian       | 0.91     | 0.93   | 0.92     | 254     |
    | japanese     | 0.70     | 0.75   | 0.72     | 220     |
    | korean       | 0.86     | 0.76   | 0.81     | 242     |
    | thai         | 0.79     | 0.85   | 0.82     | 254     |
    | nauwkeurigheid| 0.80     | 1199   |          |         |
    | macro gem.   | 0.80     | 0.80   | 0.80     | 1199    |
    | gewogen gem. | 0.80     | 0.80   | 0.80     | 1199    |

## 🚀Uitdaging

In deze les heb je je schoongemaakte gegevens gebruikt om een machine learning-model te bouwen dat een nationale keuken kan voorspellen op basis van een reeks ingrediënten. Neem de tijd om de vele opties te bekijken die Scikit-learn biedt om gegevens te classificeren. Verdiep je in het concept van 'solver' om te begrijpen wat er achter de schermen gebeurt.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Verdiep je wat meer in de wiskunde achter logistische regressie in [deze les](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Opdracht 

[Bestudeer de solvers](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.