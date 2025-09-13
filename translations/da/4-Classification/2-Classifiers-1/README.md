<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T00:43:18+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "da"
}
-->
# Klassifikatorer for køkkener 1

I denne lektion vil du bruge det datasæt, du gemte fra den sidste lektion, fyldt med balancerede og rene data om køkkener.

Du vil bruge dette datasæt med en række klassifikatorer til _at forudsige et givet nationalt køkken baseret på en gruppe ingredienser_. Mens du gør dette, vil du lære mere om nogle af de måder, algoritmer kan bruges til klassifikationsopgaver.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)
# Forberedelse

Forudsat at du har gennemført [Lektion 1](../1-Introduction/README.md), skal du sikre dig, at en _cleaned_cuisines.csv_-fil findes i rodmappen `/data` for disse fire lektioner.

## Øvelse - forudsige et nationalt køkken

1. Arbejd i denne lektions _notebook.ipynb_-mappe, og importer filen sammen med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dataene ser sådan ud:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importer nu flere biblioteker:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Del X- og y-koordinaterne op i to dataframes til træning. `cuisine` kan være labels-dataframen:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det vil se sådan ud:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Fjern kolonnen `Unnamed: 0` og kolonnen `cuisine` ved at kalde `drop()`. Gem resten af dataene som træningsfunktioner:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Dine funktioner ser sådan ud:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nu er du klar til at træne din model!

## Valg af klassifikator

Nu hvor dine data er rene og klar til træning, skal du beslutte, hvilken algoritme du vil bruge til opgaven. 

Scikit-learn grupperer klassifikation under Supervised Learning, og i den kategori finder du mange måder at klassificere på. [Udvalget](https://scikit-learn.org/stable/supervised_learning.html) kan virke overvældende ved første øjekast. Følgende metoder inkluderer alle klassifikationsteknikker:

- Lineære modeller
- Support Vector Machines
- Stokastisk gradientnedstigning
- Nærmeste naboer
- Gaussian-processer
- Beslutningstræer
- Ensemble-metoder (voting Classifier)
- Multiclass- og multioutput-algoritmer (multiclass- og multilabel-klassifikation, multiclass-multioutput-klassifikation)

> Du kan også bruge [neural netværk til at klassificere data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det ligger uden for denne lektions omfang.

### Hvilken klassifikator skal du vælge?

Så, hvilken klassifikator skal du vælge? Ofte kan det være en god idé at prøve flere og se, hvilken der giver det bedste resultat. Scikit-learn tilbyder en [side-by-side sammenligning](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på et oprettet datasæt, hvor KNeighbors, SVC på to måder, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB og QuadraticDiscriminationAnalysis sammenlignes og visualiseres:

![sammenligning af klassifikatorer](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Diagrammer genereret fra Scikit-learns dokumentation

> AutoML løser dette problem elegant ved at køre disse sammenligninger i skyen, så du kan vælge den bedste algoritme til dine data. Prøv det [her](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### En bedre tilgang

En bedre tilgang end blot at gætte er at følge ideerne på dette downloadbare [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Her opdager vi, at vi for vores multiclass-problem har nogle valgmuligheder:

![cheatsheet for multiclass-problemer](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> En sektion af Microsofts Algorithm Cheat Sheet, der beskriver muligheder for multiclass-klassifikation

✅ Download dette cheat sheet, print det ud, og hæng det op på din væg!

### Overvejelser

Lad os se, om vi kan ræsonnere os frem til forskellige tilgange givet de begrænsninger, vi har:

- **Neural netværk er for tunge**. Givet vores rene, men minimale datasæt, og det faktum at vi kører træning lokalt via notebooks, er neural netværk for tunge til denne opgave.
- **Ingen to-klassifikator**. Vi bruger ikke en to-klassifikator, så det udelukker one-vs-all.
- **Beslutningstræ eller logistisk regression kunne fungere**. Et beslutningstræ kunne fungere, eller logistisk regression for multiclass-data.
- **Multiclass Boosted Decision Trees løser et andet problem**. Multiclass Boosted Decision Tree er mest egnet til ikke-parametriske opgaver, f.eks. opgaver designet til at opbygge rangeringer, så det er ikke nyttigt for os.

### Brug af Scikit-learn 

Vi vil bruge Scikit-learn til at analysere vores data. Der er dog mange måder at bruge logistisk regression i Scikit-learn. Se på [parametrene, der kan angives](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Grundlæggende er der to vigtige parametre - `multi_class` og `solver` - som vi skal angive, når vi beder Scikit-learn om at udføre en logistisk regression. `multi_class`-værdien anvender en bestemt adfærd. Værdien af solver angiver, hvilken algoritme der skal bruges. Ikke alle solvers kan kombineres med alle `multi_class`-værdier.

Ifølge dokumentationen, i multiclass-tilfældet, træningsalgoritmen:

- **Bruger one-vs-rest (OvR)-skemaet**, hvis `multi_class`-indstillingen er sat til `ovr`
- **Bruger krydsentropitab**, hvis `multi_class`-indstillingen er sat til `multinomial`. (I øjeblikket understøttes `multinomial`-indstillingen kun af ‘lbfgs’, ‘sag’, ‘saga’ og ‘newton-cg’-solvers.)

> 🎓 'Skemaet' her kan enten være 'ovr' (one-vs-rest) eller 'multinomial'. Da logistisk regression egentlig er designet til at understøtte binær klassifikation, giver disse skemaer den mulighed for bedre at håndtere multiclass-klassifikationsopgaver. [kilde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solveren' defineres som "den algoritme, der skal bruges i optimeringsproblemet". [kilde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tilbyder denne tabel til at forklare, hvordan solvers håndterer forskellige udfordringer præsenteret af forskellige typer datastrukturer:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Øvelse - del dataene

Vi kan fokusere på logistisk regression til vores første træningsforsøg, da du for nylig har lært om sidstnævnte i en tidligere lektion.
Del dine data i trænings- og testgrupper ved at kalde `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Øvelse - anvend logistisk regression

Da du bruger multiclass-tilfældet, skal du vælge, hvilket _skema_ du vil bruge, og hvilken _solver_ du vil angive. Brug LogisticRegression med en multiclass-indstilling og **liblinear**-solveren til at træne.

1. Opret en logistisk regression med multi_class sat til `ovr` og solver sat til `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prøv en anden solver som `lbfgs`, som ofte er angivet som standard
> Bemærk, brug Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)-funktionen til at flade dine data ud, når det er nødvendigt.
Nøjagtigheden er god ved over **80%**!

1. Du kan se denne model i aktion ved at teste en række data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet bliver udskrevet:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prøv et andet rækkenummer og tjek resultaterne

1. Gå dybere, og undersøg nøjagtigheden af denne forudsigelse:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet bliver udskrevet - indisk køkken er dens bedste gæt, med god sandsynlighed:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du forklare, hvorfor modellen er ret sikker på, at dette er et indisk køkken?

1. Få flere detaljer ved at udskrive en klassifikationsrapport, som du gjorde i regression-lektionerne:

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

## 🚀Udfordring

I denne lektion brugte du dine rensede data til at bygge en maskinlæringsmodel, der kan forudsige en national køkkenstil baseret på en række ingredienser. Tag dig tid til at læse om de mange muligheder, Scikit-learn tilbyder til at klassificere data. Gå dybere ned i konceptet 'solver' for at forstå, hvad der sker bag kulisserne.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Undersøg lidt mere om matematikken bag logistisk regression i [denne lektion](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Opgave 

[Undersøg solvers](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.