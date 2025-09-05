<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T21:50:40+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "no"
}
-->
# Klassifisering av matretter 1

I denne leksjonen skal du bruke datasettet du lagret fra forrige leksjon, som inneholder balanserte og rene data om matretter.

Du vil bruke dette datasettet med ulike klassifiseringsmetoder for å _forutsi en nasjonal matrett basert på en gruppe ingredienser_. Mens du gjør dette, vil du lære mer om hvordan algoritmer kan brukes til klassifiseringsoppgaver.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)
# Forberedelse

Forutsatt at du fullførte [Leksjon 1](../1-Introduction/README.md), sørg for at en _cleaned_cuisines.csv_-fil finnes i rotmappen `/data` for disse fire leksjonene.

## Øvelse - forutsi en nasjonal matrett

1. Arbeid i denne leksjonens _notebook.ipynb_-mappe, og importer filen sammen med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Dataen ser slik ut:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importer nå flere biblioteker:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Del X- og y-koordinatene inn i to dataframes for trening. `cuisine` kan være etikett-datasettet:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det vil se slik ut:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Fjern kolonnen `Unnamed: 0` og kolonnen `cuisine` ved å bruke `drop()`. Lagre resten av dataene som trenbare funksjoner:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Funksjonene dine ser slik ut:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nå er du klar til å trene modellen din!

## Velge klassifiseringsmetode

Nå som dataene dine er rene og klare for trening, må du bestemme hvilken algoritme du skal bruke til oppgaven.

Scikit-learn grupperer klassifisering under Supervised Learning, og i den kategorien finner du mange måter å klassifisere på. [Utvalget](https://scikit-learn.org/stable/supervised_learning.html) kan virke overveldende ved første øyekast. Følgende metoder inkluderer klassifiseringsteknikker:

- Lineære modeller
- Support Vector Machines
- Stokastisk gradientnedstigning
- Nærmeste naboer
- Gaussiske prosesser
- Beslutningstrær
- Ensemble-metoder (voting Classifier)
- Multiklasse- og multioutput-algoritmer (multiklasse- og multilabel-klassifisering, multiklasse-multioutput-klassifisering)

> Du kan også bruke [nevrale nettverk til å klassifisere data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det er utenfor omfanget av denne leksjonen.

### Hvilken klassifiseringsmetode skal du velge?

Så, hvilken klassifiseringsmetode bør du velge? Ofte kan det være nyttig å teste flere metoder og se etter gode resultater. Scikit-learn tilbyr en [side-ved-side-sammenligning](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på et opprettet datasett, som sammenligner KNeighbors, SVC på to måter, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB og QuadraticDiscriminationAnalysis, og viser resultatene visuelt:

![sammenligning av klassifiseringsmetoder](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Diagrammer generert fra Scikit-learns dokumentasjon

> AutoML løser dette problemet elegant ved å kjøre disse sammenligningene i skyen, slik at du kan velge den beste algoritmen for dataene dine. Prøv det [her](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### En bedre tilnærming

En bedre tilnærming enn å gjette vilkårlig er å følge ideene i dette nedlastbare [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Her oppdager vi at vi har noen valg for vårt multiklasse-problem:

![jukselapp for multiklasse-problemer](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> En del av Microsofts Algorithm Cheat Sheet, som beskriver alternativer for multiklasse-klassifisering

✅ Last ned denne jukselappen, skriv den ut, og heng den på veggen!

### Resonnement

La oss se om vi kan resonnere oss frem til ulike tilnærminger gitt de begrensningene vi har:

- **Nevrale nettverk er for tunge**. Gitt vårt rene, men minimale datasett, og det faktum at vi kjører trening lokalt via notebooks, er nevrale nettverk for tunge for denne oppgaven.
- **Ingen to-klasse klassifiserer**. Vi bruker ikke en to-klasse klassifiserer, så det utelukker one-vs-all.
- **Beslutningstre eller logistisk regresjon kan fungere**. Et beslutningstre kan fungere, eller logistisk regresjon for multiklasse-data.
- **Multiklasse Boosted Decision Trees løser et annet problem**. Multiklasse Boosted Decision Trees er mest egnet for ikke-parametriske oppgaver, f.eks. oppgaver designet for å bygge rangeringer, så det er ikke nyttig for oss.

### Bruke Scikit-learn 

Vi skal bruke Scikit-learn til å analysere dataene våre. Det finnes imidlertid mange måter å bruke logistisk regresjon i Scikit-learn. Ta en titt på [parametrene du kan sende](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

I hovedsak er det to viktige parametere - `multi_class` og `solver` - som vi må spesifisere når vi ber Scikit-learn om å utføre en logistisk regresjon. `multi_class`-verdien gir en viss oppførsel. Verdien av solver angir hvilken algoritme som skal brukes. Ikke alle solvers kan kombineres med alle `multi_class`-verdier.

Ifølge dokumentasjonen, i multiklasse-tilfellet, treningsalgoritmen:

- **Bruker one-vs-rest (OvR)-skjemaet**, hvis `multi_class`-alternativet er satt til `ovr`
- **Bruker kryssentropitap**, hvis `multi_class`-alternativet er satt til `multinomial`. (For øyeblikket støttes `multinomial`-alternativet kun av solverne ‘lbfgs’, ‘sag’, ‘saga’ og ‘newton-cg’.)

> 🎓 'Skjemaet' her kan enten være 'ovr' (one-vs-rest) eller 'multinomial'. Siden logistisk regresjon egentlig er designet for å støtte binær klassifisering, lar disse skjemaene den håndtere multiklasse-klassifiseringsoppgaver bedre. [kilde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' er definert som "algoritmen som skal brukes i optimaliseringsproblemet". [kilde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tilbyr denne tabellen for å forklare hvordan solvers håndterer ulike utfordringer presentert av forskjellige typer datastrukturer:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Øvelse - del dataene

Vi kan fokusere på logistisk regresjon for vår første treningsrunde siden du nylig lærte om dette i en tidligere leksjon.
Del dataene dine inn i trenings- og testgrupper ved å bruke `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Øvelse - bruk logistisk regresjon

Siden du bruker multiklasse-tilfellet, må du velge hvilket _skjema_ du skal bruke og hvilken _solver_ du skal sette. Bruk LogisticRegression med en multiklasse-innstilling og **liblinear** solver for å trene.

1. Opprett en logistisk regresjon med multi_class satt til `ovr` og solver satt til `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prøv en annen solver som `lbfgs`, som ofte er satt som standard
> Merk, bruk Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)-funksjonen for å flate ut dataene dine når det er nødvendig.
Nøyaktigheten er god på over **80%**!

1. Du kan se denne modellen i aksjon ved å teste én rad med data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet skrives ut:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prøv et annet radnummer og sjekk resultatene

1. For å gå dypere, kan du sjekke nøyaktigheten til denne prediksjonen:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet skrives ut - indisk mat er modellens beste gjetning, med høy sannsynlighet:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du forklare hvorfor modellen er ganske sikker på at dette er indisk mat?

1. Få mer detaljert informasjon ved å skrive ut en klassifikasjonsrapport, slik du gjorde i regresjonsleksjonene:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | presisjon | recall | f1-score | støtte |
    | ------------ | --------- | ------ | -------- | ------ |
    | chinese      | 0.73      | 0.71   | 0.72     | 229    |
    | indian       | 0.91      | 0.93   | 0.92     | 254    |
    | japanese     | 0.70      | 0.75   | 0.72     | 220    |
    | korean       | 0.86      | 0.76   | 0.81     | 242    |
    | thai         | 0.79      | 0.85   | 0.82     | 254    |
    | nøyaktighet  | 0.80      | 1199   |          |        |
    | makro snitt  | 0.80      | 0.80   | 0.80     | 1199   |
    | vektet snitt | 0.80      | 0.80   | 0.80     | 1199   |

## 🚀Utfordring

I denne leksjonen brukte du de rensede dataene dine til å bygge en maskinlæringsmodell som kan forutsi en nasjonal matrett basert på en rekke ingredienser. Ta deg tid til å lese gjennom de mange alternativene Scikit-learn tilbyr for å klassifisere data. Gå dypere inn i konseptet 'solver' for å forstå hva som skjer i bakgrunnen.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Grav litt dypere i matematikken bak logistisk regresjon i [denne leksjonen](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Oppgave 

[Studer solverne](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.