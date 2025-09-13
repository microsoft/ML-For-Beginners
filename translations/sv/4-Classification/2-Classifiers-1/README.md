<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T21:50:01+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "sv"
}
-->
# Klassificering av kök 1

I den här lektionen kommer du att använda datasetet som du sparade från den senaste lektionen, fullt av balanserad och ren data om olika kök.

Du kommer att använda detta dataset med en mängd olika klassificerare för att _förutsäga ett visst nationellt kök baserat på en grupp ingredienser_. Under tiden kommer du att lära dig mer om hur algoritmer kan användas för klassificeringsuppgifter.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)
# Förberedelse

Om du har slutfört [Lektionen 1](../1-Introduction/README.md), se till att en _cleaned_cuisines.csv_-fil finns i rotmappen `/data` för dessa fyra lektioner.

## Övning - förutsäg ett nationellt kök

1. Arbeta i den här lektionens _notebook.ipynb_-mapp och importera filen tillsammans med Pandas-biblioteket:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Datat ser ut så här:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Importera nu flera andra bibliotek:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Dela upp X- och y-koordinaterna i två dataframes för träning. `cuisine` kan vara etikett-databasen:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Det kommer att se ut så här:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Ta bort kolumnen `Unnamed: 0` och kolumnen `cuisine` genom att använda `drop()`. Spara resten av datan som träningsbara funktioner:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Dina funktioner ser ut så här:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nu är du redo att träna din modell!

## Välja klassificerare

Nu när din data är ren och redo för träning måste du bestämma vilken algoritm du ska använda för uppgiften.

Scikit-learn grupperar klassificering under Supervised Learning, och inom den kategorin hittar du många sätt att klassificera. [Variationen](https://scikit-learn.org/stable/supervised_learning.html) kan verka överväldigande vid första anblicken. Följande metoder inkluderar alla klassificeringstekniker:

- Linjära modeller
- Support Vector Machines
- Stokastisk gradientnedstigning
- Närmaste grannar
- Gaussiska processer
- Beslutsträd
- Ensemblemetoder (röstningsklassificerare)
- Multiklass- och multioutput-algoritmer (multiklass- och multilabel-klassificering, multiklass-multioutput-klassificering)

> Du kan också använda [neurala nätverk för att klassificera data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), men det ligger utanför denna lektions omfattning.

### Vilken klassificerare ska man välja?

Så, vilken klassificerare ska du välja? Ofta kan man testa flera och leta efter ett bra resultat. Scikit-learn erbjuder en [jämförelse sida vid sida](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) på ett skapat dataset, där KNeighbors, SVC på två sätt, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB och QuadraticDiscriminationAnalysis jämförs och resultaten visualiseras:

![jämförelse av klassificerare](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Diagram genererade från Scikit-learns dokumentation

> AutoML löser detta problem smidigt genom att köra dessa jämförelser i molnet, vilket gör att du kan välja den bästa algoritmen för din data. Prova det [här](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### En bättre metod

En bättre metod än att gissa vilt är att följa idéerna i detta nedladdningsbara [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Här upptäcker vi att vi för vårt multiklassproblem har några alternativ:

![fusklapp för multiklassproblem](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> En del av Microsofts Algorithm Cheat Sheet, som beskriver alternativ för multiklassklassificering

✅ Ladda ner denna fusklapp, skriv ut den och häng upp den på väggen!

### Resonemang

Låt oss se om vi kan resonera oss fram till olika metoder med tanke på de begränsningar vi har:

- **Neurala nätverk är för tunga**. Med tanke på vårt rena men minimala dataset och det faktum att vi kör träning lokalt via notebooks, är neurala nätverk för resurskrävande för denna uppgift.
- **Ingen tvåklassklassificerare**. Vi använder inte en tvåklassklassificerare, så det utesluter one-vs-all.
- **Beslutsträd eller logistisk regression kan fungera**. Ett beslutsträd kan fungera, eller logistisk regression för multiklassdata.
- **Multiklass Boosted Decision Trees löser ett annat problem**. Multiklass Boosted Decision Tree är mest lämplig för icke-parametriska uppgifter, t.ex. uppgifter som är utformade för att skapa rankningar, så det är inte användbart för oss.

### Använda Scikit-learn 

Vi kommer att använda Scikit-learn för att analysera vår data. Det finns dock många sätt att använda logistisk regression i Scikit-learn. Ta en titt på [parametrarna att skicka](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

I huvudsak finns det två viktiga parametrar - `multi_class` och `solver` - som vi behöver specificera när vi ber Scikit-learn att utföra en logistisk regression. Värdet på `multi_class` tillämpar ett visst beteende. Värdet på solver är vilken algoritm som ska användas. Inte alla solvers kan kombineras med alla `multi_class`-värden.

Enligt dokumentationen, i multiklassfallet, träningsalgoritmen:

- **Använder one-vs-rest (OvR)-schemat**, om `multi_class`-alternativet är inställt på `ovr`
- **Använder korsentropiförlust**, om `multi_class`-alternativet är inställt på `multinomial`. (För närvarande stöds `multinomial`-alternativet endast av solvers ‘lbfgs’, ‘sag’, ‘saga’ och ‘newton-cg’.)

> 🎓 'Schemat' här kan antingen vara 'ovr' (one-vs-rest) eller 'multinomial'. Eftersom logistisk regression egentligen är utformad för att stödja binär klassificering, tillåter dessa scheman den att bättre hantera multiklassklassificeringsuppgifter. [källa](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' definieras som "algoritmen som ska användas i optimeringsproblemet". [källa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn erbjuder denna tabell för att förklara hur solvers hanterar olika utmaningar som presenteras av olika typer av datastrukturer:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Övning - dela upp datan

Vi kan fokusera på logistisk regression för vår första träningsförsök eftersom du nyligen lärde dig om den i en tidigare lektion.
Dela upp din data i tränings- och testgrupper genom att kalla på `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Övning - tillämpa logistisk regression

Eftersom du använder multiklassfallet behöver du välja vilket _schema_ du ska använda och vilken _solver_ du ska ställa in. Använd LogisticRegression med en multiklassinställning och **liblinear**-solver för att träna.

1. Skapa en logistisk regression med multi_class inställd på `ovr` och solver inställd på `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Prova en annan solver som `lbfgs`, som ofta är inställd som standard
> Observera, använd Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)-funktionen för att platta ut dina data vid behov.
Noggrannheten är bra på över **80%**!

1. Du kan se denna modell i aktion genom att testa en rad data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Resultatet skrivs ut:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Prova ett annat radnummer och kontrollera resultaten

1. För att gå djupare kan du kontrollera noggrannheten för denna förutsägelse:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Resultatet skrivs ut - indisk mat är dess bästa gissning, med hög sannolikhet:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kan du förklara varför modellen är ganska säker på att detta är indisk mat?

1. Få mer detaljer genom att skriva ut en klassificeringsrapport, precis som du gjorde i regression-lektionerna:

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

## 🚀Utmaning

I denna lektion använde du dina rensade data för att bygga en maskininlärningsmodell som kan förutsäga ett nationellt kök baserat på en serie ingredienser. Ta lite tid att läsa igenom de många alternativ som Scikit-learn erbjuder för att klassificera data. Gå djupare in i konceptet 'solver' för att förstå vad som händer bakom kulisserna.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Gräv lite djupare i matematiken bakom logistisk regression i [denna lektion](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)  
## Uppgift 

[Studera lösningsmetoderna](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör du vara medveten om att automatiserade översättningar kan innehålla fel eller inexaktheter. Det ursprungliga dokumentet på dess ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.