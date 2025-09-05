<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T16:17:35+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "hu"
}
-->
# Konyhai osztályozók 1

Ebben a leckében az előző leckében elmentett, kiegyensúlyozott és tiszta adatokkal teli adatállományt fogod használni, amely a különböző konyhákról szól.

Ezt az adatállományt különféle osztályozókkal fogod használni, hogy _egy adott nemzeti konyhát megjósolj egy összetevőcsoport alapján_. Eközben többet megtudhatsz arról, hogyan lehet algoritmusokat alkalmazni osztályozási feladatokhoz.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)
# Felkészülés

Feltételezve, hogy befejezted az [1. leckét](../1-Introduction/README.md), győződj meg róla, hogy egy _cleaned_cuisines.csv_ fájl létezik a gyökér `/data` mappában ehhez a négy leckéhez.

## Gyakorlat - egy nemzeti konyha megjóslása

1. Dolgozz ebben a lecke _notebook.ipynb_ mappájában, és importáld a fájlt a Pandas könyvtárral együtt:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Az adatok így néznek ki:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Most importálj több könyvtárat:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Oszd fel az X és y koordinátákat két adatkeretre a tanításhoz. A `cuisine` lehet a címkéket tartalmazó adatkeret:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ez így fog kinézni:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Dobd el az `Unnamed: 0` és a `cuisine` oszlopokat a `drop()` hívásával. A többi adatot mentsd el tanítható jellemzőként:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    A jellemzők így néznek ki:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Most készen állsz arra, hogy betanítsd a modelledet!

## Osztályozó kiválasztása

Most, hogy az adataid tiszták és készen állnak a tanításra, el kell döntened, melyik algoritmust használod a feladathoz.

A Scikit-learn a felügyelt tanulás kategóriájába sorolja az osztályozást, és ebben a kategóriában számos módszert találsz az osztályozáshoz. [A választék](https://scikit-learn.org/stable/supervised_learning.html) elsőre meglehetősen zavarba ejtő. Az alábbi módszerek mind tartalmaznak osztályozási technikákat:

- Lineáris modellek
- Támogató vektorgépek
- Stochasztikus gradiens csökkenés
- Legközelebbi szomszédok
- Gauss-folyamatok
- Döntési fák
- Együttes módszerek (szavazó osztályozó)
- Többosztályos és többkimenetes algoritmusok (többosztályos és többcímkés osztályozás, többosztályos-többkimenetes osztályozás)

> Az adatok osztályozására [neurális hálózatokat is használhatsz](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), de ez a lecke keretein kívül esik.

### Melyik osztályozót válasszuk?

Tehát, melyik osztályozót válasszuk? Gyakran az a módszer, hogy több algoritmust kipróbálunk, és keresünk egy jó eredményt. A Scikit-learn kínál egy [összehasonlítást](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) egy létrehozott adatállományon, amely összehasonlítja a KNeighbors, SVC két módját, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB és QuadraticDiscriminationAnalysis algoritmusokat, és vizualizálja az eredményeket:

![osztályozók összehasonlítása](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Diagramok a Scikit-learn dokumentációjából

> Az AutoML elegánsan megoldja ezt a problémát azáltal, hogy ezeket az összehasonlításokat a felhőben futtatja, lehetővé téve, hogy kiválaszd a legjobb algoritmust az adataidhoz. Próbáld ki [itt](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Egy jobb megközelítés

Egy jobb módszer, mint a vad találgatás, az, hogy követjük az ötleteket ezen letölthető [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) alapján. Itt felfedezzük, hogy a többosztályos problémánkhoz van néhány választási lehetőség:

![cheatsheet többosztályos problémákhoz](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> A Microsoft Algoritmus Cheat Sheet egy szakasza, amely részletezi a többosztályos osztályozási lehetőségeket

✅ Töltsd le ezt a cheat sheetet, nyomtasd ki, és tedd ki a faladra!

### Érvelés

Nézzük meg, hogy az adott korlátok alapján milyen megközelítéseket választhatunk:

- **A neurális hálózatok túl nehezek**. Tekintve, hogy az adatállományunk tiszta, de minimális, és hogy a tanítást helyben, notebookokon keresztül végezzük, a neurális hálózatok túl nehézkesek ehhez a feladathoz.
- **Nincs kétosztályos osztályozó**. Nem használunk kétosztályos osztályozót, így az egy-vs-minden kizárható.
- **Döntési fa vagy logisztikus regresszió működhet**. Egy döntési fa működhet, vagy logisztikus regresszió többosztályos adatokhoz.
- **Többosztályos Boosted Decision Trees más problémát old meg**. A többosztályos Boosted Decision Tree leginkább nemparaméteres feladatokhoz alkalmas, például rangsorok létrehozására, így számunkra nem hasznos.

### Scikit-learn használata 

A Scikit-learn-t fogjuk használni az adataink elemzésére. Azonban számos módja van annak, hogy logisztikus regressziót használjunk a Scikit-learn-ben. Nézd meg a [paramétereket](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression), amelyeket megadhatsz.

Lényegében két fontos paraméter van - `multi_class` és `solver` -, amelyeket meg kell határoznunk, amikor arra kérjük a Scikit-learn-t, hogy végezzen logisztikus regressziót. A `multi_class` érték egy bizonyos viselkedést alkalmaz. A solver értéke az algoritmus, amelyet használni kell. Nem minden solver párosítható minden `multi_class` értékkel.

A dokumentáció szerint a többosztályos esetben a tanítási algoritmus:

- **Az egy-vs-minden (OvR) sémát használja**, ha a `multi_class` opció `ovr`-re van állítva
- **A keresztentrópia veszteséget használja**, ha a `multi_class` opció `multinomial`-ra van állítva. (Jelenleg a `multinomial` opciót csak az ‘lbfgs’, ‘sag’, ‘saga’ és ‘newton-cg’ solvers támogatják.)

> 🎓 A 'séma' itt lehet 'ovr' (egy-vs-minden) vagy 'multinomial'. Mivel a logisztikus regresszió valójában bináris osztályozás támogatására lett tervezve, ezek a sémák lehetővé teszik, hogy jobban kezelje a többosztályos osztályozási feladatokat. [forrás](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 A 'solver' úgy van definiálva, mint "az algoritmus, amelyet az optimalizálási problémában használni kell". [forrás](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

A Scikit-learn ezt a táblázatot kínálja, hogy megmagyarázza, hogyan kezelik a solvers a különböző kihívásokat, amelyeket a különböző adatstruktúrák jelentenek:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Gyakorlat - az adatok felosztása

Koncentráljunk a logisztikus regresszióra az első tanítási próbánk során, mivel nemrég tanultál róla egy korábbi leckében.
Oszd fel az adataidat tanítási és tesztelési csoportokra a `train_test_split()` hívásával:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Gyakorlat - logisztikus regresszió alkalmazása

Mivel a többosztályos esetet használod, ki kell választanod, hogy milyen _sémát_ és milyen _solvert_ állíts be. Használj LogisticRegression-t többosztályos beállítással és a **liblinear** solverrel a tanításhoz.

1. Hozz létre egy logisztikus regressziót, ahol a multi_class `ovr`-re van állítva, és a solver `liblinear`-re:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Próbálj ki egy másik solvert, például `lbfgs`, amelyet gyakran alapértelmezettként állítanak be
> Megjegyzés: Használja a Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) függvényt az adatok lapításához, amikor szükséges.
A pontosság **80%** felett jó!

1. Ezt a modellt működés közben láthatod, ha tesztelsz egy adat sort (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Az eredmény ki van nyomtatva:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Próbálj ki egy másik sor számot, és ellenőrizd az eredményeket.

1. Mélyebbre ásva ellenőrizheted ennek az előrejelzésnek a pontosságát:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Az eredmény ki van nyomtatva - az indiai konyha a legjobb tippje, jó valószínűséggel:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Meg tudod magyarázni, miért gondolja a modell, hogy ez biztosan indiai konyha?

1. Szerezz több részletet egy osztályozási jelentés kinyomtatásával, ahogy a regressziós leckékben tetted:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | pontosság | visszahívás | f1-érték | támogatás |
    | ------------ | --------- | ----------- | -------- | --------- |
    | chinese      | 0.73      | 0.71        | 0.72     | 229       |
    | indian       | 0.91      | 0.93        | 0.92     | 254       |
    | japanese     | 0.70      | 0.75        | 0.72     | 220       |
    | korean       | 0.86      | 0.76        | 0.81     | 242       |
    | thai         | 0.79      | 0.85        | 0.82     | 254       |
    | pontosság    | 0.80      | 1199        |          |           |
    | makro átlag  | 0.80      | 0.80        | 0.80     | 1199      |
    | súlyozott átlag | 0.80   | 0.80        | 0.80     | 1199      |

## 🚀Kihívás

Ebben a leckében a megtisztított adataidat használtad egy gépi tanulási modell felépítéséhez, amely képes nemzeti konyhát előre jelezni egy sor összetevő alapján. Szánj időt arra, hogy átnézd a Scikit-learn által kínált számos lehetőséget az adatok osztályozására. Áss mélyebbre a 'solver' fogalmába, hogy megértsd, mi zajlik a háttérben.

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Áss egy kicsit mélyebbre a logisztikus regresszió matematikájában [ebben a leckében](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Feladat 

[Tanulmányozd a solvereket](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.