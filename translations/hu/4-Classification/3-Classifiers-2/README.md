<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T16:23:30+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "hu"
}
-->
# Konyhai osztályozók 2

Ebben a második osztályozási leckében további módszereket fedezhetsz fel a numerikus adatok osztályozására. Megismerheted azt is, hogy milyen következményekkel jár, ha egyik osztályozót választod a másik helyett.

## [Előzetes kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Előfeltétel

Feltételezzük, hogy elvégezted az előző leckéket, és van egy megtisztított adatállományod a `data` mappában, amely _cleaned_cuisines.csv_ néven található a 4-leckés mappa gyökérkönyvtárában.

### Előkészület

Betöltöttük a _notebook.ipynb_ fájlodat a megtisztított adatállománnyal, és X és y adatkeretekre osztottuk, készen állva a modellépítési folyamatra.

## Egy osztályozási térkép

Korábban megismerkedtél a különböző lehetőségekkel, amelyekkel adatokat osztályozhatsz a Microsoft csalólapja segítségével. A Scikit-learn egy hasonló, de részletesebb csalólapot kínál, amely tovább segíthet az osztályozók (más néven becslők) szűkítésében:

![ML térkép a Scikit-learn-től](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tipp: [nézd meg ezt a térképet online](https://scikit-learn.org/stable/tutorial/machine_learning_map/), és kattints az útvonalakon, hogy elolvashasd a dokumentációt.

### A terv

Ez a térkép nagyon hasznos, ha tisztában vagy az adataiddal, mivel „végigjárhatod” az útvonalait, hogy döntést hozz:

- Több mint 50 mintánk van
- Kategóriát szeretnénk előre jelezni
- Címkézett adataink vannak
- Kevesebb mint 100 ezer mintánk van
- ✨ Választhatunk egy Linear SVC-t
- Ha ez nem működik, mivel numerikus adataink vannak
    - Kipróbálhatunk egy ✨ KNeighbors Classifiert 
      - Ha ez sem működik, próbáljuk ki a ✨ SVC-t és ✨ Ensemble Classifiert

Ez egy nagyon hasznos útvonal, amit követhetünk.

## Gyakorlat - az adatok felosztása

Ezt az útvonalat követve kezdjük azzal, hogy importálunk néhány szükséges könyvtárat.

1. Importáld a szükséges könyvtárakat:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Oszd fel a tanuló és tesztadatokat:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC osztályozó

A Support-Vector clustering (SVC) a Support-Vector gépek ML technikáinak családjába tartozik (további információk alább). Ebben a módszerben egy 'kernel'-t választhatsz, amely meghatározza, hogyan csoportosítja a címkéket. A 'C' paraméter a 'regularizációt' jelenti, amely szabályozza a paraméterek hatását. A kernel lehet [többféle](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); itt 'linear'-re állítjuk, hogy lineáris SVC-t használjunk. Az alapértelmezett valószínűség 'false'; itt 'true'-ra állítjuk, hogy valószínűségi becsléseket kapjunk. A random state '0'-ra van állítva, hogy az adatokat keverjük a valószínűségek eléréséhez.

### Gyakorlat - alkalmazz lineáris SVC-t

Kezdj egy osztályozók tömbjének létrehozásával. Ehhez fokozatosan hozzáadunk elemeket, ahogy tesztelünk.

1. Kezdj egy Linear SVC-vel:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Tanítsd be a modelledet a Linear SVC-vel, és nyomtass ki egy jelentést:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Az eredmény elég jó:

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

## K-Neighbors osztályozó

A K-Neighbors az ML módszerek "szomszédok" családjába tartozik, amelyeket felügyelt és nem felügyelt tanulásra is lehet használni. Ebben a módszerben előre meghatározott számú pontot hozunk létre, és az adatok ezek köré gyűlnek, hogy általánosított címkéket lehessen előre jelezni az adatokhoz.

### Gyakorlat - alkalmazd a K-Neighbors osztályozót

Az előző osztályozó jó volt, és jól működött az adatokkal, de talán jobb pontosságot érhetünk el. Próbálj ki egy K-Neighbors osztályozót.

1. Adj hozzá egy sort az osztályozók tömbjéhez (tegyél vesszőt a Linear SVC elem után):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Az eredmény kicsit rosszabb:

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

    ✅ Tudj meg többet a [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) módszerről.

## Support Vector osztályozó

A Support-Vector osztályozók az ML módszerek [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) családjába tartoznak, amelyeket osztályozási és regressziós feladatokra használnak. Az SVM-ek "a tanuló példákat pontokká térképezik az űrben", hogy maximalizálják a távolságot két kategória között. A későbbi adatok ebbe az űrbe kerülnek, hogy előre jelezzék a kategóriájukat.

### Gyakorlat - alkalmazz Support Vector osztályozót

Próbáljunk meg egy kicsit jobb pontosságot elérni egy Support Vector osztályozóval.

1. Tegyél vesszőt a K-Neighbors elem után, majd add hozzá ezt a sort:

    ```python
    'SVC': SVC(),
    ```

    Az eredmény elég jó!

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

    ✅ Tudj meg többet a [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) módszerről.

## Ensemble osztályozók

Kövessük az útvonalat egészen a végéig, még akkor is, ha az előző teszt elég jó volt. Próbáljunk ki néhány 'Ensemble osztályozót', különösen a Random Forest és AdaBoost módszereket:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Az eredmény nagyon jó, különösen a Random Forest esetében:

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

✅ Tudj meg többet az [Ensemble osztályozókról](https://scikit-learn.org/stable/modules/ensemble.html).

Ez a gépi tanulási módszer "több alapbecslő előrejelzéseit kombinálja", hogy javítsa a modell minőségét. Példánkban Random Trees és AdaBoost módszereket használtunk.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), egy átlagolási módszer, amely 'döntési fák' 'erdőjét' építi fel véletlenszerűséggel, hogy elkerülje a túltanulást. Az n_estimators paraméter a fák számát határozza meg.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) egy osztályozót illeszt az adatállományhoz, majd ennek másolatait illeszti ugyanarra az adatállományra. Azokra az elemekre összpontosít, amelyeket helytelenül osztályoztak, és a következő osztályozó illesztését úgy állítja be, hogy javítsa azokat.

---

## 🚀Kihívás

Ezeknek a technikáknak számos paramétere van, amelyeket módosíthatsz. Kutass utána mindegyik alapértelmezett paramétereinek, és gondold át, hogy ezek módosítása mit jelentene a modell minőségére nézve.

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Ezekben a leckékben sok a szakzsargon, ezért szánj egy percet arra, hogy átnézd [ezt a listát](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) a hasznos terminológiáról!

## Feladat 

[Paraméterek játéka](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.