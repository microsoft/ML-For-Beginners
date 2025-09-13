<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T08:00:00+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "lt"
}
-->
# Virtuvės klasifikatoriai 2

Šioje antroje klasifikavimo pamokoje jūs tyrinėsite daugiau būdų, kaip klasifikuoti skaitmeninius duomenis. Taip pat sužinosite apie pasekmes, renkantis vieną klasifikatorių vietoj kito.

## [Prieš paskaitą: testas](https://ff-quizzes.netlify.app/en/ml/)

### Būtinos žinios

Daroma prielaida, kad jau baigėte ankstesnes pamokas ir turite išvalytą duomenų rinkinį savo `data` aplanke, pavadintą _cleaned_cuisines.csv_, esančiame šio 4 pamokų aplanko šaknyje.

### Pasiruošimas

Mes įkėlėme jūsų _notebook.ipynb_ failą su išvalytu duomenų rinkiniu ir padalijome jį į X ir y duomenų rėmelius, paruoštus modelio kūrimo procesui.

## Klasifikavimo žemėlapis

Ankstesnėje pamokoje sužinojote apie įvairias galimybes klasifikuoti duomenis, naudodamiesi „Microsoft“ apgaulės lapu. Scikit-learn siūlo panašų, bet detalesnį apgaulės lapą, kuris gali dar labiau padėti susiaurinti jūsų pasirinkimą (kitaip vadinamą klasifikatoriais):

![ML žemėlapis iš Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Patarimas: [aplankykite šį žemėlapį internete](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ir spustelėkite kelią, kad perskaitytumėte dokumentaciją.

### Planas

Šis žemėlapis yra labai naudingas, kai aiškiai suprantate savo duomenis, nes galite „eiti“ jo keliais iki sprendimo:

- Turime >50 pavyzdžių
- Norime prognozuoti kategoriją
- Turime pažymėtus duomenis
- Turime mažiau nei 100 tūkst. pavyzdžių
- ✨ Galime pasirinkti Linear SVC
- Jei tai neveikia, kadangi turime skaitmeninius duomenis
    - Galime išbandyti ✨ KNeighbors Classifier 
      - Jei tai neveikia, išbandykite ✨ SVC ir ✨ Ensemble Classifiers

Tai labai naudingas kelias, kurio verta laikytis.

## Užduotis – padalykite duomenis

Sekdami šį kelią, turėtume pradėti importuodami kai kurias reikalingas bibliotekas.

1. Importuokite reikalingas bibliotekas:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Padalykite savo mokymo ir testavimo duomenis:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klasifikatorius

Support-Vector Clustering (SVC) yra Support-Vector Machines šeimos ML technikų dalis (daugiau apie jas sužinosite žemiau). Šiame metode galite pasirinkti „branduolį“ (kernel), kuris nusprendžia, kaip suskirstyti etiketes. Parametras „C“ reiškia „reguliavimą“, kuris reguliuoja parametrų įtaką. Branduolys gali būti vienas iš [kelių](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); čia mes nustatome jį kaip „linear“, kad užtikrintume Linear SVC naudojimą. Tikimybė pagal nutylėjimą yra „false“; čia mes nustatome ją kaip „true“, kad gautume tikimybių įvertinimus. Atsitiktinę būseną nustatome kaip „0“, kad sumaišytume duomenis ir gautume tikimybes.

### Užduotis – pritaikykite Linear SVC

Pradėkite kurdami klasifikatorių masyvą. Jūs palaipsniui pridėsite prie šio masyvo, kai testuosite.

1. Pradėkite nuo Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Treniruokite savo modelį naudodami Linear SVC ir išspausdinkite ataskaitą:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultatas yra gana geras:

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

## K-Neighbors klasifikatorius

K-Neighbors yra „kaimynų“ šeimos ML metodų dalis, kuriuos galima naudoti tiek prižiūrimam, tiek neprižiūrimam mokymuisi. Šiame metode sukuriamas iš anksto nustatytas taškų skaičius, o duomenys renkami aplink šiuos taškus, kad būtų galima prognozuoti apibendrintas etiketes.

### Užduotis – pritaikykite K-Neighbors klasifikatorių

Ankstesnis klasifikatorius buvo geras ir gerai veikė su duomenimis, bet galbūt galime pasiekti geresnį tikslumą. Išbandykite K-Neighbors klasifikatorių.

1. Pridėkite eilutę prie savo klasifikatorių masyvo (po Linear SVC elemento pridėkite kablelį):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultatas yra šiek tiek blogesnis:

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

    ✅ Sužinokite daugiau apie [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikatoriai yra [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) šeimos ML metodų dalis, naudojama klasifikavimo ir regresijos užduotims. SVM „sudeda mokymo pavyzdžius į taškus erdvėje“, kad maksimaliai padidintų atstumą tarp dviejų kategorijų. Vėlesni duomenys yra sudedami į šią erdvę, kad būtų galima prognozuoti jų kategoriją.

### Užduotis – pritaikykite Support Vector Classifier

Pabandykime pasiekti šiek tiek geresnį tikslumą naudodami Support Vector Classifier.

1. Po K-Neighbors elemento pridėkite kablelį, tada pridėkite šią eilutę:

    ```python
    'SVC': SVC(),
    ```

    Rezultatas yra gana geras!

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

    ✅ Sužinokite daugiau apie [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Sekime kelią iki galo, nors ankstesnis testas buvo gana geras. Išbandykime „Ensemble Classifiers“, konkrečiai Random Forest ir AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultatas yra labai geras, ypač Random Forest:

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

✅ Sužinokite daugiau apie [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Šis mašininio mokymosi metodas „sujungia kelių bazinių įvertintojų prognozes“, kad pagerintų modelio kokybę. Mūsų pavyzdyje naudojome Random Trees ir AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), vidurkinimo metodas, sukuria „mišką“ iš „sprendimų medžių“, įterptų su atsitiktinumu, kad būtų išvengta per didelio pritaikymo. Parametras n_estimators nustatomas kaip medžių skaičius.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) pritaiko klasifikatorių duomenų rinkiniui, o tada pritaiko šio klasifikatoriaus kopijas tam pačiam duomenų rinkiniui. Jis sutelkia dėmesį į neteisingai klasifikuotų elementų svorius ir koreguoja kitą klasifikatorių, kad ištaisytų klaidas.

---

## 🚀Iššūkis

Kiekviena iš šių technikų turi daug parametrų, kuriuos galite koreguoti. Ištyrinėkite kiekvieno numatytuosius parametrus ir pagalvokite, ką šių parametrų koregavimas reikštų modelio kokybei.

## [Po paskaitos: testas](https://ff-quizzes.netlify.app/en/ml/)

## Peržiūra ir savarankiškas mokymasis

Šiose pamokose yra daug terminologijos, todėl skirkite minutę peržiūrėti [šį sąrašą](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) naudingų terminų!

## Užduotis 

[Parametrų žaidimas](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudotis profesionalių vertėjų paslaugomis. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.