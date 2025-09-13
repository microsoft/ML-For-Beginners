<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T13:14:38+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "sl"
}
-->
# Razvrščevalniki kuhinj 2

V tej drugi lekciji o razvrščanju boste raziskali več načinov za razvrščanje numeričnih podatkov. Prav tako boste spoznali posledice izbire enega razvrščevalnika namesto drugega.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

### Predpogoji

Predvidevamo, da ste zaključili prejšnje lekcije in imate očiščen nabor podatkov v svoji mapi `data`, imenovan _cleaned_cuisines.csv_, ki se nahaja v korenski mapi tega 4-lekcijskega sklopa.

### Priprava

Vaša datoteka _notebook.ipynb_ je bila naložena z očiščenim naborom podatkov, ki je razdeljen na podatkovna okvira X in y, pripravljena za proces gradnje modela.

## Zemljevid razvrščanja

Prej ste se naučili o različnih možnostih razvrščanja podatkov z uporabo Microsoftovega priročnika. Scikit-learn ponuja podoben, vendar bolj podroben priročnik, ki vam lahko dodatno pomaga zožiti izbiro ocenjevalnikov (drugi izraz za razvrščevalnike):

![ML Zemljevid iz Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Nasvet: [obiskujte ta zemljevid na spletu](https://scikit-learn.org/stable/tutorial/machine_learning_map/) in kliknite po poti za branje dokumentacije.

### Načrt

Ta zemljevid je zelo koristen, ko imate jasno predstavo o svojih podatkih, saj lahko 'hodite' po njegovih poteh do odločitve:

- Imamo >50 vzorcev
- Želimo napovedati kategorijo
- Imamo označene podatke
- Imamo manj kot 100K vzorcev
- ✨ Lahko izberemo Linear SVC
- Če to ne deluje, ker imamo numerične podatke
    - Lahko poskusimo ✨ KNeighbors Classifier 
      - Če to ne deluje, poskusimo ✨ SVC in ✨ Ensemble Classifiers

To je zelo koristna pot za sledenje.

## Naloga - razdelite podatke

Sledimo tej poti in začnemo z uvozom nekaterih knjižnic za uporabo.

1. Uvozite potrebne knjižnice:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Razdelite svoje podatke na trening in test:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linearni SVC razvrščevalnik

Support-Vector clustering (SVC) je del družine tehnik strojnega učenja Support-Vector Machines (več o tem spodaj). Pri tej metodi lahko izberete 'jedro' za odločanje, kako razvrstiti oznake. Parameter 'C' se nanaša na 'regularizacijo', ki uravnava vpliv parametrov. Jedro je lahko eno izmed [več možnosti](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tukaj ga nastavimo na 'linearno', da zagotovimo uporabo linearnega SVC. Privzeta vrednost za verjetnost je 'false'; tukaj jo nastavimo na 'true', da pridobimo ocene verjetnosti. Naključno stanje nastavimo na '0', da premešamo podatke za pridobitev verjetnosti.

### Naloga - uporabite linearni SVC

Začnite z ustvarjanjem matrike razvrščevalnikov. Postopoma boste dodajali tej matriki, ko bomo testirali.

1. Začnite z Linearnim SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Natrenirajte svoj model z Linearnim SVC in natisnite poročilo:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultat je precej dober:

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

## K-Neighbors razvrščevalnik

K-Neighbors je del družine metod strojnega učenja "neighbors", ki se lahko uporablja za nadzorovano in nenadzorovano učenje. Pri tej metodi se ustvari vnaprej določeno število točk, okoli katerih se zbirajo podatki, da se lahko za podatke napovejo posplošene oznake.

### Naloga - uporabite K-Neighbors razvrščevalnik

Prejšnji razvrščevalnik je bil dober in je dobro deloval s podatki, vendar morda lahko dosežemo boljšo natančnost. Poskusite K-Neighbors razvrščevalnik.

1. Dodajte vrstico v svojo matriko razvrščevalnikov (dodajte vejico za element Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultat je nekoliko slabši:

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

    ✅ Preberite več o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector razvrščevalnik

Support-Vector razvrščevalniki so del družine metod strojnega učenja [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), ki se uporabljajo za naloge razvrščanja in regresije. SVM "preslika primere treninga v točke v prostoru", da maksimizira razdaljo med dvema kategorijama. Naknadni podatki so preslikani v ta prostor, da se lahko napove njihova kategorija.

### Naloga - uporabite Support Vector razvrščevalnik

Poskusimo doseči nekoliko boljšo natančnost s Support Vector razvrščevalnikom.

1. Dodajte vejico za element K-Neighbors in nato dodajte to vrstico:

    ```python
    'SVC': SVC(),
    ```

    Rezultat je zelo dober!

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

    ✅ Preberite več o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble razvrščevalniki

Sledimo poti do samega konca, čeprav je bil prejšnji test zelo dober. Poskusimo nekaj 'Ensemble razvrščevalnikov', natančneje Random Forest in AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultat je zelo dober, še posebej za Random Forest:

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

✅ Preberite več o [Ensemble razvrščevalnikih](https://scikit-learn.org/stable/modules/ensemble.html)

Ta metoda strojnega učenja "združuje napovedi več osnovnih ocenjevalnikov", da izboljša kakovost modela. V našem primeru smo uporabili Random Trees in AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda povprečenja, gradi 'gozd' 'odločilnih dreves', ki so prežeta z naključnostjo, da se izogne prekomernemu prileganju. Parameter n_estimators je nastavljen na število dreves.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) prilagodi razvrščevalnik naboru podatkov in nato prilagodi kopije tega razvrščevalnika istemu naboru podatkov. Osredotoča se na uteži nepravilno razvrščenih elementov in prilagodi prileganje za naslednji razvrščevalnik, da jih popravi.

---

## 🚀Izziv

Vsaka od teh tehnik ima veliko število parametrov, ki jih lahko prilagodite. Raziskujte privzete parametre vsake metode in razmislite, kaj bi pomenilo prilagajanje teh parametrov za kakovost modela.

## [Zaključni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

V teh lekcijah je veliko žargona, zato si vzemite trenutek za pregled [tega seznama](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) uporabne terminologije!

## Naloga 

[Parameter play](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni prevod s strani človeka. Ne prevzemamo odgovornosti za morebitna napačna razumevanja ali napačne interpretacije, ki bi nastale zaradi uporabe tega prevoda.