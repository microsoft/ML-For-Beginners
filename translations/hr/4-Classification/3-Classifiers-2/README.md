<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T13:14:16+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "hr"
}
-->
# Klasifikatori kuhinja 2

U ovoj drugoj lekciji o klasifikaciji istražit ćete više načina za klasifikaciju numeričkih podataka. Također ćete naučiti o posljedicama odabira jednog klasifikatora u odnosu na drugi.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Preduvjeti

Pretpostavljamo da ste završili prethodne lekcije i da imate očišćeni skup podataka u svojoj mapi `data` pod nazivom _cleaned_cuisines.csv_ u korijenu ove mape s 4 lekcije.

### Priprema

Učitali smo vašu datoteku _notebook.ipynb_ s očišćenim skupom podataka i podijelili je u X i y podatkovne okvire, spremne za proces izgradnje modela.

## Karta klasifikacije

Ranije ste naučili o raznim opcijama koje imate pri klasifikaciji podataka koristeći Microsoftov vodič. Scikit-learn nudi sličan, ali detaljniji vodič koji može dodatno pomoći u sužavanju izbora procjenitelja (drugi naziv za klasifikatore):

![ML Karta iz Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Savjet: [posjetite ovu kartu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) i kliknite na putanju kako biste pročitali dokumentaciju.

### Plan

Ova karta je vrlo korisna kada imate jasno razumijevanje svojih podataka, jer možete 'hodati' njenim stazama do odluke:

- Imamo >50 uzoraka
- Želimo predvidjeti kategoriju
- Imamo označene podatke
- Imamo manje od 100K uzoraka
- ✨ Možemo odabrati Linear SVC
- Ako to ne uspije, budući da imamo numeričke podatke
    - Možemo pokušati ✨ KNeighbors Classifier 
      - Ako to ne uspije, pokušajte ✨ SVC i ✨ Ensemble Classifiers

Ovo je vrlo koristan put za slijediti.

## Vježba - podijelite podatke

Slijedeći ovu putanju, trebali bismo započeti uvozom nekih biblioteka koje ćemo koristiti.

1. Uvezite potrebne biblioteke:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Podijelite svoje podatke za treniranje i testiranje:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klasifikator

Support-Vector clustering (SVC) je dio obitelji tehnika strojnog učenja Support-Vector machines (SVM) (saznajte više o njima dolje). U ovoj metodi možete odabrati 'kernel' kako biste odlučili kako grupirati oznake. Parametar 'C' odnosi se na 'regularizaciju' koja regulira utjecaj parametara. Kernel može biti jedan od [nekoliko](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); ovdje ga postavljamo na 'linear' kako bismo osigurali korištenje linearne SVC. Vjerojatnost je prema zadanim postavkama 'false'; ovdje je postavljamo na 'true' kako bismo dobili procjene vjerojatnosti. Postavljamo random state na '0' kako bismo promiješali podatke za dobivanje vjerojatnosti.

### Vježba - primijenite linearni SVC

Započnite stvaranjem niza klasifikatora. Postupno ćete dodavati ovom nizu dok testiramo.

1. Započnite s Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Trenirajte svoj model koristeći Linear SVC i ispišite izvještaj:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultat je prilično dobar:

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

## K-Neighbors klasifikator

K-Neighbors je dio obitelji metoda strojnog učenja "neighbors", koje se mogu koristiti za nadzirano i nenadzirano učenje. U ovoj metodi unaprijed definirani broj točaka se stvara, a podaci se prikupljaju oko tih točaka kako bi se predvidjele generalizirane oznake za podatke.

### Vježba - primijenite K-Neighbors klasifikator

Prethodni klasifikator je bio dobar i dobro je radio s podacima, ali možda možemo postići bolju točnost. Pokušajte s K-Neighbors klasifikatorom.

1. Dodajte liniju svom nizu klasifikatora (dodajte zarez nakon stavke Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultat je malo lošiji:

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

    ✅ Saznajte više o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikatori su dio obitelji metoda strojnog učenja [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) koje se koriste za zadatke klasifikacije i regresije. SVM "mapira primjere za treniranje na točke u prostoru" kako bi maksimizirao udaljenost između dvije kategorije. Naknadni podaci se mapiraju u ovaj prostor kako bi se predvidjela njihova kategorija.

### Vježba - primijenite Support Vector Classifier

Pokušajmo postići malo bolju točnost s Support Vector Classifier.

1. Dodajte zarez nakon stavke K-Neighbors, a zatim dodajte ovu liniju:

    ```python
    'SVC': SVC(),
    ```

    Rezultat je vrlo dobar!

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

    ✅ Saznajte više o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Slijedimo put do samog kraja, iako je prethodni test bio vrlo dobar. Pokušajmo s 'Ensemble Classifiers', konkretno Random Forest i AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultat je vrlo dobar, posebno za Random Forest:

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

✅ Saznajte više o [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Ova metoda strojnog učenja "kombinira predviđanja nekoliko osnovnih procjenitelja" kako bi poboljšala kvalitetu modela. U našem primjeru koristili smo Random Trees i AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda prosjeka, gradi 'šumu' 'odlučujućih stabala' s dodanom slučajnošću kako bi se izbjeglo prekomjerno prilagođavanje. Parametar n_estimators postavljen je na broj stabala.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) prilagođava klasifikator skupu podataka, a zatim prilagođava kopije tog klasifikatora istom skupu podataka. Fokusira se na težine pogrešno klasificiranih stavki i prilagođava fit za sljedeći klasifikator kako bi ih ispravio.

---

## 🚀Izazov

Svaka od ovih tehnika ima veliki broj parametara koje možete prilagoditi. Istražite zadane parametre svake od njih i razmislite što bi prilagodba tih parametara značila za kvalitetu modela.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

U ovim lekcijama ima puno stručnih izraza, pa odvojite trenutak da pregledate [ovaj popis](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) korisne terminologije!

## Zadatak 

[Parametarska igra](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.