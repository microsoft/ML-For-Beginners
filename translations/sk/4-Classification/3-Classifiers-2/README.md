<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T16:23:53+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "sk"
}
-->
# Klasifikátory kuchýň 2

V tejto druhej lekcii o klasifikácii preskúmate ďalšie spôsoby klasifikácie číselných údajov. Tiež sa dozviete o dôsledkoch výberu jedného klasifikátora oproti druhému.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Predpoklady

Predpokladáme, že ste dokončili predchádzajúce lekcie a máte vyčistený dataset vo vašom priečinku `data` s názvom _cleaned_cuisines.csv_ v koreňovom adresári tejto 4-lekciovej zložky.

### Príprava

Načítali sme váš súbor _notebook.ipynb_ s vyčisteným datasetom a rozdelili ho na dátové rámce X a y, pripravené na proces vytvárania modelu.

## Mapa klasifikácie

Predtým ste sa naučili o rôznych možnostiach klasifikácie údajov pomocou cheat sheetu od Microsoftu. Scikit-learn ponúka podobný, ale podrobnejší cheat sheet, ktorý vám môže pomôcť ešte viac zúžiť výber odhadovačov (iný termín pre klasifikátory):

![ML Map from Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [navštívte túto mapu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) a kliknite na cesty, aby ste si prečítali dokumentáciu.

### Plán

Táto mapa je veľmi užitočná, keď máte jasný prehľad o svojich údajoch, pretože sa môžete „prejsť“ po jej cestách k rozhodnutiu:

- Máme >50 vzoriek
- Chceme predpovedať kategóriu
- Máme označené údaje
- Máme menej ako 100K vzoriek
- ✨ Môžeme zvoliť Linear SVC
- Ak to nefunguje, keďže máme číselné údaje
    - Môžeme skúsiť ✨ KNeighbors Classifier 
      - Ak to nefunguje, skúste ✨ SVC a ✨ Ensemble Classifiers

Toto je veľmi užitočná cesta, ktorú treba sledovať.

## Cvičenie - rozdelenie údajov

Podľa tejto cesty by sme mali začať importovaním niektorých knižníc na použitie.

1. Importujte potrebné knižnice:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Rozdeľte svoje tréningové a testovacie údaje:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klasifikátor

Support-Vector clustering (SVC) je súčasťou rodiny techník strojového učenia Support-Vector Machines (viac o nich nižšie). V tejto metóde si môžete vybrať „kernel“, ktorý rozhoduje o tom, ako sa budú označenia zoskupovať. Parameter 'C' sa týka 'regularizácie', ktorá reguluje vplyv parametrov. Kernel môže byť jeden z [niekoľkých](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tu ho nastavíme na 'linear', aby sme využili Linear SVC. Pravdepodobnosť je predvolene nastavená na 'false'; tu ju nastavíme na 'true', aby sme získali odhady pravdepodobnosti. Random state nastavíme na '0', aby sme premiešali údaje na získanie pravdepodobností.

### Cvičenie - aplikácia Linear SVC

Začnite vytvorením poľa klasifikátorov. Postupne budete pridávať do tohto poľa, ako budeme testovať.

1. Začnite s Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Natrénujte svoj model pomocou Linear SVC a vytlačte správu:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Výsledok je celkom dobrý:

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

## K-Neighbors klasifikátor

K-Neighbors je súčasťou rodiny metód strojového učenia „neighbors“, ktoré sa dajú použiť na riadené aj neriadené učenie. V tejto metóde sa vytvorí preddefinovaný počet bodov a údaje sa zhromažďujú okolo týchto bodov tak, aby sa dali predpovedať všeobecné označenia pre údaje.

### Cvičenie - aplikácia K-Neighbors klasifikátora

Predchádzajúci klasifikátor bol dobrý a fungoval dobre s údajmi, ale možno môžeme dosiahnuť lepšiu presnosť. Skúste K-Neighbors klasifikátor.

1. Pridajte riadok do svojho poľa klasifikátorov (pridajte čiarku za položku Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Výsledok je o niečo horší:

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

    ✅ Zistite viac o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector klasifikátor

Support-Vector klasifikátory sú súčasťou rodiny metód strojového učenia [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), ktoré sa používajú na klasifikačné a regresné úlohy. SVM „mapujú tréningové príklady na body v priestore“, aby maximalizovali vzdialenosť medzi dvoma kategóriami. Následné údaje sa mapujú do tohto priestoru, aby sa dala predpovedať ich kategória.

### Cvičenie - aplikácia Support Vector klasifikátora

Skúsme dosiahnuť o niečo lepšiu presnosť pomocou Support Vector klasifikátora.

1. Pridajte čiarku za položku K-Neighbors a potom pridajte tento riadok:

    ```python
    'SVC': SVC(),
    ```

    Výsledok je veľmi dobrý!

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

    ✅ Zistite viac o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble klasifikátory

Poďme sledovať cestu až do konca, aj keď predchádzajúci test bol veľmi dobrý. Skúsme niektoré 'Ensemble Classifiers', konkrétne Random Forest a AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Výsledok je veľmi dobrý, najmä pre Random Forest:

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

✅ Zistite viac o [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Táto metóda strojového učenia „kombinuje predpovede niekoľkých základných odhadovačov“, aby zlepšila kvalitu modelu. V našom príklade sme použili Random Trees a AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metóda priemerovania, vytvára „les“ z „rozhodovacích stromov“ naplnených náhodnosťou, aby sa zabránilo pretrénovaniu. Parameter n_estimators je nastavený na počet stromov.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) prispôsobí klasifikátor datasetu a potom prispôsobí kópie tohto klasifikátora rovnakému datasetu. Zameriava sa na váhy nesprávne klasifikovaných položiek a upravuje fit pre ďalší klasifikátor, aby ich opravil.

---

## 🚀Výzva

Každá z týchto techník má veľké množstvo parametrov, ktoré môžete upraviť. Preskúmajte predvolené parametre každého z nich a premýšľajte o tom, čo by znamenalo upravenie týchto parametrov pre kvalitu modelu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

V týchto lekciách je veľa odborných výrazov, takže si chvíľu preštudujte [tento zoznam](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) užitočnej terminológie!

## Zadanie 

[Parameter play](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.