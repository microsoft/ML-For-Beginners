<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T00:49:40+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "cs"
}
-->
# Klasifikátory kuchyní 2

V této druhé lekci o klasifikaci se podíváte na další způsoby klasifikace číselných dat. Také se dozvíte o důsledcích volby jednoho klasifikátoru oproti jinému.

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Předpoklady

Předpokládáme, že jste dokončili předchozí lekce a máte vyčištěný dataset ve složce `data` s názvem _cleaned_cuisines.csv_ v kořenovém adresáři této čtyřlekční složky.

### Příprava

Načtený soubor _notebook.ipynb_ obsahuje vyčištěný dataset, který jsme rozdělili na datové rámce X a y, připravené pro proces tvorby modelu.

## Mapa klasifikace

Dříve jste se naučili o různých možnostech klasifikace dat pomocí cheat sheetu od Microsoftu. Scikit-learn nabízí podobný, ale podrobnější cheat sheet, který vám může pomoci ještě více zúžit výběr odhadovačů (další termín pro klasifikátory):

![ML Map from Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [navštivte tuto mapu online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) a klikněte na cestu, abyste si přečetli dokumentaci.

### Plán

Tato mapa je velmi užitečná, jakmile máte jasnou představu o svých datech, protože se můžete „procházet“ jejími cestami k rozhodnutí:

- Máme >50 vzorků
- Chceme předpovědět kategorii
- Máme označená data
- Máme méně než 100K vzorků
- ✨ Můžeme zvolit Linear SVC
- Pokud to nefunguje, protože máme číselná data
    - Můžeme zkusit ✨ KNeighbors Classifier 
      - Pokud to nefunguje, zkusíme ✨ SVC a ✨ Ensemble Classifiers

Toto je velmi užitečná cesta, kterou se můžeme řídit.

## Cvičení - rozdělení dat

Podle této cesty bychom měli začít importem některých knihoven, které budeme používat.

1. Importujte potřebné knihovny:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Rozdělte svá tréninková a testovací data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klasifikátor

Support-Vector clustering (SVC) je součástí rodiny technik strojového učení Support-Vector Machines (více o nich níže). V této metodě můžete zvolit „kernel“, který určuje, jak se budou štítky seskupovat. Parametr 'C' se týká 'regularizace', která reguluje vliv parametrů. Kernel může být jeden z [několika](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); zde jej nastavíme na 'linear', abychom využili lineární SVC. Pravděpodobnost je ve výchozím nastavení 'false'; zde ji nastavíme na 'true', abychom získali odhady pravděpodobnosti. Náhodný stav nastavíme na '0', abychom data zamíchali a získali pravděpodobnosti.

### Cvičení - aplikace lineárního SVC

Začněte vytvořením pole klasifikátorů. Do tohoto pole budete postupně přidávat, jak budeme testovat.

1. Začněte s Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Natrénujte svůj model pomocí Linear SVC a vytiskněte zprávu:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Výsledek je docela dobrý:

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

K-Neighbors je součástí rodiny metod strojového učení „neighbors“, které lze použít pro řízené i neřízené učení. V této metodě je vytvořen předem definovaný počet bodů a data se shromažďují kolem těchto bodů tak, aby bylo možné předpovědět obecné štítky pro data.

### Cvičení - aplikace K-Neighbors klasifikátoru

Předchozí klasifikátor byl dobrý a fungoval dobře s daty, ale možná můžeme dosáhnout lepší přesnosti. Zkuste K-Neighbors klasifikátor.

1. Přidejte řádek do svého pole klasifikátorů (přidejte čárku za položku Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Výsledek je o něco horší:

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

    ✅ Naučte se více o [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klasifikátory jsou součástí rodiny metod strojového učení [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), které se používají pro klasifikační a regresní úlohy. SVM „mapují tréninkové příklady na body v prostoru“, aby maximalizovaly vzdálenost mezi dvěma kategoriemi. Následná data jsou mapována do tohoto prostoru, aby bylo možné předpovědět jejich kategorii.

### Cvičení - aplikace Support Vector Classifier

Zkusme dosáhnout o něco lepší přesnosti pomocí Support Vector Classifier.

1. Přidejte čárku za položku K-Neighbors a poté přidejte tento řádek:

    ```python
    'SVC': SVC(),
    ```

    Výsledek je velmi dobrý!

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

    ✅ Naučte se více o [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Pojďme se vydat cestou až na konec, i když předchozí test byl velmi dobrý. Zkusme některé „Ensemble Classifiers“, konkrétně Random Forest a AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Výsledek je velmi dobrý, zejména u Random Forest:

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

✅ Naučte se více o [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Tato metoda strojového učení „kombinuje předpovědi několika základních odhadovačů“, aby zlepšila kvalitu modelu. V našem příkladu jsme použili Random Trees a AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metoda průměrování, vytváří „les“ „rozhodovacích stromů“ naplněných náhodností, aby se zabránilo přeučení. Parametr n_estimators je nastaven na počet stromů.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) přizpůsobí klasifikátor datasetu a poté přizpůsobí kopie tohoto klasifikátoru stejnému datasetu. Zaměřuje se na váhy nesprávně klasifikovaných položek a upravuje fit pro další klasifikátor, aby provedl opravu.

---

## 🚀Výzva

Každá z těchto technik má velké množství parametrů, které můžete upravit. Prozkoumejte výchozí parametry každé z nich a přemýšlejte o tom, co by znamenalo upravení těchto parametrů pro kvalitu modelu.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

V těchto lekcích je hodně odborných termínů, takže si udělejte chvíli na přehled [tohoto seznamu](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) užitečné terminologie!

## Zadání 

[Hra s parametry](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.