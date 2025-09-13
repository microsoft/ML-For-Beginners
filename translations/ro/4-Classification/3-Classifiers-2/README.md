<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T16:24:14+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ro"
}
-->
# Clasificatori culinari 2

În această a doua lecție despre clasificare, vei explora mai multe modalități de a clasifica date numerice. De asemenea, vei învăța despre implicațiile alegerii unui clasificator în detrimentul altuia.

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

### Cerințe preliminare

Presupunem că ai finalizat lecțiile anterioare și ai un set de date curățat în folderul `data`, numit _cleaned_cuisines.csv_, în rădăcina acestui folder cu 4 lecții.

### Pregătire

Am încărcat fișierul tău _notebook.ipynb_ cu setul de date curățat și l-am împărțit în cadre de date X și y, pregătite pentru procesul de construire a modelului.

## O hartă a clasificării

Anterior, ai învățat despre diversele opțiuni pe care le ai atunci când clasifici date folosind fișa de ajutor de la Microsoft. Scikit-learn oferă o fișă similară, dar mai detaliată, care te poate ajuta să restrângi și mai mult alegerea estimatoarelor (un alt termen pentru clasificatori):

![Harta ML de la Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Sfat: [vizitează această hartă online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) și explorează căile pentru a citi documentația.

### Planul

Această hartă este foarte utilă odată ce ai o înțelegere clară a datelor tale, deoarece poți „parcurge” căile pentru a lua o decizie:

- Avem >50 de mostre
- Vrem să prezicem o categorie
- Avem date etichetate
- Avem mai puțin de 100K mostre
- ✨ Putem alege un Linear SVC
- Dacă acest lucru nu funcționează, deoarece avem date numerice
    - Putem încerca un ✨ KNeighbors Classifier 
      - Dacă nici acesta nu funcționează, încercăm ✨ SVC și ✨ Ensemble Classifiers

Aceasta este o cale foarte utilă de urmat.

## Exercițiu - împarte datele

Urmând această cale, ar trebui să începem prin a importa câteva biblioteci necesare.

1. Importă bibliotecile necesare:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Împarte datele de antrenament și test:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Clasificator Linear SVC

Support-Vector Clustering (SVC) face parte din familia tehnicilor ML Support-Vector Machines (află mai multe despre acestea mai jos). În această metodă, poți alege un „kernel” pentru a decide cum să grupezi etichetele. Parametrul 'C' se referă la 'regularizare', care reglează influența parametrilor. Kernel-ul poate fi unul dintre [mai multe](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); aici îl setăm la 'linear' pentru a ne asigura că folosim Linear SVC. Probabilitatea este implicit 'false'; aici o setăm la 'true' pentru a obține estimări de probabilitate. Setăm starea aleatorie la '0' pentru a amesteca datele și a obține probabilități.

### Exercițiu - aplică un Linear SVC

Începe prin a crea un array de clasificatori. Vei adăuga progresiv la acest array pe măsură ce testăm.

1. Începe cu un Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Antrenează modelul folosind Linear SVC și afișează un raport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Rezultatul este destul de bun:

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

## Clasificator K-Neighbors

K-Neighbors face parte din familia „neighbors” a metodelor ML, care pot fi utilizate atât pentru învățare supravegheată, cât și nesupravegheată. În această metodă, se creează un număr predefinit de puncte, iar datele sunt grupate în jurul acestor puncte astfel încât să se poată prezice etichete generalizate pentru date.

### Exercițiu - aplică clasificatorul K-Neighbors

Clasificatorul anterior a fost bun și a funcționat bine cu datele, dar poate putem obține o acuratețe mai bună. Încearcă un clasificator K-Neighbors.

1. Adaugă o linie în array-ul de clasificatori (adaugă o virgulă după elementul Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Rezultatul este puțin mai slab:

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

    ✅ Află mai multe despre [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Clasificator Support Vector

Clasificatorii Support-Vector fac parte din familia [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) a metodelor ML utilizate pentru sarcini de clasificare și regresie. SVM-urile „mapează exemplele de antrenament în puncte din spațiu” pentru a maximiza distanța dintre două categorii. Datele ulterioare sunt mapate în acest spațiu astfel încât categoria lor să poată fi prezisă.

### Exercițiu - aplică un Support Vector Classifier

Să încercăm să obținem o acuratețe puțin mai bună cu un Support Vector Classifier.

1. Adaugă o virgulă după elementul K-Neighbors, apoi adaugă această linie:

    ```python
    'SVC': SVC(),
    ```

    Rezultatul este destul de bun!

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

    ✅ Află mai multe despre [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Clasificatori Ensemble

Să urmăm calea până la capăt, chiar dacă testul anterior a fost destul de bun. Să încercăm câțiva 'Clasificatori Ensemble', în special Random Forest și AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Rezultatul este foarte bun, mai ales pentru Random Forest:

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

✅ Află mai multe despre [Clasificatori Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Această metodă de învățare automată „combină predicțiile mai multor estimatori de bază” pentru a îmbunătăți calitatea modelului. În exemplul nostru, am folosit Random Trees și AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), o metodă de mediere, construiește o „pădure” de „arbori de decizie” infuzați cu aleatoriu pentru a evita supraînvățarea. Parametrul n_estimators este setat la numărul de arbori.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ajustează un clasificator pe un set de date și apoi ajustează copii ale acelui clasificator pe același set de date. Se concentrează pe greutățile elementelor clasificate incorect și ajustează potrivirea pentru următorul clasificator pentru a corecta.

---

## 🚀Provocare

Fiecare dintre aceste tehnici are un număr mare de parametri pe care îi poți ajusta. Cercetează parametrii impliciți ai fiecărei metode și gândește-te ce ar însemna ajustarea acestor parametri pentru calitatea modelului.

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Există mulți termeni tehnici în aceste lecții, așa că ia-ți un moment pentru a revizui [această listă](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) de terminologie utilă!

## Temă 

[Joacă-te cu parametrii](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa maternă ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.