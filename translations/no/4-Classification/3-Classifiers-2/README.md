<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T21:54:02+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "no"
}
-->
# Klassifisering av matretter 2

I denne andre leksjonen om klassifisering vil du utforske flere måter å klassifisere numeriske data på. Du vil også lære om konsekvensene ved å velge én klassifikator fremfor en annen.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

### Forutsetninger

Vi antar at du har fullført de tidligere leksjonene og har et renset datasett i `data`-mappen kalt _cleaned_cuisines.csv_ i roten av denne 4-leksjonsmappen.

### Forberedelse

Vi har lastet inn filen din _notebook.ipynb_ med det rensede datasettet og har delt det inn i X- og y-dataframes, klare for modellbyggingsprosessen.

## Et klassifiseringskart

Tidligere lærte du om de ulike alternativene du har når du klassifiserer data ved hjelp av Microsofts jukselapp. Scikit-learn tilbyr en lignende, men mer detaljert jukselapp som kan hjelpe deg med å snevre inn valget av estimatoren (et annet ord for klassifikator):

![ML-kart fra Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tips: [besøk dette kartet online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) og klikk langs stien for å lese dokumentasjonen.

### Planen

Dette kartet er veldig nyttig når du har en klar forståelse av dataene dine, da du kan 'gå' langs stiene til en beslutning:

- Vi har >50 prøver
- Vi ønsker å forutsi en kategori
- Vi har merket data
- Vi har færre enn 100K prøver
- ✨ Vi kan velge en Linear SVC
- Hvis det ikke fungerer, siden vi har numeriske data
    - Vi kan prøve en ✨ KNeighbors Classifier 
      - Hvis det ikke fungerer, prøv ✨ SVC og ✨ Ensemble Classifiers

Dette er en veldig nyttig sti å følge.

## Øvelse - del opp dataene

Ved å følge denne stien bør vi starte med å importere noen biblioteker vi skal bruke.

1. Importer de nødvendige bibliotekene:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Del opp trenings- og testdataene:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC klassifikator

Support-Vector clustering (SVC) er en del av Support-Vector-maskiner-familien av ML-teknikker (lær mer om disse nedenfor). I denne metoden kan du velge en 'kernel' for å bestemme hvordan etikettene skal grupperes. Parameteren 'C' refererer til 'regularisering', som regulerer påvirkningen av parametere. Kernel kan være en av [flere](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); her setter vi den til 'linear' for å sikre at vi bruker lineær SVC. Sannsynlighet er som standard satt til 'false'; her setter vi den til 'true' for å samle sannsynlighetsestimater. Vi setter den tilfeldige tilstanden til '0' for å blande dataene for å få sannsynligheter.

### Øvelse - bruk en lineær SVC

Start med å lage en array av klassifikatorer. Du vil legge til gradvis i denne arrayen mens vi tester.

1. Start med en Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Tren modellen din ved hjelp av Linear SVC og skriv ut en rapport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Resultatet er ganske bra:

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

## K-Neighbors klassifikator

K-Neighbors er en del av "neighbors"-familien av ML-metoder, som kan brukes til både overvåket og ikke-overvåket læring. I denne metoden opprettes et forhåndsdefinert antall punkter, og data samles rundt disse punktene slik at generaliserte etiketter kan forutsies for dataene.

### Øvelse - bruk K-Neighbors klassifikator

Den forrige klassifikatoren var god og fungerte bra med dataene, men kanskje vi kan få bedre nøyaktighet. Prøv en K-Neighbors klassifikator.

1. Legg til en linje i klassifikator-arrayen din (legg til en komma etter Linear SVC-elementet):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Resultatet er litt dårligere:

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

    ✅ Lær om [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector klassifikatorer er en del av [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine)-familien av ML-metoder som brukes til klassifiserings- og regresjonsoppgaver. SVMs "kartlegger treningsprøver til punkter i rommet" for å maksimere avstanden mellom to kategorier. Påfølgende data kartlegges inn i dette rommet slik at deres kategori kan forutsies.

### Øvelse - bruk en Support Vector Classifier

La oss prøve å få litt bedre nøyaktighet med en Support Vector Classifier.

1. Legg til en komma etter K-Neighbors-elementet, og legg deretter til denne linjen:

    ```python
    'SVC': SVC(),
    ```

    Resultatet er ganske bra!

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

    ✅ Lær om [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

La oss følge stien helt til slutten, selv om den forrige testen var ganske bra. La oss prøve noen 'Ensemble Classifiers', spesielt Random Forest og AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Resultatet er veldig bra, spesielt for Random Forest:

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

✅ Lær om [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Denne metoden for maskinlæring "kombinerer prediksjonene fra flere base-estimatorer" for å forbedre modellens kvalitet. I vårt eksempel brukte vi Random Trees og AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), en gjennomsnittsmetode, bygger en 'skog' av 'beslutningstrær' med innført tilfeldighet for å unngå overtilpasning. Parameteren n_estimators er satt til antall trær.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) tilpasser en klassifikator til et datasett og deretter tilpasser kopier av den klassifikatoren til det samme datasettet. Den fokuserer på vektene til feilklassifiserte elementer og justerer tilpasningen for neste klassifikator for å korrigere.

---

## 🚀Utfordring

Hver av disse teknikkene har et stort antall parametere som du kan justere. Undersøk standardparametrene for hver av dem og tenk på hva justering av disse parameterne vil bety for modellens kvalitet.

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Det er mye sjargong i disse leksjonene, så ta et øyeblikk til å gå gjennom [denne listen](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) med nyttig terminologi!

## Oppgave 

[Parameterlek](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.