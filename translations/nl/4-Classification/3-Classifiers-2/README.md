<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T19:55:53+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "nl"
}
-->
# Culinair Classificators 2

In deze tweede les over classificatie ga je meer manieren verkennen om numerieke gegevens te classificeren. Je leert ook over de gevolgen van het kiezen van de ene classifier boven de andere.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Vereisten

We gaan ervan uit dat je de vorige lessen hebt voltooid en een opgeschoonde dataset hebt in je `data` map genaamd _cleaned_cuisines.csv_ in de hoofdmap van deze 4-lessen folder.

### Voorbereiding

We hebben je _notebook.ipynb_ bestand geladen met de opgeschoonde dataset en deze verdeeld in X- en y-dataframes, klaar voor het modelbouwproces.

## Een classificatiekaart

Eerder heb je geleerd over de verschillende opties die je hebt bij het classificeren van gegevens met behulp van Microsoft's cheat sheet. Scikit-learn biedt een vergelijkbare, maar meer gedetailleerde cheat sheet die je verder kan helpen bij het verfijnen van je keuzes voor classifiers (ook wel estimators genoemd):

![ML Map van Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [bezoek deze kaart online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) en klik langs het pad om documentatie te lezen.

### Het plan

Deze kaart is erg handig zodra je een goed begrip hebt van je gegevens, omdat je langs de paden kunt 'wandelen' naar een beslissing:

- We hebben >50 samples
- We willen een categorie voorspellen
- We hebben gelabelde gegevens
- We hebben minder dan 100K samples
- ✨ We kunnen een Linear SVC kiezen
- Als dat niet werkt, omdat we numerieke gegevens hebben:
    - Kunnen we een ✨ KNeighbors Classifier proberen
      - Als dat niet werkt, probeer ✨ SVC en ✨ Ensemble Classifiers

Dit is een zeer nuttig pad om te volgen.

## Oefening - splits de gegevens

Volgens dit pad moeten we beginnen met het importeren van enkele bibliotheken die we gaan gebruiken.

1. Importeer de benodigde bibliotheken:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Splits je trainings- en testgegevens:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector clustering (SVC) is een onderdeel van de Support-Vector machines familie van ML-technieken (leer meer hierover hieronder). Bij deze methode kun je een 'kernel' kiezen om te bepalen hoe de labels worden gegroepeerd. De 'C'-parameter verwijst naar 'regularisatie', wat de invloed van parameters reguleert. De kernel kan een van [meerdere](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) zijn; hier stellen we deze in op 'linear' om ervoor te zorgen dat we gebruik maken van lineaire SVC. Probability staat standaard op 'false'; hier stellen we deze in op 'true' om waarschijnlijkheidschattingen te verzamelen. We stellen de random state in op '0' om de gegevens te schudden en waarschijnlijkheden te verkrijgen.

### Oefening - pas een lineaire SVC toe

Begin met het maken van een array van classifiers. Je zult deze array geleidelijk uitbreiden terwijl we testen.

1. Begin met een Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Train je model met de Linear SVC en print een rapport:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Het resultaat is behoorlijk goed:

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

## K-Neighbors classifier

K-Neighbors is onderdeel van de "neighbors" familie van ML-methoden, die zowel voor supervised als unsupervised learning kunnen worden gebruikt. Bij deze methode wordt een vooraf bepaald aantal punten gecreëerd en worden gegevens verzameld rond deze punten, zodat gegeneraliseerde labels kunnen worden voorspeld voor de gegevens.

### Oefening - pas de K-Neighbors classifier toe

De vorige classifier was goed en werkte goed met de gegevens, maar misschien kunnen we een betere nauwkeurigheid behalen. Probeer een K-Neighbors classifier.

1. Voeg een regel toe aan je classifier array (voeg een komma toe na het Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Het resultaat is iets slechter:

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

    ✅ Leer meer over [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector classifiers zijn onderdeel van de [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) familie van ML-methoden die worden gebruikt voor classificatie- en regressietaken. SVMs "mappen trainingsvoorbeelden naar punten in de ruimte" om de afstand tussen twee categorieën te maximaliseren. Vervolgens worden gegevens in deze ruimte gemapt zodat hun categorie kan worden voorspeld.

### Oefening - pas een Support Vector Classifier toe

Laten we proberen een iets betere nauwkeurigheid te behalen met een Support Vector Classifier.

1. Voeg een komma toe na het K-Neighbors item en voeg vervolgens deze regel toe:

    ```python
    'SVC': SVC(),
    ```

    Het resultaat is behoorlijk goed!

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

    ✅ Leer meer over [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Laten we het pad helemaal volgen, ook al was de vorige test behoorlijk goed. Laten we enkele 'Ensemble Classifiers' proberen, specifiek Random Forest en AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Het resultaat is erg goed, vooral voor Random Forest:

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

✅ Leer meer over [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Deze methode van Machine Learning "combineert de voorspellingen van verschillende basisestimators" om de kwaliteit van het model te verbeteren. In ons voorbeeld hebben we Random Trees en AdaBoost gebruikt.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), een gemiddelde methode, bouwt een 'forest' van 'decision trees' met willekeurigheid om overfitting te voorkomen. De n_estimators parameter wordt ingesteld op het aantal bomen.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) past een classifier toe op een dataset en past vervolgens kopieën van die classifier toe op dezelfde dataset. Het richt zich op de gewichten van verkeerd geclassificeerde items en past de fit aan voor de volgende classifier om correcties aan te brengen.

---

## 🚀Uitdaging

Elk van deze technieken heeft een groot aantal parameters die je kunt aanpassen. Onderzoek de standaardparameters van elk en denk na over wat het aanpassen van deze parameters zou betekenen voor de kwaliteit van het model.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Er is veel jargon in deze lessen, dus neem even de tijd om [deze lijst](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) met nuttige terminologie te bekijken!

## Opdracht 

[Parameter spelen](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.