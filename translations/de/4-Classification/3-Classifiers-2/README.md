# Küchenklassifizierer 2

In dieser zweiten Klassifikationslektion werden Sie weitere Möglichkeiten erkunden, numerische Daten zu klassifizieren. Sie werden auch die Auswirkungen der Wahl eines Klassifizierers gegenüber einem anderen kennenlernen.

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23/)

### Voraussetzung

Wir gehen davon aus, dass Sie die vorherigen Lektionen abgeschlossen haben und einen bereinigten Datensatz in Ihrem `data`-Ordner haben, der _cleaned_cuisines.csv_ im Wurzelverzeichnis dieses 4-Lektionen-Ordners heißt.

### Vorbereitung

Wir haben Ihre _notebook.ipynb_-Datei mit dem bereinigten Datensatz geladen und in X- und y-Datenrahmen unterteilt, bereit für den Modellierungsprozess.

## Eine Klassifikationskarte

Früher haben Sie die verschiedenen Optionen kennengelernt, die Sie beim Klassifizieren von Daten mit Microsofts Spickzettel haben. Scikit-learn bietet einen ähnlichen, aber detaillierteren Spickzettel, der Ihnen helfen kann, Ihre Schätzer (ein anderer Begriff für Klassifizierer) weiter einzugrenzen:

![ML-Karte von Scikit-learn](../../../../translated_images/map.e963a6a51349425ab107b38f6c7307eb4c0d0c7ccdd2e81a5e1919292bab9ac7.de.png)
> Tipp: [Besuchen Sie diese Karte online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) und klicken Sie entlang des Pfades, um die Dokumentation zu lesen.

### Der Plan

Diese Karte ist sehr hilfreich, sobald Sie ein klares Verständnis Ihrer Daten haben, da Sie 'entlang ihrer Pfade' zu einer Entscheidung 'gehen' können:

- Wir haben >50 Proben
- Wir möchten eine Kategorie vorhersagen
- Wir haben beschriftete Daten
- Wir haben weniger als 100K Proben
- ✨ Wir können einen linearen SVC wählen
- Wenn das nicht funktioniert, da wir numerische Daten haben
    - Können wir einen ✨ KNeighbors-Klassifizierer ausprobieren 
      - Wenn das nicht funktioniert, versuchen Sie ✨ SVC und ✨ Ensemble-Klassifizierer

Das ist ein sehr hilfreicher Weg, dem man folgen kann.

## Übung - Daten aufteilen

Folgen Sie diesem Pfad, sollten wir zunächst einige Bibliotheken importieren.

1. Importieren Sie die benötigten Bibliotheken:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Teilen Sie Ihre Trainings- und Testdaten auf:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linearer SVC-Klassifizierer

Support-Vektor-Klassifizierung (SVC) ist ein Teil der Familie der Support-Vektor-Maschinen von ML-Techniken (erfahren Sie mehr darüber weiter unten). Bei dieser Methode können Sie einen 'Kernel' wählen, um zu entscheiden, wie die Labels gruppiert werden. Der Parameter 'C' bezieht sich auf die 'Regularisierung', die den Einfluss der Parameter reguliert. Der Kernel kann einer von [mehreren](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) sein; hier setzen wir ihn auf 'linear', um sicherzustellen, dass wir den linearen SVC nutzen. Die Wahrscheinlichkeit ist standardmäßig auf 'false' gesetzt; hier setzen wir sie auf 'true', um Wahrscheinlichkeitsabschätzungen zu sammeln. Wir setzen den Zufallsstatus auf '0', um die Daten zu mischen und Wahrscheinlichkeiten zu erhalten.

### Übung - einen linearen SVC anwenden

Beginnen Sie damit, ein Array von Klassifizierern zu erstellen. Sie werden dieses Array schrittweise erweitern, während wir testen. 

1. Beginnen Sie mit einem linearen SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Trainieren Sie Ihr Modell mit dem linearen SVC und drucken Sie einen Bericht aus:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Das Ergebnis ist ziemlich gut:

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

## K-Neighbors-Klassifizierer

K-Neighbors gehört zur Familie der "Nachbarn"-Methoden von ML, die sowohl für überwachtes als auch für unüberwachtes Lernen verwendet werden können. Bei dieser Methode wird eine vordefinierte Anzahl von Punkten erstellt, und Daten werden um diese Punkte herum gesammelt, sodass verallgemeinerte Labels für die Daten vorhergesagt werden können.

### Übung - den K-Neighbors-Klassifizierer anwenden

Der vorherige Klassifizierer war gut und hat gut mit den Daten funktioniert, aber vielleicht können wir eine bessere Genauigkeit erzielen. Probieren Sie einen K-Neighbors-Klassifizierer aus.

1. Fügen Sie eine Zeile zu Ihrem Klassifizierer-Array hinzu (fügen Sie ein Komma nach dem Element des linearen SVC hinzu):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Das Ergebnis ist etwas schlechter:

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

    ✅ Erfahren Sie mehr über [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support-Vektor-Klassifizierer

Support-Vektor-Klassifizierer sind Teil der [Support-Vektor-Maschinen](https://wikipedia.org/wiki/Support-vector_machine) Familie von ML-Methoden, die für Klassifikations- und Regressionsaufgaben verwendet werden. SVMs "karten Trainingsbeispiele in Punkte im Raum" ab, um den Abstand zwischen zwei Kategorien zu maximieren. Nachfolgende Daten werden in diesen Raum abgebildet, damit ihre Kategorie vorhergesagt werden kann.

### Übung - einen Support-Vektor-Klassifizierer anwenden

Versuchen wir, eine etwas bessere Genauigkeit mit einem Support-Vektor-Klassifizierer zu erzielen.

1. Fügen Sie ein Komma nach dem K-Neighbors-Element hinzu und fügen Sie dann diese Zeile hinzu:

    ```python
    'SVC': SVC(),
    ```

    Das Ergebnis ist ziemlich gut!

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

    ✅ Erfahren Sie mehr über [Support-Vektoren](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble-Klassifizierer

Lassen Sie uns den Weg bis zum Ende verfolgen, auch wenn der vorherige Test ziemlich gut war. Lassen Sie uns einige 'Ensemble-Klassifizierer, speziell Random Forest und AdaBoost, ausprobieren:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Das Ergebnis ist sehr gut, insbesondere für Random Forest:

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

✅ Erfahren Sie mehr über [Ensemble-Klassifizierer](https://scikit-learn.org/stable/modules/ensemble.html)

Diese Methode des maschinellen Lernens "kombiniert die Vorhersagen mehrerer Basis-Schätzer", um die Qualität des Modells zu verbessern. In unserem Beispiel haben wir Random Trees und AdaBoost verwendet.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), eine Durchschnittsmethode, erstellt einen 'Wald' von 'Entscheidungsbäumen', die mit Zufälligkeit durchsetzt sind, um Überanpassung zu vermeiden. Der Parameter n_estimators wird auf die Anzahl der Bäume gesetzt.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) passt einen Klassifizierer an einen Datensatz an und passt dann Kopien dieses Klassifizierers an denselben Datensatz an. Es konzentriert sich auf die Gewichte falsch klassifizierter Elemente und passt die Anpassung für den nächsten Klassifizierer an, um dies zu korrigieren.

---

## 🚀Herausforderung

Jede dieser Techniken hat eine große Anzahl von Parametern, die Sie anpassen können. Recherchieren Sie die Standardparameter jedes einzelnen und überlegen Sie, was es für die Qualität des Modells bedeuten würde, diese Parameter anzupassen.

## [Nachlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24/)

## Überprüfung & Selbststudium

In diesen Lektionen gibt es eine Menge Fachbegriffe, also nehmen Sie sich einen Moment Zeit, um [diese Liste](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) nützlicher Terminologie zu überprüfen!

## Aufgabe 

[Parameter spielen](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, bitten wir zu beachten, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner Ursprungssprache sollte als die maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.