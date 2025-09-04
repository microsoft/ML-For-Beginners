<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "808a71076f76ae8f5458862a8edd9215",
  "translation_date": "2025-09-03T21:55:21+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "de"
}
-->
# Küchenklassifikatoren 2

In dieser zweiten Lektion zur Klassifikation wirst du weitere Möglichkeiten zur Klassifikation numerischer Daten erkunden. Außerdem wirst du die Auswirkungen kennenlernen, die die Wahl eines Klassifikators gegenüber einem anderen haben kann.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23/)

### Voraussetzungen

Wir gehen davon aus, dass du die vorherigen Lektionen abgeschlossen hast und einen bereinigten Datensatz in deinem `data`-Ordner namens _cleaned_cuisines.csv_ im Hauptverzeichnis dieses 4-Lektionen-Ordners hast.

### Vorbereitung

Wir haben deine _notebook.ipynb_-Datei mit dem bereinigten Datensatz geladen und in X- und y-Datenframes unterteilt, die bereit für den Modellierungsprozess sind.

## Eine Klassifikationskarte

Zuvor hast du die verschiedenen Optionen kennengelernt, die dir bei der Klassifikation von Daten mithilfe des Cheat Sheets von Microsoft zur Verfügung stehen. Scikit-learn bietet ein ähnliches, aber detaillierteres Cheat Sheet, das dir dabei helfen kann, deine Auswahl an Schätzern (ein anderer Begriff für Klassifikatoren) weiter einzugrenzen:

![ML-Karte von Scikit-learn](../../../../translated_images/map.e963a6a51349425ab107b38f6c7307eb4c0d0c7ccdd2e81a5e1919292bab9ac7.de.png)  
> Tipp: [Besuche diese Karte online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) und klicke entlang des Pfades, um die Dokumentation zu lesen.

### Der Plan

Diese Karte ist sehr hilfreich, sobald du ein klares Verständnis deiner Daten hast, da du entlang ihrer Pfade zu einer Entscheidung „gehen“ kannst:

- Wir haben >50 Stichproben
- Wir möchten eine Kategorie vorhersagen
- Wir haben beschriftete Daten
- Wir haben weniger als 100.000 Stichproben
- ✨ Wir können einen Linear SVC wählen
- Falls das nicht funktioniert, da wir numerische Daten haben:
    - Können wir einen ✨ KNeighbors-Klassifikator ausprobieren
      - Falls das nicht funktioniert, probiere ✨ SVC und ✨ Ensemble-Klassifikatoren

Dies ist ein sehr hilfreicher Weg, dem man folgen kann.

## Übung - Daten aufteilen

Entlang dieses Pfades sollten wir zunächst einige Bibliotheken importieren, die wir verwenden möchten.

1. Importiere die benötigten Bibliotheken:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Teile deine Trainings- und Testdaten:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC-Klassifikator

Support-Vector Clustering (SVC) ist ein Mitglied der Familie der Support-Vector-Maschinen-Techniken des maschinellen Lernens (mehr dazu unten). Bei dieser Methode kannst du einen 'Kernel' wählen, um zu entscheiden, wie die Labels gruppiert werden. Der Parameter 'C' bezieht sich auf die 'Regularisierung', die den Einfluss der Parameter reguliert. Der Kernel kann einer von [mehreren](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) sein; hier setzen wir ihn auf 'linear', um sicherzustellen, dass wir Linear SVC nutzen. Die Wahrscheinlichkeit ist standardmäßig auf 'false' gesetzt; hier setzen wir sie auf 'true', um Wahrscheinlichkeitsabschätzungen zu erhalten. Wir setzen den Zufallszustand auf '0', um die Daten zu mischen und Wahrscheinlichkeiten zu erhalten.

### Übung - Linear SVC anwenden

Beginne damit, ein Array von Klassifikatoren zu erstellen. Du wirst dieses Array schrittweise erweitern, während wir testen.

1. Beginne mit einem Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Trainiere dein Modell mit dem Linear SVC und drucke einen Bericht aus:

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

## K-Neighbors-Klassifikator

K-Neighbors ist Teil der Familie der "Nachbarn"-Methoden des maschinellen Lernens, die sowohl für überwachtes als auch für unüberwachtes Lernen verwendet werden können. Bei dieser Methode wird eine vordefinierte Anzahl von Punkten erstellt, und Daten werden um diese Punkte herum gesammelt, sodass generalisierte Labels für die Daten vorhergesagt werden können.

### Übung - K-Neighbors-Klassifikator anwenden

Der vorherige Klassifikator war gut und hat gut mit den Daten funktioniert, aber vielleicht können wir eine bessere Genauigkeit erzielen. Probiere einen K-Neighbors-Klassifikator aus.

1. Füge eine Zeile zu deinem Klassifikator-Array hinzu (füge ein Komma nach dem Linear SVC-Element hinzu):

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

    ✅ Erfahre mehr über [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support-Vector-Klassifikator

Support-Vector-Klassifikatoren sind Teil der [Support-Vector-Maschinen](https://wikipedia.org/wiki/Support-vector_machine)-Familie von ML-Methoden, die für Klassifikations- und Regressionstasks verwendet werden. SVMs "mappen Trainingsbeispiele auf Punkte im Raum", um den Abstand zwischen zwei Kategorien zu maximieren. Nachfolgende Daten werden in diesen Raum gemappt, sodass ihre Kategorie vorhergesagt werden kann.

### Übung - Support-Vector-Klassifikator anwenden

Lass uns versuchen, eine etwas bessere Genauigkeit mit einem Support-Vector-Klassifikator zu erzielen.

1. Füge ein Komma nach dem K-Neighbors-Element hinzu und füge dann diese Zeile hinzu:

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

    ✅ Erfahre mehr über [Support-Vektoren](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble-Klassifikatoren

Lass uns den Pfad bis zum Ende verfolgen, auch wenn der vorherige Test ziemlich gut war. Lass uns einige 'Ensemble-Klassifikatoren' ausprobieren, insbesondere Random Forest und AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Das Ergebnis ist sehr gut, besonders bei Random Forest:

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

✅ Erfahre mehr über [Ensemble-Klassifikatoren](https://scikit-learn.org/stable/modules/ensemble.html)

Diese Methode des maschinellen Lernens "kombiniert die Vorhersagen mehrerer Basis-Schätzer", um die Qualität des Modells zu verbessern. In unserem Beispiel haben wir Random Trees und AdaBoost verwendet.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), eine Durchschnittsmethode, erstellt einen 'Wald' aus 'Entscheidungsbäumen', die mit Zufälligkeit durchsetzt sind, um Überanpassung zu vermeiden. Der Parameter n_estimators wird auf die Anzahl der Bäume gesetzt.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) passt einen Klassifikator an einen Datensatz an und passt dann Kopien dieses Klassifikators an denselben Datensatz an. Es konzentriert sich auf die Gewichte von falsch klassifizierten Elementen und passt die Anpassung für den nächsten Klassifikator an, um diese zu korrigieren.

---

## 🚀 Herausforderung

Jede dieser Techniken hat eine große Anzahl von Parametern, die du anpassen kannst. Recherchiere die Standardparameter jeder Technik und überlege, was das Anpassen dieser Parameter für die Qualität des Modells bedeuten würde.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24/)

## Rückblick & Selbststudium

Es gibt viele Fachbegriffe in diesen Lektionen, also nimm dir einen Moment Zeit, um [diese Liste](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) nützlicher Begriffe zu überprüfen!

## Aufgabe 

[Parameter-Spiel](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.