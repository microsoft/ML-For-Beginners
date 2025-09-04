<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "f80e513b3279869e7661e3190cc83076",
  "translation_date": "2025-09-03T21:44:27+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "de"
}
-->
# Zeitreihenprognose mit Support Vector Regressor

In der vorherigen Lektion hast du gelernt, wie man mit dem ARIMA-Modell Zeitreihen vorhersagt. Jetzt wirst du das Support Vector Regressor-Modell kennenlernen, ein Regressionsmodell, das zur Vorhersage kontinuierlicher Daten verwendet wird.

## [Quiz vor der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/51/) 

## Einführung

In dieser Lektion wirst du eine spezifische Methode entdecken, um Modelle mit [**SVM**: **S**upport **V**ector **M**achine](https://de.wikipedia.org/wiki/Support_Vector_Machine) für Regression zu erstellen, oder **SVR: Support Vector Regressor**.

### SVR im Kontext von Zeitreihen [^1]

Bevor du die Bedeutung von SVR für die Vorhersage von Zeitreihen verstehst, sind hier einige wichtige Konzepte, die du kennen solltest:

- **Regression:** Eine überwachte Lerntechnik zur Vorhersage kontinuierlicher Werte aus einem gegebenen Satz von Eingaben. Die Idee ist, eine Kurve (oder Linie) im Merkmalsraum zu finden, die die maximale Anzahl von Datenpunkten enthält. [Klicke hier](https://de.wikipedia.org/wiki/Regressionsanalyse) für weitere Informationen.
- **Support Vector Machine (SVM):** Ein Typ von überwachten maschinellen Lernmodellen, die für Klassifikation, Regression und Ausreißererkennung verwendet werden. Das Modell ist eine Hyperebene im Merkmalsraum, die im Fall der Klassifikation als Grenze und im Fall der Regression als beste Anpassungslinie fungiert. In SVM wird häufig eine Kernel-Funktion verwendet, um den Datensatz in einen Raum mit höherer Dimension zu transformieren, sodass sie leichter trennbar sind. [Klicke hier](https://de.wikipedia.org/wiki/Support_Vector_Machine) für weitere Informationen zu SVMs.
- **Support Vector Regressor (SVR):** Ein Typ von SVM, der die beste Anpassungslinie (die im Fall von SVM eine Hyperebene ist) findet, die die maximale Anzahl von Datenpunkten enthält.

### Warum SVR? [^1]

In der letzten Lektion hast du ARIMA kennengelernt, ein sehr erfolgreiches statistisches lineares Modell zur Vorhersage von Zeitreihendaten. Allerdings weisen Zeitreihendaten in vielen Fällen *Nichtlinearität* auf, die von linearen Modellen nicht abgebildet werden kann. In solchen Fällen macht die Fähigkeit von SVM, Nichtlinearität in den Daten für Regression zu berücksichtigen, SVR erfolgreich bei der Vorhersage von Zeitreihen.

## Übung - Erstelle ein SVR-Modell

Die ersten Schritte zur Datenvorbereitung sind die gleichen wie in der vorherigen Lektion über [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Öffne den [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working)-Ordner in dieser Lektion und finde die Datei [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Führe das Notebook aus und importiere die notwendigen Bibliotheken: [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Lade die Daten aus der Datei `/data/energy.csv` in ein Pandas-DataFrame und schaue sie dir an: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Zeichne alle verfügbaren Energiedaten von Januar 2012 bis Dezember 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![vollständige Daten](../../../../translated_images/full-data.a82ec9957e580e976f651a4fc38f280b9229c6efdbe3cfe7c60abaa9486d2cbe.de.png)

   Jetzt erstellen wir unser SVR-Modell.

### Erstelle Trainings- und Testdatensätze

Jetzt sind deine Daten geladen, sodass du sie in Trainings- und Testdatensätze aufteilen kannst. Anschließend formst du die Daten um, um einen zeitbasierten Datensatz zu erstellen, der für das SVR benötigt wird. Du trainierst dein Modell mit dem Trainingsdatensatz. Nachdem das Modell das Training abgeschlossen hat, bewertest du seine Genauigkeit anhand des Trainingsdatensatzes, des Testdatensatzes und dann des vollständigen Datensatzes, um die Gesamtleistung zu sehen. Du musst sicherstellen, dass der Testdatensatz einen späteren Zeitraum als der Trainingsdatensatz abdeckt, um sicherzustellen, dass das Modell keine Informationen aus zukünftigen Zeiträumen erhält [^2] (eine Situation, die als *Overfitting* bekannt ist).

1. Weise dem Trainingsdatensatz einen Zeitraum von zwei Monaten vom 1. September bis zum 31. Oktober 2014 zu. Der Testdatensatz umfasst den Zeitraum von zwei Monaten vom 1. November bis zum 31. Dezember 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisiere die Unterschiede: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![Trainings- und Testdaten](../../../../translated_images/train-test.ead0cecbfc341921d4875eccf25fed5eefbb860cdbb69cabcc2276c49e4b33e5.de.png)

### Bereite die Daten für das Training vor

Jetzt musst du die Daten für das Training vorbereiten, indem du die Daten filterst und skalierst. Filtere deinen Datensatz, um nur die benötigten Zeiträume und Spalten einzuschließen, und skaliere ihn, um sicherzustellen, dass die Daten im Intervall 0,1 projiziert werden.

1. Filtere den ursprünglichen Datensatz, um nur die oben genannten Zeiträume pro Satz und nur die benötigte Spalte 'load' plus das Datum einzuschließen: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. Skaliere die Trainingsdaten, um sie in den Bereich (0, 1) zu bringen: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Jetzt skalierst du die Testdaten: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Erstelle Daten mit Zeit-Schritten [^1]

Für das SVR transformierst du die Eingabedaten in die Form `[batch, timesteps]`. Du formst die vorhandenen `train_data` und `test_data` so um, dass eine neue Dimension entsteht, die sich auf die Zeit-Schritte bezieht.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Für dieses Beispiel nehmen wir `timesteps = 5`. Die Eingaben für das Modell sind die Daten der ersten 4 Zeit-Schritte, und die Ausgabe sind die Daten des 5. Zeit-Schritts.

```python
timesteps=5
```

Umwandlung der Trainingsdaten in einen 2D-Tensor mithilfe von verschachtelter Listenkomprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Umwandlung der Testdaten in einen 2D-Tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Auswahl von Eingaben und Ausgaben aus Trainings- und Testdaten:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Implementiere SVR [^1]

Jetzt ist es Zeit, SVR zu implementieren. Um mehr über diese Implementierung zu erfahren, kannst du [diese Dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) lesen. Für unsere Implementierung folgen wir diesen Schritten:

  1. Definiere das Modell, indem du `SVR()` aufrufst und die Hyperparameter des Modells übergibst: Kernel, Gamma, C und Epsilon
  2. Bereite das Modell für die Trainingsdaten vor, indem du die Funktion `fit()` aufrufst
  3. Erstelle Vorhersagen, indem du die Funktion `predict()` aufrufst

Jetzt erstellen wir ein SVR-Modell. Hier verwenden wir den [RBF-Kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) und setzen die Hyperparameter Gamma, C und Epsilon auf 0.5, 10 und 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Trainiere das Modell mit Trainingsdaten [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Erstelle Modellvorhersagen [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Du hast dein SVR erstellt! Jetzt müssen wir es bewerten.

### Bewerte dein Modell [^1]

Für die Bewertung skalieren wir zuerst die Daten zurück auf unsere ursprüngliche Skala. Dann überprüfen wir die Leistung, indem wir die ursprüngliche und die vorhergesagte Zeitreihe plotten und auch das MAPE-Ergebnis ausgeben.

Skaliere die vorhergesagten und ursprünglichen Ausgaben:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Überprüfe die Modellleistung auf Trainings- und Testdaten [^1]

Wir extrahieren die Zeitstempel aus dem Datensatz, um sie auf der x-Achse unseres Plots anzuzeigen. Beachte, dass wir die ersten ```timesteps-1``` Werte als Eingabe für die erste Ausgabe verwenden, sodass die Zeitstempel für die Ausgabe danach beginnen.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Zeichne die Vorhersagen für Trainingsdaten:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![Vorhersage der Trainingsdaten](../../../../translated_images/train-data-predict.3c4ef4e78553104ffdd53d47a4c06414007947ea328e9261ddf48d3eafdefbbf.de.png)

Gib MAPE für Trainingsdaten aus:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Zeichne die Vorhersagen für Testdaten:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![Vorhersage der Testdaten](../../../../translated_images/test-data-predict.8afc47ee7e52874f514ebdda4a798647e9ecf44a97cc927c535246fcf7a28aa9.de.png)

Gib MAPE für Testdaten aus:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Du hast ein sehr gutes Ergebnis auf dem Testdatensatz!

### Überprüfe die Modellleistung auf dem vollständigen Datensatz [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![Vorhersage der vollständigen Daten](../../../../translated_images/full-data-predict.4f0fed16a131c8f3bcc57a3060039dc7f2f714a05b07b68c513e0fe7fb3d8964.de.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Sehr schöne Plots, die ein Modell mit guter Genauigkeit zeigen. Gut gemacht!

---

## 🚀 Herausforderung

- Versuche, die Hyperparameter (Gamma, C, Epsilon) beim Erstellen des Modells zu ändern und die Daten zu bewerten, um herauszufinden, welche Hyperparameter die besten Ergebnisse auf den Testdaten liefern. Weitere Informationen zu diesen Hyperparametern findest du [hier](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Versuche, verschiedene Kernel-Funktionen für das Modell zu verwenden und analysiere deren Leistung auf dem Datensatz. Eine hilfreiche Dokumentation findest du [hier](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Versuche, verschiedene Werte für `timesteps` zu verwenden, damit das Modell zurückblicken kann, um Vorhersagen zu treffen.

## [Quiz nach der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/52/)

## Rückblick & Selbststudium

Diese Lektion sollte die Anwendung von SVR für Zeitreihenprognosen einführen. Um mehr über SVR zu erfahren, kannst du [diesen Blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) lesen. Diese [Dokumentation zu scikit-learn](https://scikit-learn.org/stable/modules/svm.html) bietet eine umfassendere Erklärung zu SVMs im Allgemeinen, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) und auch andere Implementierungsdetails wie die verschiedenen [Kernel-Funktionen](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), die verwendet werden können, und deren Parameter.

## Aufgabe

[Ein neues SVR-Modell](assignment.md)

## Credits

[^1]: Der Text, Code und die Ausgabe in diesem Abschnitt wurden von [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) beigesteuert.
[^2]: Der Text, Code und die Ausgabe in diesem Abschnitt wurden aus [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) übernommen.

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.