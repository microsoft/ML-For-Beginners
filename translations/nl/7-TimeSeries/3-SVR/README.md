<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T19:07:00+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "nl"
}
-->
# Tijdreeksvoorspelling met Support Vector Regressor

In de vorige les heb je geleerd hoe je het ARIMA-model kunt gebruiken om voorspellingen te maken voor tijdreeksen. Nu ga je kijken naar het Support Vector Regressor-model, een regressiemodel dat wordt gebruikt om continue gegevens te voorspellen.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/) 

## Introductie

In deze les ontdek je een specifieke manier om modellen te bouwen met [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) voor regressie, oftewel **SVR: Support Vector Regressor**. 

### SVR in de context van tijdreeksen [^1]

Voordat je het belang van SVR in tijdreeksvoorspelling begrijpt, zijn hier enkele belangrijke concepten die je moet kennen:

- **Regressie:** Een techniek voor begeleid leren om continue waarden te voorspellen op basis van een gegeven set invoerwaarden. Het idee is om een curve (of lijn) in de kenmerkenruimte te passen die het maximale aantal datapunten bevat. [Klik hier](https://en.wikipedia.org/wiki/Regression_analysis) voor meer informatie.
- **Support Vector Machine (SVM):** Een type begeleid machine learning-model dat wordt gebruikt voor classificatie, regressie en het detecteren van uitschieters. Het model is een hypervlak in de kenmerkenruimte, dat in het geval van classificatie fungeert als een grens en in het geval van regressie als de best passende lijn. In SVM wordt meestal een kernfunctie gebruikt om de dataset te transformeren naar een ruimte met een hoger aantal dimensies, zodat ze gemakkelijker te scheiden zijn. [Klik hier](https://en.wikipedia.org/wiki/Support-vector_machine) voor meer informatie over SVM's.
- **Support Vector Regressor (SVR):** Een type SVM dat de best passende lijn (die in het geval van SVM een hypervlak is) vindt die het maximale aantal datapunten bevat.

### Waarom SVR? [^1]

In de vorige les heb je geleerd over ARIMA, een zeer succesvol statistisch lineair model om tijdreeksgegevens te voorspellen. Echter, in veel gevallen hebben tijdreeksgegevens *non-lineariteit*, die niet kan worden gemodelleerd door lineaire modellen. In dergelijke gevallen maakt het vermogen van SVM om non-lineariteit in de gegevens te overwegen voor regressietaken SVR succesvol in tijdreeksvoorspelling.

## Oefening - bouw een SVR-model

De eerste paar stappen voor gegevensvoorbereiding zijn hetzelfde als die van de vorige les over [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Open de [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) map in deze les en vind het [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) bestand.[^2]

1. Voer de notebook uit en importeer de benodigde bibliotheken:  [^2]

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

2. Laad de gegevens uit het `/data/energy.csv` bestand in een Pandas dataframe en bekijk ze:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plot alle beschikbare energiedata van januari 2012 tot december 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![volledige data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nu gaan we ons SVR-model bouwen.

### Maak trainings- en testdatasets

Nu je gegevens zijn geladen, kun je ze scheiden in trainings- en testsets. Vervolgens vorm je de gegevens om tot een dataset op basis van tijdstappen, wat nodig zal zijn voor de SVR. Je traint je model op de trainingsset. Nadat het model is getraind, evalueer je de nauwkeurigheid op de trainingsset, testset en vervolgens de volledige dataset om de algehele prestaties te zien. Je moet ervoor zorgen dat de testset een latere periode in de tijd omvat dan de trainingsset om te voorkomen dat het model informatie uit toekomstige tijdsperioden verkrijgt [^2] (een situatie die bekend staat als *overfitting*).

1. Wijs een periode van twee maanden toe van 1 september tot 31 oktober 2014 aan de trainingsset. De testset omvat de periode van twee maanden van 1 november tot 31 december 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiseer de verschillen: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![trainings- en testdata](../../../../7-TimeSeries/3-SVR/images/train-test.png)



### Bereid de gegevens voor op training

Nu moet je de gegevens voorbereiden op training door filtering en schaling van je gegevens uit te voeren. Filter je dataset om alleen de benodigde tijdsperioden en kolommen op te nemen, en schaal de gegevens zodat ze worden geprojecteerd in het interval 0,1.

1. Filter de originele dataset om alleen de eerder genoemde tijdsperioden per set op te nemen en alleen de benodigde kolom 'load' plus de datum: [^2]

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
   
2. Schaal de trainingsgegevens naar het bereik (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Nu schaal je de testgegevens: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Maak gegevens met tijdstappen [^1]

Voor de SVR transformeer je de invoergegevens naar de vorm `[batch, timesteps]`. Dus, je herschikt de bestaande `train_data` en `test_data` zodat er een nieuwe dimensie is die verwijst naar de tijdstappen. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Voor dit voorbeeld nemen we `timesteps = 5`. Dus, de invoer voor het model zijn de gegevens voor de eerste 4 tijdstappen, en de uitvoer zal de gegevens voor de 5e tijdstap zijn.

```python
timesteps=5
```

Converteer trainingsgegevens naar een 2D-tensor met behulp van geneste lijstbegrippen:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Converteer testgegevens naar een 2D-tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Selecteer invoer en uitvoer uit trainings- en testgegevens:

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

### Implementeer SVR [^1]

Nu is het tijd om SVR te implementeren. Voor meer informatie over deze implementatie kun je [deze documentatie](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) raadplegen. Voor onze implementatie volgen we deze stappen:

  1. Definieer het model door `SVR()` aan te roepen en de modelhyperparameters door te geven: kernel, gamma, c en epsilon
  2. Bereid het model voor op de trainingsgegevens door de functie `fit()` aan te roepen
  3. Maak voorspellingen door de functie `predict()` aan te roepen

Nu maken we een SVR-model. Hier gebruiken we de [RBF-kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), en stellen we de hyperparameters gamma, C en epsilon in op respectievelijk 0.5, 10 en 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Pas het model toe op trainingsgegevens [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Maak modelvoorspellingen [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Je hebt je SVR gebouwd! Nu moeten we het evalueren.

### Evalueer je model [^1]

Voor evaluatie schalen we eerst de gegevens terug naar onze originele schaal. Vervolgens, om de prestaties te controleren, plotten we de originele en voorspelde tijdreeksplot en printen we ook het MAPE-resultaat.

Schaal de voorspelde en originele uitvoer:

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

#### Controleer modelprestaties op trainings- en testgegevens [^1]

We halen de tijdstempels uit de dataset om te tonen op de x-as van onze plot. Merk op dat we de eerste ```timesteps-1``` waarden gebruiken als invoer voor de eerste uitvoer, dus de tijdstempels voor de uitvoer beginnen daarna.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plot de voorspellingen voor trainingsgegevens:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![voorspelling trainingsgegevens](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Print MAPE voor trainingsgegevens

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plot de voorspellingen voor testgegevens

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![voorspelling testgegevens](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Print MAPE voor testgegevens

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Je hebt een zeer goed resultaat op de testdataset!

### Controleer modelprestaties op volledige dataset [^1]

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

![voorspelling volledige data](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```



🏆 Zeer mooie plots, die een model met goede nauwkeurigheid laten zien. Goed gedaan!

---

## 🚀Uitdaging

- Probeer de hyperparameters (gamma, C, epsilon) aan te passen bij het maken van het model en evalueer op de gegevens om te zien welke set hyperparameters de beste resultaten geeft op de testgegevens. Voor meer informatie over deze hyperparameters kun je [deze documentatie](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) raadplegen. 
- Probeer verschillende kernfuncties te gebruiken voor het model en analyseer hun prestaties op de dataset. Een nuttige documentatie is te vinden [hier](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Probeer verschillende waarden voor `timesteps` te gebruiken zodat het model terugkijkt om voorspellingen te maken.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Deze les was bedoeld om de toepassing van SVR voor tijdreeksvoorspelling te introduceren. Voor meer informatie over SVR kun je [deze blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) raadplegen. Deze [documentatie over scikit-learn](https://scikit-learn.org/stable/modules/svm.html) biedt een meer uitgebreide uitleg over SVM's in het algemeen, [SVR's](https://scikit-learn.org/stable/modules/svm.html#regression) en ook andere implementatiedetails zoals de verschillende [kernfuncties](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) die kunnen worden gebruikt, en hun parameters.

## Opdracht

[Een nieuw SVR-model](assignment.md)



## Credits


[^1]: De tekst, code en output in deze sectie is bijgedragen door [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: De tekst, code en output in deze sectie is afkomstig van [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen voor nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.