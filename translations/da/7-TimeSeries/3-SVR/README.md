<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T23:54:49+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "da"
}
-->
# Tidsserieforudsigelse med Support Vector Regressor

I den forrige lektion lærte du, hvordan man bruger ARIMA-modellen til at lave tidsserieforudsigelser. Nu skal du se på Support Vector Regressor-modellen, som er en regressionsmodel, der bruges til at forudsige kontinuerlige data.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/) 

## Introduktion

I denne lektion vil du opdage en specifik måde at bygge modeller med [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) til regression, eller **SVR: Support Vector Regressor**. 

### SVR i konteksten af tidsserier [^1]

Før du forstår vigtigheden af SVR i tidsserieforudsigelse, er her nogle af de vigtige begreber, du skal kende:

- **Regression:** En superviseret læringsteknik til at forudsige kontinuerlige værdier ud fra et givet sæt input. Ideen er at tilpasse en kurve (eller linje) i funktionsrummet, der har det maksimale antal datapunkter. [Klik her](https://en.wikipedia.org/wiki/Regression_analysis) for mere information.
- **Support Vector Machine (SVM):** En type superviseret maskinlæringsmodel, der bruges til klassifikation, regression og detektion af outliers. Modellen er et hyperplan i funktionsrummet, som i tilfælde af klassifikation fungerer som en grænse, og i tilfælde af regression fungerer som den bedst tilpassede linje. I SVM bruges en Kernel-funktion generelt til at transformere datasættet til et rum med et højere antal dimensioner, så de kan adskilles lettere. [Klik her](https://en.wikipedia.org/wiki/Support-vector_machine) for mere information om SVM'er.
- **Support Vector Regressor (SVR):** En type SVM, der finder den bedst tilpassede linje (som i SVM er et hyperplan), der har det maksimale antal datapunkter.

### Hvorfor SVR? [^1]

I den sidste lektion lærte du om ARIMA, som er en meget succesfuld statistisk lineær metode til at forudsige tidsseriedata. Men i mange tilfælde har tidsseriedata *ikke-linearitet*, som ikke kan kortlægges af lineære modeller. I sådanne tilfælde gør SVM's evne til at tage højde for ikke-linearitet i data til regression SVR succesfuld i tidsserieforudsigelse.

## Øvelse - byg en SVR-model

De første trin til datapreparation er de samme som i den forrige lektion om [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Åbn mappen [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) i denne lektion og find filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Kør notebooken og importer de nødvendige biblioteker: [^2]

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

2. Indlæs data fra filen `/data/energy.csv` til en Pandas dataframe og kig på dem: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plot alle de tilgængelige energidata fra januar 2012 til december 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![fulde data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nu skal vi bygge vores SVR-model.

### Opret trænings- og testdatasæt

Nu er dine data indlæst, så du kan opdele dem i trænings- og testdatasæt. Derefter skal du omforme dataene for at skabe et tidsbaseret datasæt, som vil være nødvendigt for SVR. Du træner din model på træningssættet. Når modellen er færdig med at træne, evaluerer du dens nøjagtighed på træningssættet, testsættet og derefter det fulde datasæt for at se den samlede ydeevne. Du skal sikre dig, at testsættet dækker en senere periode end træningssættet for at sikre, at modellen ikke får information fra fremtidige tidsperioder [^2] (en situation kendt som *overfitting*).

1. Tildel en to-måneders periode fra 1. september til 31. oktober 2014 til træningssættet. Testsættet vil inkludere to-måneders perioden fra 1. november til 31. december 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiser forskellene: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![trænings- og testdata](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Forbered dataene til træning

Nu skal du forberede dataene til træning ved at udføre filtrering og skalering af dine data. Filtrer dit datasæt, så det kun inkluderer de tidsperioder og kolonner, du har brug for, og skaler dataene for at sikre, at de projiceres i intervallet 0,1.

1. Filtrer det originale datasæt, så det kun inkluderer de nævnte tidsperioder pr. sæt og kun inkluderer den nødvendige kolonne 'load' plus datoen: [^2]

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
   
2. Skaler træningsdataene til at være i intervallet (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Skaler nu testdataene: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Opret data med tidssteg [^1]

For SVR transformerer du inputdataene til formen `[batch, timesteps]`. Så du omformer de eksisterende `train_data` og `test_data`, så der er en ny dimension, der refererer til tidsstegene.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

I dette eksempel tager vi `timesteps = 5`. Så input til modellen er dataene for de første 4 tidssteg, og output vil være dataene for det 5. tidssteg.

```python
timesteps=5
```

Konvertering af træningsdata til 2D tensor ved hjælp af nested list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Konvertering af testdata til 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Udvælgelse af input og output fra trænings- og testdata:

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

### Implementer SVR [^1]

Nu er det tid til at implementere SVR. For at læse mere om denne implementering kan du referere til [denne dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). For vores implementering følger vi disse trin:

  1. Definer modellen ved at kalde `SVR()` og angive modelhyperparametrene: kernel, gamma, c og epsilon
  2. Forbered modellen til træningsdataene ved at kalde funktionen `fit()`
  3. Lav forudsigelser ved at kalde funktionen `predict()`

Nu opretter vi en SVR-model. Her bruger vi [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) og sætter hyperparametrene gamma, C og epsilon til henholdsvis 0.5, 10 og 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Tilpas modellen til træningsdata [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Lav model-forudsigelser [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Du har bygget din SVR! Nu skal vi evaluere den.

### Evaluer din model [^1]

For evaluering skal vi først skalere dataene tilbage til vores originale skala. Derefter, for at kontrollere ydeevnen, vil vi plotte den originale og forudsagte tidsserie og også udskrive MAPE-resultatet.

Skaler det forudsagte og originale output:

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

#### Kontroller modelens ydeevne på trænings- og testdata [^1]

Vi udtrækker tidsstemplerne fra datasættet for at vise dem på x-aksen i vores plot. Bemærk, at vi bruger de første ```timesteps-1``` værdier som input til det første output, så tidsstemplerne for output vil starte derefter.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plot forudsigelserne for træningsdata:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![forudsigelse af træningsdata](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Udskriv MAPE for træningsdata

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plot forudsigelserne for testdata

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![forudsigelse af testdata](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Udskriv MAPE for testdata

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Du har et meget godt resultat på testdatasættet!

### Kontroller modelens ydeevne på det fulde datasæt [^1]

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

![forudsigelse af fulde data](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Meget flotte plots, der viser en model med god nøjagtighed. Godt arbejde!

---

## 🚀Udfordring

- Prøv at justere hyperparametrene (gamma, C, epsilon) under oprettelsen af modellen og evaluer på dataene for at se, hvilket sæt hyperparametre giver de bedste resultater på testdataene. For at lære mere om disse hyperparametre kan du referere til dokumentet [her](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Prøv at bruge forskellige kernel-funktioner til modellen og analyser deres ydeevne på datasættet. Et nyttigt dokument kan findes [her](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Prøv at bruge forskellige værdier for `timesteps` for modellen til at kigge tilbage for at lave forudsigelser.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Denne lektion introducerede anvendelsen af SVR til tidsserieforudsigelse. For at læse mere om SVR kan du referere til [denne blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Denne [dokumentation om scikit-learn](https://scikit-learn.org/stable/modules/svm.html) giver en mere omfattende forklaring om SVM'er generelt, [SVR'er](https://scikit-learn.org/stable/modules/svm.html#regression) og også andre implementeringsdetaljer såsom de forskellige [kernel-funktioner](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), der kan bruges, og deres parametre.

## Opgave

[En ny SVR-model](assignment.md)

## Credits

[^1]: Teksten, koden og output i dette afsnit blev bidraget af [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Teksten, koden og output i dette afsnit blev taget fra [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.