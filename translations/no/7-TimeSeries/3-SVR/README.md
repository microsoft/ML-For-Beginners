<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T21:24:09+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "no"
}
-->
# Tidsserieprognoser med Support Vector Regressor

I forrige leksjon lærte du hvordan du bruker ARIMA-modellen til å lage tidsserieprediksjoner. Nå skal vi se på Support Vector Regressor-modellen, som er en regresjonsmodell brukt til å forutsi kontinuerlige data.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/) 

## Introduksjon

I denne leksjonen vil du oppdage en spesifikk måte å bygge modeller med [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) for regresjon, eller **SVR: Support Vector Regressor**. 

### SVR i konteksten av tidsserier [^1]

Før du forstår viktigheten av SVR i tidsserieprediksjon, er det noen viktige konsepter du bør kjenne til:

- **Regresjon:** En overvåket læringsteknikk for å forutsi kontinuerlige verdier basert på et gitt sett med input. Ideen er å tilpasse en kurve (eller linje) i funksjonsrommet som har flest mulig datapunkter. [Klikk her](https://en.wikipedia.org/wiki/Regression_analysis) for mer informasjon.
- **Support Vector Machine (SVM):** En type overvåket maskinlæringsmodell brukt til klassifisering, regresjon og deteksjon av avvik. Modellen er et hyperplan i funksjonsrommet, som i tilfelle klassifisering fungerer som en grense, og i tilfelle regresjon fungerer som den beste tilpassede linjen. I SVM brukes vanligvis en Kernel-funksjon for å transformere datasettet til et rom med høyere dimensjoner, slik at de blir lettere separerbare. [Klikk her](https://en.wikipedia.org/wiki/Support-vector_machine) for mer informasjon om SVM.
- **Support Vector Regressor (SVR):** En type SVM som finner den beste tilpassede linjen (som i tilfelle SVM er et hyperplan) som har flest mulig datapunkter.

### Hvorfor SVR? [^1]

I forrige leksjon lærte du om ARIMA, som er en svært vellykket statistisk lineær metode for å forutsi tidsseriedata. Men i mange tilfeller har tidsseriedata *ikke-linearitet*, som ikke kan modelleres av lineære metoder. I slike tilfeller gjør SVMs evne til å ta hensyn til ikke-linearitet i dataene for regresjonsoppgaver SVR vellykket i tidsserieprognoser.

## Øvelse - bygg en SVR-modell

De første stegene for datapreparering er de samme som i forrige leksjon om [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Åpne [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working)-mappen i denne leksjonen og finn [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb)-filen.[^2]

1. Kjør notebooken og importer de nødvendige bibliotekene: [^2]

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

2. Last inn dataene fra `/data/energy.csv`-filen til en Pandas-datastruktur og se på dem: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plott alle tilgjengelige energidata fra januar 2012 til desember 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nå skal vi bygge vår SVR-modell.

### Opprett trenings- og testdatasett

Nå er dataene dine lastet inn, så du kan dele dem opp i trenings- og testsett. Deretter vil du omforme dataene for å lage et tidsstegbasert datasett som vil være nødvendig for SVR. Du vil trene modellen på treningssettet. Etter at modellen er ferdig trent, vil du evaluere dens nøyaktighet på treningssettet, testsettet og deretter hele datasettet for å se den generelle ytelsen. Du må sørge for at testsettet dekker en senere tidsperiode enn treningssettet for å sikre at modellen ikke får informasjon fra fremtidige tidsperioder [^2] (en situasjon kjent som *Overfitting*).

1. Tildel en to-måneders periode fra 1. september til 31. oktober 2014 til treningssettet. Testsettet vil inkludere to-måneders perioden fra 1. november til 31. desember 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualiser forskjellene: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Forbered dataene for trening

Nå må du forberede dataene for trening ved å filtrere og skalere dataene dine. Filtrer datasettet for kun å inkludere de tidsperiodene og kolonnene du trenger, og skaler for å sikre at dataene projiseres i intervallet 0,1.

1. Filtrer det originale datasettet for kun å inkludere de nevnte tidsperiodene per sett og kun inkludere den nødvendige kolonnen 'load' pluss datoen: [^2]

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
   
2. Skaler treningsdataene til å være i området (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Nå skalerer du testdataene: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Opprett data med tidssteg [^1]

For SVR transformerer du inputdataene til formen `[batch, timesteps]`. Så du omformer de eksisterende `train_data` og `test_data` slik at det er en ny dimensjon som refererer til tidsstegene. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

For dette eksempelet tar vi `timesteps = 5`. Så input til modellen er dataene for de første 4 tidsstegene, og output vil være dataene for det 5. tidssteget.

```python
timesteps=5
```

Konverter treningsdata til 2D tensor ved hjelp av nested list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Konverter testdata til 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Velg input og output fra trenings- og testdata:

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

Nå er det på tide å implementere SVR. For å lese mer om denne implementeringen, kan du referere til [denne dokumentasjonen](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). For vår implementering følger vi disse stegene:

  1. Definer modellen ved å kalle `SVR()` og sende inn modellens hyperparametere: kernel, gamma, c og epsilon
  2. Forbered modellen for treningsdataene ved å kalle funksjonen `fit()`
  3. Lag prediksjoner ved å kalle funksjonen `predict()`

Nå oppretter vi en SVR-modell. Her bruker vi [RBF-kjernen](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), og setter hyperparameterne gamma, C og epsilon til henholdsvis 0.5, 10 og 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Tren modellen på treningsdata [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Lag modellprediksjoner [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Du har bygget din SVR! Nå må vi evaluere den.

### Evaluer modellen din [^1]

For evaluering skal vi først skalere dataene tilbake til vår originale skala. Deretter, for å sjekke ytelsen, skal vi plotte den originale og predikerte tidsserien, og også skrive ut MAPE-resultatet.

Skaler den predikerte og originale outputen:

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

#### Sjekk modellens ytelse på trenings- og testdata [^1]

Vi henter tidsstemplene fra datasettet for å vise på x-aksen i vårt plot. Merk at vi bruker de første ```timesteps-1``` verdiene som input for den første outputen, så tidsstemplene for outputen vil starte etter det.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plot prediksjonene for treningsdata:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![training data prediction](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Skriv ut MAPE for treningsdata

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plot prediksjonene for testdata

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Skriv ut MAPE for testdata

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Du har et veldig godt resultat på testdatasettet!

### Sjekk modellens ytelse på hele datasettet [^1]

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

![full data prediction](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Veldig fine plott som viser en modell med god nøyaktighet. Bra jobbet!

---

## 🚀Utfordring

- Prøv å justere hyperparameterne (gamma, C, epsilon) mens du oppretter modellen og evaluer på dataene for å se hvilke sett med hyperparametere som gir de beste resultatene på testdataene. For å lære mer om disse hyperparameterne, kan du referere til dokumentet [her](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Prøv å bruke forskjellige kjernefunksjoner for modellen og analyser deres ytelse på datasettet. Et nyttig dokument kan finnes [her](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Prøv å bruke forskjellige verdier for `timesteps` for modellen for å se tilbake og lage prediksjoner.

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudie

Denne leksjonen var en introduksjon til bruken av SVR for tidsserieprognoser. For å lese mer om SVR, kan du referere til [denne bloggen](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Denne [dokumentasjonen på scikit-learn](https://scikit-learn.org/stable/modules/svm.html) gir en mer omfattende forklaring om SVM generelt, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) og også andre implementeringsdetaljer som de forskjellige [kjernefunksjonene](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) som kan brukes, og deres parametere.

## Oppgave

[En ny SVR-modell](assignment.md)

## Krediteringer

[^1]: Teksten, koden og outputen i denne seksjonen ble bidratt av [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Teksten, koden og outputen i denne seksjonen ble hentet fra [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.