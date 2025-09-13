<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T21:23:41+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "sv"
}
-->
# Tidsserieprognoser med Support Vector Regressor

I den föregående lektionen lärde du dig hur man använder ARIMA-modellen för att göra tidsserieprognoser. Nu ska vi titta på Support Vector Regressor-modellen, som är en regressionsmodell som används för att förutsäga kontinuerliga data.

## [Quiz före lektionen](https://ff-quizzes.netlify.app/en/ml/) 

## Introduktion

I denna lektion kommer du att upptäcka ett specifikt sätt att bygga modeller med [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) för regression, eller **SVR: Support Vector Regressor**. 

### SVR i kontexten av tidsserier [^1]

Innan vi förstår vikten av SVR för tidsserieprognoser, är här några viktiga begrepp du behöver känna till:

- **Regression:** En teknik för övervakad inlärning som används för att förutsäga kontinuerliga värden från en given uppsättning indata. Idén är att passa en kurva (eller linje) i funktionsutrymmet som har maximalt antal datapunkter. [Klicka här](https://en.wikipedia.org/wiki/Regression_analysis) för mer information.
- **Support Vector Machine (SVM):** En typ av övervakad maskininlärningsmodell som används för klassificering, regression och detektering av avvikelser. Modellen är ett hyperplan i funktionsutrymmet, som i fallet med klassificering fungerar som en gräns, och i fallet med regression fungerar som den bästa anpassade linjen. I SVM används vanligtvis en kärnfunktion för att transformera datasetet till ett utrymme med högre dimensioner, så att de kan separeras enklare. [Klicka här](https://en.wikipedia.org/wiki/Support-vector_machine) för mer information om SVM.
- **Support Vector Regressor (SVR):** En typ av SVM som används för att hitta den bästa anpassade linjen (som i fallet med SVM är ett hyperplan) som har maximalt antal datapunkter.

### Varför SVR? [^1]

I den senaste lektionen lärde du dig om ARIMA, som är en mycket framgångsrik statistisk linjär metod för att förutsäga tidsseriedata. Men i många fall har tidsseriedata *icke-linjäritet*, vilket inte kan modelleras av linjära metoder. I sådana fall gör SVM:s förmåga att hantera icke-linjäritet i data för regressionsuppgifter SVR framgångsrik för tidsserieprognoser.

## Övning - bygg en SVR-modell

De första stegen för databeredning är desamma som i den föregående lektionen om [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Öppna mappen [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) i denna lektion och hitta filen [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Kör notebooken och importera de nödvändiga biblioteken: [^2]

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

2. Ladda data från filen `/data/energy.csv` till en Pandas-dataram och granska den: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plotta all tillgänglig energidata från januari 2012 till december 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nu ska vi bygga vår SVR-modell.

### Skapa tränings- och testdataset

Nu är din data laddad, så du kan dela upp den i tränings- och testdataset. Därefter omformar du datan för att skapa ett dataset baserat på tidssteg, vilket kommer att behövas för SVR. Du tränar din modell på träningsdatasetet. När modellen har tränats klart utvärderar du dess noggrannhet på träningsdatasetet, testdatasetet och sedan hela datasetet för att se den övergripande prestandan. Du måste säkerställa att testdatasetet täcker en senare tidsperiod än träningsdatasetet för att säkerställa att modellen inte får information från framtida tidsperioder [^2] (en situation som kallas *överanpassning*).

1. Tilldela en tvåmånadersperiod från 1 september till 31 oktober 2014 till träningsdatasetet. Testdatasetet kommer att inkludera tvåmånadersperioden från 1 november till 31 december 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisera skillnaderna: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Förbered data för träning

Nu behöver du förbereda data för träning genom att filtrera och skala din data. Filtrera ditt dataset för att endast inkludera de tidsperioder och kolumner du behöver, och skala för att säkerställa att datan projiceras inom intervallet 0,1.

1. Filtrera det ursprungliga datasetet för att endast inkludera de nämnda tidsperioderna per set och endast inkludera den nödvändiga kolumnen 'load' plus datum: [^2]

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
   
2. Skala träningsdatan till intervallet (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Skala nu testdatan: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Skapa data med tidssteg [^1]

För SVR omvandlar du indata till formen `[batch, timesteps]`. Så du omformar den befintliga `train_data` och `test_data` så att det finns en ny dimension som hänvisar till tidsstegen. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

För detta exempel tar vi `timesteps = 5`. Så indata till modellen är datan för de första 4 tidsstegen, och utdata kommer att vara datan för det 5:e tidssteget.

```python
timesteps=5
```

Omvandla träningsdata till en 2D-tensor med hjälp av nästlad listkomprimering:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Omvandla testdata till en 2D-tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Välja indata och utdata från tränings- och testdata:

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

### Implementera SVR [^1]

Nu är det dags att implementera SVR. För att läsa mer om denna implementering kan du referera till [denna dokumentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). För vår implementering följer vi dessa steg:

  1. Definiera modellen genom att kalla på `SVR()` och skicka in modellens hyperparametrar: kernel, gamma, c och epsilon
  2. Förbered modellen för träningsdata genom att kalla på funktionen `fit()`
  3. Gör förutsägelser genom att kalla på funktionen `predict()`

Nu skapar vi en SVR-modell. Här använder vi [RBF-kärnan](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), och ställer in hyperparametrarna gamma, C och epsilon till 0.5, 10 och 0.05 respektive.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Träna modellen på träningsdata [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Gör modellförutsägelser [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Du har byggt din SVR! Nu behöver vi utvärdera den.

### Utvärdera din modell [^1]

För utvärdering kommer vi först att skala tillbaka datan till vår ursprungliga skala. Sedan, för att kontrollera prestandan, kommer vi att plotta den ursprungliga och förutsagda tidsserieplotten, och även skriva ut MAPE-resultatet.

Skala tillbaka den förutsagda och ursprungliga utdata:

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

#### Kontrollera modellens prestanda på tränings- och testdata [^1]

Vi extraherar tidsstämplarna från datasetet för att visa på x-axeln i vår plot. Observera att vi använder de första ```timesteps-1``` värdena som indata för den första utdata, så tidsstämplarna för utdata börjar efter det.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plotta förutsägelser för träningsdata:

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

Skriv ut MAPE för träningsdata

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plotta förutsägelser för testdata

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Skriv ut MAPE för testdata

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Du har ett mycket bra resultat på testdatasetet!

### Kontrollera modellens prestanda på hela datasetet [^1]

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

🏆 Mycket fina plottar som visar en modell med god noggrannhet. Bra jobbat!

---

## 🚀Utmaning

- Försök att justera hyperparametrarna (gamma, C, epsilon) när du skapar modellen och utvärdera på datan för att se vilka uppsättningar av hyperparametrar som ger de bästa resultaten på testdatan. För att veta mer om dessa hyperparametrar kan du referera till dokumentet [här](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Försök att använda olika kärnfunktioner för modellen och analysera deras prestanda på datasetet. Ett hjälpsamt dokument kan hittas [här](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Försök att använda olika värden för `timesteps` för modellen att titta tillbaka för att göra förutsägelser.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Denna lektion introducerade tillämpningen av SVR för tidsserieprognoser. För att läsa mer om SVR kan du referera till [denna blogg](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Denna [dokumentation om scikit-learn](https://scikit-learn.org/stable/modules/svm.html) ger en mer omfattande förklaring om SVM i allmänhet, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) och även andra implementeringsdetaljer såsom de olika [kärnfunktionerna](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) som kan användas, och deras parametrar.

## Uppgift

[En ny SVR-modell](assignment.md)

## Krediter

[^1]: Text, kod och resultat i denna sektion bidrog av [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Text, kod och resultat i denna sektion togs från [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.