<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T23:55:59+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "fi"
}
-->
# Aikasarjojen ennustaminen Support Vector Regressor -mallilla

Edellisessä osiossa opit käyttämään ARIMA-mallia aikasarjojen ennustamiseen. Nyt tutustut Support Vector Regressor -malliin, joka on regressiomalli jatkuvien arvojen ennustamiseen.

## [Ennakkovisa](https://ff-quizzes.netlify.app/en/ml/) 

## Johdanto

Tässä osiossa opit rakentamaan malleja käyttäen [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) -menetelmää regressiotehtäviin, eli **SVR: Support Vector Regressor**.

### SVR aikasarjojen yhteydessä [^1]

Ennen kuin ymmärrät SVR:n merkityksen aikasarjojen ennustamisessa, on tärkeää tuntea seuraavat käsitteet:

- **Regressio:** Ohjattu oppimismenetelmä, jolla ennustetaan jatkuvia arvoja annetusta syötedatasta. Tavoitteena on sovittaa käyrä (tai viiva) piirreavaruuteen siten, että se sisältää mahdollisimman monta datapistettä. [Lisätietoa](https://en.wikipedia.org/wiki/Regression_analysis).
- **Support Vector Machine (SVM):** Ohjattu koneoppimismalli, jota käytetään luokitteluun, regressioon ja poikkeamien tunnistamiseen. Malli muodostaa hypertason piirreavaruuteen, joka toimii luokittelussa rajana ja regressiossa parhaana sovitusviivana. SVM:ssä käytetään yleensä ydinfunktiota (Kernel), joka muuntaa datan korkeampaan ulottuvuuteen, jotta se olisi helpommin eroteltavissa. [Lisätietoa](https://en.wikipedia.org/wiki/Support-vector_machine).
- **Support Vector Regressor (SVR):** SVM:n tyyppi, joka etsii parhaan sovitusviivan (SVM:n tapauksessa hypertason), joka sisältää mahdollisimman monta datapistettä.

### Miksi SVR? [^1]

Edellisessä osiossa opit ARIMA-mallista, joka on erittäin menestyksekäs tilastollinen lineaarinen menetelmä aikasarjojen ennustamiseen. Kuitenkin monissa tapauksissa aikasarjadatassa on *epälineaarisuutta*, jota lineaariset mallit eivät pysty mallintamaan. Tällaisissa tilanteissa SVM:n kyky huomioida epälineaarisuus tekee SVR:stä menestyksekkään aikasarjojen ennustamisessa.

## Harjoitus - rakenna SVR-malli

Ensimmäiset vaiheet datan valmistelussa ovat samat kuin edellisessä [ARIMA-osiossa](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Avaa tämän osion [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) -kansio ja etsi [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) -tiedosto.[^2]

1. Suorita notebook ja tuo tarvittavat kirjastot: [^2]

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

2. Lataa data `/data/energy.csv` -tiedostosta Pandas-dataframeen ja tarkastele sitä: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Piirrä kaikki saatavilla oleva energiadata tammikuusta 2012 joulukuuhun 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![koko data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nyt rakennetaan SVR-malli.

### Luo harjoitus- ja testidatasetit

Kun data on ladattu, jaa se harjoitus- ja testidatasetteihin. Muotoile data aikaväleihin perustuvaksi datasetiksi, jota tarvitaan SVR:ää varten. Koulutat mallin harjoitusdatalla. Kun malli on koulutettu, arvioit sen tarkkuutta harjoitusdatalla, testidatalla ja koko datasetillä nähdäksesi kokonaisvaltaisen suorituskyvyn. Varmista, että testidata kattaa ajanjakson, joka on harjoitusdatan jälkeinen, jotta malli ei saa tietoa tulevista ajanjaksoista [^2] (tilanne, jota kutsutaan *ylisovittamiseksi*).

1. Allokoi kahden kuukauden ajanjakso 1. syyskuuta - 31. lokakuuta 2014 harjoitusdataksi. Testidata sisältää kahden kuukauden ajanjakson 1. marraskuuta - 31. joulukuuta 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisoi erot: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![harjoitus- ja testidata](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Valmistele data koulutusta varten

Nyt sinun täytyy valmistella data koulutusta varten suodattamalla ja skaalaamalla se. Suodata datasetti sisältämään vain tarvittavat ajanjaksot ja sarakkeet, ja skaalaa data välille 0,1.

1. Suodata alkuperäinen datasetti sisältämään vain edellä mainitut ajanjaksot ja tarvittava sarake 'load' sekä päivämäärä: [^2]

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
   
2. Skaalaa harjoitusdata välille (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Skaalaa nyt testidata: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Luo data aikaväleillä [^1]

SVR:ää varten muunnat syötteen muotoon `[batch, timesteps]`. Muotoile olemassa oleva `train_data` ja `test_data` siten, että niihin lisätään uusi ulottuvuus, joka viittaa aikaväleihin.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Tässä esimerkissä otetaan `timesteps = 5`. Mallin syötteet ovat ensimmäisten neljän aikavälin data, ja ulostulo on viidennen aikavälin data.

```python
timesteps=5
```

Muunna harjoitusdata 2D-tensoriksi käyttäen sisäkkäistä listan ymmärrystä:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Muunna testidata 2D-tensoriksi:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Valitse syötteet ja ulostulot harjoitus- ja testidatasta:

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

### Toteuta SVR [^1]

Nyt on aika toteuttaa SVR. Lisätietoa toteutuksesta löydät [tästä dokumentaatiosta](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Toteutuksessa noudatetaan seuraavia vaiheita:

1. Määrittele malli kutsumalla `SVR()` ja syöttämällä mallin hyperparametrit: kernel, gamma, c ja epsilon
2. Valmistele malli harjoitusdataa varten kutsumalla `fit()`-funktiota
3. Tee ennusteita kutsumalla `predict()`-funktiota

Nyt luodaan SVR-malli. Tässä käytetään [RBF-ydintä](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), ja asetetaan hyperparametrit gamma, C ja epsilon arvoihin 0.5, 10 ja 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Sovita malli harjoitusdataan [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Tee malliennusteita [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Olet rakentanut SVR-mallin! Nyt arvioidaan sen suorituskykyä.

### Arvioi mallisi [^1]

Arviointia varten skaalaamme datan takaisin alkuperäiseen mittakaavaan. Suorituskyvyn tarkistamiseksi piirrämme alkuperäisen ja ennustetun aikasarjan sekä tulostamme MAPE-tuloksen.

Skaalaa ennustettu ja alkuperäinen ulostulo:

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

#### Tarkista mallin suorituskyky harjoitus- ja testidatalla [^1]

Poimimme aikaleimat datasetistä, jotta ne voidaan näyttää x-akselilla. Huomaa, että käytämme ensimmäisiä ```timesteps-1``` arvoja ensimmäisen ulostulon syötteenä, joten ulostulon aikaleimat alkavat vasta sen jälkeen.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Piirrä ennusteet harjoitusdatasta:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![harjoitusdatan ennuste](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Tulosta MAPE harjoitusdatasta

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Piirrä ennusteet testidatasta

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testidatan ennuste](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Tulosta MAPE testidatasta

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Sait erittäin hyvän tuloksen testidatalla!

### Tarkista mallin suorituskyky koko datasetillä [^1]

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

![koko datan ennuste](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Erittäin hienot kaaviot, jotka osoittavat mallin hyvän tarkkuuden. Hyvin tehty!

---

## 🚀Haaste

- Kokeile säätää hyperparametreja (gamma, C, epsilon) mallia luodessasi ja arvioi niiden vaikutusta testidatan tuloksiin. Lisätietoa hyperparametreista löydät [täältä](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Kokeile käyttää erilaisia ydinfunktioita mallissa ja analysoi niiden suorituskykyä datasetillä. Hyödyllinen dokumentti löytyy [täältä](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Kokeile käyttää erilaisia `timesteps`-arvoja, jotta malli voi katsoa taaksepäin ennustetta tehdessään.

## [Jälkivisa](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Tässä osiossa esiteltiin SVR:n käyttö aikasarjojen ennustamiseen. Lisätietoa SVR:stä löydät [tästä blogista](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Tämä [scikit-learn-dokumentaatio](https://scikit-learn.org/stable/modules/svm.html) tarjoaa kattavamman selityksen SVM:stä yleisesti, [SVR:stä](https://scikit-learn.org/stable/modules/svm.html#regression) ja muista toteutuksen yksityiskohdista, kuten eri [ydinfunktioista](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) ja niiden parametreista.

## Tehtävä

[Uusi SVR-malli](assignment.md)

## Kiitokset

[^1]: Tämän osion teksti, koodi ja tulokset on kirjoittanut [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Tämän osion teksti, koodi ja tulokset on otettu [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) -osiosta

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.