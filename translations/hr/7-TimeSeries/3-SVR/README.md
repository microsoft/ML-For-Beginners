<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T12:04:47+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "hr"
}
-->
# Predviđanje vremenskih serija pomoću Support Vector Regressor-a

U prethodnoj lekciji naučili ste kako koristiti ARIMA model za predviđanje vremenskih serija. Sada ćemo se fokusirati na model Support Vector Regressor, koji se koristi za predviđanje kontinuiranih podataka.

## [Pre-lecture kviz](https://ff-quizzes.netlify.app/en/ml/) 

## Uvod

U ovoj lekciji otkrit ćete specifičan način izgradnje modela pomoću [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) za regresiju, ili **SVR: Support Vector Regressor**. 

### SVR u kontekstu vremenskih serija [^1]

Prije nego što razumijete važnost SVR-a u predviđanju vremenskih serija, evo nekoliko važnih pojmova koje trebate znati:

- **Regresija:** Tehnika nadziranog učenja za predviđanje kontinuiranih vrijednosti iz skupa ulaznih podataka. Ideja je prilagoditi krivulju (ili liniju) u prostoru značajki koja ima maksimalan broj podataka. [Kliknite ovdje](https://en.wikipedia.org/wiki/Regression_analysis) za više informacija.
- **Support Vector Machine (SVM):** Vrsta modela nadziranog strojnog učenja koji se koristi za klasifikaciju, regresiju i otkrivanje odstupanja. Model je hiperravnina u prostoru značajki, koja u slučaju klasifikacije djeluje kao granica, a u slučaju regresije kao linija najboljeg pristajanja. U SVM-u se obično koristi Kernel funkcija za transformaciju skupa podataka u prostor s većim brojem dimenzija, kako bi se podaci lakše razdvojili. [Kliknite ovdje](https://en.wikipedia.org/wiki/Support-vector_machine) za više informacija o SVM-ovima.
- **Support Vector Regressor (SVR):** Vrsta SVM-a, koja pronalazi liniju najboljeg pristajanja (koja je u slučaju SVM-a hiperravnina) s maksimalnim brojem podataka.

### Zašto SVR? [^1]

U prethodnoj lekciji naučili ste o ARIMA modelu, koji je vrlo uspješna statistička linearna metoda za predviđanje vremenskih serija. Međutim, u mnogim slučajevima podaci vremenskih serija imaju *nelinearnost*, koju linearni modeli ne mogu mapirati. U takvim slučajevima sposobnost SVM-a da uzme u obzir nelinearnost podataka za regresijske zadatke čini SVR uspješnim u predviđanju vremenskih serija.

## Vježba - izgradnja SVR modela

Prvi koraci za pripremu podataka isti su kao u prethodnoj lekciji o [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Otvorite mapu [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) u ovoj lekciji i pronađite datoteku [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Pokrenite notebook i uvezite potrebne biblioteke: [^2]

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

2. Učitajte podatke iz datoteke `/data/energy.csv` u Pandas dataframe i pogledajte ih: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Prikažite sve dostupne podatke o energiji od siječnja 2012. do prosinca 2014.: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![puni podaci](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Sada ćemo izgraditi naš SVR model.

### Kreiranje skupa za treniranje i testiranje

Sada su vaši podaci učitani, pa ih možete podijeliti na skup za treniranje i skup za testiranje. Zatim ćete preoblikovati podatke kako biste stvorili skup podataka temeljen na vremenskim koracima, što će biti potrebno za SVR. Model ćete trenirati na skupu za treniranje. Nakon što model završi s treniranjem, procijenit ćete njegovu točnost na skupu za treniranje, skupu za testiranje i zatim na cijelom skupu podataka kako biste vidjeli ukupnu izvedbu. Morate osigurati da skup za testiranje pokriva kasniji vremenski period od skupa za treniranje kako biste osigurali da model ne dobije informacije iz budućih vremenskih perioda [^2] (situacija poznata kao *Overfitting*).

1. Dodijelite dvomjesečno razdoblje od 1. rujna do 31. listopada 2014. skupu za treniranje. Skup za testiranje uključivat će dvomjesečno razdoblje od 1. studenog do 31. prosinca 2014.: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Vizualizirajte razlike: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![podaci za treniranje i testiranje](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Priprema podataka za treniranje

Sada trebate pripremiti podatke za treniranje filtriranjem i skaliranjem podataka. Filtrirajte svoj skup podataka tako da uključuje samo potrebne vremenske periode i stupce, te skalirajte podatke kako biste osigurali da su projicirani u interval 0,1.

1. Filtrirajte originalni skup podataka tako da uključuje samo prethodno navedene vremenske periode po skupu i samo potrebni stupac 'load' plus datum: [^2]

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
   
2. Skalirajte podatke za treniranje u raspon (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Sada skalirajte podatke za testiranje: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Kreiranje podataka s vremenskim koracima [^1]

Za SVR, transformirate ulazne podatke u oblik `[batch, timesteps]`. Dakle, preoblikujete postojeće `train_data` i `test_data` tako da postoji nova dimenzija koja se odnosi na vremenske korake.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Za ovaj primjer uzimamo `timesteps = 5`. Dakle, ulazi u model su podaci za prva 4 vremenska koraka, a izlaz će biti podaci za 5. vremenski korak.

```python
timesteps=5
```

Pretvaranje podataka za treniranje u 2D tensor pomoću ugniježđenih list comprehensions:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Pretvaranje podataka za testiranje u 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Odabir ulaza i izlaza iz podataka za treniranje i testiranje:

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

### Implementacija SVR-a [^1]

Sada je vrijeme za implementaciju SVR-a. Za više informacija o ovoj implementaciji, možete se referirati na [ovu dokumentaciju](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Za našu implementaciju slijedimo ove korake:

  1. Definirajte model pozivanjem `SVR()` i prosljeđivanjem hiperparametara modela: kernel, gamma, c i epsilon
  2. Pripremite model za podatke za treniranje pozivanjem funkcije `fit()`
  3. Napravite predviđanja pozivanjem funkcije `predict()`

Sada kreiramo SVR model. Ovdje koristimo [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), i postavljamo hiperparametre gamma, C i epsilon na 0.5, 10 i 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Treniranje modela na podacima za treniranje [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Izrada predviđanja modela [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Izgradili ste svoj SVR! Sada ga trebamo procijeniti.

### Procjena modela [^1]

Za procjenu, prvo ćemo skalirati podatke natrag na našu originalnu skalu. Zatim, kako bismo provjerili izvedbu, prikazat ćemo originalni i predviđeni graf vremenskih serija, te ispisati rezultat MAPE-a.

Skaliranje predviđenih i originalnih izlaza:

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

#### Provjera izvedbe modela na podacima za treniranje i testiranje [^1]

Izvlačimo vremenske oznake iz skupa podataka kako bismo ih prikazali na x-osi našeg grafa. Napominjemo da koristimo prvih ```timesteps-1``` vrijednosti kao ulaz za prvi izlaz, tako da vremenske oznake za izlaz počinju nakon toga.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Prikaz predviđanja za podatke za treniranje:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![predviđanje podataka za treniranje](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Ispis MAPE-a za podatke za treniranje

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Prikaz predviđanja za podatke za testiranje

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![predviđanje podataka za testiranje](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Ispis MAPE-a za podatke za testiranje

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Imate vrlo dobar rezultat na skupu podataka za testiranje!

### Provjera izvedbe modela na cijelom skupu podataka [^1]

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

![predviđanje cijelog skupa podataka](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Vrlo lijepi grafovi, koji pokazuju model s dobrom točnošću. Bravo!

---

## 🚀Izazov

- Pokušajte prilagoditi hiperparametre (gamma, C, epsilon) prilikom kreiranja modela i procijeniti na podacima kako biste vidjeli koji skup hiperparametara daje najbolje rezultate na skupu podataka za testiranje. Za više informacija o ovim hiperparametrima, možete se referirati na dokument [ovdje](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Pokušajte koristiti različite kernel funkcije za model i analizirati njihove izvedbe na skupu podataka. Koristan dokument možete pronaći [ovdje](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Pokušajte koristiti različite vrijednosti za `timesteps` kako bi model gledao unatrag za predviđanje.

## [Post-lecture kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Ova lekcija je bila uvod u primjenu SVR-a za predviđanje vremenskih serija. Za više informacija o SVR-u, možete se referirati na [ovaj blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ova [dokumentacija o scikit-learn](https://scikit-learn.org/stable/modules/svm.html) pruža sveobuhvatnije objašnjenje o SVM-ovima općenito, [SVR-ima](https://scikit-learn.org/stable/modules/svm.html#regression) i također drugim detaljima implementacije kao što su različite [kernel funkcije](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) koje se mogu koristiti, i njihovi parametri.

## Zadatak

[Novi SVR model](assignment.md)

## Zasluge

[^1]: Tekst, kod i izlaz u ovom odjeljku doprinio je [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Tekst, kod i izlaz u ovom odjeljku preuzet je iz [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.