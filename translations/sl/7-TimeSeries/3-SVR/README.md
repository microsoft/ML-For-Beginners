<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T12:05:29+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "sl"
}
-->
# Napovedovanje časovnih vrst z regresorjem podpornih vektorjev

V prejšnji lekciji ste se naučili uporabljati model ARIMA za napovedovanje časovnih vrst. Zdaj si bomo ogledali model Support Vector Regressor, ki je regresijski model za napovedovanje neprekinjenih podatkov.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/) 

## Uvod

V tej lekciji boste spoznali specifičen način gradnje modelov z [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) za regresijo, ali **SVR: Support Vector Regressor**. 

### SVR v kontekstu časovnih vrst [^1]

Preden razumemo pomen SVR pri napovedovanju časovnih vrst, je pomembno poznati naslednje koncepte:

- **Regresija:** Tehnika nadzorovanega učenja za napovedovanje neprekinjenih vrednosti na podlagi danih vhodnih podatkov. Ideja je prilagoditi krivuljo (ali premico) v prostoru značilnosti, ki zajame največje število podatkovnih točk. [Kliknite tukaj](https://en.wikipedia.org/wiki/Regression_analysis) za več informacij.
- **Support Vector Machine (SVM):** Vrsta modela nadzorovanega strojnega učenja, ki se uporablja za klasifikacijo, regresijo in zaznavanje odstopanj. Model je hiperploskev v prostoru značilnosti, ki v primeru klasifikacije deluje kot meja, v primeru regresije pa kot najboljša prilagoditvena premica. Pri SVM se pogosto uporablja funkcija jedra (Kernel function), ki transformira podatkovni niz v prostor z več dimenzijami, da postane lažje ločljiv. [Kliknite tukaj](https://en.wikipedia.org/wiki/Support-vector_machine) za več informacij o SVM.
- **Support Vector Regressor (SVR):** Vrsta SVM, ki najde najboljšo prilagoditveno premico (ki je v primeru SVM hiperploskev), ki zajame največje število podatkovnih točk.

### Zakaj SVR? [^1]

V prejšnji lekciji ste spoznali ARIMA, ki je zelo uspešna statistična linearna metoda za napovedovanje časovnih vrst. Vendar pa so podatki časovnih vrst pogosto *nelinearni*, kar linearni modeli ne morejo zajeti. V takih primerih sposobnost SVM, da upošteva nelinearnost podatkov pri regresijskih nalogah, naredi SVR uspešnega pri napovedovanju časovnih vrst.

## Naloga - izdelava modela SVR

Prvi koraki za pripravo podatkov so enaki kot v prejšnji lekciji o [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Odprite mapo [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) v tej lekciji in poiščite datoteko [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Zaženite beležko in uvozite potrebne knjižnice: [^2]

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

2. Naložite podatke iz datoteke `/data/energy.csv` v Pandas dataframe in si jih oglejte: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Prikažite vse razpoložljive podatke o energiji od januarja 2012 do decembra 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![celotni podatki](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Zdaj pa izdelajmo naš model SVR.

### Ustvarjanje učnih in testnih podatkovnih nizov

Ko so podatki naloženi, jih lahko ločite na učni in testni niz. Nato jih boste preoblikovali v podatkovni niz, ki temelji na časovnih korakih, kar bo potrebno za SVR. Model boste trenirali na učnem nizu. Ko bo model končal s treniranjem, boste ocenili njegovo natančnost na učnem nizu, testnem nizu in nato na celotnem podatkovnem nizu, da preverite splošno zmogljivost. Poskrbeti morate, da testni niz zajema poznejše obdobje v času od učnega niza, da zagotovite, da model ne pridobi informacij iz prihodnjih časovnih obdobij [^2] (situacija, znana kot *Overfitting*).

1. Dodelite dvomesečno obdobje od 1. septembra do 31. oktobra 2014 učnemu nizu. Testni niz bo vključeval dvomesečno obdobje od 1. novembra do 31. decembra 2014: [^2]

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

   ![učni in testni podatki](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Priprava podatkov za treniranje

Zdaj morate pripraviti podatke za treniranje z izvajanjem filtriranja in skaliranja podatkov. Filtrirajte svoj podatkovni niz, da vključite samo potrebna časovna obdobja in stolpce, ter skalirajte podatke, da jih projicirate v interval 0,1.

1. Filtrirajte izvirni podatkovni niz, da vključite samo omenjena časovna obdobja na nizih in samo potrebni stolpec 'load' ter datum: [^2]

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
   
2. Skalirajte učne podatke, da bodo v razponu (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Zdaj skalirajte testne podatke: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Ustvarjanje podatkov s časovnimi koraki [^1]

Za SVR preoblikujete vhodne podatke v obliko `[batch, timesteps]`. Tako obstoječe `train_data` in `test_data` preoblikujete tako, da dodate novo dimenzijo, ki se nanaša na časovne korake.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Za ta primer vzamemo `timesteps = 5`. Tako so vhodni podatki za model podatki za prve 4 časovne korake, izhod pa bodo podatki za 5. časovni korak.

```python
timesteps=5
```

Pretvorba učnih podatkov v 2D tensor z uporabo ugnezdene listne komprehencije:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Pretvorba testnih podatkov v 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Izbor vhodov in izhodov iz učnih in testnih podatkov:

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

### Implementacija SVR [^1]

Zdaj je čas za implementacijo SVR. Za več informacij o tej implementaciji si lahko ogledate [to dokumentacijo](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Za našo implementacijo sledimo tem korakom:

  1. Definirajte model z uporabo `SVR()` in podajte hiperparametre modela: kernel, gamma, c in epsilon
  2. Pripravite model za učne podatke z uporabo funkcije `fit()`
  3. Izvedite napovedi z uporabo funkcije `predict()`

Zdaj ustvarimo model SVR. Tukaj uporabimo [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) in nastavimo hiperparametre gamma, C in epsilon na 0.5, 10 in 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Prilagoditev modela na učne podatke [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Izvedba napovedi modela [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Izdelali ste svoj SVR! Zdaj ga moramo oceniti.

### Ocenjevanje modela [^1]

Za ocenjevanje najprej skaliramo podatke nazaj na našo izvirno lestvico. Nato za preverjanje zmogljivosti prikažemo izvirni in napovedani časovni niz ter natisnemo rezultat MAPE.

Skaliranje napovedanih in izvirnih izhodov:

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

#### Preverjanje zmogljivosti modela na učnih in testnih podatkih [^1]

Iz podatkovnega niza izvlečemo časovne oznake za prikaz na x-osi našega grafa. Upoštevajte, da uporabljamo prvih ```timesteps-1``` vrednosti kot vhod za prvi izhod, zato se časovne oznake za izhod začnejo po tem.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Prikaz napovedi za učne podatke:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![napoved učnih podatkov](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Natisnite MAPE za učne podatke

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Prikaz napovedi za testne podatke

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![napoved testnih podatkov](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Natisnite MAPE za testne podatke

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Na testnem podatkovnem nizu imate zelo dober rezultat!

### Preverjanje zmogljivosti modela na celotnem podatkovnem nizu [^1]

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

![napoved celotnih podatkov](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Zelo lepi grafi, ki prikazujejo model z dobro natančnostjo. Odlično opravljeno!

---

## 🚀Izziv

- Poskusite prilagoditi hiperparametre (gamma, C, epsilon) med ustvarjanjem modela in ocenite podatke, da vidite, kateri nabor hiperparametrov daje najboljše rezultate na testnih podatkih. Za več informacij o teh hiperparametrih si lahko ogledate dokument [tukaj](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Poskusite uporabiti različne funkcije jedra za model in analizirajte njihovo zmogljivost na podatkovnem nizu. Koristen dokument najdete [tukaj](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Poskusite uporabiti različne vrednosti za `timesteps`, da model pogleda nazaj za napovedovanje.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Ta lekcija je bila namenjena predstavitvi uporabe SVR za napovedovanje časovnih vrst. Za več informacij o SVR si lahko ogledate [ta blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ta [dokumentacija o scikit-learn](https://scikit-learn.org/stable/modules/svm.html) ponuja bolj celovito razlago o SVM na splošno, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) in tudi druge podrobnosti implementacije, kot so različne [funkcije jedra](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), ki jih je mogoče uporabiti, ter njihovi parametri.

## Naloga

[Novi model SVR](assignment.md)

## Zasluge

[^1]: Besedilo, koda in izhod v tem razdelku je prispeval [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Besedilo, koda in izhod v tem razdelku je vzeto iz [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.