<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T15:37:17+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "sk"
}
-->
# Predpovedanie časových radov pomocou Support Vector Regressor

V predchádzajúcej lekcii ste sa naučili používať model ARIMA na predpovedanie časových radov. Teraz sa pozrieme na model Support Vector Regressor, ktorý je regresný model používaný na predpovedanie spojitých údajov.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/) 

## Úvod

V tejto lekcii objavíte špecifický spôsob budovania modelov pomocou [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) pre regresiu, alebo **SVR: Support Vector Regressor**. 

### SVR v kontexte časových radov [^1]

Predtým, než pochopíte význam SVR pri predpovedaní časových radov, je dôležité poznať nasledujúce koncepty:

- **Regresia:** Technika učenia s učiteľom na predpovedanie spojitých hodnôt z daného súboru vstupov. Ide o prispôsobenie krivky (alebo čiary) v priestore vlastností, ktorá obsahuje maximálny počet dátových bodov. [Kliknite sem](https://en.wikipedia.org/wiki/Regression_analysis) pre viac informácií.
- **Support Vector Machine (SVM):** Typ modelu strojového učenia s učiteľom používaný na klasifikáciu, regresiu a detekciu odľahlých hodnôt. Model je hyperplocha v priestore vlastností, ktorá v prípade klasifikácie funguje ako hranica a v prípade regresie ako najlepšie prispôsobená čiara. V SVM sa zvyčajne používa funkcia Kernel na transformáciu dátového súboru do priestoru s vyšším počtom dimenzií, aby boli ľahšie oddeliteľné. [Kliknite sem](https://en.wikipedia.org/wiki/Support-vector_machine) pre viac informácií o SVM.
- **Support Vector Regressor (SVR):** Typ SVM, ktorý hľadá najlepšie prispôsobenú čiaru (ktorá je v prípade SVM hyperplocha) obsahujúcu maximálny počet dátových bodov.

### Prečo SVR? [^1]

V poslednej lekcii ste sa naučili o ARIMA, čo je veľmi úspešná štatistická lineárna metóda na predpovedanie časových radov. Avšak v mnohých prípadoch majú časové rady *nelinearitu*, ktorú lineárne modely nedokážu zachytiť. V takýchto prípadoch schopnosť SVM zohľadniť nelinearitu v údajoch pri regresných úlohách robí SVR úspešným pri predpovedaní časových radov.

## Cvičenie - vytvorenie modelu SVR

Prvé kroky na prípravu údajov sú rovnaké ako v predchádzajúcej lekcii o [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Otvorte priečinok [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) v tejto lekcii a nájdite súbor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Spustite notebook a importujte potrebné knižnice: [^2]

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

2. Načítajte údaje zo súboru `/data/energy.csv` do Pandas dataframe a pozrite sa na ne: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Vykreslite všetky dostupné údaje o energii od januára 2012 do decembra 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![úplné údaje](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Teraz vytvoríme náš model SVR.

### Vytvorenie tréningových a testovacích dátových súborov

Keď sú vaše údaje načítané, môžete ich rozdeliť na tréningovú a testovaciu množinu. Potom údaje upravíte tak, aby vytvorili dataset založený na časových krokoch, ktorý bude potrebný pre SVR. Model budete trénovať na tréningovej množine. Po dokončení tréningu modelu vyhodnotíte jeho presnosť na tréningovej množine, testovacej množine a potom na celom datasete, aby ste videli celkový výkon. Musíte zabezpečiť, že testovacia množina pokrýva neskoršie obdobie v čase oproti tréningovej množine, aby ste zabezpečili, že model nezíska informácie z budúcich časových období [^2] (situácia známa ako *Overfitting*).

1. Priraďte dvojmesačné obdobie od 1. septembra do 31. októbra 2014 tréningovej množine. Testovacia množina bude zahŕňať dvojmesačné obdobie od 1. novembra do 31. decembra 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Vizualizujte rozdiely: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![tréningové a testovacie údaje](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Príprava údajov na tréning

Teraz musíte pripraviť údaje na tréning vykonaním filtrovania a škálovania údajov. Filtrovanie datasetu zahŕňa iba časové obdobia a stĺpce, ktoré potrebujete, a škálovanie zabezpečí, že údaje budú premietnuté do intervalu 0,1.

1. Filtrovanie pôvodného datasetu tak, aby zahŕňal iba uvedené časové obdobia na množinu a iba potrebný stĺpec 'load' plus dátum: [^2]

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
   
2. Škálovanie tréningových údajov do rozsahu (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Teraz škálujte testovacie údaje: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Vytvorenie údajov s časovými krokmi [^1]

Pre SVR transformujete vstupné údaje do formy `[batch, timesteps]`. Takže existujúce `train_data` a `test_data` upravíte tak, aby obsahovali nový rozmer, ktorý sa týka časových krokov. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Pre tento príklad berieme `timesteps = 5`. Takže vstupy do modelu sú údaje za prvé 4 časové kroky a výstup budú údaje za 5. časový krok.

```python
timesteps=5
```

Konverzia tréningových údajov na 2D tensor pomocou vnoreného zoznamového porozumenia:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Konverzia testovacích údajov na 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Výber vstupov a výstupov z tréningových a testovacích údajov:

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

### Implementácia SVR [^1]

Teraz je čas implementovať SVR. Prečítajte si viac o tejto implementácii v [dokumentácii](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Pre našu implementáciu postupujeme podľa týchto krokov:

  1. Definujte model volaním `SVR()` a zadaním hyperparametrov modelu: kernel, gamma, c a epsilon
  2. Pripravte model na tréningové údaje volaním funkcie `fit()`
  3. Vytvorte predpovede volaním funkcie `predict()`

Teraz vytvoríme model SVR. Použijeme [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) a nastavíme hyperparametre gamma, C a epsilon na hodnoty 0.5, 10 a 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Tréning modelu na tréningových údajoch [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Vytvorenie predpovedí modelu [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Postavili ste svoj SVR! Teraz ho musíme vyhodnotiť.

### Vyhodnotenie modelu [^1]

Na vyhodnotenie najskôr škálujeme údaje späť na pôvodnú škálu. Potom, aby sme skontrolovali výkon, vykreslíme pôvodný a predpovedaný časový rad a tiež vytlačíme výsledok MAPE.

Škálovanie predpovedaného a pôvodného výstupu:

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

#### Kontrola výkonu modelu na tréningových a testovacích údajoch [^1]

Z datasetu extrahujeme časové značky, aby sme ich zobrazili na osi x nášho grafu. Všimnite si, že používame prvých ```timesteps-1``` hodnôt ako vstup pre prvý výstup, takže časové značky pre výstup začnú až po tom.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Vykreslenie predpovedí pre tréningové údaje:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![predpoveď tréningových údajov](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Tlač MAPE pre tréningové údaje

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Vykreslenie predpovedí pre testovacie údaje

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![predpoveď testovacích údajov](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Tlač MAPE pre testovacie údaje

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Dosiahli ste veľmi dobrý výsledok na testovacom datasete!

### Kontrola výkonu modelu na celom datasete [^1]

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

![predpoveď celých údajov](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Veľmi pekné grafy, ktoré ukazujú model s dobrou presnosťou. Skvelá práca!

---

## 🚀Výzva

- Skúste upraviť hyperparametre (gamma, C, epsilon) pri vytváraní modelu a vyhodnoťte údaje, aby ste zistili, ktorá sada hyperparametrov poskytuje najlepšie výsledky na testovacích údajoch. Viac o týchto hyperparametroch sa dozviete v [dokumente](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Skúste použiť rôzne funkcie kernelu pre model a analyzujte ich výkonnosť na datasete. Užitočný dokument nájdete [tu](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Skúste použiť rôzne hodnoty pre `timesteps`, aby model mohol pozerať späť na predpoveď.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Táto lekcia bola zameraná na predstavenie aplikácie SVR pre predpovedanie časových radov. Viac o SVR si môžete prečítať v [tomto blogu](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Táto [dokumentácia na scikit-learn](https://scikit-learn.org/stable/modules/svm.html) poskytuje komplexnejšie vysvetlenie o SVM všeobecne, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) a tiež ďalšie detaily implementácie, ako sú rôzne [funkcie kernelu](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), ktoré je možné použiť, a ich parametre.

## Zadanie

[Nový model SVR](assignment.md)

## Kredity

[^1]: Text, kód a výstup v tejto sekcii prispel [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Text, kód a výstup v tejto sekcii bol prevzatý z [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.