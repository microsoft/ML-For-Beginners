<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T15:36:39+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "hu"
}
-->
# Idősoros előrejelzés Support Vector Regressor segítségével

Az előző leckében megtanultad, hogyan használhatod az ARIMA modellt idősoros előrejelzések készítésére. Most a Support Vector Regressor modellel fogsz megismerkedni, amely egy regressziós modell folyamatos adatok előrejelzésére.

## [Előzetes kvíz](https://ff-quizzes.netlify.app/en/ml/) 

## Bevezetés

Ebben a leckében felfedezheted, hogyan építhetsz modelleket [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) segítségével regresszióhoz, vagyis **SVR: Support Vector Regressor**.

### SVR az idősoros előrejelzés kontextusában [^1]

Mielőtt megértenéd az SVR fontosságát az idősoros előrejelzésben, íme néhány fontos fogalom, amelyet ismerned kell:

- **Regresszió:** Felügyelt tanulási technika, amely folyamatos értékeket jósol meg egy adott bemeneti halmazból. Az ötlet az, hogy egy görbét (vagy egyenes vonalat) illesszünk a jellemzők terébe, amely a legtöbb adatpontot tartalmazza. [Kattints ide](https://en.wikipedia.org/wiki/Regression_analysis) további információért.
- **Support Vector Machine (SVM):** Egy felügyelt gépi tanulási modell, amelyet osztályozásra, regresszióra és kiugró értékek detektálására használnak. A modell egy hipersík a jellemzők terében, amely osztályozás esetén határként, regresszió esetén pedig legjobban illeszkedő vonalként működik. Az SVM-ben általában Kernel függvényt használnak az adathalmaz magasabb dimenziójú térbe történő átalakítására, hogy könnyebben elválaszthatóak legyenek. [Kattints ide](https://en.wikipedia.org/wiki/Support-vector_machine) további információért az SVM-ekről.
- **Support Vector Regressor (SVR):** Az SVM egy típusa, amely megtalálja a legjobban illeszkedő vonalat (ami az SVM esetében egy hipersík), amely a legtöbb adatpontot tartalmazza.

### Miért SVR? [^1]

Az előző leckében megismerkedtél az ARIMA modellel, amely egy nagyon sikeres statisztikai lineáris módszer idősoros adatok előrejelzésére. Azonban sok esetben az idősoros adatok *nemlinearitást* mutatnak, amelyet a lineáris modellek nem tudnak leképezni. Ilyen esetekben az SVM képessége, hogy figyelembe vegye az adatok nemlinearitását a regressziós feladatok során, sikeressé teszi az SVR-t az idősoros előrejelzésben.

## Gyakorlat - SVR modell építése

Az első néhány adat-előkészítési lépés megegyezik az előző [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) leckében tanultakkal.

Nyisd meg a [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) mappát ebben a leckében, és keresd meg a [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) fájlt. [^2]

1. Futtasd a notebookot, és importáld a szükséges könyvtárakat: [^2]

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

2. Töltsd be az adatokat a `/data/energy.csv` fájlból egy Pandas dataframe-be, és nézd meg: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Ábrázold az összes elérhető energiaadatot 2012 januárjától 2014 decemberéig: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![teljes adatok](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Most építsük meg az SVR modellünket.

### Képzési és tesztelési adathalmazok létrehozása

Most, hogy az adatok betöltve vannak, szétválaszthatod őket képzési és tesztelési halmazokra. Ezután átalakítod az adatokat időlépés-alapú adathalmazzá, amelyre szükség lesz az SVR-hez. A modell képzését a képzési halmazon végzed. Miután a modell befejezte a képzést, kiértékeled a pontosságát a képzési halmazon, a tesztelési halmazon, majd az egész adathalmazon, hogy láthasd az általános teljesítményt. Biztosítanod kell, hogy a tesztelési halmaz egy későbbi időszakot fedjen le, mint a képzési halmaz, hogy elkerüld, hogy a modell információt szerezzen a jövőbeli időszakokból [^2] (ezt a helyzetet *túltanulásnak* nevezzük).

1. Jelölj ki egy két hónapos időszakot 2014. szeptember 1-től október 31-ig a képzési halmaz számára. A tesztelési halmaz a 2014. november 1-től december 31-ig tartó két hónapos időszakot fogja tartalmazni: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Vizualizáld a különbségeket: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![képzési és tesztelési adatok](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Adatok előkészítése a képzéshez

Most elő kell készítened az adatokat a képzéshez, szűréssel és skálázással. Szűrd az adathalmazt, hogy csak a szükséges időszakokat és oszlopokat tartalmazza, majd skálázd az adatokat, hogy az értékek a 0 és 1 közötti intervallumba essenek.

1. Szűrd az eredeti adathalmazt, hogy csak az említett időszakokat és a szükséges 'load' oszlopot, valamint a dátumot tartalmazza: [^2]

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
   
2. Skálázd a képzési adatokat a (0, 1) tartományba: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Most skálázd a tesztelési adatokat: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Adatok létrehozása időlépésekkel [^1]

Az SVR-hez az input adatokat `[batch, timesteps]` formára kell átalakítani. Ezért átalakítod a meglévő `train_data` és `test_data` adatokat úgy, hogy legyen egy új dimenzió, amely az időlépéseket jelöli.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Ebben a példában `timesteps = 5` értéket veszünk. Tehát a modell bemenete az első 4 időlépés adatai lesznek, a kimenet pedig az 5. időlépés adatai.

```python
timesteps=5
```

A képzési adatok 2D tensorra való átalakítása beágyazott listakomprehenzióval:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

A tesztelési adatok 2D tensorra való átalakítása:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

A képzési és tesztelési adatok bemeneteinek és kimeneteinek kiválasztása:

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

### SVR megvalósítása [^1]

Most itt az ideje, hogy megvalósítsd az SVR-t. További információért erről a megvalósításról, olvasd el [ezt a dokumentációt](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). A mi megvalósításunkban a következő lépéseket követjük:

  1. Definiáld a modellt az `SVR()` meghívásával, és add meg a modell hiperparamétereit: kernel, gamma, c és epsilon
  2. Készítsd elő a modellt a képzési adatokhoz az `fit()` függvény meghívásával
  3. Végezz előrejelzéseket az `predict()` függvény meghívásával

Most létrehozzuk az SVR modellt. Itt az [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) használatát választjuk, és a hiperparamétereket gamma, C és epsilon értékekre állítjuk: 0.5, 10 és 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Modell illesztése a képzési adatokra [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Modell előrejelzések készítése [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Elkészítetted az SVR-t! Most ki kell értékelni.

### Modell kiértékelése [^1]

A kiértékeléshez először visszaskálázzuk az adatokat az eredeti skálára. Ezután az eredmény ellenőrzéséhez ábrázoljuk az eredeti és előrejelzett idősoros adatokat, valamint kiírjuk a MAPE eredményt.

Az előrejelzett és eredeti kimenet skálázása:

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

#### Modell teljesítményének ellenőrzése a képzési és tesztelési adatokon [^1]

Kinyerjük az időbélyegeket az adathalmazból, hogy az x-tengelyen megjelenítsük őket. Ne feledd, hogy az első ```timesteps-1``` értékeket használjuk bemenetként az első kimenethez, így a kimenet időbélyegei ezután kezdődnek.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

A képzési adatok előrejelzéseinek ábrázolása:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![képzési adatok előrejelzése](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

MAPE kiírása a képzési adatokra:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

A tesztelési adatok előrejelzéseinek ábrázolása:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![tesztelési adatok előrejelzése](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

MAPE kiírása a tesztelési adatokra:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Nagyon jó eredményt értél el a tesztelési adathalmazon!

### Modell teljesítményének ellenőrzése a teljes adathalmazon [^1]

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

![teljes adatok előrejelzése](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Nagyon szép ábrák, amelyek egy jó pontosságú modellt mutatnak. Szép munka!

---

## 🚀Kihívás

- Próbáld meg módosítani a hiperparamétereket (gamma, C, epsilon) a modell létrehozásakor, és értékeld ki az adatokat, hogy lásd, melyik hiperparaméter-készlet adja a legjobb eredményeket a tesztelési adatokon. További információért ezekről a hiperparaméterekről olvasd el [ezt a dokumentumot](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Próbálj ki különböző kernel függvényeket a modellhez, és elemezd a teljesítményüket az adathalmazon. Egy hasznos dokumentumot itt találhatsz: [kernel függvények](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Próbálj ki különböző értékeket a `timesteps` paraméterhez, hogy a modell visszatekintve készítsen előrejelzést.

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Ez a lecke az SVR alkalmazását mutatta be idősoros előrejelzéshez. További információért az SVR-ről olvasd el [ezt a blogot](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ez a [scikit-learn dokumentáció](https://scikit-learn.org/stable/modules/svm.html) átfogóbb magyarázatot nyújt az SVM-ekről általában, az [SVR-ekről](https://scikit-learn.org/stable/modules/svm.html#regression), valamint más megvalósítási részletekről, például a különböző [kernel függvényekről](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), amelyek használhatók, és azok paramétereiről.

## Feladat

[Egy új SVR modell](assignment.md)

## Köszönet

[^1]: A szöveget, kódot és kimenetet ebben a szakaszban [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) készítette
[^2]: A szöveget, kódot és kimenetet ebben a szakaszban az [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) leckéből vettük

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget az ebből a fordításból eredő félreértésekért vagy téves értelmezésekért.