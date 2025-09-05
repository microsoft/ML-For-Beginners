<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T23:54:21+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "cs"
}
-->
# Předpověď časových řad pomocí Support Vector Regressor

V předchozí lekci jste se naučili používat model ARIMA k předpovědi časových řad. Nyní se podíváme na model Support Vector Regressor, což je regresní model používaný k předpovědi spojitých dat.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/) 

## Úvod

V této lekci objevíte specifický způsob, jak vytvářet modely pomocí [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) pro regresi, tedy **SVR: Support Vector Regressor**. 

### SVR v kontextu časových řad [^1]

Než pochopíte význam SVR při předpovědi časových řad, je důležité se seznámit s následujícími koncepty:

- **Regrese:** Technika učení s učitelem, která předpovídá spojité hodnoty na základě dané sady vstupů. Cílem je najít křivku (nebo přímku) v prostoru vlastností, která obsahuje maximální počet datových bodů. [Klikněte zde](https://en.wikipedia.org/wiki/Regression_analysis) pro více informací.
- **Support Vector Machine (SVM):** Typ modelu strojového učení s učitelem používaný pro klasifikaci, regresi a detekci odlehlých hodnot. Model je hyperrovina v prostoru vlastností, která v případě klasifikace funguje jako hranice a v případě regrese jako nejlepší přímka. V SVM se obvykle používá funkce jádra k transformaci datové sady do prostoru s vyšším počtem dimenzí, aby byly snadněji oddělitelné. [Klikněte zde](https://en.wikipedia.org/wiki/Support-vector_machine) pro více informací o SVM.
- **Support Vector Regressor (SVR):** Typ SVM, který hledá nejlepší přímku (v případě SVM hyperrovinu), která obsahuje maximální počet datových bodů.

### Proč SVR? [^1]

V minulé lekci jste se naučili o ARIMA, což je velmi úspěšná statistická lineární metoda pro předpověď časových řad. Nicméně v mnoha případech mají časové řady *nelinearitu*, kterou nelze mapovat pomocí lineárních modelů. V takových případech schopnost SVM zohlednit nelinearitu dat při regresních úlohách činí SVR úspěšným při předpovědi časových řad.

## Cvičení - vytvoření modelu SVR

První kroky přípravy dat jsou stejné jako v předchozí lekci o [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Otevřete složku [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) v této lekci a najděte soubor [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Spusťte notebook a importujte potřebné knihovny: [^2]

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

2. Načtěte data ze souboru `/data/energy.csv` do Pandas dataframe a podívejte se na ně: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Vykreslete všechna dostupná data o energii od ledna 2012 do prosince 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Nyní vytvoříme model SVR.

### Vytvoření trénovacích a testovacích datových sad

Nyní máte data načtená, takže je můžete rozdělit na trénovací a testovací sady. Poté data upravíte tak, aby vytvořila datovou sadu založenou na časových krocích, což bude potřeba pro SVR. Model budete trénovat na trénovací sadě. Po dokončení trénování modelu vyhodnotíte jeho přesnost na trénovací sadě, testovací sadě a poté na celé datové sadě, abyste viděli celkový výkon. Musíte zajistit, aby testovací sada pokrývala pozdější období než trénovací sada, aby model nezískal informace z budoucích časových období [^2] (situace známá jako *přeučení*).

1. Vyčleňte dvouměsíční období od 1. září do 31. října 2014 pro trénovací sadu. Testovací sada bude zahrnovat dvouměsíční období od 1. listopadu do 31. prosince 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Vizualizujte rozdíly: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Příprava dat pro trénování

Nyní je potřeba připravit data pro trénování provedením filtrování a škálování dat. Filtrovat budete datovou sadu tak, aby zahrnovala pouze potřebná časová období a sloupce, a škálovat, aby byla data projektována do intervalu 0,1.

1. Filtrovat původní datovou sadu tak, aby zahrnovala pouze zmíněná časová období na sadu a pouze potřebný sloupec 'load' plus datum: [^2]

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
   
2. Škálovat trénovací data do rozsahu (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Nyní škálujte testovací data: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Vytvoření dat s časovými kroky [^1]

Pro SVR transformujete vstupní data do formy `[batch, timesteps]`. Takže upravíte existující `train_data` a `test_data` tak, aby existovala nová dimenze, která odkazuje na časové kroky.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Pro tento příklad bereme `timesteps = 5`. Vstupy do modelu jsou tedy data pro první 4 časové kroky a výstup budou data pro 5. časový krok.

```python
timesteps=5
```

Konverze trénovacích dat na 2D tensor pomocí vnořené list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Konverze testovacích dat na 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Výběr vstupů a výstupů z trénovacích a testovacích dat:

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

### Implementace SVR [^1]

Nyní je čas implementovat SVR. Pro více informací o této implementaci můžete navštívit [tuto dokumentaci](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Pro naši implementaci postupujeme podle těchto kroků:

  1. Definujte model voláním `SVR()` a předáním hyperparametrů modelu: kernel, gamma, c a epsilon
  2. Připravte model pro trénovací data voláním funkce `fit()`
  3. Proveďte předpovědi voláním funkce `predict()`

Nyní vytvoříme model SVR. Použijeme [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) a nastavíme hyperparametry gamma, C a epsilon na hodnoty 0.5, 10 a 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Trénování modelu na trénovacích datech [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Předpovědi modelu [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Postavili jste svůj SVR! Nyní je potřeba jej vyhodnotit.

### Vyhodnocení modelu [^1]

Pro vyhodnocení nejprve škálujeme data zpět na původní měřítko. Poté, abychom zkontrolovali výkon, vykreslíme původní a předpovězený graf časových řad a také vytiskneme výsledek MAPE.

Škálování předpovězených a původních výstupů:

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

#### Kontrola výkonu modelu na trénovacích a testovacích datech [^1]

Z datové sady extrahujeme časové značky, které zobrazíme na ose x našeho grafu. Všimněte si, že používáme prvních ```timesteps-1``` hodnot jako vstup pro první výstup, takže časové značky pro výstup začnou až poté.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Vykreslení předpovědí pro trénovací data:

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

Tisk MAPE pro trénovací data

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Vykreslení předpovědí pro testovací data

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Tisk MAPE pro testovací data

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Máte velmi dobrý výsledek na testovací datové sadě!

### Kontrola výkonu modelu na celé datové sadě [^1]

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

🏆 Velmi pěkné grafy, ukazující model s dobrou přesností. Skvělá práce!

---

## 🚀Výzva

- Zkuste upravit hyperparametry (gamma, C, epsilon) při vytváření modelu a vyhodnoťte data, abyste zjistili, která sada hyperparametrů poskytuje nejlepší výsledky na testovacích datech. Více o těchto hyperparametrech se dozvíte v [dokumentaci zde](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Zkuste použít různé funkce jádra pro model a analyzujte jejich výkon na datové sadě. Užitečný dokument najdete [zde](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Zkuste použít různé hodnoty `timesteps`, aby model mohl zpětně předpovídat.

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Tato lekce měla za cíl představit aplikaci SVR pro předpověď časových řad. Pro více informací o SVR můžete navštívit [tento blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Tato [dokumentace na scikit-learn](https://scikit-learn.org/stable/modules/svm.html) poskytuje komplexnější vysvětlení o SVM obecně, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) a také další detaily implementace, jako jsou různé [funkce jádra](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), které lze použít, a jejich parametry.

## Zadání

[Nový model SVR](assignment.md)

## Poděkování

[^1]: Text, kód a výstup v této sekci přispěl [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Text, kód a výstup v této sekci byl převzat z [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.