<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T12:04:07+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "sr"
}
-->
# Прогнозирање временских серија помоћу модела Support Vector Regressor

У претходној лекцији научили сте како да користите ARIMA модел за предвиђање временских серија. Сада ћете се упознати са моделом Support Vector Regressor, који је регресиони модел за предвиђање континуираних података.

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/) 

## Увод

У овој лекцији ћете открити специфичан начин за изградњу модела помоћу [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) за регресију, односно **SVR: Support Vector Regressor**. 

### SVR у контексту временских серија [^1]

Пре него што разумете значај SVR-а у предвиђању временских серија, ево неких важних концепата које треба да знате:

- **Регресија:** Надгледана техника учења за предвиђање континуираних вредности из датог скупа улаза. Идеја је да се уклопи крива (или права) у простор карактеристика која садржи максималан број података. [Кликните овде](https://en.wikipedia.org/wiki/Regression_analysis) за више информација.
- **Support Vector Machine (SVM):** Тип модела машинског учења са надзором који се користи за класификацију, регресију и детекцију одступања. Модел представља хиперраван у простору карактеристика, која у случају класификације делује као граница, а у случају регресије као најбоље уклопљена линија. У SVM-у се обично користи Kernel функција за трансформацију скупа података у простор са већим бројем димензија, како би били лакше раздвојиви. [Кликните овде](https://en.wikipedia.org/wiki/Support-vector_machine) за више информација о SVM-у.
- **Support Vector Regressor (SVR):** Тип SVM-а који проналази најбоље уклопљену линију (која је у случају SVM-а хиперраван) са максималним бројем података.

### Зашто SVR? [^1]

У претходној лекцији научили сте о ARIMA моделу, који је веома успешан статистички линеарни метод за прогнозирање временских серија. Међутим, у многим случајевима, временске серије садрже *нелинеарности*, које линеарни модели не могу да обраде. У таквим случајевима, способност SVM-а да узме у обзир нелинеарности у подацима за задатке регресије чини SVR успешним у прогнозирању временских серија.

## Вежба - изградња SVR модела

Први кораци у припреми података исти су као у претходној лекцији о [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Отворите [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) фасциклу у овој лекцији и пронађите [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) датотеку.[^2]

1. Покрените бележницу и увезите неопходне библиотеке:  [^2]

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

2. Учитајте податке из `/data/energy.csv` датотеке у Pandas dataframe и погледајте их:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Прикажите све доступне податке о енергији од јануара 2012. до децембра 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![пуни подаци](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Сада, хајде да изградимо наш SVR модел.

### Креирање скупова за обуку и тестирање

Сада када су ваши подаци учитани, можете их поделити на скуп за обуку и скуп за тестирање. Затим ћете преобликовати податке како бисте креирали скуп података заснован на временским корацима, што ће бити потребно за SVR. Обучаваћете модел на скупу за обуку. Након што модел заврши обуку, проценићете његову тачност на скупу за обуку, скупу за тестирање и затим на целом скупу података како бисте видели укупне перформансе. Морате осигурати да скуп за тестирање покрива каснији временски период у односу на скуп за обуку како бисте спречили да модел добије информације из будућих временских периода [^2] (ситуација позната као *претерано уклапање*).

1. Доделите двомесечни период од 1. септембра до 31. октобра 2014. скупу за обуку. Скуп за тестирање ће укључивати двомесечни период од 1. новембра до 31. децембра 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Визуелизујте разлике: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![подаци за обуку и тестирање](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Припрема података за обуку

Сада треба да припремите податке за обуку тако што ћете извршити филтрирање и скалирање података. Филтрирајте скуп података тако да укључује само потребне временске периоде и колоне, а затим извршите скалирање како би подаци били пројектовани у интервалу 0,1.

1. Филтрирајте оригинални скуп података тако да укључује само претходно наведене временске периоде по скупу и само потребну колону 'load' плус датум: [^2]

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
   
2. Скалирајте податке за обуку тако да буду у опсегу (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Сада скалирајте податке за тестирање: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Креирање података са временским корацима [^1]

За SVR, трансформишете улазне податке у облик `[batch, timesteps]`. Дакле, преобликујете постојеће `train_data` и `test_data` тако да постоји нова димензија која се односи на временске кораке. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

За овај пример, узимамо `timesteps = 5`. Дакле, улази у модел су подаци за прва 4 временска корака, а излаз ће бити подаци за 5. временски корак.

```python
timesteps=5
```

Претварање података за обуку у 2D тензор помоћу угнежђене листе:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Претварање података за тестирање у 2D тензор:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Одабир улаза и излаза из података за обуку и тестирање:

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

### Имплементација SVR [^1]

Сада је време за имплементацију SVR-а. За више информација о овој имплементацији, можете погледати [ову документацију](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). За нашу имплементацију, следимо ове кораке:

1. Дефинишите модел позивањем `SVR()` и прослеђивањем хиперпараметара модела: kernel, gamma, c и epsilon
2. Припремите модел за податке за обуку позивањем функције `fit()`
3. Направите предвиђања позивањем функције `predict()`

Сада креирамо SVR модел. Овде користимо [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), и постављамо хиперпараметре gamma, C и epsilon на 0.5, 10 и 0.05 респективно.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Обучите модел на подацима за обуку [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Направите предвиђања модела [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Изградили сте свој SVR! Сада треба да га процените.

### Процена вашег модела [^1]

За процену, прво ћемо вратити податке на оригиналну скалу. Затим, како бисмо проверили перформансе, приказаћемо оригинални и предвиђени график временских серија, као и исписати MAPE резултат.

Скалирајте предвиђени и оригинални излаз:

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

#### Проверите перформансе модела на подацима за обуку и тестирање [^1]

Извлачимо временске ознаке из скупа података како бисмо их приказали на x-оси нашег графика. Имајте у виду да користимо првих ```timesteps-1``` вредности као улаз за први излаз, тако да временске ознаке за излаз почињу након тога.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Прикажите предвиђања за податке за обуку:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![предвиђање података за обуку](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Испишите MAPE за податке за обуку

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Прикажите предвиђања за податке за тестирање

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![предвиђање података за тестирање](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Испишите MAPE за податке за тестирање

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Имате веома добар резултат на скупу података за тестирање!

### Проверите перформансе модела на целом скупу података [^1]

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

![предвиђање целог скупа података](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Врло лепи графици, који показују модел са добром тачношћу. Браво!

---

## 🚀Изазов

- Покушајте да подесите хиперпараметре (gamma, C, epsilon) приликом креирања модела и процените на подацима како бисте видели који сет хиперпараметара даје најбоље резултате на скупу података за тестирање. За више информација о овим хиперпараметрима, можете погледати [документацију овде](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Покушајте да користите различите kernel функције за модел и анализирајте њихове перформансе на скупу података. Корисна документација се налази [овде](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Покушајте да користите различите вредности за `timesteps` како би модел гледао уназад приликом предвиђања.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Ова лекција је била увод у примену SVR-а за прогнозирање временских серија. За више информација о SVR-у, можете погледати [овај блог](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ова [документација на scikit-learn](https://scikit-learn.org/stable/modules/svm.html) пружа свеобухватније објашњење о SVM-овима уопште, [SVR-овима](https://scikit-learn.org/stable/modules/svm.html#regression) и такође другим детаљима имплементације као што су различите [kernel функције](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) које се могу користити и њихови параметри.

## Задатак

[Нови SVR модел](assignment.md)

## Захвалнице

[^1]: Текст, код и излаз у овом одељку допринео је [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)  
[^2]: Текст, код и излаз у овом одељку преузет је из [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако тежимо тачности, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не сносимо одговорност за било каква неспоразумевања или погрешна тумачења која могу произаћи из коришћења овог превода.