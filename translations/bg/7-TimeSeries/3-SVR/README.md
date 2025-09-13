<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T23:53:49+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "bg"
}
-->
# Прогнозиране на времеви серии със Support Vector Regressor

В предишния урок научихте как да използвате модела ARIMA за прогнозиране на времеви серии. Сега ще разгледате модела Support Vector Regressor, който е регресионен модел, използван за прогнозиране на непрекъснати данни.

## [Тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/) 

## Въведение

В този урок ще откриете специфичен начин за изграждане на модели с [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) за регресия, или **SVR: Support Vector Regressor**. 

### SVR в контекста на времеви серии [^1]

Преди да разберете значението на SVR за прогнозиране на времеви серии, ето някои важни концепции, които трябва да знаете:

- **Регресия:** Техника за обучение с надзор, която предсказва непрекъснати стойности от даден набор от входни данни. Идеята е да се намери крива (или линия) в пространството на характеристиките, която има максимален брой точки от данни. [Кликнете тук](https://en.wikipedia.org/wiki/Regression_analysis) за повече информация.
- **Support Vector Machine (SVM):** Вид модел за машинно обучение с надзор, използван за класификация, регресия и откриване на аномалии. Моделът представлява хиперплоскост в пространството на характеристиките, която в случай на класификация действа като граница, а в случай на регресия - като линия на най-добро съответствие. В SVM обикновено се използва Kernel функция за трансформиране на набора от данни в пространство с по-голям брой измерения, така че те да бъдат лесно разделими. [Кликнете тук](https://en.wikipedia.org/wiki/Support-vector_machine) за повече информация за SVM.
- **Support Vector Regressor (SVR):** Вид SVM, който намира линия на най-добро съответствие (която в случая на SVM е хиперплоскост), която има максимален брой точки от данни.

### Защо SVR? [^1]

В последния урок научихте за ARIMA, който е много успешен статистически линеен метод за прогнозиране на времеви серии. Въпреки това, в много случаи времевите серии имат *нелинейност*, която не може да бъде моделирана от линейни модели. В такива случаи способността на SVM да отчита нелинейността в данните за задачи по регресия прави SVR успешен в прогнозиране на времеви серии.

## Упражнение - изграждане на SVR модел

Първите няколко стъпки за подготовка на данните са същите като тези от предишния урок за [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Отворете папката [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) в този урок и намерете файла [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Стартирайте notebook-а и импортирайте необходимите библиотеки: [^2]

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

2. Заредете данните от файла `/data/energy.csv` в Pandas dataframe и разгледайте ги: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Начертайте всички налични данни за енергия от януари 2012 до декември 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![пълни данни](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Сега нека изградим нашия SVR модел.

### Създаване на тренировъчни и тестови набори от данни

Сега данните ви са заредени, така че можете да ги разделите на тренировъчен и тестов набор. След това ще преформатирате данните, за да създадете набор от данни, базиран на времеви стъпки, който ще бъде необходим за SVR. Ще обучите модела си върху тренировъчния набор. След като моделът приключи обучението, ще оцените неговата точност върху тренировъчния набор, тестовия набор и след това върху целия набор от данни, за да видите цялостното представяне. Трябва да се уверите, че тестовият набор обхваща по-късен период от време спрямо тренировъчния набор, за да гарантирате, че моделът не получава информация от бъдещи времеви периоди [^2] (ситуация, известна като *Overfitting*).

1. Отделете двумесечен период от 1 септември до 31 октомври 2014 за тренировъчния набор. Тестовият набор ще включва двумесечния период от 1 ноември до 31 декември 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Визуализирайте разликите: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![тренировъчни и тестови данни](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Подготовка на данните за обучение

Сега трябва да подготвите данните за обучение, като извършите филтриране и скалиране на данните. Филтрирайте набора от данни, за да включите само необходимите времеви периоди и колони, и скалирайте, за да гарантирате, че данните са проектирани в интервала 0,1.

1. Филтрирайте оригиналния набор от данни, за да включите само споменатите времеви периоди за всеки набор и само необходимата колона 'load' плюс датата: [^2]

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
   
2. Скалирайте тренировъчните данни, за да бъдат в диапазона (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Сега скалирайте тестовите данни: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Създаване на данни с времеви стъпки [^1]

За SVR трансформирате входните данни, за да бъдат във формата `[batch, timesteps]`. Така преформатирате съществуващите `train_data` и `test_data`, така че да има ново измерение, което се отнася до времевите стъпки. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

За този пример вземаме `timesteps = 5`. Така входовете към модела са данните за първите 4 времеви стъпки, а изходът ще бъде данните за 5-тата времева стъпка.

```python
timesteps=5
```

Преобразуване на тренировъчните данни в 2D тензор с помощта на вложени списъчни разбирания:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Преобразуване на тестовите данни в 2D тензор:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Избор на входове и изходи от тренировъчните и тестовите данни:

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

### Имплементиране на SVR [^1]

Сега е време да имплементирате SVR. За повече информация относно тази имплементация можете да се обърнете към [тази документация](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). За нашата имплементация следваме тези стъпки:

  1. Дефинирайте модела, като извикате `SVR()` и зададете хиперпараметрите на модела: kernel, gamma, c и epsilon
  2. Подгответе модела за тренировъчните данни, като извикате функцията `fit()`
  3. Направете прогнози, като извикате функцията `predict()`

Сега създаваме SVR модел. Тук използваме [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) и задаваме хиперпараметрите gamma, C и epsilon като 0.5, 10 и 0.05 съответно.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Обучение на модела върху тренировъчни данни [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Направете прогнози с модела [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Създадохте SVR! Сега трябва да го оцените.

### Оценка на модела [^1]

За оценка първо ще скалираме обратно данните към оригиналната скала. След това, за да проверим представянето, ще начертаем графика на оригиналните и прогнозирани времеви серии и ще отпечатаме резултата от MAPE.

Скалирайте обратно прогнозните и оригиналните изходи:

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

#### Проверка на представянето на модела върху тренировъчни и тестови данни [^1]

Извличаме времевите марки от набора от данни, за да ги покажем на x-оста на графиката. Забележете, че използваме първите ```timesteps-1``` стойности като вход за първия изход, така че времевите марки за изхода ще започнат след това.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Начертайте прогнозите за тренировъчни данни:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![прогноза за тренировъчни данни](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Отпечатайте MAPE за тренировъчни данни

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Начертайте прогнозите за тестови данни

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![прогноза за тестови данни](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Отпечатайте MAPE за тестови данни

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Имате много добър резултат върху тестовия набор от данни!

### Проверка на представянето на модела върху целия набор от данни [^1]

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

![прогноза за целия набор от данни](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Много хубави графики, показващи модел с добра точност. Браво!

---

## 🚀Предизвикателство

- Опитайте да промените хиперпараметрите (gamma, C, epsilon) при създаването на модела и оценете данните, за да видите кой набор от хиперпараметри дава най-добри резултати върху тестовия набор от данни. За повече информация относно тези хиперпараметри можете да се обърнете към документа [тук](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Опитайте да използвате различни функции на kernel за модела и анализирайте тяхното представяне върху набора от данни. Полезен документ можете да намерите [тук](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Опитайте да използвате различни стойности за `timesteps`, за да накарате модела да се върне назад и да направи прогноза.

## [Тест след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостоятелно обучение

Този урок беше за въвеждане на приложението на SVR за прогнозиране на времеви серии. За повече информация относно SVR можете да се обърнете към [този блог](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Тази [документация за scikit-learn](https://scikit-learn.org/stable/modules/svm.html) предоставя по-изчерпателно обяснение за SVM като цяло, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) и също така други детайли за имплементация, като различните [функции на kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), които могат да бъдат използвани, и техните параметри.

## Задача

[Нов SVR модел](assignment.md)

## Благодарности

[^1]: Текстът, кодът и резултатите в този раздел са предоставени от [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Текстът, кодът и резултатите в този раздел са взети от [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.