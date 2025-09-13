<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T08:28:13+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ru"
}
-->
# Прогнозирование временных рядов с использованием Support Vector Regressor

В предыдущем уроке вы узнали, как использовать модель ARIMA для прогнозирования временных рядов. Теперь мы рассмотрим модель Support Vector Regressor, которая используется для предсказания непрерывных данных.

## [Тест перед лекцией](https://ff-quizzes.netlify.app/en/ml/) 

## Введение

В этом уроке вы узнаете, как строить модели с использованием [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) для регрессии, или **SVR: Support Vector Regressor**. 

### SVR в контексте временных рядов [^1]

Прежде чем понять важность SVR для прогнозирования временных рядов, необходимо ознакомиться с некоторыми ключевыми концепциями:

- **Регрессия:** Метод обучения с учителем, используемый для предсказания непрерывных значений на основе заданного набора входных данных. Идея заключается в нахождении кривой (или линии) в пространстве признаков, которая проходит через максимальное количество точек данных. [Подробнее здесь](https://en.wikipedia.org/wiki/Regression_analysis).
- **Support Vector Machine (SVM):** Тип модели машинного обучения с учителем, используемой для классификации, регрессии и обнаружения выбросов. Модель представляет собой гиперплоскость в пространстве признаков, которая в случае классификации действует как граница, а в случае регрессии — как линия наилучшего соответствия. В SVM обычно используется функция ядра для преобразования набора данных в пространство с большим количеством измерений, чтобы данные стали более разделимыми. [Подробнее здесь](https://en.wikipedia.org/wiki/Support-vector_machine).
- **Support Vector Regressor (SVR):** Тип SVM, который находит линию наилучшего соответствия (в случае SVM это гиперплоскость), проходящую через максимальное количество точек данных.

### Почему SVR? [^1]

В прошлом уроке вы изучили ARIMA — очень успешный статистический линейный метод прогнозирования временных рядов. Однако во многих случаях данные временных рядов обладают *нелинейностью*, которую линейные модели не могут отразить. В таких случаях способность SVM учитывать нелинейность данных делает SVR успешным инструментом для прогнозирования временных рядов.

## Упражнение — создание модели SVR

Первые шаги подготовки данных такие же, как и в предыдущем уроке про [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Откройте папку [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) в этом уроке и найдите файл [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. Запустите ноутбук и импортируйте необходимые библиотеки: [^2]

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

2. Загрузите данные из файла `/data/energy.csv` в Pandas DataFrame и ознакомьтесь с ними: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Постройте график всех доступных данных об энергии с января 2012 года по декабрь 2014 года: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![полные данные](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Теперь создадим модель SVR.

### Создание обучающих и тестовых наборов данных

После загрузки данных разделите их на обучающий и тестовый наборы. Затем преобразуйте данные, чтобы создать набор данных с временными шагами, который потребуется для SVR. Вы обучите модель на обучающем наборе. После завершения обучения вы оцените её точность на обучающем наборе, тестовом наборе и на полном наборе данных, чтобы оценить общую производительность. Убедитесь, что тестовый набор охватывает более поздний период времени, чем обучающий, чтобы модель не получала информацию из будущих временных периодов [^2] (ситуация, известная как *переобучение*).

1. Выделите двухмесячный период с 1 сентября по 31 октября 2014 года для обучающего набора. Тестовый набор будет включать двухмесячный период с 1 ноября по 31 декабря 2014 года: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Визуализируйте различия: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![обучающие и тестовые данные](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Подготовка данных для обучения

Теперь необходимо подготовить данные для обучения, выполнив фильтрацию и масштабирование. Отфильтруйте набор данных, чтобы включить только нужные временные периоды и столбцы, а также выполните масштабирование, чтобы данные находились в интервале 0,1.

1. Отфильтруйте исходный набор данных, чтобы включить только указанные временные периоды и нужный столбец 'load' плюс дату: [^2]

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
   
2. Масштабируйте обучающие данные в диапазон (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Теперь масштабируйте тестовые данные: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Создание данных с временными шагами [^1]

Для SVR преобразуйте входные данные в форму `[batch, timesteps]`. Таким образом, преобразуйте существующие `train_data` и `test_data`, добавив новое измерение, которое будет соответствовать временным шагам. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

В этом примере мы берем `timesteps = 5`. Таким образом, входные данные для модели — это данные за первые 4 временных шага, а выходные данные — за 5-й временной шаг.

```python
timesteps=5
```

Преобразование обучающих данных в 2D-тензор с использованием вложенного спискового включения:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Преобразование тестовых данных в 2D-тензор:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Выбор входных и выходных данных из обучающих и тестовых данных:

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

### Реализация SVR [^1]

Теперь пришло время реализовать SVR. Чтобы узнать больше об этой реализации, вы можете обратиться к [этой документации](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Для нашей реализации следуем следующим шагам:

1. Определите модель, вызвав `SVR()` и передав гиперпараметры модели: kernel, gamma, c и epsilon.
2. Подготовьте модель для обучающих данных, вызвав функцию `fit()`.
3. Сделайте предсказания, вызвав функцию `predict()`.

Теперь создадим модель SVR. Здесь мы используем [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) и задаем гиперпараметры gamma, C и epsilon как 0.5, 10 и 0.05 соответственно.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Обучение модели на обучающих данных [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Выполнение предсказаний модели [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Вы создали SVR! Теперь нужно его оценить.

### Оценка модели [^1]

Для оценки сначала вернем данные к исходному масштабу. Затем, чтобы проверить производительность, построим график исходного и предсказанного временного ряда, а также выведем результат MAPE.

Масштабирование предсказанных и исходных данных:

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

#### Проверка производительности модели на обучающих и тестовых данных [^1]

Извлекаем временные метки из набора данных для отображения на оси x нашего графика. Обратите внимание, что мы используем первые ```timesteps-1``` значения в качестве входных данных для первого выхода, поэтому временные метки для выхода начнутся после этого.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Построение предсказаний для обучающих данных:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![предсказания обучающих данных](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Вывод MAPE для обучающих данных:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Построение предсказаний для тестовых данных:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![предсказания тестовых данных](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Вывод MAPE для тестовых данных:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Отличный результат на тестовом наборе данных!

### Проверка производительности модели на полном наборе данных [^1]

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

![предсказания полного набора данных](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Отличные графики, показывающие модель с хорошей точностью. Отличная работа!

---

## 🚀Задание

- Попробуйте изменить гиперпараметры (gamma, C, epsilon) при создании модели и оцените результаты на данных, чтобы определить, какой набор гиперпараметров дает лучшие результаты на тестовом наборе данных. Подробнее о гиперпараметрах можно узнать [здесь](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Попробуйте использовать разные функции ядра для модели и проанализируйте их производительность на наборе данных. Полезный документ можно найти [здесь](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Попробуйте использовать разные значения `timesteps`, чтобы модель могла учитывать больше данных для предсказания.

## [Тест после лекции](https://ff-quizzes.netlify.app/en/ml/)

## Обзор и самостоятельное изучение

Этот урок был посвящен применению SVR для прогнозирования временных рядов. Чтобы узнать больше о SVR, вы можете обратиться к [этому блогу](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Эта [документация scikit-learn](https://scikit-learn.org/stable/modules/svm.html) предоставляет более полное объяснение SVM в целом, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression), а также других деталей реализации, таких как различные [функции ядра](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) и их параметры.

## Задание

[Новая модель SVR](assignment.md)

## Благодарности

[^1]: Текст, код и результаты в этом разделе были предоставлены [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Текст, код и результаты в этом разделе были взяты из [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Несмотря на наши усилия обеспечить точность, автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.