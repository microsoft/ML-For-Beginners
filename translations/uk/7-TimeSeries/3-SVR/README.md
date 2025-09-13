<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T12:06:12+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "uk"
}
-->
# Прогнозування часових рядів за допомогою регресора опорних векторів

У попередньому уроці ви дізналися, як використовувати модель ARIMA для прогнозування часових рядів. Тепер ви ознайомитеся з моделлю регресора опорних векторів (Support Vector Regressor), яка використовується для прогнозування безперервних даних.

## [Тест перед лекцією](https://ff-quizzes.netlify.app/en/ml/) 

## Вступ

У цьому уроці ви дізнаєтеся, як створювати моделі за допомогою [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) для регресії, або **SVR: Support Vector Regressor**. 

### SVR у контексті часових рядів [^1]

Перед тим як зрозуміти важливість SVR у прогнозуванні часових рядів, ось кілька важливих концепцій, які вам потрібно знати:

- **Регресія:** Техніка навчання з учителем для прогнозування безперервних значень на основі заданого набору вхідних даних. Ідея полягає у побудові кривої (або лінії) у просторі ознак, яка має максимальну кількість точок даних. [Натисніть тут](https://en.wikipedia.org/wiki/Regression_analysis) для отримання додаткової інформації.
- **Машина опорних векторів (SVM):** Тип моделі машинного навчання з учителем, яка використовується для класифікації, регресії та виявлення аномалій. Модель є гіперплощиною у просторі ознак, яка у випадку класифікації виступає як межа, а у випадку регресії — як лінія найкращого підходу. У SVM зазвичай використовується функція ядра для перетворення набору даних у простір з більшою кількістю вимірів, щоб вони могли бути легко розділені. [Натисніть тут](https://en.wikipedia.org/wiki/Support-vector_machine) для отримання додаткової інформації про SVM.
- **Регресор опорних векторів (SVR):** Тип SVM, який знаходить лінію найкращого підходу (яка у випадку SVM є гіперплощиною) з максимальною кількістю точок даних.

### Чому SVR? [^1]

У попередньому уроці ви дізналися про ARIMA, яка є дуже успішним статистичним лінійним методом для прогнозування даних часових рядів. Однак у багатьох випадках дані часових рядів мають *нелінійність*, яку неможливо відобразити за допомогою лінійних моделей. У таких випадках здатність SVM враховувати нелінійність даних для задач регресії робить SVR успішним у прогнозуванні часових рядів.

## Вправа - створення моделі SVR

Перші кілька кроків підготовки даних такі ж, як у попередньому уроці про [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Відкрийте папку [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) у цьому уроці та знайдіть файл [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Запустіть ноутбук та імпортуйте необхідні бібліотеки: [^2]

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

2. Завантажте дані з файлу `/data/energy.csv` у Pandas DataFrame та перегляньте їх: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Побудуйте графік усіх доступних даних про енергію з січня 2012 року до грудня 2014 року: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![повні дані](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Тепер давайте створимо нашу модель SVR.

### Створення навчальних і тестових наборів даних

Тепер ваші дані завантажені, тому ви можете розділити їх на навчальний і тестовий набори. Потім ви переформатуєте дані, щоб створити набір даних на основі часових кроків, який буде потрібен для SVR. Ви навчите свою модель на навчальному наборі. Після завершення навчання моделі ви оціните її точність на навчальному наборі, тестовому наборі, а потім на повному наборі даних, щоб побачити загальну продуктивність. Ви повинні переконатися, що тестовий набір охоплює більш пізній період часу, ніж навчальний набір, щоб гарантувати, що модель не отримує інформацію з майбутніх періодів часу [^2] (ситуація, відома як *перенавчання*).

1. Виділіть двомісячний період з 1 вересня по 31 жовтня 2014 року для навчального набору. Тестовий набір включатиме двомісячний період з 1 листопада по 31 грудня 2014 року: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Візуалізуйте відмінності: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![навчальні та тестові дані](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Підготовка даних для навчання

Тепер вам потрібно підготувати дані для навчання, виконавши фільтрацію та масштабування даних. Відфільтруйте ваш набір даних, щоб включити лише потрібні періоди часу та стовпці, а також виконайте масштабування, щоб дані були представлені в інтервалі 0,1.

1. Відфільтруйте оригінальний набір даних, щоб включити лише зазначені періоди часу для кожного набору та лише потрібний стовпець 'load' плюс дату: [^2]

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
   
2. Масштабуйте навчальні дані до діапазону (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Тепер масштабуйте тестові дані: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Створення даних із часовими кроками [^1]

Для SVR ви перетворюєте вхідні дані у форму `[batch, timesteps]`. Тобто ви переформатуєте існуючі `train_data` та `test_data`, щоб додати новий вимір, який відповідає часовим крокам. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Для цього прикладу ми беремо `timesteps = 5`. Отже, вхідні дані для моделі — це дані за перші 4 часові кроки, а вихідні — дані за 5-й часовий крок.

```python
timesteps=5
```

Перетворення навчальних даних у 2D-тензор за допомогою вкладеного спискового розуміння:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Перетворення тестових даних у 2D-тензор:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Вибір вхідних і вихідних даних із навчальних і тестових даних:

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

### Реалізація SVR [^1]

Тепер настав час реалізувати SVR. Щоб дізнатися більше про цю реалізацію, ви можете звернутися до [цієї документації](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Для нашої реалізації ми дотримуємося таких кроків:

  1. Визначте модель, викликавши `SVR()` і передавши гіперпараметри моделі: kernel, gamma, c та epsilon
  2. Підготуйте модель для навчальних даних, викликавши функцію `fit()`
  3. Зробіть прогнози, викликавши функцію `predict()`

Тепер ми створюємо модель SVR. Тут ми використовуємо [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) і встановлюємо гіперпараметри gamma, C та epsilon як 0.5, 10 та 0.05 відповідно.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Навчання моделі на навчальних даних [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Прогнозування за допомогою моделі [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Ви створили SVR! Тепер потрібно оцінити його.

### Оцінка вашої моделі [^1]

Для оцінки спочатку ми повернемо дані до оригінального масштабу. Потім, щоб перевірити продуктивність, ми побудуємо графік оригінальних і прогнозованих часових рядів, а також виведемо результат MAPE.

Масштабування прогнозованих і оригінальних вихідних даних:

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

#### Перевірка продуктивності моделі на навчальних і тестових даних [^1]

Ми витягуємо часові мітки з набору даних, щоб показати їх на осі x нашого графіка. Зверніть увагу, що ми використовуємо перші ```timesteps-1``` значення як вхідні дані для першого вихідного значення, тому часові мітки для вихідних даних почнуться після цього.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Побудова графіка прогнозів для навчальних даних:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![прогноз навчальних даних](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Виведення MAPE для навчальних даних

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Побудова графіка прогнозів для тестових даних

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![прогноз тестових даних](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Виведення MAPE для тестових даних

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Ви отримали дуже хороший результат на тестовому наборі даних!

### Перевірка продуктивності моделі на повному наборі даних [^1]

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

![прогноз повних даних](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Дуже гарні графіки, які показують модель з хорошою точністю. Чудова робота!

---

## 🚀Виклик

- Спробуйте змінити гіперпараметри (gamma, C, epsilon) під час створення моделі та оцінити дані, щоб побачити, який набір гіперпараметрів дає найкращі результати на тестових даних. Щоб дізнатися більше про ці гіперпараметри, ви можете звернутися до документа [тут](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Спробуйте використовувати різні функції ядра для моделі та аналізувати їх продуктивність на наборі даних. Корисний документ можна знайти [тут](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Спробуйте використовувати різні значення для `timesteps`, щоб модель могла дивитися назад для прогнозування.

## [Тест після лекції](https://ff-quizzes.netlify.app/en/ml/)

## Огляд і самостійне навчання

Цей урок був присвячений застосуванню SVR для прогнозування часових рядів. Щоб дізнатися більше про SVR, ви можете звернутися до [цього блогу](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Ця [документація scikit-learn](https://scikit-learn.org/stable/modules/svm.html) надає більш детальне пояснення про SVM загалом, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression), а також інші деталі реалізації, такі як різні [функції ядра](https://scikit-learn.org/stable/modules/svm.html#kernel-functions), які можна використовувати, та їх параметри.

## Завдання

[Нова модель SVR](assignment.md)

## Подяки

[^1]: Текст, код і результати в цьому розділі були надані [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Текст, код і результати в цьому розділі були взяті з [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, зверніть увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для критично важливої інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.