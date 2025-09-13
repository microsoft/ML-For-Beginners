<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T08:48:05+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ur"
}
-->
# وقت سیریز کی پیش گوئی سپورٹ ویکٹر ریگریسر کے ساتھ

پچھلے سبق میں، آپ نے ARIMA ماڈل کا استعمال کرتے ہوئے وقت سیریز کی پیش گوئی کرنا سیکھا۔ اب آپ سپورٹ ویکٹر ریگریسر ماڈل کے بارے میں جانیں گے، جو مسلسل ڈیٹا کی پیش گوئی کے لیے استعمال ہونے والا ایک ریگریسر ماڈل ہے۔

## [پری لیکچر کوئز](https://ff-quizzes.netlify.app/en/ml/) 

## تعارف

اس سبق میں، آپ ایک مخصوص طریقہ دریافت کریں گے جس کے ذریعے [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) کو ریگریشن کے لیے استعمال کیا جاتا ہے، جسے **SVR: Support Vector Regressor** کہا جاتا ہے۔

### وقت سیریز کے تناظر میں SVR [^1]

وقت سیریز کی پیش گوئی میں SVR کی اہمیت کو سمجھنے سے پہلے، یہاں کچھ اہم تصورات ہیں جنہیں جاننا ضروری ہے:

- **ریگریشن:** سپروائزڈ لرننگ تکنیک جو دیے گئے ان پٹ سیٹ سے مسلسل اقدار کی پیش گوئی کرتی ہے۔ اس کا مقصد فیچر اسپیس میں ایک لائن یا منحنی کو فٹ کرنا ہے جس پر زیادہ سے زیادہ ڈیٹا پوائنٹس ہوں۔ [مزید معلومات کے لیے یہاں کلک کریں](https://en.wikipedia.org/wiki/Regression_analysis)۔
- **سپورٹ ویکٹر مشین (SVM):** سپروائزڈ مشین لرننگ ماڈل کی ایک قسم جو کلاسیفیکیشن، ریگریشن اور آؤٹ لائرز ڈیٹیکشن کے لیے استعمال ہوتی ہے۔ یہ ماڈل فیچر اسپیس میں ایک ہائپرپلین ہوتا ہے، جو کلاسیفیکیشن کے معاملے میں ایک حد کے طور پر کام کرتا ہے، اور ریگریشن کے معاملے میں بہترین فٹ لائن کے طور پر کام کرتا ہے۔ SVM میں، عام طور پر ایک کرنل فنکشن استعمال کیا جاتا ہے تاکہ ڈیٹا سیٹ کو زیادہ ڈائمینشنز والے اسپیس میں تبدیل کیا جا سکے، تاکہ وہ آسانی سے الگ ہو سکیں۔ [SVMs پر مزید معلومات کے لیے یہاں کلک کریں](https://en.wikipedia.org/wiki/Support-vector_machine)۔
- **سپورٹ ویکٹر ریگریسر (SVR):** SVM کی ایک قسم، جو بہترین فٹ لائن (جو SVM کے معاملے میں ایک ہائپرپلین ہے) تلاش کرتی ہے جس پر زیادہ سے زیادہ ڈیٹا پوائنٹس ہوں۔

### SVR کیوں؟ [^1]

پچھلے سبق میں آپ نے ARIMA کے بارے میں سیکھا، جو وقت سیریز ڈیٹا کی پیش گوئی کے لیے ایک بہت کامیاب شماریاتی لکیری طریقہ ہے۔ تاہم، بہت سے معاملات میں، وقت سیریز ڈیٹا میں *غیر لکیریت* ہوتی ہے، جسے لکیری ماڈلز کے ذریعے نقشہ نہیں بنایا جا سکتا۔ ایسے معاملات میں، ریگریشن کے کاموں کے لیے ڈیٹا میں غیر لکیریت کو مدنظر رکھنے کی SVM کی صلاحیت SVR کو وقت سیریز کی پیش گوئی میں کامیاب بناتی ہے۔

## مشق - SVR ماڈل بنائیں

ڈیٹا کی تیاری کے ابتدائی چند مراحل پچھلے سبق [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) کی طرح ہیں۔

اس سبق میں [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) فولڈر کھولیں اور [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) فائل تلاش کریں۔[^2]

1. نوٹ بک چلائیں اور ضروری لائبریریاں درآمد کریں: [^2]

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

2. `/data/energy.csv` فائل سے ڈیٹا کو ایک Pandas ڈیٹا فریم میں لوڈ کریں اور دیکھیں: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. جنوری 2012 سے دسمبر 2014 تک دستیاب توانائی کے تمام ڈیٹا کو پلاٹ کریں: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![مکمل ڈیٹا](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   اب، آئیے اپنا SVR ماڈل بنائیں۔

### تربیتی اور جانچ کے ڈیٹا سیٹ بنائیں

اب آپ کا ڈیٹا لوڈ ہو چکا ہے، لہذا آپ اسے تربیت اور جانچ کے سیٹ میں تقسیم کر سکتے ہیں۔ پھر آپ ڈیٹا کو دوبارہ شکل دیں گے تاکہ وقت کے مراحل پر مبنی ڈیٹا سیٹ بنایا جا سکے، جو SVR کے لیے ضروری ہوگا۔ آپ اپنے ماڈل کو تربیتی سیٹ پر تربیت دیں گے۔ ماڈل کی تربیت مکمل ہونے کے بعد، آپ اس کی درستگی کو تربیتی سیٹ، جانچ کے سیٹ اور پھر مکمل ڈیٹا سیٹ پر جانچیں گے تاکہ مجموعی کارکردگی دیکھی جا سکے۔ آپ کو یہ یقینی بنانا ہوگا کہ جانچ کا سیٹ تربیتی سیٹ کے بعد کے وقت کی مدت کو کور کرتا ہے تاکہ یہ یقینی بنایا جا سکے کہ ماڈل مستقبل کے وقت کی مدت سے معلومات حاصل نہ کرے [^2] (جسے *اوورفٹنگ* کہا جاتا ہے)۔

1. یکم ستمبر سے 31 اکتوبر 2014 تک دو ماہ کی مدت کو تربیتی سیٹ کے لیے مختص کریں۔ جانچ کا سیٹ یکم نومبر سے 31 دسمبر 2014 تک دو ماہ کی مدت کو شامل کرے گا: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. فرق کو بصری بنائیں: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![تربیتی اور جانچ کا ڈیٹا](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### تربیت کے لیے ڈیٹا تیار کریں

اب، آپ کو تربیت کے لیے ڈیٹا تیار کرنے کی ضرورت ہے، جس میں ڈیٹا کو فلٹر کرنا اور اسکیل کرنا شامل ہے۔ اپنے ڈیٹا سیٹ کو فلٹر کریں تاکہ صرف مطلوبہ وقت کی مدت اور کالم شامل ہوں، اور اسکیلنگ کریں تاکہ ڈیٹا کو 0,1 کے وقفے میں پروجیکٹ کیا جا سکے۔

1. اصل ڈیٹا سیٹ کو فلٹر کریں تاکہ صرف مذکورہ وقت کی مدت فی سیٹ اور صرف مطلوبہ کالم 'لوڈ' اور تاریخ شامل ہوں: [^2]

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
   
2. تربیتی ڈیٹا کو (0, 1) کی حد میں اسکیل کریں: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. اب، آپ جانچ کے ڈیٹا کو اسکیل کریں: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### وقت کے مراحل کے ساتھ ڈیٹا بنائیں [^1]

SVR کے لیے، آپ ان پٹ ڈیٹا کو `[batch, timesteps]` کی شکل میں تبدیل کرتے ہیں۔ لہذا، آپ موجودہ `train_data` اور `test_data` کو دوبارہ شکل دیتے ہیں تاکہ ایک نیا ڈائمینشن ہو جو وقت کے مراحل کی نمائندگی کرے۔

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

اس مثال کے لیے، ہم `timesteps = 5` لیتے ہیں۔ لہذا، ماڈل کے ان پٹ پہلے 4 وقت کے مراحل کے ڈیٹا ہوں گے، اور آؤٹ پٹ 5ویں وقت کے مرحلے کا ڈیٹا ہوگا۔

```python
timesteps=5
```

تربیتی ڈیٹا کو 2D ٹینسر میں تبدیل کرنا:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

جانچ کے ڈیٹا کو 2D ٹینسر میں تبدیل کرنا:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

تربیتی اور جانچ کے ڈیٹا سے ان پٹ اور آؤٹ پٹ کا انتخاب:

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

### SVR نافذ کریں [^1]

اب، SVR کو نافذ کرنے کا وقت ہے۔ اس نفاذ کے بارے میں مزید پڑھنے کے لیے، آپ [اس دستاویز](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) کا حوالہ دے سکتے ہیں۔ ہمارے نفاذ کے لیے، ہم ان مراحل پر عمل کرتے ہیں:

  1. ماڈل کو `SVR()` کال کرکے اور ماڈل کے ہائپرپیرامیٹرز: کرنل، گاما، سی اور ایپسیلون پاس کرکے ڈیفائن کریں
  2. تربیتی ڈیٹا کے لیے ماڈل کو تیار کریں `fit()` فنکشن کال کرکے
  3. پیش گوئی کرنے کے لیے `predict()` فنکشن کال کریں

اب ہم ایک SVR ماڈل بناتے ہیں۔ یہاں ہم [RBF کرنل](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) استعمال کرتے ہیں، اور ہائپرپیرامیٹرز گاما، C اور ایپسیلون کو بالترتیب 0.5, 10 اور 0.05 پر سیٹ کرتے ہیں۔

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### تربیتی ڈیٹا پر ماڈل فٹ کریں [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### ماڈل کی پیش گوئی کریں [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

آپ نے اپنا SVR بنایا ہے! اب ہمیں اس کا جائزہ لینا ہوگا۔

### اپنے ماڈل کا جائزہ لیں [^1]

جائزے کے لیے، پہلے ہم ڈیٹا کو اپنی اصل اسکیل پر واپس اسکیل کریں گے۔ پھر، کارکردگی کو جانچنے کے لیے، ہم اصل اور پیش گوئی شدہ وقت سیریز پلاٹ کو پلاٹ کریں گے، اور MAPE نتیجہ بھی پرنٹ کریں گے۔

پیش گوئی شدہ اور اصل آؤٹ پٹ کو اسکیل کریں:

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

#### تربیتی اور جانچ کے ڈیٹا پر ماڈل کی کارکردگی چیک کریں [^1]

ہم ڈیٹا سیٹ سے ٹائم اسٹیمپس نکالتے ہیں تاکہ اپنے پلاٹ کے x-axis میں دکھا سکیں۔ نوٹ کریں کہ ہم پہلے ```timesteps-1``` اقدار کو پہلے آؤٹ پٹ کے لیے ان پٹ کے طور پر استعمال کر رہے ہیں، لہذا آؤٹ پٹ کے لیے ٹائم اسٹیمپس اس کے بعد شروع ہوں گے۔

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

تربیتی ڈیٹا کے لیے پیش گوئیوں کو پلاٹ کریں:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![تربیتی ڈیٹا کی پیش گوئی](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

تربیتی ڈیٹا کے لیے MAPE پرنٹ کریں

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

جانچ کے ڈیٹا کے لیے پیش گوئیوں کو پلاٹ کریں

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![جانچ کے ڈیٹا کی پیش گوئی](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

جانچ کے ڈیٹا کے لیے MAPE پرنٹ کریں

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 آپ کے جانچ کے ڈیٹا سیٹ پر بہت اچھا نتیجہ ہے!

### مکمل ڈیٹا سیٹ پر ماڈل کی کارکردگی چیک کریں [^1]

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

![مکمل ڈیٹا کی پیش گوئی](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 بہت اچھے پلاٹس، جو ایک ماڈل کی اچھی درستگی کو ظاہر کرتے ہیں۔ شاباش!

---

## 🚀چیلنج

- ماڈل بناتے وقت ہائپرپیرامیٹرز (گاما، C، ایپسیلون) کو ایڈجسٹ کرنے کی کوشش کریں اور ڈیٹا پر جائزہ لیں تاکہ دیکھیں کہ کون سا ہائپرپیرامیٹرز کا سیٹ جانچ کے ڈیٹا پر بہترین نتائج دیتا ہے۔ ان ہائپرپیرامیٹرز کے بارے میں مزید جاننے کے لیے، آپ [یہ دستاویز](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) دیکھ سکتے ہیں۔
- ماڈل کے لیے مختلف کرنل فنکشنز استعمال کرنے کی کوشش کریں اور ان کی کارکردگی کا تجزیہ کریں۔ ایک مددگار دستاویز [یہاں](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) مل سکتی ہے۔
- ماڈل کے لیے `timesteps` کے مختلف اقدار استعمال کرنے کی کوشش کریں تاکہ پیش گوئی کے لیے پیچھے دیکھ سکیں۔

## [پوسٹ لیکچر کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ

یہ سبق وقت سیریز کی پیش گوئی کے لیے SVR کے اطلاق کو متعارف کرانے کے لیے تھا۔ SVR کے بارے میں مزید پڑھنے کے لیے، آپ [اس بلاگ](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) کا حوالہ دے سکتے ہیں۔ یہ [scikit-learn پر دستاویز](https://scikit-learn.org/stable/modules/svm.html) SVMs کے بارے میں زیادہ جامع وضاحت فراہم کرتی ہے، [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) اور دیگر نفاذ کی تفصیلات جیسے مختلف [کرنل فنکشنز](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) جو استعمال کیے جا سکتے ہیں، اور ان کے پیرامیٹرز۔

## اسائنمنٹ

[ایک نیا SVR ماڈل](assignment.md)

## کریڈٹس

[^1]: اس سیکشن میں متن، کوڈ اور آؤٹ پٹ [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) کی طرف سے فراہم کیا گیا تھا۔
[^2]: اس سیکشن میں متن، کوڈ اور آؤٹ پٹ [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) سے لیا گیا تھا۔

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔