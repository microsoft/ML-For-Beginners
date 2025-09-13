<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T06:55:27+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "pa"
}
-->
# ਸਮੇਂ ਦੀ ਲੜੀ ਦੀ ਭਵਿੱਖਵਾਣੀ Support Vector Regressor ਨਾਲ

ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ARIMA ਮਾਡਲ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਸਮੇਂ ਦੀ ਲੜੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰਨ ਦਾ ਤਰੀਕਾ ਸਿੱਖਿਆ। ਹੁਣ ਤੁਸੀਂ Support Vector Regressor ਮਾਡਲ ਦੇਖੋਗੇ, ਜੋ ਕਿ ਇੱਕ ਰਿਗ੍ਰੈਸ਼ਨ ਮਾਡਲ ਹੈ ਜੋ ਲਗਾਤਾਰ ਡਾਟਾ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰਨ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ।

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/) 

## ਜਾਣ ਪਛਾਣ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਰਿਗ੍ਰੈਸ਼ਨ ਲਈ [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) ਨਾਲ ਮਾਡਲ ਬਣਾਉਣ ਦਾ ਇੱਕ ਵਿਸ਼ੇਸ਼ ਤਰੀਕਾ ਖੋਜੋਗੇ, ਜਿਸਨੂੰ **SVR: Support Vector Regressor** ਕਿਹਾ ਜਾਂਦਾ ਹੈ।

### ਸਮੇਂ ਦੀ ਲੜੀ ਦੇ ਸੰਦਰਭ ਵਿੱਚ SVR [^1]

ਸਮੇਂ ਦੀ ਲੜੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਵਿੱਚ SVR ਦੀ ਮਹੱਤਤਾ ਨੂੰ ਸਮਝਣ ਤੋਂ ਪਹਿਲਾਂ, ਕੁਝ ਮਹੱਤਵਪੂਰਨ ਧਾਰਨਾਵਾਂ ਹਨ ਜੋ ਤੁਹਾਨੂੰ ਜਾਣਨ ਦੀ ਲੋੜ ਹੈ:

- **ਰਿਗ੍ਰੈਸ਼ਨ:** ਇੱਕ ਸੁਪਰਵਾਈਜ਼ਡ ਲਰਨਿੰਗ ਤਕਨੀਕ ਜੋ ਦਿੱਤੇ ਗਏ ਇਨਪੁਟਸ ਤੋਂ ਲਗਾਤਾਰ ਮੁੱਲਾਂ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰਨ ਲਈ ਵਰਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਵਿਚਾਰਧਾਰਾ ਇਹ ਹੈ ਕਿ ਫੀਚਰ ਸਪੇਸ ਵਿੱਚ ਇੱਕ ਵਕਰ (ਜਾਂ ਲਾਈਨ) ਫਿੱਟ ਕੀਤੀ ਜਾਵੇ ਜਿਸ ਵਿੱਚ ਜ਼ਿਆਦਾਤਰ ਡਾਟਾ ਪੌਇੰਟਸ ਹੋਣ। [ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਇੱਥੇ ਕਲਿਕ ਕਰੋ](https://en.wikipedia.org/wiki/Regression_analysis)।
- **ਸਪੋਰਟ ਵੈਕਟਰ ਮਸ਼ੀਨ (SVM):** ਇੱਕ ਸੁਪਰਵਾਈਜ਼ਡ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਜੋ ਕਲਾਸੀਫਿਕੇਸ਼ਨ, ਰਿਗ੍ਰੈਸ਼ਨ ਅਤੇ ਆਉਟਲਾਇਰ ਡਿਟੈਕਸ਼ਨ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ। ਮਾਡਲ ਫੀਚਰ ਸਪੇਸ ਵਿੱਚ ਇੱਕ ਹਾਈਪਰਪਲੇਨ ਹੈ, ਜੋ ਕਿ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਦੇ ਮਾਮਲੇ ਵਿੱਚ ਇੱਕ ਬਾਊਂਡਰੀ ਵਜੋਂ ਕੰਮ ਕਰਦਾ ਹੈ, ਅਤੇ ਰਿਗ੍ਰੈਸ਼ਨ ਦੇ ਮਾਮਲੇ ਵਿੱਚ ਬੈਸਟ-ਫਿਟ ਲਾਈਨ ਵਜੋਂ ਕੰਮ ਕਰਦਾ ਹੈ। SVM ਵਿੱਚ, ਇੱਕ Kernel ਫੰਕਸ਼ਨ ਆਮ ਤੌਰ 'ਤੇ ਡਾਟਾਸੈਟ ਨੂੰ ਵਧੇਰੇ ਡਾਇਮੈਂਸ਼ਨ ਵਾਲੇ ਸਪੇਸ ਵਿੱਚ ਤਬਦੀਲ ਕਰਨ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ, ਤਾਂ ਜੋ ਉਹ ਆਸਾਨੀ ਨਾਲ ਵੱਖਰੇ ਹੋ ਸਕਣ। [SVMs ਬਾਰੇ ਹੋਰ ਜਾਣਕਾਰੀ ਲਈ ਇੱਥੇ ਕਲਿਕ ਕਰੋ](https://en.wikipedia.org/wiki/Support-vector_machine)।
- **ਸਪੋਰਟ ਵੈਕਟਰ ਰਿਗ੍ਰੈਸਰ (SVR):** SVM ਦੀ ਇੱਕ ਕਿਸਮ, ਜੋ ਬੈਸਟ-ਫਿਟ ਲਾਈਨ (ਜੋ SVM ਦੇ ਮਾਮਲੇ ਵਿੱਚ ਇੱਕ ਹਾਈਪਰਪਲੇਨ ਹੈ) ਲੱਭਣ ਲਈ ਵਰਤੀ ਜਾਂਦੀ ਹੈ ਜਿਸ ਵਿੱਚ ਜ਼ਿਆਦਾਤਰ ਡਾਟਾ ਪੌਇੰਟਸ ਹੁੰਦੇ ਹਨ।

### ਕਿਉਂ SVR? [^1]

ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਤੁਸੀਂ ARIMA ਬਾਰੇ ਸਿੱਖਿਆ, ਜੋ ਕਿ ਸਮੇਂ ਦੀ ਲੜੀ ਦੇ ਡਾਟਾ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰਨ ਲਈ ਇੱਕ ਬਹੁਤ ਹੀ ਸਫਲ ਸਾਂਖਿਕ ਰੇਖੀ ਮੈਥਡ ਹੈ। ਹਾਲਾਂਕਿ, ਕਈ ਮਾਮਲਿਆਂ ਵਿੱਚ, ਸਮੇਂ ਦੀ ਲੜੀ ਦੇ ਡਾਟਾ ਵਿੱਚ *ਗੈਰ-ਰੇਖੀਤਾ* ਹੁੰਦੀ ਹੈ, ਜਿਸਨੂੰ ਰੇਖੀ ਮਾਡਲਾਂ ਦੁਆਰਾ ਮੈਪ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਦਾ। ਇਸ ਤਰ੍ਹਾਂ ਦੇ ਮਾਮਲਿਆਂ ਵਿੱਚ, ਰਿਗ੍ਰੈਸ਼ਨ ਟਾਸਕ ਲਈ ਡਾਟਾ ਵਿੱਚ ਗੈਰ-ਰੇਖੀਤਾ ਨੂੰ ਧਿਆਨ ਵਿੱਚ ਰੱਖਣ ਦੀ SVM ਦੀ ਸਮਰੱਥਾ SVR ਨੂੰ ਸਮੇਂ ਦੀ ਲੜੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਵਿੱਚ ਸਫਲ ਬਣਾਉਂਦੀ ਹੈ।

## ਅਭਿਆਸ - SVR ਮਾਡਲ ਬਣਾਉਣਾ

ਡਾਟਾ ਤਿਆਰ ਕਰਨ ਦੇ ਪਹਿਲੇ ਕੁਝ ਕਦਮ ਪਿਛਲੇ ਪਾਠ [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) ਦੇ ਸਮਾਨ ਹਨ।

ਇਸ ਪਾਠ ਵਿੱਚ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) ਫੋਲਡਰ ਖੋਲ੍ਹੋ ਅਤੇ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) ਫਾਈਲ ਲੱਭੋ।[^2]

1. ਨੋਟਬੁੱਕ ਚਲਾਓ ਅਤੇ ਜ਼ਰੂਰੀ ਲਾਇਬ੍ਰੇਰੀਜ਼ ਇੰਪੋਰਟ ਕਰੋ: [^2]

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

2. `/data/energy.csv` ਫਾਈਲ ਤੋਂ ਡਾਟਾ Pandas ਡਾਟਾਫ੍ਰੇਮ ਵਿੱਚ ਲੋਡ ਕਰੋ ਅਤੇ ਡਾਟਾ ਵੇਖੋ: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. ਜਨਵਰੀ 2012 ਤੋਂ ਦਸੰਬਰ 2014 ਤੱਕ ਉਪਲਬਧ ਸਾਰੀ energy ਡਾਟਾ ਪਲਾਟ ਕਰੋ: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![ਪੂਰਾ ਡਾਟਾ](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   ਹੁਣ, ਆਓ ਆਪਣਾ SVR ਮਾਡਲ ਬਣਾਈਏ।

### ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਡਾਟਾਸੈਟ ਬਣਾਉਣਾ

ਹੁਣ ਤੁਹਾਡਾ ਡਾਟਾ ਲੋਡ ਹੋ ਗਿਆ ਹੈ, ਇਸ ਨੂੰ ਟ੍ਰੇਨ ਅਤੇ ਟੈਸਟ ਸੈਟ ਵਿੱਚ ਵੱਖਰਾ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ। ਫਿਰ ਤੁਸੀਂ ਡਾਟਾ ਨੂੰ ਰੀਸ਼ੇਪ ਕਰਕੇ ਇੱਕ time-step ਅਧਾਰਿਤ ਡਾਟਾਸੈਟ ਬਣਾਉਗੇ, ਜੋ SVR ਲਈ ਲੋੜੀਂਦਾ ਹੋਵੇਗਾ। ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਸੈਟ 'ਤੇ ਟ੍ਰੇਨ ਕਰੋਗੇ। ਮਾਡਲ ਟ੍ਰੇਨਿੰਗ ਪੂਰੀ ਹੋਣ ਤੋਂ ਬਾਅਦ, ਤੁਸੀਂ ਇਸਦੀ ਸਹੀਤਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਸੈਟ, ਟੈਸਟਿੰਗ ਸੈਟ ਅਤੇ ਫਿਰ ਪੂਰੇ ਡਾਟਾਸੈਟ 'ਤੇ ਮਾਪੋਗੇ ਤਾਂ ਜੋ ਕੁੱਲ ਪ੍ਰਦਰਸ਼ਨ ਦੇਖਿਆ ਜਾ ਸਕੇ। ਤੁਹਾਨੂੰ ਇਹ ਯਕੀਨੀ ਬਣਾਉਣਾ ਹੋਵੇਗਾ ਕਿ ਟੈਸਟ ਸੈਟ ਟ੍ਰੇਨਿੰਗ ਸੈਟ ਤੋਂ ਬਾਅਦ ਦੇ ਸਮੇਂ ਦੀ ਮਿਆਦ ਨੂੰ ਕਵਰ ਕਰਦਾ ਹੈ ਤਾਂ ਜੋ ਮਾਡਲ ਭਵਿੱਖ ਦੇ ਸਮੇਂ ਦੀ ਮਿਆਦ ਤੋਂ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤ ਨਾ ਕਰੇ [^2] (ਇਸ ਸਥਿਤੀ ਨੂੰ *Overfitting* ਕਿਹਾ ਜਾਂਦਾ ਹੈ)।

1. 1 ਸਤੰਬਰ ਤੋਂ 31 ਅਕਤੂਬਰ 2014 ਤੱਕ ਦੀ ਦੋ ਮਹੀਨੇ ਦੀ ਮਿਆਦ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਸੈਟ ਲਈ ਅਲਾਟ ਕਰੋ। ਟੈਸਟ ਸੈਟ ਵਿੱਚ 1 ਨਵੰਬਰ ਤੋਂ 31 ਦਸੰਬਰ 2014 ਤੱਕ ਦੀ ਦੋ ਮਹੀਨੇ ਦੀ ਮਿਆਦ ਸ਼ਾਮਲ ਹੋਵੇਗੀ: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. ਅੰਤਰਾਂ ਨੂੰ ਵਿਜੁਅਲਾਈਜ਼ ਕਰੋ: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਡਾਟਾ](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### ਟ੍ਰੇਨਿੰਗ ਲਈ ਡਾਟਾ ਤਿਆਰ ਕਰੋ

ਹੁਣ, ਤੁਹਾਨੂੰ ਫਿਲਟਰੀਂਗ ਅਤੇ ਸਕੇਲਿੰਗ ਕਰਕੇ ਆਪਣੇ ਡਾਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਲਈ ਤਿਆਰ ਕਰਨ ਦੀ ਲੋੜ ਹੈ। ਆਪਣੇ ਡਾਟਾਸੈਟ ਨੂੰ ਫਿਲਟਰ ਕਰੋ ਤਾਂ ਜੋ ਸਿਰਫ਼ ਲੋੜੀਂਦੇ ਸਮੇਂ ਦੀ ਮਿਆਦ ਅਤੇ ਕਾਲਮ ਸ਼ਾਮਲ ਕੀਤੇ ਜਾ ਸਕਣ, ਅਤੇ ਡਾਟਾ ਨੂੰ 0,1 ਦੇ ਇੰਟਰਵਾਲ ਵਿੱਚ ਪ੍ਰੋਜੈਕਟ ਕਰਨ ਲਈ ਸਕੇਲਿੰਗ ਕਰੋ।

1. ਮੂਲ ਡਾਟਾਸੈਟ ਨੂੰ ਸਿਰਫ਼ ਉਪਰੋਕਤ ਸਮੇਂ ਦੀ ਮਿਆਦ ਅਤੇ ਸਿਰਫ਼ ਲੋੜੀਂਦਾ ਕਾਲਮ 'load' ਅਤੇ ਤਾਰੀਖ ਸ਼ਾਮਲ ਕਰਨ ਲਈ ਫਿਲਟਰ ਕਰੋ: [^2]

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
   
2. ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਨੂੰ (0, 1) ਦੀ ਰੇਂਜ ਵਿੱਚ ਸਕੇਲ ਕਰੋ: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. ਹੁਣ, ਟੈਸਟਿੰਗ ਡਾਟਾ ਨੂੰ ਸਕੇਲ ਕਰੋ: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### ਟਾਈਮ-ਸਟੈਪਸ ਨਾਲ ਡਾਟਾ ਬਣਾਉਣਾ [^1]

SVR ਲਈ, ਤੁਸੀਂ ਇਨਪੁਟ ਡਾਟਾ ਨੂੰ `[batch, timesteps]` ਦੇ ਰੂਪ ਵਿੱਚ ਤਬਦੀਲ ਕਰਦੇ ਹੋ। ਇਸ ਲਈ, ਤੁਸੀਂ ਮੌਜੂਦਾ `train_data` ਅਤੇ `test_data` ਨੂੰ ਰੀਸ਼ੇਪ ਕਰਦੇ ਹੋ ਤਾਂ ਕਿ ਇੱਕ ਨਵਾਂ ਡਾਇਮੈਂਸ਼ਨ ਹੋਵੇ ਜੋ ਟਾਈਮ-ਸਟੈਪਸ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ।

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

ਇਸ ਉਦਾਹਰਨ ਲਈ, ਅਸੀਂ `timesteps = 5` ਲੈਂਦੇ ਹਾਂ। ਇਸ ਲਈ, ਮਾਡਲ ਲਈ ਇਨਪੁਟ ਪਹਿਲੇ 4 ਟਾਈਮ-ਸਟੈਪਸ ਲਈ ਡਾਟਾ ਹੈ, ਅਤੇ ਆਉਟਪੁਟ 5ਵੇਂ ਟਾਈਮ-ਸਟੈਪ ਲਈ ਡਾਟਾ ਹੋਵੇਗਾ।

```python
timesteps=5
```

Nested list comprehension ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਨੂੰ 2D ਟੈਂਸਰ ਵਿੱਚ ਤਬਦੀਲ ਕਰਨਾ:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

ਟੈਸਟਿੰਗ ਡਾਟਾ ਨੂੰ 2D ਟੈਂਸਰ ਵਿੱਚ ਤਬਦੀਲ ਕਰਨਾ:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਡਾਟਾ ਤੋਂ ਇਨਪੁਟਸ ਅਤੇ ਆਉਟਪੁਟਸ ਦੀ ਚੋਣ:

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

### SVR ਲਾਗੂ ਕਰੋ [^1]

ਹੁਣ, SVR ਲਾਗੂ ਕਰਨ ਦਾ ਸਮਾਂ ਹੈ। ਇਸ ਲਾਗੂ ਕਰਨ ਬਾਰੇ ਹੋਰ ਪੜ੍ਹਨ ਲਈ, ਤੁਸੀਂ [ਇਹ ਡੌਕੂਮੈਂਟੇਸ਼ਨ](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) ਨੂੰ ਰਿਫਰ ਕਰ ਸਕਦੇ ਹੋ। ਸਾਡੇ ਲਾਗੂ ਕਰਨ ਲਈ, ਅਸੀਂ ਇਹ ਕਦਮ ਅਨੁਸਰਣ ਕਰਦੇ ਹਾਂ:

  1. ਮਾਡਲ ਨੂੰ `SVR()` ਕਾਲ ਕਰਕੇ ਅਤੇ ਮਾਡਲ ਹਾਈਪਰਪੈਰਾਮੀਟਰਜ਼: kernel, gamma, c ਅਤੇ epsilon ਪਾਸ ਕਰਕੇ ਡਿਫਾਈਨ ਕਰੋ।
  2. `fit()` ਫੰਕਸ਼ਨ ਕਾਲ ਕਰਕੇ ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਲਈ ਮਾਡਲ ਤਿਆਰ ਕਰੋ।
  3. `predict()` ਫੰਕਸ਼ਨ ਕਾਲ ਕਰਕੇ ਭਵਿੱਖਵਾਣੀ ਕਰੋ।

ਹੁਣ ਅਸੀਂ SVR ਮਾਡਲ ਬਣਾਉਂਦੇ ਹਾਂ। ਇੱਥੇ ਅਸੀਂ [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹਾਂ, ਅਤੇ gamma, C ਅਤੇ epsilon ਨੂੰ 0.5, 10 ਅਤੇ 0.05 ਵਜੋਂ ਸੈਟ ਕਰਦੇ ਹਾਂ।

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ 'ਤੇ ਮਾਡਲ ਫਿਟ ਕਰੋ [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### ਮਾਡਲ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰੋ [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

ਤੁਹਾਡਾ SVR ਬਣ ਗਿਆ ਹੈ! ਹੁਣ ਅਸੀਂ ਇਸਦੀ ਮੁਲਾਂਕਣ ਕਰਦੇ ਹਾਂ।

### ਆਪਣੇ ਮਾਡਲ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ [^1]

ਮੁਲਾਂਕਣ ਲਈ, ਪਹਿਲਾਂ ਅਸੀਂ ਡਾਟਾ ਨੂੰ ਆਪਣੇ ਮੂਲ ਸਕੇਲ ਵਿੱਚ ਵਾਪਸ ਸਕੇਲ ਕਰਦੇ ਹਾਂ। ਫਿਰ, ਪ੍ਰਦਰਸ਼ਨ ਦੀ ਜਾਂਚ ਕਰਨ ਲਈ, ਅਸੀਂ ਮੂਲ ਅਤੇ ਭਵਿੱਖਵਾਣੀ ਸਮੇਂ ਦੀ ਲੜੀ ਦਾ ਪਲਾਟ ਬਣਾਉਂਦੇ ਹਾਂ, ਅਤੇ MAPE ਨਤੀਜਾ ਪ੍ਰਿੰਟ ਕਰਦੇ ਹਾਂ।

ਭਵਿੱਖਵਾਣੀ ਕੀਤੀ ਅਤੇ ਮੂਲ ਆਉਟਪੁਟ ਨੂੰ ਸਕੇਲ ਕਰੋ:

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

#### ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਡਾਟਾ 'ਤੇ ਮਾਡਲ ਪ੍ਰਦਰਸ਼ਨ ਦੀ ਜਾਂਚ ਕਰੋ [^1]

ਅਸੀਂ ਡਾਟਾਸੈਟ ਤੋਂ ਟਾਈਮਸਟੈਂਪਸ ਨੂੰ ਕੱਢਦੇ ਹਾਂ ਤਾਂ ਜੋ ਆਪਣੇ ਪਲਾਟ ਦੇ x-axis ਵਿੱਚ ਦਿਖਾ ਸਕੀਏ। ਧਿਆਨ ਦਿਓ ਕਿ ਅਸੀਂ ਪਹਿਲੇ ```timesteps-1``` ਮੁੱਲਾਂ ਨੂੰ ਪਹਿਲੇ ਆਉਟਪੁਟ ਲਈ ਇਨਪੁਟ ਵਜੋਂ ਵਰਤ ਰਹੇ ਹਾਂ, ਇਸ ਲਈ ਆਉਟਪੁਟ ਲਈ ਟਾਈਮਸਟੈਂਪਸ ਇਸ ਤੋਂ ਬਾਅਦ ਸ਼ੁਰੂ ਹੋਣਗੇ।

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਲਈ ਭਵਿੱਖਵਾਣੀ ਪਲਾਟ ਕਰੋ:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਭਵਿੱਖਵਾਣੀ](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

ਟ੍ਰੇਨਿੰਗ ਡਾਟਾ ਲਈ MAPE ਪ੍ਰਿੰਟ ਕਰੋ

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

ਟੈਸਟਿੰਗ ਡਾਟਾ ਲਈ ਭਵਿੱਖਵਾਣੀ ਪਲਾਟ ਕਰੋ

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![ਟੈਸਟਿੰਗ ਡਾਟਾ ਭਵਿੱਖਵਾਣੀ](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

ਟੈਸਟਿੰਗ ਡਾਟਾ ਲਈ MAPE ਪ੍ਰਿੰਟ ਕਰੋ

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 ਤੁਹਾਡੇ ਕੋਲ ਟੈਸਟਿੰਗ ਡਾਟਾਸੈਟ 'ਤੇ ਬਹੁਤ ਵਧੀਆ ਨਤੀਜਾ ਹੈ!

### ਪੂਰੇ ਡਾਟਾਸੈਟ 'ਤੇ ਮਾਡਲ ਪ੍ਰਦਰਸ਼ਨ ਦੀ ਜਾਂਚ ਕਰੋ [^1]

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

![ਪੂਰੇ ਡਾਟਾ ਦੀ ਭਵਿੱਖਵਾਣੀ](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 ਬਹੁਤ ਵਧੀਆ ਪਲਾਟ, ਜੋ ਇੱਕ ਮਾਡਲ ਨੂੰ ਚੰਗੀ ਸਹੀਤਾ ਨਾਲ ਦਰਸਾਉਂਦੇ ਹਨ। ਸ਼ਾਬਾਸ਼!

---

## 🚀ਚੁਣੌਤੀ

- ਮਾਡਲ ਬਣਾਉਣ ਦੌਰਾਨ ਹਾਈਪਰਪੈਰਾਮੀਟਰਜ਼ (gamma, C, epsilon) ਨੂੰ ਬਦਲਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਅਤੇ ਡਾਟਾ 'ਤੇ ਮੁਲਾਂਕਣ ਕਰੋ ਤਾਂ ਜੋ ਪਤਾ ਲੱਗੇ ਕਿ ਕਿਹੜੇ ਹਾਈਪਰਪੈਰਾਮੀਟਰਜ਼ ਦਾ ਸੈੱਟ ਟੈਸਟਿੰਗ ਡਾਟਾ 'ਤੇ ਸਭ ਤੋਂ ਵਧੀਆ ਨਤੀਜੇ ਦਿੰਦਾ ਹੈ। ਇਨ੍ਹਾਂ ਹਾਈਪਰਪੈਰਾਮੀਟਰਜ਼ ਬਾਰੇ ਹੋਰ ਜਾਣਨ ਲਈ, ਤੁਸੀਂ [ਇਹ ਡੌਕੂਮੈਂਟ](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) ਨੂੰ ਰਿਫਰ ਕਰ ਸਕਦੇ ਹੋ।
- ਮਾਡਲ ਲਈ ਵੱਖ-ਵੱਖ kernel functions ਦੀ ਵਰਤੋਂ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਅਤੇ ਡਾਟਾਸੈਟ 'ਤੇ ਉਨ੍ਹਾਂ ਦੇ ਪ੍ਰਦਰਸ਼ਨ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ। ਇੱਕ ਮਦਦਗਾਰ ਡੌਕੂਮੈਂਟ [ਇੱਥੇ](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) ਮਿਲ ਸਕਦਾ ਹੈ।
- ਮਾਡਲ ਨੂੰ ਭਵਿੱਖਵਾਣੀ ਕਰਨ ਲਈ ਵੱਖ-ਵੱਖ `timesteps` ਮੁੱਲਾਂ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।

## [ਪੋਸਟ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ-ਅਧਿਐਨ

ਇਹ ਪਾਠ ਸਮੇਂ ਦੀ ਲੜੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਲਈ SVR ਦੀ ਅਰਜ਼ੀ ਨੂੰ ਪੇਸ਼ ਕਰਨ ਲਈ ਸੀ। SVR ਬਾਰੇ ਹੋਰ ਪੜ੍ਹਨ ਲਈ, ਤੁਸੀਂ [ਇਹ ਬਲੌਗ](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) ਨੂੰ ਰਿਫਰ ਕਰ ਸਕਦੇ ਹੋ। ਇਹ [scikit-learn ਡੌਕੂਮੈਂਟੇਸ਼ਨ](https://scikit-learn.org/stable/modules/svm.html) SVMs ਦੇ ਬਾਰੇ ਵਿੱਚ ਇੱਕ ਵਧੇਰੇ ਵਿਸਤ੍ਰਿਤ ਵਿਆਖਿਆ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) ਅਤੇ ਹੋਰ ਲਾਗੂ ਕਰਨ ਦੇ ਵੇਰਵੇ ਜਿਵੇਂ ਕਿ ਵੱਖ-ਵੱਖ [kernel functions](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) ਜੋ ਵਰਤੇ ਜਾ ਸਕਦੇ ਹਨ, ਅਤੇ ਉਨ੍ਹਾਂ ਦੇ ਪੈਰਾਮੀਟਰਜ਼।



---

**ਅਸਵੀਕਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਨਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।