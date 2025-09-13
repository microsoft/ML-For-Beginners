<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T06:53:15+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "pa"
}
-->
# ARIMA ਨਾਲ ਟਾਈਮ ਸੀਰੀਜ਼ ਫੋਰਕਾਸਟਿੰਗ

ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਟਾਈਮ ਸੀਰੀਜ਼ ਫੋਰਕਾਸਟਿੰਗ ਬਾਰੇ ਕੁਝ ਸਿੱਖਿਆ ਅਤੇ ਇੱਕ ਡਾਟਾਸੈੱਟ ਲੋਡ ਕੀਤਾ ਜੋ ਇੱਕ ਸਮੇਂ ਦੀ ਮਿਆਦ ਦੌਰਾਨ ਬਿਜਲੀ ਦੇ ਲੋਡ ਵਿੱਚ ਉਤਾਰ-ਚੜ੍ਹਾਵ ਦਿਖਾਉਂਦਾ ਹੈ।

[![ARIMA ਦਾ ਪਰਿਚਯ](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA ਮਾਡਲਾਂ ਦਾ ਪਰਿਚਯ")

> 🎥 ਉਪਰੋਕਤ ਚਿੱਤਰ 'ਤੇ ਕਲਿਕ ਕਰੋ ਇੱਕ ਵੀਡੀਓ ਲਈ: ARIMA ਮਾਡਲਾਂ ਦਾ ਸੰਖੇਪ ਪਰਿਚਯ। ਉਦਾਹਰਣ R ਵਿੱਚ ਕੀਤਾ ਗਿਆ ਹੈ, ਪਰ ਸੰਕਲਪ ਸਾਰਵਭੌਮ ਹਨ।

## [ਪਿਛਲੇ ਪਾਠ ਦਾ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਪਰਿਚਯ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) ਨਾਲ ਮਾਡਲ ਬਣਾਉਣ ਦਾ ਇੱਕ ਵਿਸ਼ੇਸ਼ ਤਰੀਕਾ ਖੋਜੋਗੇ। ARIMA ਮਾਡਲ ਖਾਸ ਤੌਰ 'ਤੇ [non-stationarity](https://wikipedia.org/wiki/Stationary_process) ਦਿਖਾਉਣ ਵਾਲੇ ਡਾਟਾ ਲਈ ਸੁਟੇਬਲ ਹਨ।

## ਆਮ ਸੰਕਲਪ

ARIMA ਨਾਲ ਕੰਮ ਕਰਨ ਲਈ, ਕੁਝ ਸੰਕਲਪਾਂ ਨੂੰ ਸਮਝਣਾ ਜ਼ਰੂਰੀ ਹੈ:

- 🎓 **Stationarity**. ਸਾਂਖਿਆਕੀ ਸੰਦਰਭ ਵਿੱਚ, stationarity ਉਸ ਡਾਟਾ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ ਜਿਸ ਦਾ ਵੰਡਨ ਸਮੇਂ ਵਿੱਚ ਬਦਲਦਾ ਨਹੀਂ। Non-stationary ਡਾਟਾ ਵਿੱਚ ਰੁਝਾਨਾਂ ਦੇ ਕਾਰਨ ਉਤਾਰ-ਚੜ੍ਹਾਵ ਹੁੰਦੇ ਹਨ, ਜਿਨ੍ਹਾਂ ਨੂੰ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਨ ਲਈ ਤਬਦੀਲ ਕੀਤਾ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਉਦਾਹਰਣ ਲਈ, ਮੌਸਮੀਤਾ ਡਾਟਾ ਵਿੱਚ ਉਤਾਰ-ਚੜ੍ਹਾਵ ਪੈਦਾ ਕਰ ਸਕਦੀ ਹੈ, ਜਿਸ ਨੂੰ 'seasonal-differencing' ਦੀ ਪ੍ਰਕਿਰਿਆ ਦੁਆਰਾ ਹਟਾਇਆ ਜਾ ਸਕਦਾ ਹੈ।

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. ਸਾਂਖਿਆਕੀ ਸੰਦਰਭ ਵਿੱਚ, differencing ਡਾਟਾ ਨੂੰ non-stationary ਤੋਂ stationarity ਵਿੱਚ ਤਬਦੀਲ ਕਰਨ ਦੀ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। "Differencing ਇੱਕ ਟਾਈਮ ਸੀਰੀਜ਼ ਦੇ ਪੱਧਰ ਵਿੱਚ ਬਦਲਾਅ ਨੂੰ ਹਟਾਉਂਦਾ ਹੈ, ਰੁਝਾਨ ਅਤੇ ਮੌਸਮੀਤਾ ਨੂੰ ਖਤਮ ਕਰਦਾ ਹੈ ਅਤੇ ਇਸ ਤਰ੍ਹਾਂ ਟਾਈਮ ਸੀਰੀਜ਼ ਦੇ ਮੀਨ ਨੂੰ ਸਥਿਰ ਕਰਦਾ ਹੈ।" [Shixiong et al ਦੁਆਰਾ ਪੇਪਰ](https://arxiv.org/abs/1904.07632)

## ਟਾਈਮ ਸੀਰੀਜ਼ ਦੇ ਸੰਦਰਭ ਵਿੱਚ ARIMA

ਆਓ ARIMA ਦੇ ਹਿੱਸਿਆਂ ਨੂੰ ਖੋਲ੍ਹ ਕੇ ਸਮਝੀਏ ਕਿ ਇਹ ਟਾਈਮ ਸੀਰੀਜ਼ ਨੂੰ ਮਾਡਲ ਕਰਨ ਵਿੱਚ ਕਿਵੇਂ ਮਦਦ ਕਰਦਾ ਹੈ ਅਤੇ ਇਸ ਦੇ ਖਿਲਾਫ ਅਨੁਮਾਨ ਲਗਾਉਣ ਵਿੱਚ ਸਹਾਇਕ ਹੈ।

- **AR - AutoRegressive ਲਈ**. Autoregressive ਮਾਡਲ, ਜਿਵੇਂ ਕਿ ਨਾਮ ਦਰਸਾਉਂਦਾ ਹੈ, ਪਿਛਲੇ ਸਮੇਂ ਵਿੱਚ ਡਾਟਾ ਦੇ ਪਿਛਲੇ ਮੁੱਲਾਂ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਦੇ ਹਨ ਅਤੇ ਉਨ੍ਹਾਂ ਬਾਰੇ ਧਾਰਨਾ ਬਣਾਉਂਦੇ ਹਨ। ਇਹ ਪਿਛਲੇ ਮੁੱਲਾਂ 'lags' ਕਹਿੰਦੇ ਹਨ। ਉਦਾਹਰਣ ਲਈ, ਪੈਂਸਲਾਂ ਦੀ ਮਾਸਿਕ ਵਿਕਰੀ ਦਿਖਾਉਣ ਵਾਲਾ ਡਾਟਾ। ਹਰ ਮਹੀਨੇ ਦੀ ਵਿਕਰੀ ਦੀ ਕੁੱਲ ਰਕਮ ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਇੱਕ 'evolving variable' ਮੰਨੀ ਜਾਵੇਗੀ। ਇਹ ਮਾਡਲ ਇਸ ਤਰ੍ਹਾਂ ਬਣਾਇਆ ਜਾਂਦਾ ਹੈ ਕਿ "ਵਿਕਾਸਸ਼ੀਲ ਚਰ ਨੂੰ ਆਪਣੇ ਪਿਛਲੇ (ਅਰਥਾਤ, ਪਿਛਲੇ) ਮੁੱਲਾਂ 'ਤੇ regress ਕੀਤਾ ਜਾਂਦਾ ਹੈ।" [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrated ਲਈ**. 'ARMA' ਮਾਡਲਾਂ ਦੇ ਸਮਾਨ, ARIMA ਵਿੱਚ 'I' ਇਸ ਦੇ *[integrated](https://wikipedia.org/wiki/Order_of_integration)* ਪੱਖ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ। ਡਾਟਾ ਨੂੰ 'integrated' ਕੀਤਾ ਜਾਂਦਾ ਹੈ ਜਦੋਂ differencing ਕਦਮ non-stationarity ਨੂੰ ਖਤਮ ਕਰਨ ਲਈ ਲਾਗੂ ਕੀਤੇ ਜਾਂਦੇ ਹਨ।

- **MA - Moving Average ਲਈ**. ਇਸ ਮਾਡਲ ਦੇ [moving-average](https://wikipedia.org/wiki/Moving-average_model) ਪੱਖ ਦਾ ਅਰਥ ਹੈ ਕਿ ਆਉਟਪੁੱਟ ਚਰ ਨੂੰ ਪਿਛਲੇ ਅਤੇ ਮੌਜੂਦਾ 'lags' ਦੇ ਮੁੱਲਾਂ ਨੂੰ ਦੇਖ ਕੇ ਨਿਰਧਾਰਤ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।

ਸਾਰ: ARIMA ਨੂੰ ਟਾਈਮ ਸੀਰੀਜ਼ ਡਾਟਾ ਦੇ ਵਿਸ਼ੇਸ਼ ਰੂਪ ਨੂੰ ਜਿੰਨਾ ਹੋ ਸਕੇ ਨੇੜੇ ਫਿੱਟ ਕਰਨ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ।

## ਅਭਿਆਸ - ARIMA ਮਾਡਲ ਬਣਾਓ

ਇਸ ਪਾਠ ਵਿੱਚ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) ਫੋਲਡਰ ਖੋਲ੍ਹੋ ਅਤੇ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) ਫਾਈਲ ਲੱਭੋ।

1. `statsmodels` Python ਲਾਇਬ੍ਰੇਰੀ ਨੂੰ ਲੋਡ ਕਰਨ ਲਈ notebook ਚਲਾਓ; ਤੁਹਾਨੂੰ ARIMA ਮਾਡਲਾਂ ਲਈ ਇਸ ਦੀ ਲੋੜ ਹੋਵੇਗੀ।

1. ਜ਼ਰੂਰੀ ਲਾਇਬ੍ਰੇਰੀਆਂ ਲੋਡ ਕਰੋ।

1. ਹੁਣ, ਡਾਟਾ ਪਲਾਟ ਕਰਨ ਲਈ ਕੁਝ ਹੋਰ ਲਾਇਬ੍ਰੇਰੀਆਂ ਲੋਡ ਕਰੋ:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. `/data/energy.csv` ਫਾਈਲ ਤੋਂ ਡਾਟਾ Pandas dataframe ਵਿੱਚ ਲੋਡ ਕਰੋ ਅਤੇ ਇਸ ਨੂੰ ਵੇਖੋ:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. ਜਨਵਰੀ 2012 ਤੋਂ ਦਸੰਬਰ 2014 ਤੱਕ ਸਾਰੇ ਉਪਲਬਧ energy ਡਾਟਾ ਪਲਾਟ ਕਰੋ। ਕੋਈ ਹੈਰਾਨੀ ਨਹੀਂ ਹੋਣੀ ਚਾਹੀਦੀ ਕਿਉਂਕਿ ਅਸੀਂ ਇਹ ਡਾਟਾ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਵੇਖਿਆ ਸੀ:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ਹੁਣ, ਆਓ ਇੱਕ ਮਾਡਲ ਬਣਾਈਏ!

### ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਡਾਟਾਸੈੱਟ ਬਣਾਓ

ਹੁਣ ਤੁਹਾਡਾ ਡਾਟਾ ਲੋਡ ਹੋ ਗਿਆ ਹੈ, ਇਸ ਨੂੰ train ਅਤੇ test sets ਵਿੱਚ ਵੰਡੋ। ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ train set 'ਤੇ ਟ੍ਰੇਨ ਕਰੋਗੇ। ਹਮੇਸ਼ਾ ਦੀ ਤਰ੍ਹਾਂ, ਮਾਡਲ ਦੇ ਟ੍ਰੇਨਿੰਗ ਪੂਰੀ ਹੋਣ ਤੋਂ ਬਾਅਦ, ਤੁਸੀਂ ਇਸ ਦੀ accuracy ਨੂੰ test set ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਮਾਪੋਗੇ। ਤੁਹਾਨੂੰ ਇਹ ਯਕੀਨੀ ਬਣਾਉਣਾ ਚਾਹੀਦਾ ਹੈ ਕਿ test set train set ਤੋਂ ਬਾਅਦ ਦੇ ਸਮੇਂ ਦੀ ਮਿਆਦ ਨੂੰ ਕਵਰ ਕਰਦਾ ਹੈ ਤਾਂ ਜੋ ਮਾਡਲ ਭਵਿੱਖ ਦੇ ਸਮੇਂ ਦੀ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤ ਨਾ ਕਰੇ।

1. 1 ਸਤੰਬਰ ਤੋਂ 31 ਅਕਤੂਬਰ 2014 ਤੱਕ ਦੇ ਦੋ ਮਹੀਨੇ train set ਲਈ ਵੰਡੋ। test set ਵਿੱਚ 1 ਨਵੰਬਰ ਤੋਂ 31 ਦਸੰਬਰ 2014 ਤੱਕ ਦੇ ਦੋ ਮਹੀਨੇ ਸ਼ਾਮਲ ਹੋਣਗੇ:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    ਕਿਉਂਕਿ ਇਹ ਡਾਟਾ energy ਦੀ ਦਿਨ-ਦਰ-ਦਿਨ ਖਪਤ ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ, ਇਸ ਵਿੱਚ ਇੱਕ ਮਜ਼ਬੂਤ ਮੌਸਮੀ ਪੈਟਰਨ ਹੈ, ਪਰ ਖਪਤ ਹਾਲ ਹੀ ਦੇ ਦਿਨਾਂ ਵਿੱਚ ਖਪਤ ਦੇ ਨਾਲ ਸਭ ਤੋਂ ਜ਼ਿਆਦਾ ਮਿਲਦੀ ਜੁਲਦੀ ਹੈ।

1. ਅੰਤਰਾਂ ਨੂੰ ਵਿਜ਼ੁਅਲਾਈਜ਼ ਕਰੋ:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![training and testing data](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    ਇਸ ਲਈ, ਡਾਟਾ ਨੂੰ train ਕਰਨ ਲਈ ਇੱਕ ਛੋਟੀ ਸਮੇਂ ਦੀ ਮਿਆਦ ਦੀ ਵਰਤੋਂ ਕਰਨਾ ਕਾਫ਼ੀ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ।

    > Note: ਕਿਉਂਕਿ ਅਸੀਂ ARIMA ਮਾਡਲ ਨੂੰ fit ਕਰਨ ਲਈ ਵਰਤਿਆ ਜਾਣ ਵਾਲੀ ਫੰਕਸ਼ਨ in-sample validation ਦੀ ਵਰਤੋਂ ਕਰਦਾ ਹੈ, ਅਸੀਂ validation ਡਾਟਾ ਨੂੰ ਛੱਡ ਦੇਵਾਂਗੇ।

### ਟ੍ਰੇਨਿੰਗ ਲਈ ਡਾਟਾ ਤਿਆਰ ਕਰੋ

ਹੁਣ, ਤੁਹਾਨੂੰ filtering ਅਤੇ scaling ਦੀ ਪ੍ਰਕਿਰਿਆ ਦੁਆਰਾ ਆਪਣੇ ਡਾਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਲਈ ਤਿਆਰ ਕਰਨ ਦੀ ਲੋੜ ਹੈ। ਆਪਣੇ ਡਾਟਾਸੈੱਟ ਨੂੰ ਸਿਰਫ਼ ਜ਼ਰੂਰੀ ਸਮੇਂ ਦੀ ਮਿਆਦ ਅਤੇ ਕਾਲਮਾਂ ਨੂੰ ਸ਼ਾਮਲ ਕਰਨ ਲਈ ਫਿਲਟਰ ਕਰੋ ਅਤੇ ਡਾਟਾ ਨੂੰ 0,1 ਦੇ ਅੰਤਰਾਲ ਵਿੱਚ ਪ੍ਰੋਜੈਕਟ ਕਰਨ ਲਈ scale ਕਰੋ।

1. ਮੂਲ ਡਾਟਾਸੈੱਟ ਨੂੰ ਸਿਰਫ਼ ਉਪਰੋਕਤ ਸਮੇਂ ਦੀ ਮਿਆਦ ਅਤੇ 'load' ਕਾਲਮ ਪਲਸ date ਨੂੰ ਸ਼ਾਮਲ ਕਰਨ ਲਈ ਫਿਲਟਰ ਕਰੋ:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    ਤੁਸੀਂ ਡਾਟਾ ਦਾ shape ਦੇਖ ਸਕਦੇ ਹੋ:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. ਡਾਟਾ ਨੂੰ (0, 1) ਦੀ ਰੇਂਜ ਵਿੱਚ scale ਕਰੋ।

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. ਮੂਲ ਅਤੇ scale ਕੀਤੇ ਡਾਟਾ ਨੂੰ ਵਿਜ਼ੁਅਲਾਈਜ਼ ਕਰੋ:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![original](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > ਮੂਲ ਡਾਟਾ

    ![scaled](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > scale ਕੀਤਾ ਡਾਟਾ

1. ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ scale ਕੀਤੇ ਡਾਟਾ ਨੂੰ ਕੈਲੀਬਰੇਟ ਕਰ ਲਿਆ ਹੈ, ਤੁਸੀਂ test ਡਾਟਾ ਨੂੰ scale ਕਰ ਸਕਦੇ ਹੋ:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA ਲਾਗੂ ਕਰੋ

ਹੁਣ ARIMA ਲਾਗੂ ਕਰਨ ਦਾ ਸਮਾਂ ਹੈ! ਤੁਸੀਂ ਹੁਣ `statsmodels` ਲਾਇਬ੍ਰੇਰੀ ਦੀ ਵਰਤੋਂ ਕਰੋਗੇ ਜੋ ਤੁਸੀਂ ਪਹਿਲਾਂ install ਕੀਤੀ ਸੀ।

ਹੁਣ ਤੁਹਾਨੂੰ ਕੁਝ ਕਦਮਾਂ ਦੀ ਪਾਲਣਾ ਕਰਨ ਦੀ ਲੋੜ ਹੈ:

   1. ਮਾਡਲ ਨੂੰ `SARIMAX()` ਕਾਲ ਕਰਕੇ ਅਤੇ ਮਾਡਲ ਪੈਰਾਮੀਟਰ: p, d, ਅਤੇ q ਪੈਰਾਮੀਟਰ, ਅਤੇ P, D, ਅਤੇ Q ਪੈਰਾਮੀਟਰ ਪਾਸ ਕਰਕੇ define ਕਰੋ।
   2. fit() ਫੰਕਸ਼ਨ ਕਾਲ ਕਰਕੇ ਮਾਡਲ ਨੂੰ train ਡਾਟਾ ਲਈ ਤਿਆਰ ਕਰੋ।
   3. `forecast()` ਫੰਕਸ਼ਨ ਕਾਲ ਕਰਕੇ ਅਤੇ forecast ਕਰਨ ਲਈ ਕਦਮਾਂ ਦੀ ਗਿਣਤੀ (horizon) ਨੂੰ specify ਕਰਕੇ ਅਨੁਮਾਨ ਲਗਾਓ।

> 🎓 ਇਹ ਸਾਰੇ ਪੈਰਾਮੀਟਰ ਕਿਸ ਲਈ ਹਨ? ARIMA ਮਾਡਲ ਵਿੱਚ 3 ਪੈਰਾਮੀਟਰ ਹਨ ਜੋ ਟਾਈਮ ਸੀਰੀਜ਼ ਦੇ ਮੁੱਖ ਪੱਖਾਂ: ਮੌਸਮੀਤਾ, ਰੁਝਾਨ, ਅਤੇ ਸ਼ੋਰ ਨੂੰ ਮਾਡਲ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰਦੇ ਹਨ। ਇਹ ਪੈਰਾਮੀਟਰ ਹਨ:

`p`: ਮਾਡਲ ਦੇ auto-regressive ਪੱਖ ਨਾਲ ਸੰਬੰਧਿਤ ਪੈਰਾਮੀਟਰ, ਜੋ *ਪਿਛਲੇ* ਮੁੱਲਾਂ ਨੂੰ ਸ਼ਾਮਲ ਕਰਦਾ ਹੈ।
`d`: ਮਾਡਲ ਦੇ integrated ਹਿੱਸੇ ਨਾਲ ਸੰਬੰਧਿਤ ਪੈਰਾਮੀਟਰ, ਜੋ *differencing* (🎓 differencing ਨੂੰ ਯਾਦ ਕਰੋ 👆?) ਦੀ ਮਾਤਰਾ ਨੂੰ ਪ੍ਰਭਾਵਿਤ ਕਰਦਾ ਹੈ।
`q`: ਮਾਡਲ ਦੇ moving-average ਹਿੱਸੇ ਨਾਲ ਸੰਬੰਧਿਤ ਪੈਰਾਮੀਟਰ।

> Note: ਜੇ ਤੁਹਾਡੇ ਡਾਟਾ ਵਿੱਚ ਮੌਸਮੀ ਪੱਖ ਹੈ - ਜੋ ਇਸ ਡਾਟਾ ਵਿੱਚ ਹੈ - , ਅਸੀਂ ਇੱਕ ਮੌਸਮੀ ARIMA ਮਾਡਲ (SARIMA) ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹਾਂ। ਇਸ ਮਾਮਲੇ ਵਿੱਚ ਤੁਹਾਨੂੰ ਹੋਰ ਪੈਰਾਮੀਟਰ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦੀ ਲੋੜ ਹੈ: `P`, `D`, ਅਤੇ `Q` ਜੋ `p`, `d`, ਅਤੇ `q` ਦੇ ਸਮਾਨ ਸੰਬੰਧਨ ਨੂੰ ਦਰਸਾਉਂਦੇ ਹਨ, ਪਰ ਮਾਡਲ ਦੇ ਮੌਸਮੀ ਹਿੱਸਿਆਂ ਨਾਲ ਸੰਬੰਧਿਤ ਹਨ।

1. ਪਹਿਲਾਂ ਆਪਣੀ ਪਸੰਦੀਦਾ horizon value ਸੈਟ ਕਰੋ। ਆਓ 3 ਘੰਟਿਆਂ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੀਏ:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA ਮਾਡਲ ਦੇ ਪੈਰਾਮੀਟਰਾਂ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਮੁੱਲਾਂ ਦੀ ਚੋਣ ਕਰਨਾ ਚੁਣੌਤੀਪੂਰਨ ਹੋ ਸਕਦਾ ਹੈ ਕਿਉਂਕਿ ਇਹ ਕੁਝ ਹੱਦ ਤੱਕ subjective ਅਤੇ ਸਮੇਂ ਲੈਣ ਵਾਲਾ ਹੈ। ਤੁਸੀਂ [`pyramid` ਲਾਇਬ੍ਰੇਰੀ](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) ਤੋਂ `auto_arima()` ਫੰਕਸ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰਨ ਬਾਰੇ ਸੋਚ ਸਕਦੇ ਹੋ।

1. ਹੁਣ ਕੁਝ manual ਚੋਣਾਂ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਤਾਂ ਜੋ ਇੱਕ ਵਧੀਆ ਮਾਡਲ ਲੱਭ ਸਕੋ।

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    ਨਤੀਜਿਆਂ ਦੀ ਇੱਕ ਟੇਬਲ ਪ੍ਰਿੰਟ ਕੀਤੀ ਜਾਂਦੀ ਹੈ।

ਤੁਸੀਂ ਆਪਣਾ ਪਹਿਲਾ ਮਾਡਲ ਬਣਾਇਆ ਹੈ! ਹੁਣ ਅਸੀਂ ਇਸਨੂੰ evaluate ਕਰਨ ਦਾ ਤਰੀਕਾ ਲੱਭਣਾ ਹੈ।

### ਆਪਣੇ ਮਾਡਲ ਨੂੰ evaluate ਕਰੋ

ਆਪਣੇ ਮਾਡਲ ਨੂੰ evaluate ਕਰਨ ਲਈ, ਤੁਸੀਂ `walk forward` validation ਕਰ ਸਕਦੇ ਹੋ। ਅਮਲ ਵਿੱਚ, ਟਾਈਮ ਸੀਰੀਜ਼ ਮਾਡਲ ਹਰ ਵਾਰ ਜਦੋਂ ਨਵਾਂ ਡਾਟਾ ਉਪਲਬਧ ਹੁੰਦਾ ਹੈ, ਦੁਬਾਰਾ train ਕੀਤਾ ਜਾਂਦਾ ਹੈ। ਇਹ ਮਾਡਲ ਨੂੰ ਹਰ ਸਮੇਂ ਦੇ ਕਦਮ 'ਤੇ ਸਭ ਤੋਂ ਵਧੀਆ forecast ਕਰਨ ਦੀ ਆਗਿਆ ਦਿੰਦਾ ਹੈ।

ਇਸ ਤਕਨੀਕ ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹੋਏ ਟਾਈਮ ਸੀਰੀਜ਼ ਦੇ ਸ਼ੁਰੂ 'ਤੇ train ਡਾਟਾ ਸੈਟ 'ਤੇ ਮਾਡਲ train ਕਰੋ। ਫਿਰ ਅਗਲੇ ਸਮੇਂ ਦੇ ਕਦਮ 'ਤੇ ਅਨੁਮਾਨ ਲਗਾਓ। ਅਨੁਮਾਨ ਨੂੰ ਜਾਣੇ ਮੁੱਲ ਦੇ ਖਿਲਾਫ evaluate ਕੀਤਾ ਜਾਂਦਾ ਹੈ। train ਸੈਟ ਨੂੰ ਜਾਣੇ ਮੁੱਲ ਨੂੰ ਸ਼ਾਮਲ ਕਰਨ ਲਈ ਵਧਾਇਆ ਜਾਂਦਾ ਹੈ ਅਤੇ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਦੁਹਰਾਇਆ ਜਾਂਦਾ ਹੈ।

> Note: ਤੁਹਾਨੂੰ train ਸੈਟ window ਨੂੰ fixed ਰੱਖਣਾ ਚਾਹੀਦਾ ਹੈ ਤਾਂ ਕਿ train ਕਰਨ ਦੀ ਕੁਸ਼ਲਤਾ ਵਧੇਰੇ ਹੋਵੇ। ਹਰ ਵਾਰ ਜਦੋਂ ਤੁਸੀਂ train ਸੈਟ ਵਿੱਚ ਇੱਕ ਨਵਾਂ observation ਸ਼ਾਮਲ ਕਰਦੇ ਹੋ, ਤੁਸੀਂ ਸੈਟ ਦੇ ਸ਼ੁਰੂ ਤੋਂ observation ਨੂੰ ਹਟਾ ਦਿੰਦੇ ਹੋ।

ਇਹ ਪ੍ਰਕਿਰਿਆ ਇਸ ਗੱਲ ਦਾ ਵਧੇਰੇ ਮਜ਼ਬੂਤ ਅੰਦਾਜ਼ਾ ਦਿੰਦੀ ਹੈ ਕਿ ਮਾਡਲ ਅਮਲ ਵਿੱਚ ਕਿਵੇਂ ਕੰਮ ਕਰੇਗਾ। ਹਾਲਾਂਕਿ, ਇਹ ਬਹੁਤ ਸਾਰੇ ਮਾਡਲ ਬਣਾਉਣ ਦੀ ਗਣਨਾ ਦੀ ਲਾਗਤ 'ਤੇ ਆਉਂਦਾ ਹੈ। ਇਹ ਛੋਟੇ ਡਾਟਾ ਜਾਂ ਸਧਾਰਨ ਮਾਡਲ ਦੇ ਮਾਮਲੇ ਵਿੱਚ ਸਵੀਕਾਰਯੋਗ ਹੈ, ਪਰ ਵੱਡੇ ਪੱਧਰ 'ਤੇ ਸਮੱਸਿਆ ਹੋ ਸਕਦੀ ਹੈ।

Walk-forward validation ਟਾਈਮ ਸੀਰੀਜ਼ ਮਾਡਲ evaluation ਦਾ gold standard ਹੈ ਅਤੇ ਤੁਹਾਡੇ ਆਪਣੇ ਪ੍ਰੋਜੈਕਟਾਂ ਲਈ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ।

1. ਪਹਿਲਾਂ, HORIZON ਕਦਮਾਂ ਲਈ ਹਰ test ਡਾਟਾ ਪੌਇੰਟ ਬਣਾਓ।

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    ਡਾਟਾ ਨੂੰ ਇਸ ਦੇ horizon point ਦੇ ਅਨੁਸਾਰ ਹੋਰਿਜ਼ਾਂਟਲੀ ਸ਼ਿਫਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।

1. test ਡਾਟਾ 'ਤੇ ਇਸ sliding window ਤਰੀਕੇ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਮਾਨ ਲਗਾਓ, test ਡਾਟਾ ਦੀ ਲੰਬਾਈ ਦੇ ਆਕਾਰ ਵਿੱਚ ਇੱਕ ਲੂਪ ਵਿੱਚ:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    ਤੁਸੀਂ train ਹੋਣ ਨੂੰ ਦੇਖ ਸਕਦੇ ਹੋ:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. ਅਸਲ load ਦੇ ਨਾਲ ਅਨੁਮਾਨਾਂ ਦੀ ਤੁਲਨਾ ਕਰੋ:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Output
    |     |            |
> **🧮 ਗਣਿਤ ਦਿਖਾਓ**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) ਦੀ ਵਰਤੋਂ ਅਨੁਮਾਨ ਸਹੀਤਾ ਨੂੰ ਦਰਸਾਉਣ ਲਈ ਕੀਤੀ ਜਾਂਦੀ ਹੈ, ਜੋ ਉਪਰੋਕਤ ਫਾਰਮੂਲੇ ਦੁਆਰਾ ਪਰਿਭਾਸ਼ਿਤ ਅਨੁਪਾਤ ਹੈ। ਅਸਲ ਅਤੇ ਅਨੁਮਾਨਿਤ ਦੇ ਵਿਚਕਾਰ ਅੰਤਰ ਨੂੰ ਅਸਲ ਨਾਲ ਵੰਡਿਆ ਜਾਂਦਾ ਹੈ।
>
> "ਇਸ ਗਣਨਾ ਵਿੱਚ ਮੂਲਮਾਨ ਹਰ ਅਨੁਮਾਨਿਤ ਸਮੇਂ ਦੇ ਬਿੰਦੂ ਲਈ ਜੋੜਿਆ ਜਾਂਦਾ ਹੈ ਅਤੇ ਫਿੱਟ ਕੀਤੇ ਗਏ ਬਿੰਦੂਆਂ ਦੀ ਗਿਣਤੀ n ਨਾਲ ਵੰਡਿਆ ਜਾਂਦਾ ਹੈ।" [ਵਿਕੀਪੀਡੀਆ](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. ਕੋਡ ਵਿੱਚ ਸਮੀਕਰਨ ਪ੍ਰਗਟ ਕਰੋ:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. ਇੱਕ ਕਦਮ ਦਾ MAPE ਗਣਨਾ ਕਰੋ:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    ਇੱਕ ਕਦਮ ਦੀ ਭਵਿੱਖਵਾਣੀ MAPE:  0.5570581332313952 %

1. ਬਹੁ-ਕਦਮ ਦੀ ਭਵਿੱਖਵਾਣੀ MAPE ਪ੍ਰਿੰਟ ਕਰੋ:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    ਇੱਕ ਚੰਗਾ ਘੱਟ ਸੰਖਿਆ ਸਭ ਤੋਂ ਵਧੀਆ ਹੈ: ਧਿਆਨ ਦਿਓ ਕਿ ਜੇਕਰ ਭਵਿੱਖਵਾਣੀ ਦਾ MAPE 10 ਹੈ, ਤਾਂ ਇਹ 10% ਤੱਕ ਗਲਤ ਹੈ।

1. ਪਰ ਹਮੇਸ਼ਾ ਦੀ ਤਰ੍ਹਾਂ, ਇਸ ਤਰ੍ਹਾਂ ਦੀ ਸਹੀਤਾ ਮਾਪਣ ਨੂੰ ਵਿਜੁਅਲ ਤੌਰ 'ਤੇ ਦੇਖਣਾ ਆਸਾਨ ਹੁੰਦਾ ਹੈ, ਤਾਂ ਆਓ ਇਸਨੂੰ ਪਲਾਟ ਕਰੀਏ:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![ਇੱਕ ਟਾਈਮ ਸੀਰੀਜ਼ ਮਾਡਲ](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 ਇੱਕ ਬਹੁਤ ਵਧੀਆ ਪਲਾਟ, ਜੋ ਇੱਕ ਮਾਡਲ ਨੂੰ ਚੰਗੀ ਸਹੀਤਾ ਨਾਲ ਦਿਖਾਉਂਦਾ ਹੈ। ਸ਼ਾਬਾਸ਼!

---

## 🚀ਚੁਣੌਤੀ

ਟਾਈਮ ਸੀਰੀਜ਼ ਮਾਡਲ ਦੀ ਸਹੀਤਾ ਦੀ ਜਾਂਚ ਕਰਨ ਦੇ ਤਰੀਕਿਆਂ ਵਿੱਚ ਖੋਜ ਕਰੋ। ਅਸੀਂ ਇਸ ਪਾਠ ਵਿੱਚ MAPE ਬਾਰੇ ਗੱਲ ਕਰਦੇ ਹਾਂ, ਪਰ ਕੀ ਤੁਸੀਂ ਹੋਰ ਤਰੀਕੇ ਵਰਤ ਸਕਦੇ ਹੋ? ਉਨ੍ਹਾਂ ਦੀ ਖੋਜ ਕਰੋ ਅਤੇ ਉਨ੍ਹਾਂ ਨੂੰ ਦਰਜ ਕਰੋ। ਇੱਕ ਮਦਦਗਾਰ ਦਸਤਾਵੇਜ਼ [ਇੱਥੇ](https://otexts.com/fpp2/accuracy.html) ਮਿਲ ਸਕਦਾ ਹੈ।

## [ਪਾਠ-ਪ੍ਰਸ਼ਨੋਤਰੀ](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

ਇਹ ਪਾਠ ARIMA ਨਾਲ ਟਾਈਮ ਸੀਰੀਜ਼ ਫੋਰਕਾਸਟਿੰਗ ਦੇ ਕੇਵਲ ਬੁਨਿਆਦੀ ਪੱਖਾਂ ਨੂੰ ਛੂਹਦਾ ਹੈ। ਆਪਣੇ ਗਿਆਨ ਨੂੰ ਗਹਿਰਾ ਕਰਨ ਲਈ ਕੁਝ ਸਮਾਂ ਲਓ ਅਤੇ [ਇਸ ਰਿਪੋਜ਼ਟਰੀ](https://microsoft.github.io/forecasting/) ਅਤੇ ਇਸਦੇ ਵੱਖ-ਵੱਖ ਮਾਡਲ ਪ੍ਰਕਾਰਾਂ ਵਿੱਚ ਖੋਜ ਕਰੋ ਤਾਂ ਜੋ ਹੋਰ ਤਰੀਕਿਆਂ ਨਾਲ ਟਾਈਮ ਸੀਰੀਜ਼ ਮਾਡਲ ਬਣਾਉਣ ਬਾਰੇ ਸਿੱਖਿਆ ਜਾ ਸਕੇ।

## ਅਸਾਈਨਮੈਂਟ

[ਇੱਕ ਨਵਾਂ ARIMA ਮਾਡਲ](assignment.md)

---

**ਅਸਵੀਕਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤ ਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।