<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-04T21:01:18+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "bn"
}
-->
# ARIMA দিয়ে টাইম সিরিজ পূর্বাভাস

পূর্ববর্তী পাঠে, আপনি টাইম সিরিজ পূর্বাভাস সম্পর্কে কিছু শিখেছেন এবং একটি ডেটাসেট লোড করেছেন যা একটি নির্দিষ্ট সময়কালে বৈদ্যুতিক লোডের ওঠানামা দেখায়।

[![ARIMA পরিচিতি](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA মডেলের পরিচিতি")

> 🎥 উপরের ছবিতে ক্লিক করুন একটি ভিডিওর জন্য: ARIMA মডেলের সংক্ষিপ্ত পরিচিতি। উদাহরণটি R-এ করা হয়েছে, তবে ধারণাগুলি সার্বজনীন।

## [পূর্ব-পাঠ কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পরিচিতি

এই পাঠে, আপনি একটি নির্দিষ্ট পদ্ধতি আবিষ্কার করবেন [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) ব্যবহার করে মডেল তৈরি করার। ARIMA মডেল বিশেষভাবে উপযুক্ত [non-stationarity](https://wikipedia.org/wiki/Stationary_process) প্রদর্শনকারী ডেটার জন্য।

## সাধারণ ধারণা

ARIMA নিয়ে কাজ করতে হলে কিছু ধারণা সম্পর্কে জানা প্রয়োজন:

- 🎓 **Stationarity**। পরিসংখ্যানগত প্রেক্ষাপটে, স্টেশনারিটি এমন ডেটাকে বোঝায় যার বিতরণ সময়ে স্থানান্তরিত হলেও পরিবর্তিত হয় না। Non-stationary ডেটা, তখন প্রবণতার কারণে ওঠানামা দেখায় যা বিশ্লেষণের জন্য রূপান্তরিত করতে হয়। উদাহরণস্বরূপ, মৌসুমী প্রবণতা ডেটায় ওঠানামা আনতে পারে এবং 'seasonal-differencing' প্রক্রিয়ার মাধ্যমে এটি দূর করা যায়।

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**। পরিসংখ্যানগত প্রেক্ষাপটে, ডেটাকে স্টেশনারি করতে রূপান্তর করার প্রক্রিয়াকে ডিফারেন্সিং বলা হয়। এটি ডেটার non-constant প্রবণতা সরিয়ে দেয়। "ডিফারেন্সিং টাইম সিরিজের স্তরের পরিবর্তনগুলি সরিয়ে দেয়, প্রবণতা এবং মৌসুমীতা দূর করে এবং টাইম সিরিজের গড়কে স্থিতিশীল করে।" [Shixiong et al-এর পেপার](https://arxiv.org/abs/1904.07632)

## টাইম সিরিজের প্রেক্ষাপটে ARIMA

ARIMA-এর অংশগুলো বিশ্লেষণ করে দেখা যাক এটি কীভাবে টাইম সিরিজ মডেল করতে সাহায্য করে এবং পূর্বাভাস দিতে সহায়তা করে।

- **AR - AutoRegressive**। অটো-রিগ্রেসিভ মডেল, নাম থেকেই বোঝা যায়, ডেটার পূর্ববর্তী মানগুলো বিশ্লেষণ করে এবং সেগুলোর উপর ভিত্তি করে অনুমান করে। এই পূর্ববর্তী মানগুলোকে 'lags' বলা হয়। উদাহরণস্বরূপ, মাসিক পেন্সিল বিক্রির ডেটা। প্রতিটি মাসের বিক্রির মোট পরিমাণ ডেটাসেটে একটি 'evolving variable' হিসেবে বিবেচিত হবে। এই মডেলটি তৈরি হয় যেখানে "evolving variable of interest তার নিজস্ব lagged (অর্থাৎ পূর্ববর্তী) মানগুলোর উপর রিগ্রেস করা হয়।" [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Integrated**। 'ARMA' মডেলের সাথে তুলনায়, ARIMA-তে 'I' এর অর্থ *[integrated](https://wikipedia.org/wiki/Order_of_integration)*। ডেটা 'integrated' হয় যখন ডিফারেন্সিং ধাপগুলো প্রয়োগ করা হয় non-stationarity দূর করতে।

- **MA - Moving Average**। এই মডেলের [moving-average](https://wikipedia.org/wiki/Moving-average_model) অংশটি বর্তমান এবং পূর্ববর্তী ল্যাগ মানগুলো পর্যবেক্ষণ করে আউটপুট ভেরিয়েবল নির্ধারণ করে।

মূল কথা: ARIMA ব্যবহার করা হয় টাইম সিরিজ ডেটার বিশেষ ফর্মের সাথে মডেলকে যতটা সম্ভব ঘনিষ্ঠভাবে ফিট করার জন্য।

## অনুশীলন - একটি ARIMA মডেল তৈরি করুন

এই পাঠের [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) ফোল্ডার খুলুন এবং [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) ফাইলটি খুঁজুন।

1. নোটবুক চালান `statsmodels` Python লাইব্রেরি লোড করতে; ARIMA মডেলের জন্য এটি প্রয়োজন।

1. প্রয়োজনীয় লাইব্রেরি লোড করুন।

1. এখন, ডেটা প্লট করার জন্য আরও কিছু দরকারী লাইব্রেরি লোড করুন:

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

1. `/data/energy.csv` ফাইল থেকে ডেটা একটি Pandas ডেটাফ্রেমে লোড করুন এবং এটি দেখুন:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. জানুয়ারি 2012 থেকে ডিসেম্বর 2014 পর্যন্ত সমস্ত উপলব্ধ এনার্জি ডেটা প্লট করুন। এখানে কোনো চমক থাকবে না কারণ আমরা এই ডেটা আগের পাঠে দেখেছি:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    এখন, একটি মডেল তৈরি করা যাক!

### প্রশিক্ষণ এবং পরীক্ষার ডেটাসেট তৈরি করুন

এখন আপনার ডেটা লোড হয়েছে, তাই আপনি এটি প্রশিক্ষণ এবং পরীক্ষার সেটে ভাগ করতে পারেন। আপনি আপনার মডেলটি প্রশিক্ষণ সেটে প্রশিক্ষণ দেবেন। যথারীতি, মডেলটি প্রশিক্ষণ শেষ করার পরে, আপনি পরীক্ষার সেট ব্যবহার করে এর সঠিকতা মূল্যায়ন করবেন। নিশ্চিত করুন যে পরীক্ষার সেটটি প্রশিক্ষণ সেটের চেয়ে পরবর্তী সময়কালের ডেটা কভার করে যাতে মডেল ভবিষ্যতের সময়কাল থেকে তথ্য না পায়।

1. সেপ্টেম্বর 1 থেকে অক্টোবর 31, 2014 পর্যন্ত দুই মাসের সময়কাল প্রশিক্ষণ সেটে বরাদ্দ করুন। পরীক্ষার সেটে নভেম্বর 1 থেকে ডিসেম্বর 31, 2014 পর্যন্ত দুই মাসের সময়কাল অন্তর্ভুক্ত থাকবে:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    যেহেতু এই ডেটা দৈনিক এনার্জি ব্যবহারের প্রতিফলন করে, তাই এখানে একটি শক্তিশালী মৌসুমী প্যাটার্ন রয়েছে, তবে সাম্প্রতিক দিনের ব্যবহারের সাথে এটি সবচেয়ে বেশি সাদৃশ্যপূর্ণ।

1. পার্থক্যগুলো চিত্রিত করুন:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![প্রশিক্ষণ এবং পরীক্ষার ডেটা](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    সুতরাং, ডেটা প্রশিক্ষণের জন্য একটি তুলনামূলকভাবে ছোট সময় উইন্ডো ব্যবহার করা যথেষ্ট হওয়া উচিত।

    > নোট: যেহেতু আমরা ARIMA মডেল ফিট করার জন্য যে ফাংশনটি ব্যবহার করি তা ফিটিংয়ের সময় ইন-স্যাম্পল ভ্যালিডেশন ব্যবহার করে, আমরা ভ্যালিডেশন ডেটা বাদ দেব।

### প্রশিক্ষণের জন্য ডেটা প্রস্তুত করুন

এখন, আপনাকে ডেটা ফিল্টারিং এবং স্কেলিংয়ের মাধ্যমে প্রশিক্ষণের জন্য প্রস্তুত করতে হবে। আপনার ডেটাসেটটি শুধুমাত্র প্রয়োজনীয় সময়কাল এবং কলামগুলো অন্তর্ভুক্ত করতে ফিল্টার করুন এবং ডেটা 0,1 এর মধ্যে প্রজেক্ট করার জন্য স্কেলিং করুন।

1. মূল ডেটাসেটটি শুধুমাত্র উল্লেখিত সময়কাল এবং সেট অনুযায়ী প্রয়োজনীয় কলাম 'load' এবং তারিখ অন্তর্ভুক্ত করতে ফিল্টার করুন:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    ডেটার আকার দেখতে পারেন:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. ডেটাকে (0, 1) পরিসরে স্কেল করুন।

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. মূল বনাম স্কেল করা ডেটা চিত্রিত করুন:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![মূল](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > মূল ডেটা

    ![স্কেল করা](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > স্কেল করা ডেটা

1. এখন আপনি স্কেল করা ডেটা ক্যালিব্রেট করেছেন, আপনি পরীক্ষার ডেটা স্কেল করতে পারেন:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA বাস্তবায়ন করুন

এখন ARIMA বাস্তবায়নের সময়! আপনি আগে ইনস্টল করা `statsmodels` লাইব্রেরি ব্যবহার করবেন।

এখন আপনাকে কয়েকটি ধাপ অনুসরণ করতে হবে:

   1. `SARIMAX()` কল করে এবং মডেল প্যারামিটারগুলো: p, d, এবং q প্যারামিটার, এবং P, D, এবং Q প্যারামিটার পাস করে মডেলটি সংজ্ঞায়িত করুন।
   2. `fit()` ফাংশন কল করে প্রশিক্ষণের ডেটার জন্য মডেল প্রস্তুত করুন।
   3. `forecast()` ফাংশন কল করে এবং পূর্বাভাসের ধাপের সংখ্যা (হরাইজন) নির্দিষ্ট করে পূর্বাভাস তৈরি করুন।

> 🎓 এই প্যারামিটারগুলো কী জন্য? একটি ARIMA মডেলে 3টি প্যারামিটার ব্যবহার করা হয় যা টাইম সিরিজের প্রধান দিকগুলো মডেল করতে সাহায্য করে: মৌসুমীতা, প্রবণতা, এবং শব্দ। এই প্যারামিটারগুলো হলো:

`p`: মডেলের অটো-রিগ্রেসিভ অংশের সাথে সম্পর্কিত প্যারামিটার, যা *পূর্ববর্তী* মানগুলো অন্তর্ভুক্ত করে।
`d`: মডেলের ইন্টিগ্রেটেড অংশের সাথে সম্পর্কিত প্যারামিটার, যা টাইম সিরিজে *ডিফারেন্সিং* (🎓 ডিফারেন্সিং মনে আছে 👆?) প্রয়োগের পরিমাণকে প্রভাবিত করে।
`q`: মডেলের মুভিং-অ্যাভারেজ অংশের সাথে সম্পর্কিত প্যারামিটার।

> নোট: যদি আপনার ডেটায় মৌসুমী দিক থাকে - যেমন এই ডেটায় আছে - , আমরা একটি মৌসুমী ARIMA মডেল (SARIMA) ব্যবহার করি। সেই ক্ষেত্রে আপনাকে আরেকটি প্যারামিটার সেট ব্যবহার করতে হবে: `P`, `D`, এবং `Q` যা `p`, `d`, এবং `q` এর মতো একই সম্পর্ক বর্ণনা করে, তবে মডেলের মৌসুমী উপাদানগুলোর সাথে সম্পর্কিত।

1. আপনার পছন্দের হরাইজন মান সেট করুন। চলুন 3 ঘণ্টা চেষ্টা করি:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    ARIMA মডেলের প্যারামিটারগুলোর সেরা মান নির্বাচন করা চ্যালেঞ্জিং হতে পারে কারণ এটি কিছুটা বিষয়ভিত্তিক এবং সময়সাপেক্ষ। আপনি [`pyramid` লাইব্রেরি](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) থেকে `auto_arima()` ফাংশন ব্যবহার করার কথা বিবেচনা করতে পারেন।

1. আপাতত কিছু ম্যানুয়াল নির্বাচন চেষ্টা করুন একটি ভালো মডেল খুঁজে পেতে।

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    একটি ফলাফলের টেবিল প্রিন্ট করা হয়।

আপনি আপনার প্রথম মডেল তৈরি করেছেন! এখন আমাদের এটি মূল্যায়নের একটি উপায় খুঁজে বের করতে হবে।

### আপনার মডেল মূল্যায়ন করুন

আপনার মডেল মূল্যায়ন করতে, আপনি তথাকথিত `walk forward` ভ্যালিডেশন করতে পারেন। বাস্তবে, টাইম সিরিজ মডেলগুলো প্রতিবার নতুন ডেটা উপলব্ধ হলে পুনরায় প্রশিক্ষণ দেওয়া হয়। এটি মডেলকে প্রতিটি সময় ধাপে সেরা পূর্বাভাস দিতে সক্ষম করে।

টাইম সিরিজের শুরু থেকে এই কৌশল ব্যবহার করে, প্রশিক্ষণ ডেটাসেটে মডেলটি প্রশিক্ষণ দিন। তারপর পরবর্তী সময় ধাপে একটি পূর্বাভাস তৈরি করুন। পূর্বাভাসটি পরিচিত মানের সাথে মূল্যায়ন করা হয়। প্রশিক্ষণ সেটটি পরিচিত মান অন্তর্ভুক্ত করতে প্রসারিত হয় এবং প্রক্রিয়াটি পুনরাবৃত্তি হয়।

> নোট: প্রশিক্ষণ সেট উইন্ডো স্থির রাখুন আরও দক্ষ প্রশিক্ষণের জন্য যাতে আপনি প্রতিবার প্রশিক্ষণ সেটে একটি নতুন পর্যবেক্ষণ যোগ করেন, আপনি সেটের শুরু থেকে পর্যবেক্ষণটি সরিয়ে ফেলেন।

এই প্রক্রিয়াটি মডেলটি বাস্তবে কীভাবে কাজ করবে তার আরও শক্তিশালী অনুমান প্রদান করে। তবে, এটি এতগুলো মডেল তৈরি করার গণনা খরচে আসে। যদি ডেটা ছোট হয় বা মডেলটি সহজ হয় তবে এটি গ্রহণযোগ্য, তবে বড় স্কেলে এটি একটি সমস্যা হতে পারে।

Walk-forward validation টাইম সিরিজ মডেল মূল্যায়নের সোনার মান এবং এটি আপনার নিজস্ব প্রকল্পগুলোর জন্য সুপারিশ করা হয়।

1. প্রথমে, প্রতিটি HORIZON ধাপের জন্য একটি পরীক্ষার ডেটা পয়েন্ট তৈরি করুন।

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

    ডেটা তার হরাইজন পয়েন্ট অনুযায়ী অনুভূমিকভাবে স্থানান্তরিত হয়।

1. এই স্লাইডিং উইন্ডো পদ্ধতি ব্যবহার করে একটি লুপে পরীক্ষার ডেটার দৈর্ঘ্যের আকারে পূর্বাভাস তৈরি করুন:

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

    আপনি প্রশিক্ষণটি দেখতে পারেন:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. পূর্বাভাসগুলো প্রকৃত লোডের সাথে তুলনা করুন:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    আউটপুট
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    ঘণ্টাভিত্তিক ডেটার পূর্বাভাস দেখুন, প্রকৃত লোডের সাথে তুলনা করুন। এটি কতটা সঠিক?

### মডেলের সঠিকতা পরীক্ষা করুন

আপনার মডেলের সঠিকতা পরীক্ষা করুন সমস্ত পূর্বাভাসের উপর এর mean absolute percentage error (MAPE) পরীক্ষা করে।
> **🧮 গণিত দেখুন**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) ব্যবহার করা হয় পূর্বাভাসের সঠিকতা একটি অনুপাত হিসেবে দেখানোর জন্য, যা উপরের সূত্র দ্বারা সংজ্ঞায়িত। প্রকৃত এবং পূর্বাভাসিত মানের পার্থক্য প্রকৃত মান দ্বারা ভাগ করা হয়।  
"এই গণনার মধ্যে পার্থক্যের পরম মান প্রতিটি পূর্বাভাসিত সময় বিন্দুর জন্য যোগ করা হয় এবং n সংখ্যক ফিট করা বিন্দু দ্বারা ভাগ করা হয়।" [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. কোডে সমীকরণ প্রকাশ করুন:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. এক ধাপের MAPE গণনা করুন:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    এক ধাপের পূর্বাভাস MAPE:  0.5570581332313952 %

1. বহু ধাপের পূর্বাভাস MAPE প্রিন্ট করুন:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    একটি সুন্দর কম সংখ্যা সর্বোত্তম: মনে রাখুন, যদি পূর্বাভাসের MAPE 10 হয়, তবে এটি 10% পর্যন্ত ভুল হতে পারে।

1. তবে সবসময়ই, এই ধরনের সঠিকতা পরিমাপ দৃশ্যত দেখা সহজ, তাই চলুন এটি প্লট করি:

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

    ![একটি টাইম সিরিজ মডেল](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 একটি খুব সুন্দর প্লট, যা একটি ভালো সঠিকতার মডেল দেখাচ্ছে। খুব ভালো কাজ!

---

## 🚀চ্যালেঞ্জ

টাইম সিরিজ মডেলের সঠিকতা পরীক্ষা করার বিভিন্ন পদ্ধতি নিয়ে গবেষণা করুন। এই পাঠে আমরা MAPE নিয়ে আলোচনা করেছি, তবে আপনি কি অন্য পদ্ধতি ব্যবহার করতে পারেন? সেগুলো নিয়ে গবেষণা করুন এবং মন্তব্য যোগ করুন। একটি সহায়ক ডকুমেন্ট [এখানে](https://otexts.com/fpp2/accuracy.html) পাওয়া যাবে।

## [পোস্ট-লেকচার কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা ও স্ব-অধ্যয়ন

এই পাঠে ARIMA দিয়ে টাইম সিরিজ পূর্বাভাসের শুধুমাত্র মৌলিক বিষয়গুলো আলোচনা করা হয়েছে। [এই রিপোজিটরি](https://microsoft.github.io/forecasting/) এবং এর বিভিন্ন মডেল টাইপ নিয়ে গবেষণা করে টাইম সিরিজ মডেল তৈরির অন্যান্য পদ্ধতি শিখে আপনার জ্ঞান আরও গভীর করুন।

## অ্যাসাইনমেন্ট

[একটি নতুন ARIMA মডেল](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।