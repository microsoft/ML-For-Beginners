<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T21:02:57+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "bn"
}
-->
# টাইম সিরিজ পূর্বাভাস: সাপোর্ট ভেক্টর রিগ্রেসর

পূর্ববর্তী পাঠে, আপনি ARIMA মডেল ব্যবহার করে টাইম সিরিজ পূর্বাভাস তৈরি করতে শিখেছেন। এখন আপনি সাপোর্ট ভেক্টর রিগ্রেসর মডেল সম্পর্কে জানবেন, যা একটি রিগ্রেসর মডেল এবং ধারাবাহিক ডেটা পূর্বাভাসের জন্য ব্যবহৃত হয়।

## [পূর্ব-পাঠ কুইজ](https://ff-quizzes.netlify.app/en/ml/) 

## পরিচিতি

এই পাঠে, আপনি [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) ব্যবহার করে রিগ্রেশন মডেল তৈরি করার একটি নির্দিষ্ট পদ্ধতি আবিষ্কার করবেন, যা **SVR: Support Vector Regressor** নামে পরিচিত। 

### টাইম সিরিজের প্রেক্ষাপটে SVR [^1]

টাইম সিরিজ পূর্বাভাসে SVR-এর গুরুত্ব বোঝার আগে, কিছু গুরুত্বপূর্ণ ধারণা সম্পর্কে জানা প্রয়োজন:

- **রিগ্রেশন:** এটি একটি সুপারভাইজড লার্নিং কৌশল যা প্রদত্ত ইনপুট সেট থেকে ধারাবাহিক মান পূর্বাভাস করে। এর মূল ধারণা হলো ফিচার স্পেসে একটি রেখা বা কার্ভ ফিট করা যা সর্বাধিক সংখ্যক ডেটা পয়েন্ট ধারণ করে। [আরও জানতে এখানে ক্লিক করুন](https://en.wikipedia.org/wiki/Regression_analysis)।
- **সাপোর্ট ভেক্টর মেশিন (SVM):** এটি একটি সুপারভাইজড মেশিন লার্নিং মডেল যা শ্রেণীবিভাজন, রিগ্রেশন এবং আউটলায়ার সনাক্তকরণের জন্য ব্যবহৃত হয়। মডেলটি ফিচার স্পেসে একটি হাইপারপ্লেন তৈরি করে, যা শ্রেণীবিভাজনের ক্ষেত্রে একটি সীমানা এবং রিগ্রেশনের ক্ষেত্রে সেরা ফিট লাইন হিসেবে কাজ করে। SVM-এ সাধারণত একটি কের্নেল ফাংশন ব্যবহার করা হয় যা ডেটাসেটকে উচ্চতর মাত্রার স্পেসে রূপান্তরিত করে, যাতে সেগুলো সহজে পৃথক করা যায়। [SVM সম্পর্কে আরও জানতে এখানে ক্লিক করুন](https://en.wikipedia.org/wiki/Support-vector_machine)।
- **সাপোর্ট ভেক্টর রিগ্রেসর (SVR):** এটি SVM-এর একটি ধরন, যা সেরা ফিট লাইন (SVM-এর ক্ষেত্রে এটি একটি হাইপারপ্লেন) খুঁজে বের করে যা সর্বাধিক সংখ্যক ডেটা পয়েন্ট ধারণ করে।

### কেন SVR? [^1]

পূর্ববর্তী পাঠে আপনি ARIMA সম্পর্কে শিখেছেন, যা টাইম সিরিজ ডেটা পূর্বাভাসের জন্য একটি অত্যন্ত সফল পরিসংখ্যানগত লিনিয়ার পদ্ধতি। তবে, অনেক ক্ষেত্রে টাইম সিরিজ ডেটায় *নন-লিনিয়ারিটি* থাকে, যা লিনিয়ার মডেল দ্বারা ম্যাপ করা যায় না। এই ধরনের ক্ষেত্রে, রিগ্রেশন টাস্কে ডেটার নন-লিনিয়ারিটি বিবেচনা করার SVM-এর ক্ষমতা SVR-কে টাইম সিরিজ পূর্বাভাসে সফল করে তোলে।

## অনুশীলন - SVR মডেল তৈরি করুন

ডেটা প্রস্তুতির প্রথম কয়েকটি ধাপ পূর্ববর্তী [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) পাঠের মতোই। 

এই পাঠের [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) ফোল্ডার খুলুন এবং [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) ফাইলটি খুঁজুন।[^2]

1. নোটবুক চালান এবং প্রয়োজনীয় লাইব্রেরি ইমপোর্ট করুন: [^2]

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

2. `/data/energy.csv` ফাইল থেকে ডেটা একটি Pandas ডেটাফ্রেমে লোড করুন এবং দেখুন: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. জানুয়ারি ২০১২ থেকে ডিসেম্বর ২০১৪ পর্যন্ত সমস্ত এনার্জি ডেটা প্লট করুন: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![সম্পূর্ণ ডেটা](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   এখন, আমাদের SVR মডেল তৈরি করা যাক।

### প্রশিক্ষণ এবং পরীক্ষার ডেটাসেট তৈরি করুন

এখন আপনার ডেটা লোড হয়েছে, তাই আপনি এটি প্রশিক্ষণ এবং পরীক্ষার সেটে ভাগ করতে পারেন। এরপর আপনি ডেটাকে রিশেপ করবেন যাতে টাইম-স্টেপ ভিত্তিক ডেটাসেট তৈরি হয়, যা SVR-এর জন্য প্রয়োজন। আপনি আপনার মডেলটি প্রশিক্ষণ সেটে প্রশিক্ষণ দেবেন। মডেলটি প্রশিক্ষণ শেষ করার পরে, আপনি এর সঠিকতা প্রশিক্ষণ সেট, পরীক্ষার সেট এবং সম্পূর্ণ ডেটাসেটে মূল্যায়ন করবেন যাতে সামগ্রিক কার্যকারিতা দেখা যায়। আপনাকে নিশ্চিত করতে হবে যে পরীক্ষার সেটটি প্রশিক্ষণ সেটের পরে সময়কাল কভার করে যাতে মডেল ভবিষ্যতের সময়কাল থেকে তথ্য না পায় [^2] (যা *ওভারফিটিং* নামে পরিচিত)।

1. ১ সেপ্টেম্বর থেকে ৩১ অক্টোবর, ২০১৪ পর্যন্ত দুই মাসের সময়কাল প্রশিক্ষণ সেটে বরাদ্দ করুন। পরীক্ষার সেটটি ১ নভেম্বর থেকে ৩১ ডিসেম্বর, ২০১৪ পর্যন্ত দুই মাসের সময়কাল অন্তর্ভুক্ত করবে: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. পার্থক্যগুলো ভিজ্যুয়ালাইজ করুন: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![প্রশিক্ষণ এবং পরীক্ষার ডেটা](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### প্রশিক্ষণের জন্য ডেটা প্রস্তুত করুন

এখন, আপনাকে ডেটা ফিল্টারিং এবং স্কেলিং করে প্রশিক্ষণের জন্য প্রস্তুত করতে হবে। আপনার ডেটাসেট ফিল্টার করুন যাতে শুধুমাত্র প্রয়োজনীয় সময়কাল এবং কলাম অন্তর্ভুক্ত থাকে, এবং স্কেলিং করুন যাতে ডেটা ০ থেকে ১ এর মধ্যে প্রজেক্ট করা যায়।

1. মূল ডেটাসেট ফিল্টার করুন যাতে শুধুমাত্র উল্লেখিত সময়কাল এবং কলাম 'load' এবং তারিখ অন্তর্ভুক্ত থাকে: [^2]

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
   
2. প্রশিক্ষণ ডেটা স্কেল করুন যাতে এটি (০, ১) রেঞ্জে থাকে: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. এখন, পরীক্ষার ডেটা স্কেল করুন: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### টাইম-স্টেপ সহ ডেটা তৈরি করুন [^1]

SVR-এর জন্য, আপনি ইনপুট ডেটাকে `[batch, timesteps]` আকারে রূপান্তর করেন। তাই, আপনি বিদ্যমান `train_data` এবং `test_data` রিশেপ করেন যাতে একটি নতুন ডাইমেনশন থাকে যা টাইমস্টেপ নির্দেশ করে।

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

এই উদাহরণের জন্য, আমরা `timesteps = 5` গ্রহণ করি। তাই, মডেলের ইনপুট হলো প্রথম ৪টি টাইমস্টেপের ডেটা, এবং আউটপুট হবে ৫ম টাইমস্টেপের ডেটা।

```python
timesteps=5
```

প্রশিক্ষণ ডেটাকে ২D টেনসরে রূপান্তর করা:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

পরীক্ষার ডেটাকে ২D টেনসরে রূপান্তর করা:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

প্রশিক্ষণ এবং পরীক্ষার ডেটা থেকে ইনপুট এবং আউটপুট নির্বাচন:

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

### SVR বাস্তবায়ন করুন [^1]

এখন, SVR বাস্তবায়নের সময়। এই বাস্তবায়ন সম্পর্কে আরও জানতে, আপনি [এই ডকুমেন্টেশন](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) দেখতে পারেন। আমাদের বাস্তবায়নের জন্য, আমরা নিম্নলিখিত ধাপগুলো অনুসরণ করি:

  1. `SVR()` কল করে এবং মডেলের হাইপারপ্যারামিটার: kernel, gamma, c এবং epsilon পাস করে মডেল সংজ্ঞায়িত করুন
  2. `fit()` ফাংশন কল করে প্রশিক্ষণ ডেটার জন্য মডেল প্রস্তুত করুন
  3. `predict()` ফাংশন কল করে পূর্বাভাস তৈরি করুন

এখন আমরা একটি SVR মডেল তৈরি করি। এখানে আমরা [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) ব্যবহার করি এবং gamma, C এবং epsilon-এর মান যথাক্রমে ০.৫, ১০ এবং ০.০৫ সেট করি।

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### প্রশিক্ষণ ডেটায় মডেল ফিট করুন [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### মডেল পূর্বাভাস তৈরি করুন [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

আপনি আপনার SVR তৈরি করেছেন! এখন আমরা এটি মূল্যায়ন করব।

### আপনার মডেল মূল্যায়ন করুন [^1]

মূল্যায়নের জন্য, প্রথমে আমরা ডেটাকে আমাদের মূল স্কেলে স্কেল ব্যাক করব। এরপর, কার্যকারিতা পরীক্ষা করতে, আমরা মূল এবং পূর্বাভাসকৃত টাইম সিরিজ প্লট করব এবং MAPE ফলাফলও প্রিন্ট করব।

পূর্বাভাসকৃত এবং মূল আউটপুট স্কেল করুন:

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

#### প্রশিক্ষণ এবং পরীক্ষার ডেটায় মডেলের কার্যকারিতা পরীক্ষা করুন [^1]

আমরা ডেটাসেট থেকে টাইমস্ট্যাম্প বের করি যাতে আমাদের প্লটের x-অক্ষে দেখানো যায়। মনে রাখবেন যে আমরা প্রথম ```timesteps-1``` মানগুলো প্রথম আউটপুটের জন্য ইনপুট হিসেবে ব্যবহার করছি, তাই আউটপুটের টাইমস্ট্যাম্পগুলো তার পরে শুরু হবে।

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

প্রশিক্ষণ ডেটার পূর্বাভাস প্লট করুন:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![প্রশিক্ষণ ডেটার পূর্বাভাস](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

প্রশিক্ষণ ডেটার জন্য MAPE প্রিন্ট করুন

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

পরীক্ষার ডেটার পূর্বাভাস প্লট করুন

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![পরীক্ষার ডেটার পূর্বাভাস](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

পরীক্ষার ডেটার জন্য MAPE প্রিন্ট করুন

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 পরীক্ষার ডেটাসেটে আপনার খুব ভালো ফলাফল হয়েছে!

### সম্পূর্ণ ডেটাসেটে মডেলের কার্যকারিতা পরীক্ষা করুন [^1]

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

![সম্পূর্ণ ডেটার পূর্বাভাস](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 খুব সুন্দর প্লট, যা একটি ভালো সঠিকতা সম্পন্ন মডেল দেখাচ্ছে। খুব ভালো কাজ!

---

## 🚀চ্যালেঞ্জ

- মডেল তৈরি করার সময় হাইপারপ্যারামিটার (gamma, C, epsilon) পরিবর্তন করার চেষ্টা করুন এবং ডেটায় মূল্যায়ন করুন যাতে দেখা যায় কোন সেটের হাইপারপ্যারামিটার পরীক্ষার ডেটায় সেরা ফলাফল দেয়। এই হাইপারপ্যারামিটার সম্পর্কে আরও জানতে, আপনি [এই ডকুমেন্ট](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) দেখতে পারেন। 
- মডেলের জন্য বিভিন্ন কের্নেল ফাংশন ব্যবহার করার চেষ্টা করুন এবং তাদের কার্যকারিতা ডেটাসেটে বিশ্লেষণ করুন। একটি সহায়ক ডকুমেন্ট [এখানে](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) পাওয়া যাবে।
- মডেলের জন্য `timesteps`-এর বিভিন্ন মান ব্যবহার করার চেষ্টা করুন যাতে পূর্বাভাস তৈরি করতে পিছনে তাকানো যায়।

## [পাঠ-পরবর্তী কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা এবং স্ব-অধ্যয়ন

এই পাঠটি টাইম সিরিজ পূর্বাভাসের জন্য SVR-এর প্রয়োগ পরিচিত করানোর জন্য ছিল। SVR সম্পর্কে আরও পড়তে, আপনি [এই ব্লগ](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) দেখতে পারেন। এই [scikit-learn ডকুমেন্টেশন](https://scikit-learn.org/stable/modules/svm.html) SVMs সম্পর্কে আরও বিস্তৃত ব্যাখ্যা প্রদান করে, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) এবং অন্যান্য বাস্তবায়ন বিবরণ যেমন বিভিন্ন [কের্নেল ফাংশন](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) যা ব্যবহার করা যেতে পারে এবং তাদের প্যারামিটার।

## অ্যাসাইনমেন্ট

[একটি নতুন SVR মডেল](assignment.md)

## ক্রেডিট

[^1]: এই অংশের টেক্সট, কোড এবং আউটপুট [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) দ্বারা অবদান রাখা হয়েছে।
[^2]: এই অংশের টেক্সট, কোড এবং আউটপুট [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) থেকে নেওয়া হয়েছে।

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।