<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T12:06:59+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "my"
}
-->
# Support Vector Regressor ကို အသုံးပြု၍ အချိန်လိုက်ခန့်မှန်းခြေ

ယခင်သင်ခန်းစာတွင် ARIMA မော်ဒယ်ကို အသုံးပြု၍ အချိန်လိုက်ခန့်မှန်းမှုများ ပြုလုပ်ပုံကို သင်လေ့လာခဲ့ပါသည်။ ယခု သင် Support Vector Regressor မော်ဒယ်ကို လေ့လာမည်ဖြစ်ပြီး၊ ၎င်းသည် ဆက်လက်တိုးတက်နေသော ဒေတာများကို ခန့်မှန်းရန် အသုံးပြုသော regression မော်ဒယ်တစ်ခုဖြစ်သည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/) 

## အကျဉ်းချုပ်

ဒီသင်ခန်းစာမှာ [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) ကို regression အတွက် အသုံးပြုပုံ၊ အထူးသဖြင့် **SVR: Support Vector Regressor** ကို လေ့လာပါမည်။

### အချိန်လိုက်ခန့်မှန်းမှုတွင် SVR [^1]

SVR ၏ အရေးပါမှုကို နားလည်ရန်မတိုင်မီ၊ သင်သိထားရမည့် အရေးကြီးသော အယူအဆများမှာ:

- **Regression:** Supervisory learning နည်းလမ်းတစ်ခုဖြစ်ပြီး၊ ပေးထားသော input set မှ ဆက်လက်တိုးတက်နေသော တန်ဖိုးများကို ခန့်မှန်းရန် အသုံးပြုသည်။ ၎င်း၏ အဓိကအကြောင်းအရာမှာ feature space တွင် အများဆုံး data points ရှိသော curve (သို့မဟုတ်) လိုင်းတစ်ခုကို fit လုပ်ရန်ဖြစ်သည်။ [ပိုမိုသိရှိရန်](https://en.wikipedia.org/wiki/Regression_analysis) နှိပ်ပါ။
- **Support Vector Machine (SVM):** Supervisory machine learning မော်ဒယ်တစ်ခုဖြစ်ပြီး classification, regression နှင့် outliers detection အတွက် အသုံးပြုသည်။ SVM တွင် Kernel function ကို dataset ကို dimension အမြင့်ရှိသော space သို့ ပြောင်းလဲရန် အသုံးပြုသည်။ [ပိုမိုသိရှိရန်](https://en.wikipedia.org/wiki/Support-vector_machine) နှိပ်ပါ။
- **Support Vector Regressor (SVR):** SVM ၏ regression အတွက် version ဖြစ်ပြီး၊ အများဆုံး data points ရှိသော best-fit line (SVM ၏ hyperplane) ကို ရှာဖွေသည်။

### SVR ကို ဘာကြောင့် အသုံးပြုသင့်သလဲ? [^1]

ယခင်သင်ခန်းစာတွင် ARIMA ကို လေ့လာခဲ့ပြီး၊ ၎င်းသည် အချိန်လိုက်ဒေတာများကို ခန့်မှန်းရန် အောင်မြင်သော statistical linear method တစ်ခုဖြစ်သည်။ သို့သော် အချို့သောအခါတွင် အချိန်လိုက်ဒေတာများတွင် *non-linearity* ရှိနိုင်ပြီး၊ linear မော်ဒယ်များဖြင့် မဖြေရှင်းနိုင်ပါ။ ဒီလိုအခြေအနေများတွင် non-linearity ကို handle လုပ်နိုင်သော SVR ၏ စွမ်းရည်သည် time series forecasting အတွက် အောင်မြင်မှုကို ရရှိစေပါသည်။

## လေ့ကျင့်မှု - SVR မော်ဒယ်တစ်ခု တည်ဆောက်ပါ

ဒေတာပြင်ဆင်မှုအဆင့်များသည် [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) သင်ခန်းစာတွင် လေ့လာခဲ့သော အဆင့်များနှင့် တူညီသည်။

ဒီသင်ခန်းစာ၏ [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) folder ကို ဖွင့်ပြီး [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) ဖိုင်ကို ရှာပါ။[^2]

1. Notebook ကို run လုပ်ပြီး လိုအပ်သော libraries များကို import လုပ်ပါ: [^2]

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

2. `/data/energy.csv` ဖိုင်မှ ဒေတာကို Pandas dataframe ထဲသို့ load လုပ်ပြီး ကြည့်ပါ: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. 2012 ခုနှစ် ဇန်နဝါရီမှ 2014 ခုနှစ် ဒီဇင်ဘာအထိရှိသော energy data အားလုံးကို plot လုပ်ပါ: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![full data](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   ယခု SVR မော်ဒယ်ကို တည်ဆောက်ပါ။

### Training နှင့် Testing datasets ဖန်တီးပါ

ဒေတာကို load လုပ်ပြီးပြီးလျှင် train နှင့် test sets သို့ ခွဲခြားပါ။ SVR အတွက် time-step based dataset ဖန်တီးရန် ဒေတာကို reshape လုပ်ပါ။ Train set တွင် မော်ဒယ်ကို train လုပ်ပြီး၊ training set, testing set နှင့် full dataset တွင် accuracy ကို စစ်ဆေးပါ။ Test set သည် training set ထက် နောက်ပိုင်းအချိန်ကာလကို ဖုံးအုပ်ထားရမည်ဖြစ်ပြီး၊ မော်ဒယ်သည် အနာဂတ်အချိန်ကာလမှ အချက်အလက်များကို မရရှိစေရန် သေချာစေရမည် [^2] (*Overfitting* ဟုခေါ်သည်)။

1. 2014 ခုနှစ် စက်တင်ဘာ 1 မှ အောက်တိုဘာ 31 အထိကို training set အဖြစ် သတ်မှတ်ပါ။ Test set သည် 2014 ခုနှစ် နိုဝင်ဘာ 1 မှ ဒီဇင်ဘာ 31 အထိကို ဖုံးအုပ်ထားမည်ဖြစ်သည်: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. ခွဲခြားမှုများကို visualization ပြုလုပ်ပါ: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Training အတွက် ဒေတာကို ပြင်ဆင်ပါ

Training အတွက် ဒေတာကို filter လုပ်ပြီး scale လုပ်ရန် လိုအပ်သည်။ Dataset ကို လိုအပ်သော ကာလများနှင့် column ('load' နှင့် date) များသာ ထည့်သွင်းရန် filter လုပ်ပြီး၊ ဒေတာကို (0, 1) interval တွင် project လုပ်ရန် scale လုပ်ပါ။

1. Original dataset ကို filter လုပ်ပြီး training နှင့် testing sets အတွက် လိုအပ်သော ကာလများနှင့် column များသာ ထည့်သွင်းပါ: [^2]

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
   
2. Training data ကို (0, 1) interval တွင် scale လုပ်ပါ: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Testing data ကို scale လုပ်ပါ: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Time-steps ဖြင့် ဒေတာကို ပြင်ဆင်ပါ [^1]

SVR အတွက် input data ကို `[batch, timesteps]` format သို့ ပြောင်းလဲရန် လိုအပ်သည်။ Training နှင့် testing data ကို reshape လုပ်ပြီး timesteps ကို အသစ်ထည့်သွင်းပါ။

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

ဒီဥပမာတွင် `timesteps = 5` ကို အသုံးပြုပါမည်။ Model inputs သည် ပထမ timesteps 4 ခု၏ ဒေတာများဖြစ်ပြီး၊ output သည် 5th timestep ၏ ဒေတာဖြစ်သည်။

```python
timesteps=5
```

Training data ကို nested list comprehension အသုံးပြု၍ 2D tensor သို့ ပြောင်းပါ:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Testing data ကို 2D tensor သို့ ပြောင်းပါ:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Training နှင့် testing data မှ inputs နှင့် outputs ကို ရွေးချယ်ပါ:

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

### SVR ကို အကောင်အထည်ဖော်ပါ [^1]

ယခု SVR ကို အကောင်အထည်ဖော်ရန် အချိန်ရောက်ပါပြီ။ ဒီ implementation အကြောင်းပိုမိုသိရှိရန် [ဒီ documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) ကို ဖတ်ရှုနိုင်ပါသည်။ 

  1. `SVR()` ကို ခေါ်ပြီး kernel, gamma, c နှင့် epsilon စသည့် hyperparameters များကို pass လုပ်ပါ။
  2. `fit()` function ကို ခေါ်ပြီး training data အတွက် model ကို ပြင်ဆင်ပါ။
  3. `predict()` function ကို ခေါ်ပြီး ခန့်မှန်းမှုများ ပြုလုပ်ပါ။

ယခု SVR မော်ဒယ်ကို ဖန်တီးပါမည်။ [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) ကို အသုံးပြုပြီး၊ gamma, C နှင့် epsilon ကို 0.5, 10 နှင့် 0.05 အဖြစ် သတ်မှတ်ပါမည်။

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Training data တွင် မော်ဒယ်ကို fit လုပ်ပါ [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Model ခန့်မှန်းမှုများ ပြုလုပ်ပါ [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

SVR ကို တည်ဆောက်ပြီးပါပြီ! ယခု ၎င်းကို အကဲဖြတ်ရန် လိုအပ်ပါသည်။

### မော်ဒယ်ကို အကဲဖြတ်ပါ [^1]

အကဲဖြတ်ရန်အတွက် ပထမဦးဆုံး original scale သို့ data ကို ပြန်လည် scale လုပ်ပါ။ Performance ကို စစ်ဆေးရန် original နှင့် predicted time series plot ကို plot လုပ်ပြီး၊ MAPE ရလဒ်ကို print လုပ်ပါ။

Predicted နှင့် original output ကို scale ပြန်လည်ပြုလုပ်ပါ:

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

#### Training နှင့် Testing data တွင် မော်ဒယ်၏ performance ကို စစ်ဆေးပါ [^1]

Dataset မှ timestamps ကို x-axis တွင် ပြရန် extract လုပ်ပါ။ Output ၏ timestamps သည် input ၏ ပထမ ```timesteps-1``` values အပြီးမှ စတင်ပါမည်။

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Training data အတွက် prediction များကို plot လုပ်ပါ:

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

Training data အတွက် MAPE ကို print လုပ်ပါ:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Testing data အတွက် prediction များကို plot လုပ်ပါ:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![testing data prediction](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Testing data အတွက် MAPE ကို print လုပ်ပါ:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Testing dataset တွင် အလွန်ကောင်းမွန်သောရလဒ် ရရှိပါသည်!

### Full dataset တွင် မော်ဒယ်၏ performance ကို စစ်ဆေးပါ [^1]

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

🏆 အလွန်ကောင်းမွန်သော accuracy ရရှိသော မော်ဒယ်ကို ပြသထားသော plot များ ဖြစ်ပါသည်။ အလုပ်ကောင်းပါသည်!

---

## 🚀Challenge

- မော်ဒယ်ကို ဖန်တီးစဉ် hyperparameters (gamma, C, epsilon) များကို ပြောင်းလဲပြီး testing data တွင် အကဲဖြတ်ပါ။ Testing data တွင် အကောင်းဆုံးရလဒ်ရရှိစေသော hyperparameters set ကို ရှာဖွေပါ။ [ဒီ documentation](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) ကို ဖတ်ရှုပါ။
- မော်ဒယ်အတွက် kernel functions များကို ပြောင်းလဲအသုံးပြုပြီး၊ dataset တွင် ၎င်းတို့၏ performance များကို စစ်ဆေးပါ။ [ဒီ documentation](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) ကို ဖတ်ရှုပါ။
- မော်ဒယ်အတွက် `timesteps` အတန်အရွယ်ကို ပြောင်းလဲအသုံးပြုပြီး ခန့်မှန်းမှုများ ပြုလုပ်ပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

ဒီသင်ခန်းစာသည် Time Series Forecasting အတွက် SVR ၏ အသုံးပြုမှုကို မိတ်ဆက်ရန် ရည်ရွယ်ပါသည်။ SVR အကြောင်းပိုမိုသိရှိရန် [ဒီ blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) ကို ဖတ်ရှုနိုင်ပါသည်။ [scikit-learn documentation](https://scikit-learn.org/stable/modules/svm.html) တွင် SVMs, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) နှင့် [kernel functions](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) အကြောင်း အပြည့်အစုံ ရှင်းလင်းထားပါသည်။

## Assignment

[A new SVR model](assignment.md)

## Credits

[^1]: ဒီအပိုင်းတွင် ပါဝင်သော စာသား၊ ကုဒ်နှင့် output များကို [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) မှ ပံ့ပိုးခဲ့သည်။
[^2]: ဒီအပိုင်းတွင် ပါဝင်သော စာသား၊ ကုဒ်နှင့် output များကို [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) မှ ယူထားသည်။

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူကောင်းမွန်သော ပရော်ဖက်ရှင်နယ်ဘာသာပြန်ဝန်ဆောင်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပါယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။