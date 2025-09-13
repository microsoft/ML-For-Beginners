<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T06:09:09+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "mr"
}
-->
# सपोर्ट व्हेक्टर रेग्रेसरसह टाइम सिरीज अंदाज

मागील धड्यात, तुम्ही ARIMA मॉडेल वापरून टाइम सिरीज अंदाज कसा करायचा ते शिकले. आता तुम्ही सपोर्ट व्हेक्टर रेग्रेसर मॉडेलकडे पाहणार आहात, जे सतत डेटा अंदाज करण्यासाठी वापरले जाते.

## [पूर्व-व्याख्यान प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/) 

## परिचय

या धड्यात, तुम्ही [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) वापरून रेग्रेशनसाठी मॉडेल तयार करण्याचा एक विशिष्ट मार्ग शोधाल, ज्याला **SVR: Support Vector Regressor** म्हणतात.

### टाइम सिरीजच्या संदर्भात SVR [^1]

टाइम सिरीज अंदाजामध्ये SVR चे महत्त्व समजण्यापूर्वी, तुम्हाला खालील महत्त्वाच्या संकल्पना माहित असणे आवश्यक आहे:

- **रेग्रेशन:** दिलेल्या इनपुट्सच्या संचातून सतत मूल्ये अंदाज करण्यासाठी सुपरवाइज्ड लर्निंग तंत्र. कल्पना अशी आहे की फीचर स्पेसमध्ये जास्तीत जास्त डेटा पॉइंट्स असलेली वक्र (किंवा रेषा) फिट करणे. [अधिक माहितीसाठी येथे क्लिक करा](https://en.wikipedia.org/wiki/Regression_analysis).
- **सपोर्ट व्हेक्टर मशीन (SVM):** वर्गीकरण, रेग्रेशन आणि आउटलाईयर डिटेक्शनसाठी वापरले जाणारे सुपरवाइज्ड मशीन लर्निंग मॉडेलचा प्रकार. मॉडेल फीचर स्पेसमधील हायपरप्लेन आहे, जे वर्गीकरणाच्या बाबतीत सीमा म्हणून कार्य करते आणि रेग्रेशनच्या बाबतीत सर्वोत्तम फिट रेषा म्हणून कार्य करते. SVM मध्ये, डेटासेटला उच्च परिमाणांच्या जागेत रूपांतरित करण्यासाठी सामान्यतः कर्नल फंक्शन वापरले जाते, जेणेकरून ते सहजपणे विभक्त होऊ शकतील. [SVM बद्दल अधिक माहितीसाठी येथे क्लिक करा](https://en.wikipedia.org/wiki/Support-vector_machine).
- **सपोर्ट व्हेक्टर रेग्रेसर (SVR):** SVM चा एक प्रकार, सर्वोत्तम फिट रेषा शोधण्यासाठी (जे SVM च्या बाबतीत हायपरप्लेन आहे) ज्यामध्ये जास्तीत जास्त डेटा पॉइंट्स असतात.

### SVR का? [^1]

मागील धड्यात तुम्ही ARIMA बद्दल शिकले, जे टाइम सिरीज डेटा अंदाज करण्यासाठी एक अतिशय यशस्वी सांख्यिकीय रेषीय पद्धत आहे. तथापि, अनेक प्रकरणांमध्ये, टाइम सिरीज डेटामध्ये *नॉन-लाइनॅरिटी* असते, जी रेषीय मॉडेलद्वारे मॅप केली जाऊ शकत नाही. अशा परिस्थितीत, रेग्रेशन कार्यांसाठी डेटामधील नॉन-लाइनॅरिटी विचारात घेण्याची SVM ची क्षमता टाइम सिरीज अंदाजामध्ये SVR ला यशस्वी बनवते.

## व्यायाम - SVR मॉडेल तयार करा

डेटा तयार करण्यासाठी सुरुवातीची काही पावले [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) धड्यातील पावलांसारखीच आहेत.

या धड्यातील [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) फोल्डर उघडा आणि [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) फाइल शोधा.[^2]

1. नोटबुक चालवा आणि आवश्यक लायब्ररी आयात करा: [^2]

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

2. `/data/energy.csv` फाइलमधून डेटा Pandas डेटा फ्रेममध्ये लोड करा आणि त्यावर नजर टाका: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. जानेवारी 2012 ते डिसेंबर 2014 पर्यंत उपलब्ध ऊर्जा डेटा प्लॉट करा: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![पूर्ण डेटा](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   आता, आपले SVR मॉडेल तयार करूया.

### प्रशिक्षण आणि चाचणी डेटासेट तयार करा

आता तुमचा डेटा लोड झाला आहे, त्यामुळे तुम्ही तो ट्रेन आणि टेस्ट सेटमध्ये विभाजित करू शकता. त्यानंतर तुम्ही डेटा टाइम-स्टेप आधारित डेटासेट तयार करण्यासाठी पुनर्रचना कराल, जे SVR साठी आवश्यक असेल. तुम्ही तुमचे मॉडेल ट्रेन सेटवर प्रशिक्षित कराल. मॉडेल प्रशिक्षण पूर्ण झाल्यानंतर, तुम्ही ट्रेनिंग सेट, टेस्टिंग सेट आणि नंतर संपूर्ण डेटासेटवर त्याची अचूकता मूल्यांकन कराल, जेणेकरून एकूण कार्यप्रदर्शन पाहता येईल. तुम्हाला हे सुनिश्चित करणे आवश्यक आहे की चाचणी संच प्रशिक्षण संचापेक्षा नंतरच्या कालावधीचा समावेश करतो, जेणेकरून मॉडेल भविष्यातील कालावधीमधून माहिती मिळवू शकत नाही [^2] (ज्याला *ओव्हरफिटिंग* म्हणतात).

1. 1 सप्टेंबर ते 31 ऑक्टोबर 2014 पर्यंतचा दोन महिन्यांचा कालावधी प्रशिक्षण संचासाठी वाटप करा. चाचणी संचामध्ये 1 नोव्हेंबर ते 31 डिसेंबर 2014 पर्यंतचा दोन महिन्यांचा कालावधी समाविष्ट असेल: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. फरकांचे व्हिज्युअलायझेशन करा: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![प्रशिक्षण आणि चाचणी डेटा](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### प्रशिक्षणासाठी डेटा तयार करा

आता, तुम्हाला तुमचा डेटा प्रशिक्षणासाठी तयार करणे आवश्यक आहे, ज्यामध्ये तुमचा डेटा फिल्टर करणे आणि स्केल करणे समाविष्ट आहे. तुमच्या डेटासेटला फक्त आवश्यक कालावधी आणि स्तंभ समाविष्ट करण्यासाठी फिल्टर करा आणि डेटा 0,1 अंतरामध्ये प्रोजेक्ट करण्यासाठी स्केलिंग करा.

1. मूळ डेटासेट फिल्टर करा, ज्यामध्ये फक्त वरील कालावधी आणि 'load' स्तंभ आणि तारीख समाविष्ट असेल: [^2]

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
   
2. प्रशिक्षण डेटा (0, 1) श्रेणीत स्केल करा: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. आता, तुम्ही चाचणी डेटा स्केल करा: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### टाइम-स्टेप्ससह डेटा तयार करा [^1]

SVR साठी, तुम्ही इनपुट डेटा `[batch, timesteps]` स्वरूपात रूपांतरित करता. त्यामुळे, तुम्ही विद्यमान `train_data` आणि `test_data` पुनर्रचना करता, ज्यामुळे एक नवीन परिमाण तयार होते, जे टाइमस्टेप्सला संदर्भित करते.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

या उदाहरणासाठी, आपण `timesteps = 5` घेतो. त्यामुळे, मॉडेलसाठी इनपुट म्हणजे पहिल्या 4 टाइमस्टेप्ससाठी डेटा असेल आणि आउटपुट म्हणजे 5व्या टाइमस्टेपसाठी डेटा असेल.

```python
timesteps=5
```

प्रशिक्षण डेटा 2D टेन्सरमध्ये रूपांतरित करणे:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

चाचणी डेटा 2D टेन्सरमध्ये रूपांतरित करणे:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

प्रशिक्षण आणि चाचणी डेटामधून इनपुट्स आणि आउटपुट्स निवडणे:

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

### SVR अंमलात आणा [^1]

आता, SVR अंमलात आणण्याची वेळ आली आहे. या अंमलबद्दल अधिक वाचण्यासाठी, तुम्ही [या दस्तऐवजाचा](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) संदर्भ घेऊ शकता. आमच्या अंमलबद्दल, आम्ही खालील चरणांचे अनुसरण करतो:

  1. `SVR()` कॉल करून आणि मॉडेल हायपरपॅरामीटर्स: kernel, gamma, c आणि epsilon पास करून मॉडेल परिभाषित करा
  2. `fit()` फंक्शन कॉल करून प्रशिक्षण डेटासाठी मॉडेल तयार करा
  3. `predict()` फंक्शन कॉल करून अंदाज तयार करा

आता आम्ही SVR मॉडेल तयार करतो. येथे आम्ही [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) वापरतो आणि हायपरपॅरामीटर्स gamma, C आणि epsilon अनुक्रमे 0.5, 10 आणि 0.05 सेट करतो.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### प्रशिक्षण डेटावर मॉडेल फिट करा [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### मॉडेल अंदाज तयार करा [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

तुम्ही तुमचे SVR तयार केले आहे! आता आम्हाला त्याचे मूल्यांकन करणे आवश्यक आहे.

### तुमचे मॉडेल मूल्यांकन करा [^1]

मूल्यांकनासाठी, प्रथम आम्ही डेटा मूळ स्केलवर परत स्केल करू. त्यानंतर, कार्यप्रदर्शन तपासण्यासाठी, आम्ही मूळ आणि अंदाजित टाइम सिरीज प्लॉट तयार करू आणि MAPE परिणाम देखील प्रिंट करू.

अंदाजित आणि मूळ आउटपुट स्केल करा:

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

#### प्रशिक्षण आणि चाचणी डेटावर मॉडेल कार्यप्रदर्शन तपासा [^1]

आम्ही आमच्या प्लॉटच्या x-अक्षावर दर्शविण्यासाठी डेटासेटमधून टाइमस्टॅम्प्स काढतो. लक्षात घ्या की आम्ही पहिल्या ```timesteps-1``` मूल्ये पहिल्या आउटपुटसाठी इनपुट म्हणून वापरत आहोत, त्यामुळे आउटपुटसाठी टाइमस्टॅम्प्स त्यानंतर सुरू होतील.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

प्रशिक्षण डेटासाठी अंदाज प्लॉट करा:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![प्रशिक्षण डेटा अंदाज](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

प्रशिक्षण डेटासाठी MAPE प्रिंट करा

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

चाचणी डेटासाठी अंदाज प्लॉट करा

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![चाचणी डेटा अंदाज](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

चाचणी डेटासाठी MAPE प्रिंट करा

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 तुम्हाला चाचणी डेटासेटवर खूप चांगला परिणाम मिळाला आहे!

### संपूर्ण डेटासेटवर मॉडेल कार्यप्रदर्शन तपासा [^1]

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

![पूर्ण डेटा अंदाज](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 खूप छान प्लॉट्स, जे चांगल्या अचूकतेसह मॉडेल दर्शवतात. उत्तम काम केले!

---

## 🚀चॅलेंज

- मॉडेल तयार करताना हायपरपॅरामीटर्स (gamma, C, epsilon) बदलण्याचा प्रयत्न करा आणि चाचणी डेटावर मूल्यांकन करा, जेणेकरून कोणते हायपरपॅरामीटर्स चाचणी डेटावर सर्वोत्तम परिणाम देतात ते पाहता येईल. या हायपरपॅरामीटर्सबद्दल अधिक जाणून घेण्यासाठी, तुम्ही [येथे](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) दस्तऐवजाचा संदर्भ घेऊ शकता. 
- मॉडेलसाठी वेगवेगळ्या कर्नल फंक्शन्स वापरण्याचा प्रयत्न करा आणि त्यांच्या कार्यप्रदर्शनाचा डेटासेटवर विश्लेषण करा. उपयुक्त दस्तऐवज [येथे](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) सापडू शकतो.
- मॉडेलसाठी अंदाज तयार करण्यासाठी `timesteps` साठी वेगवेगळ्या मूल्यांचा वापर करण्याचा प्रयत्न करा.

## [व्याख्यानानंतर प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

या धड्याचा उद्देश टाइम सिरीज अंदाजासाठी SVR चा उपयोग सादर करणे होता. SVR बद्दल अधिक वाचण्यासाठी, तुम्ही [या ब्लॉगचा](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) संदर्भ घेऊ शकता. [scikit-learn वरचा हा दस्तऐवज](https://scikit-learn.org/stable/modules/svm.html) SVMs बद्दल अधिक व्यापक स्पष्टीकरण प्रदान करतो, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) आणि इतर अंमलबद्दल तपशील जसे की वेगवेगळे [कर्नल फंक्शन्स](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) जे वापरले जाऊ शकतात आणि त्यांचे पॅरामीटर्स.

## असाइनमेंट

[नवीन SVR मॉडेल](assignment.md)

## क्रेडिट्स

[^1]: या विभागातील मजकूर, कोड आणि आउटपुट [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) यांनी योगदान दिले आहे.
[^2]: या विभागातील मजकूर, कोड आणि आउटपुट [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) मधून घेतले आहे.

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर केल्यामुळे उद्भवणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.