<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T10:18:39+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "hi"
}
-->
# सपोर्ट वेक्टर रेग्रेसर के साथ टाइम सीरीज़ फोरकास्टिंग

पिछले पाठ में, आपने ARIMA मॉडल का उपयोग करके टाइम सीरीज़ प्रेडिक्शन करना सीखा। अब आप सपोर्ट वेक्टर रेग्रेसर मॉडल पर ध्यान देंगे, जो एक रेग्रेशन मॉडल है और निरंतर डेटा की भविष्यवाणी करने के लिए उपयोग किया जाता है।

## [प्री-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## परिचय

इस पाठ में, आप [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) का उपयोग करके रेग्रेशन के लिए मॉडल बनाने का एक विशेष तरीका जानेंगे, जिसे **SVR: Support Vector Regressor** कहा जाता है।

### टाइम सीरीज़ के संदर्भ में SVR [^1]

टाइम सीरीज़ प्रेडिक्शन में SVR के महत्व को समझने से पहले, यहां कुछ महत्वपूर्ण अवधारणाएं हैं जिन्हें आपको जानना चाहिए:

- **रेग्रेशन:** यह एक सुपरवाइज़्ड लर्निंग तकनीक है जो दिए गए इनपुट से निरंतर मानों की भविष्यवाणी करती है। इसका उद्देश्य फीचर स्पेस में एक कर्व (या रेखा) फिट करना है, जिसमें अधिकतम डेटा पॉइंट्स हों। [यहां क्लिक करें](https://en.wikipedia.org/wiki/Regression_analysis) अधिक जानकारी के लिए।
- **सपोर्ट वेक्टर मशीन (SVM):** यह एक प्रकार का सुपरवाइज़्ड मशीन लर्निंग मॉडल है, जिसका उपयोग वर्गीकरण, रेग्रेशन और आउटलायर डिटेक्शन के लिए किया जाता है। यह मॉडल फीचर स्पेस में एक हाइपरप्लेन होता है, जो वर्गीकरण के मामले में एक सीमा के रूप में कार्य करता है और रेग्रेशन के मामले में बेस्ट-फिट लाइन के रूप में। SVM में, आमतौर पर एक कर्नेल फंक्शन का उपयोग किया जाता है, जो डेटा को उच्च आयाम वाले स्पेस में ट्रांसफॉर्म करता है ताकि वे आसानी से अलग किए जा सकें। [यहां क्लिक करें](https://en.wikipedia.org/wiki/Support-vector_machine) SVMs पर अधिक जानकारी के लिए।
- **सपोर्ट वेक्टर रेग्रेसर (SVR):** SVM का एक प्रकार, जो बेस्ट फिट लाइन (जो SVM के मामले में एक हाइपरप्लेन है) खोजने के लिए उपयोग किया जाता है, जिसमें अधिकतम डेटा पॉइंट्स हों।

### SVR क्यों? [^1]

पिछले पाठ में आपने ARIMA के बारे में सीखा, जो टाइम सीरीज़ डेटा की भविष्यवाणी के लिए एक बहुत ही सफल सांख्यिकीय रैखिक विधि है। हालांकि, कई मामलों में, टाइम सीरीज़ डेटा में *गैर-रैखिकता* होती है, जिसे रैखिक मॉडल द्वारा मैप नहीं किया जा सकता। ऐसे मामलों में, रेग्रेशन कार्यों के लिए डेटा में गैर-रैखिकता को ध्यान में रखने की SVM की क्षमता SVR को टाइम सीरीज़ फोरकास्टिंग में सफल बनाती है।

## अभ्यास - SVR मॉडल बनाएं

डेटा तैयार करने के पहले कुछ चरण पिछले पाठ [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) के समान हैं।

इस पाठ में [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) फ़ोल्डर खोलें और [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) फ़ाइल ढूंढें।[^2]

1. नोटबुक चलाएं और आवश्यक लाइब्रेरी आयात करें: [^2]

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

2. `/data/energy.csv` फ़ाइल से डेटा को एक Pandas डेटा फ्रेम में लोड करें और इसे देखें: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. जनवरी 2012 से दिसंबर 2014 तक उपलब्ध सभी ऊर्जा डेटा को प्लॉट करें: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![पूर्ण डेटा](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   अब, चलिए अपना SVR मॉडल बनाते हैं।

### प्रशिक्षण और परीक्षण डेटा सेट बनाएं

अब आपका डेटा लोड हो गया है, इसलिए आप इसे ट्रेन और टेस्ट सेट में विभाजित कर सकते हैं। फिर आप डेटा को टाइम-स्टेप आधारित डेटा सेट बनाने के लिए पुनः आकार देंगे, जो SVR के लिए आवश्यक होगा। आप अपने मॉडल को ट्रेन सेट पर प्रशिक्षित करेंगे। मॉडल के प्रशिक्षण के बाद, आप इसके सटीकता का मूल्यांकन ट्रेनिंग सेट, टेस्टिंग सेट और फिर पूरे डेटा सेट पर करेंगे ताकि समग्र प्रदर्शन देखा जा सके। आपको यह सुनिश्चित करना होगा कि टेस्ट सेट ट्रेनिंग सेट से समय में बाद की अवधि को कवर करता है ताकि यह सुनिश्चित हो सके कि मॉडल भविष्य की समय अवधि से जानकारी प्राप्त न करे [^2] (जिसे *ओवरफिटिंग* कहा जाता है)।

1. 1 सितंबर से 31 अक्टूबर, 2014 तक की दो महीने की अवधि को ट्रेनिंग सेट में आवंटित करें। टेस्ट सेट में 1 नवंबर से 31 दिसंबर, 2014 तक की दो महीने की अवधि शामिल होगी: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. अंतर को विज़ुअलाइज़ करें: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![ट्रेनिंग और टेस्टिंग डेटा](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### प्रशिक्षण के लिए डेटा तैयार करें

अब, आपको अपने डेटा को प्रशिक्षण के लिए तैयार करने की आवश्यकता है, जिसमें डेटा को फ़िल्टर करना और स्केल करना शामिल है। अपने डेटा सेट को केवल आवश्यक समय अवधि और कॉलम तक सीमित करें, और डेटा को 0,1 के अंतराल में प्रोजेक्ट करने के लिए स्केल करें।

1. मूल डेटा सेट को केवल उपर्युक्त समय अवधि और केवल आवश्यक कॉलम 'लोड' और तारीख तक सीमित करें: [^2]

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

2. ट्रेनिंग डेटा को (0, 1) की रेंज में स्केल करें: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```

4. अब, टेस्टिंग डेटा को स्केल करें: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### टाइम-स्टेप्स के साथ डेटा बनाएं [^1]

SVR के लिए, आप इनपुट डेटा को `[batch, timesteps]` के रूप में ट्रांसफॉर्म करते हैं। इसलिए, आप मौजूदा `train_data` और `test_data` को इस प्रकार पुनः आकार देंगे कि एक नया आयाम हो, जो टाइमस्टेप्स को संदर्भित करता है।

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

इस उदाहरण के लिए, हम `timesteps = 5` लेते हैं। तो, मॉडल के इनपुट पहले 4 टाइमस्टेप्स के डेटा होंगे, और आउटपुट 5वें टाइमस्टेप का डेटा होगा।

```python
timesteps=5
```

नेस्टेड लिस्ट कॉम्प्रिहेंशन का उपयोग करके ट्रेनिंग डेटा को 2D टेंसर में बदलना:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

टेस्टिंग डेटा को 2D टेंसर में बदलना:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

ट्रेनिंग और टेस्टिंग डेटा से इनपुट और आउटपुट का चयन:

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

### SVR लागू करें [^1]

अब, SVR को लागू करने का समय है। इस कार्यान्वयन के बारे में अधिक पढ़ने के लिए, आप [इस दस्तावेज़](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) का संदर्भ ले सकते हैं। हमारे कार्यान्वयन के लिए, हम निम्नलिखित चरणों का पालन करते हैं:

1. `SVR()` को कॉल करके और मॉडल हाइपरपैरामीटर्स: kernel, gamma, c और epsilon पास करके मॉडल को परिभाषित करें।
2. `fit()` फ़ंक्शन को कॉल करके मॉडल को ट्रेनिंग डेटा के लिए तैयार करें।
3. `predict()` फ़ंक्शन को कॉल करके भविष्यवाणियां करें।

अब हम एक SVR मॉडल बनाते हैं। यहां हम [RBF कर्नेल](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) का उपयोग करते हैं, और हाइपरपैरामीटर्स gamma, C और epsilon को क्रमशः 0.5, 10 और 0.05 पर सेट करते हैं।

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### ट्रेनिंग डेटा पर मॉडल फिट करें [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### मॉडल की भविष्यवाणियां करें [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

आपने अपना SVR बना लिया है! अब हमें इसका मूल्यांकन करना होगा।

### अपने मॉडल का मूल्यांकन करें [^1]

मूल्यांकन के लिए, पहले हम डेटा को हमारे मूल स्केल में वापस स्केल करेंगे। फिर, प्रदर्शन की जांच करने के लिए, हम मूल और भविष्यवाणी किए गए टाइम सीरीज़ प्लॉट को प्लॉट करेंगे, और MAPE परिणाम भी प्रिंट करेंगे।

भविष्यवाणी और मूल आउटपुट को स्केल करें:

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

#### ट्रेनिंग और टेस्टिंग डेटा पर मॉडल प्रदर्शन की जांच करें [^1]

हम x-अक्ष पर दिखाने के लिए डेटा सेट से टाइमस्टैम्प निकालते हैं। ध्यान दें कि हम पहले ```timesteps-1``` मानों का उपयोग पहले आउटपुट के लिए इनपुट के रूप में कर रहे हैं, इसलिए आउटपुट के लिए टाइमस्टैम्प उसके बाद शुरू होंगे।

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

ट्रेनिंग डेटा के लिए भविष्यवाणियों को प्लॉट करें:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![ट्रेनिंग डेटा भविष्यवाणी](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

ट्रेनिंग डेटा के लिए MAPE प्रिंट करें:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

टेस्टिंग डेटा के लिए भविष्यवाणियों को प्लॉट करें:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![टेस्टिंग डेटा भविष्यवाणी](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

टेस्टिंग डेटा के लिए MAPE प्रिंट करें:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 आपके पास टेस्टिंग डेटा सेट पर बहुत अच्छा परिणाम है!

### पूरे डेटा सेट पर मॉडल प्रदर्शन की जांच करें [^1]

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

![पूर्ण डेटा भविष्यवाणी](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 बहुत अच्छे प्लॉट्स, जो एक अच्छे सटीकता वाले मॉडल को दिखाते हैं। बहुत बढ़िया!

---

## 🚀चुनौती

- मॉडल बनाते समय हाइपरपैरामीटर्स (gamma, C, epsilon) को बदलने का प्रयास करें और डेटा पर मूल्यांकन करें ताकि यह देखा जा सके कि कौन सा हाइपरपैरामीटर सेट टेस्टिंग डेटा पर सबसे अच्छे परिणाम देता है। इन हाइपरपैरामीटर्स के बारे में अधिक जानने के लिए, आप [यहां](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) दस्तावेज़ देख सकते हैं।
- मॉडल के लिए विभिन्न कर्नेल फंक्शंस का उपयोग करने का प्रयास करें और उनके प्रदर्शन का विश्लेषण करें। एक सहायक दस्तावेज़ [यहां](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) पाया जा सकता है।
- मॉडल के लिए भविष्यवाणी करने के लिए `timesteps` के विभिन्न मानों का उपयोग करने का प्रयास करें।

## [पोस्ट-लेक्चर क्विज़](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

यह पाठ टाइम सीरीज़ फोरकास्टिंग के लिए SVR के अनुप्रयोग को पेश करने के लिए था। SVR के बारे में अधिक पढ़ने के लिए, आप [इस ब्लॉग](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) का संदर्भ ले सकते हैं। [scikit-learn पर यह दस्तावेज़](https://scikit-learn.org/stable/modules/svm.html) SVMs के बारे में अधिक व्यापक व्याख्या प्रदान करता है, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) और अन्य कार्यान्वयन विवरण जैसे कि विभिन्न [कर्नेल फंक्शंस](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) जो उपयोग किए जा सकते हैं, और उनके पैरामीटर्स।

## असाइनमेंट

[एक नया SVR मॉडल](assignment.md)

## क्रेडिट्स

[^1]: इस खंड में पाठ, कोड और आउटपुट [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) द्वारा योगदान किया गया था।  
[^2]: इस खंड में पाठ, कोड और आउटपुट [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) से लिया गया था।

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  