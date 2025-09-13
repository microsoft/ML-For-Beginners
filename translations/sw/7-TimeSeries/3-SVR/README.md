<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T15:36:08+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "sw"
}
-->
# Utabiri wa Mfululizo wa Wakati kwa Kutumia Support Vector Regressor

Katika somo lililopita, ulijifunza jinsi ya kutumia modeli ya ARIMA kufanya utabiri wa mfululizo wa wakati. Sasa utaangalia modeli ya Support Vector Regressor, ambayo ni modeli ya regression inayotumika kutabiri data endelevu.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/) 

## Utangulizi

Katika somo hili, utagundua njia maalum ya kujenga modeli kwa [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) kwa regression, au **SVR: Support Vector Regressor**. 

### SVR katika muktadha wa mfululizo wa wakati [^1]

Kabla ya kuelewa umuhimu wa SVR katika utabiri wa mfululizo wa wakati, hapa kuna baadhi ya dhana muhimu unazohitaji kujua:

- **Regression:** Mbinu ya kujifunza kwa usimamizi inayotabiri thamani endelevu kutoka kwa seti fulani ya pembejeo. Wazo ni kufaa mstari au curve katika nafasi ya vipengele ambayo ina idadi kubwa ya pointi za data. [Bonyeza hapa](https://en.wikipedia.org/wiki/Regression_analysis) kwa maelezo zaidi.
- **Support Vector Machine (SVM):** Aina ya modeli ya kujifunza kwa usimamizi inayotumika kwa uainishaji, regression, na kugundua data isiyo ya kawaida. Modeli ni hyperplane katika nafasi ya vipengele, ambayo katika hali ya uainishaji hufanya kama mpaka, na katika hali ya regression hufanya kama mstari bora wa kufaa. Katika SVM, kazi ya Kernel hutumika kwa kawaida kubadilisha seti ya data kuwa nafasi ya vipimo vingi zaidi, ili iweze kutenganishwa kwa urahisi. [Bonyeza hapa](https://en.wikipedia.org/wiki/Support-vector_machine) kwa maelezo zaidi kuhusu SVM.
- **Support Vector Regressor (SVR):** Aina ya SVM, inayotafuta mstari bora wa kufaa (ambayo katika hali ya SVM ni hyperplane) yenye idadi kubwa ya pointi za data.

### Kwa nini SVR? [^1]

Katika somo la mwisho ulijifunza kuhusu ARIMA, ambayo ni mbinu ya takwimu ya mstari inayofanikiwa sana kutabiri data ya mfululizo wa wakati. Hata hivyo, katika hali nyingi, data ya mfululizo wa wakati ina *kutokuwa na mstari*, ambayo haiwezi kuonyeshwa na modeli za mstari. Katika hali kama hizi, uwezo wa SVM kuzingatia kutokuwa na mstari katika data kwa kazi za regression hufanya SVR kufanikiwa katika utabiri wa mfululizo wa wakati.

## Zoezi - jenga modeli ya SVR

Hatua za awali za maandalizi ya data ni sawa na zile za somo lililopita kuhusu [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Fungua folda [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) katika somo hili na pata faili [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb). [^2]

1. Endesha notebook na uagize maktaba muhimu: [^2]

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

2. Pakia data kutoka faili `/data/energy.csv` kwenye dataframe ya Pandas na uitazame: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Chora data yote ya nishati inayopatikana kutoka Januari 2012 hadi Desemba 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data kamili](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Sasa, hebu tujenge modeli yetu ya SVR.

### Unda seti za mafunzo na majaribio

Sasa data yako imepakizwa, unaweza kuigawanya katika seti za mafunzo na majaribio. Kisha utabadilisha data ili kuunda seti ya data inayotegemea hatua za wakati ambayo itahitajika kwa SVR. Utazoeza modeli yako kwenye seti ya mafunzo. Baada ya modeli kumaliza mafunzo, utatathmini usahihi wake kwenye seti ya mafunzo, seti ya majaribio, na kisha seti kamili ya data ili kuona utendaji wa jumla. Unahitaji kuhakikisha kuwa seti ya majaribio inashughulikia kipindi cha baadaye kutoka seti ya mafunzo ili kuhakikisha kuwa modeli haipati taarifa kutoka vipindi vya baadaye [^2] (hali inayojulikana kama *Overfitting*).

1. Toa kipindi cha miezi miwili kutoka Septemba 1 hadi Oktoba 31, 2014 kwa seti ya mafunzo. Seti ya majaribio itajumuisha kipindi cha miezi miwili kutoka Novemba 1 hadi Desemba 31, 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Onyesha tofauti: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data ya mafunzo na majaribio](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Andaa data kwa mafunzo

Sasa, unahitaji kuandaa data kwa mafunzo kwa kufanya uchujaji na upimaji wa data yako. Chuja seti yako ya data ili kujumuisha tu vipindi vya wakati na safu unazohitaji, na upimaji ili kuhakikisha data inaonyeshwa katika interval 0,1.

1. Chuja seti ya data ya awali ili kujumuisha tu vipindi vya wakati vilivyotajwa kwa kila seti na kujumuisha tu safu inayohitajika 'load' pamoja na tarehe: [^2]

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
   
2. Pima data ya mafunzo kuwa katika kiwango cha (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Sasa, pima data ya majaribio: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Unda data yenye hatua za wakati [^1]

Kwa SVR, unabadilisha data ya pembejeo kuwa ya fomu `[batch, timesteps]`. Kwa hivyo, unabadilisha `train_data` na `test_data` iliyopo ili kuwe na kipimo kipya kinachorejelea hatua za wakati. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Kwa mfano huu, tunachukua `timesteps = 5`. Kwa hivyo, pembejeo kwa modeli ni data ya hatua za kwanza 4, na matokeo yatakuwa data ya hatua ya 5.

```python
timesteps=5
```

Kubadilisha data ya mafunzo kuwa tensor ya 2D kwa kutumia nested list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Kubadilisha data ya majaribio kuwa tensor ya 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Kuchagua pembejeo na matokeo kutoka data ya mafunzo na majaribio:

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

### Tekeleza SVR [^1]

Sasa, ni wakati wa kutekeleza SVR. Ili kusoma zaidi kuhusu utekelezaji huu, unaweza kurejelea [hati hii](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Kwa utekelezaji wetu, tunafuata hatua hizi:

  1. Fafanua modeli kwa kuita `SVR()` na kupitisha hyperparameters za modeli: kernel, gamma, c na epsilon
  2. Andaa modeli kwa data ya mafunzo kwa kuita kazi ya `fit()`
  3. Fanya utabiri kwa kuita kazi ya `predict()`

Sasa tunaunda modeli ya SVR. Hapa tunatumia [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), na kuweka hyperparameters gamma, C na epsilon kama 0.5, 10 na 0.05 mtawalia.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Fanya modeli ifanye mafunzo kwenye data ya mafunzo [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Fanya utabiri wa modeli [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Umejenga SVR yako! Sasa tunahitaji kuitathmini.

### Tathmini modeli yako [^1]

Kwa tathmini, kwanza tutapima tena data kwa kiwango chetu cha awali. Kisha, ili kuangalia utendaji, tutachora grafu ya mfululizo wa wakati wa asili na uliotabiriwa, na pia kuchapisha matokeo ya MAPE.

Pima tena matokeo yaliyotabiriwa na ya asili:

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

#### Angalia utendaji wa modeli kwenye data ya mafunzo na majaribio [^1]

Tunatoa timestamps kutoka seti ya data ili kuonyesha kwenye mhimili wa x wa grafu yetu. Kumbuka kuwa tunatumia ```timesteps-1``` za kwanza kama pembejeo kwa matokeo ya kwanza, kwa hivyo timestamps za matokeo zitaanza baada ya hapo.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Chora utabiri wa data ya mafunzo:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![utabiri wa data ya mafunzo](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Chapisha MAPE kwa data ya mafunzo

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Chora utabiri wa data ya majaribio

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![utabiri wa data ya majaribio](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Chapisha MAPE kwa data ya majaribio

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Una matokeo mazuri sana kwenye seti ya data ya majaribio!

### Angalia utendaji wa modeli kwenye seti kamili ya data [^1]

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

![utabiri wa data kamili](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Grafu nzuri sana, zinazoonyesha modeli yenye usahihi mzuri. Hongera!

---

## 🚀Changamoto

- Jaribu kubadilisha hyperparameters (gamma, C, epsilon) wakati wa kuunda modeli na tathmini kwenye data ili kuona ni seti gani ya hyperparameters inatoa matokeo bora kwenye data ya majaribio. Ili kujua zaidi kuhusu hyperparameters hizi, unaweza kurejelea hati [hapa](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Jaribu kutumia kazi tofauti za kernel kwa modeli na uchanganue utendaji wake kwenye seti ya data. Hati inayosaidia inaweza kupatikana [hapa](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Jaribu kutumia thamani tofauti za `timesteps` kwa modeli kuangalia nyuma ili kufanya utabiri.

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Somo hili lilikuwa la kuanzisha matumizi ya SVR kwa Utabiri wa Mfululizo wa Wakati. Ili kusoma zaidi kuhusu SVR, unaweza kurejelea [blogu hii](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Hati hii [katika scikit-learn](https://scikit-learn.org/stable/modules/svm.html) inatoa maelezo ya kina zaidi kuhusu SVM kwa ujumla, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) na pia maelezo mengine ya utekelezaji kama vile kazi tofauti za [kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) zinazoweza kutumika, na vigezo vyake.

## Kazi

[Modeli mpya ya SVR](assignment.md)

## Credits

[^1]: Maandishi, msimbo na matokeo katika sehemu hii yalichangiwa na [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Maandishi, msimbo na matokeo katika sehemu hii yalichukuliwa kutoka [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.