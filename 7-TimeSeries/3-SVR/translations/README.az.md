# Dəstək vektor reqressoru ilə zaman seriyalarının proqnozlaşdırılması

Əvvəlki dərsdə ARIMA modeli istifadə etməklə necə zaman seriyalarını proqnozlaşdıra biləcəyini öyrəndin. İndi isə Dəstək vektor reqressor modeli ilə davamlı datanın gələcəyini təxmin etməyə baxacaqsan.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/51/?loc=az)

## Giriş

Bu dərsdə sən **DVM**: **D**əstək **V**ektor **M**aşını([**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine)) istifadə etməklə reqressiya və ya **SVR: Dəstək Vektor Reqressiyası** üçün xüsusi modellər qurmağı öyrənəcəksən.

### Zaman seriyası məsələlərində SVR [^1]

SVR-ın zaman seriyaları proqnozlarında vacibliyini başa düşməzdən əvvəl bəzi vacib anlayışları bilməyin lazımdır:

- **Reqressiya:** Verilmiş dəyərlər toplusundan davamlı dəyərlərin proqnozlaşdırılması üçün nəzarətli öyrənmə texnikasıdır. Burda əsas məqsəd maksimum sayda data nöqtələrinin uyğunluq əyrisinə (və ya xəttinə) yaxın olmasıdır. Daha ətraflı məlumat üçün [bura klikləyin](https://en.wikipedia.org/wiki/Regression_analysis).

- **Dəstək Vektor Maşını (SVM):** Bu qruplaşdırma, reqressiya və uyğunsuzluqların təyin olunması üçün istifadə olunan nəzarətli maşın öyrənmə modellərindən biridir. Bu model funksiya fəzasında hiperplandır, hansı ki, qruplaşdırma tətbiqində sərhəd kimi? reqressiyada isə ən uyğun xətt kimi rol oynayır. SVM-də Kernel funksiyası əsasən dataseti çox ölçülü fəzaya çevirmək üçün istifadə olunur, beləliklə onları bölmək daha asan olur. SVM-lər barədə daha ətralı məlumat üçün bura [bura klikləyin](https://en.wikipedia.org/wiki/Support-vector_machine).

- **Dəstək Vektor Reqressoru (SVR):** Maksimum sayda data nöqtəsinin ən uyğun gələn xətti (SVM halında hiperplan) tapmaq üçün istifadə olunan SVM tiplərindən biridir.

### Nə üçün SVR? [^1]

Son dərsdə zaman seriya datalarının proqnozlaşdırılması üçün tətbiq olunan uğurlu statistik xətti üsul ARIMA barədə öyrəndin. Lakin bir çox hallarda zaman seriya dataları *qeyri-xətti* olurlar və onları xətti modellərlə uyğunlaşdırmaq olmur Bu hallarda SVM-in qeyri-xətti dataların reqressiya tapşırıqlarda bacarıqlarını nəzərə alaraq SVR-i zaman seriyası proqnozlaşdırılmasında da uğurla istifadə etmək olar.

## Tapşırıq - SVR modeli qur

Data hazırlanmasında ilk addımlar əvvəlki [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) dərsindəki kimidir.

Bu dərsin [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) qovluğunu açın və [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) faylını tapın.[^2]

1. Notbuku icra edin və lazımi kitabxanaları daxil edin:  [^2]

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

2. `/data/energy.csv` faylından dataları Pandas datafreyminə yığın və nəticəyə baxın:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. 2012-ci il yanvar ayından 2014-ci il dekabr ayına kimi olan bütün enerji datalarının qrafikini çəkin: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![bütün data](../images/full-data.png)

   İndi isə SVR modelini quraq.

### Öyrətmə və test datasetlərini yaradın

İndi data yüklənib və sən onu öyrətmə və test setlərinə ayıra bilərsən. Sonra sənin datanı SVR üçün lazım olan zaman addımları ilə ayrılmış datasetə çevirməyin lazım olacaq. Modelini öyrətmə seti ilə öyrədəcəksən. Model öyrənməsi bitdikdən sonra onun dəqiqliyini əvvəlcə öyrənmə və test seti ilə, sonra isə bütün dataset ilə yoxlayıb ümumi performansını ölçəcəksən. Sən əmin olmalısan ki, test setin öyrənmə setin əhatə etdiyi zaman periodundan sonrakı gələcəyi də əhatə edir və modelin gələcək zamandan informasiya almır [^2] (bu vəziyyətə *Overfitting* deyilir).

1. 2014-cü il sentabrın 1-dən oktyabrın 31-ə kimi 2 aylıq periodu öyrənmə seti kimi ayır. Test seti isə növbəti iki ayı, noyabrın 1-dən dekabrın 31-ə kimi olan dataları əhatə edəcək: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Fərqi göstər: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![öyrətmə and test setləri](../images/train-test.png)

### Öyrətmə üçün data hazırla

İndi sənin datanı filtrasiya və miqyasını dəyişmə əməliyyatları tətbiq etməklə öyrətmə üçün hazırlamağın lazımdır. Dataseti filtrasiya edərək yalnız lazım olan zaman periodları və sütunlarını saxla və datanın 0 ilə 1 arasında yerləşdiyinə əmin olacaq şəkildə miqyasını dəyiş.

1. Verilən dataseti əvvəl nəzərdə tutulan zaman periodlarına əsasən filtr et və lazım olan 'load' və tarix sütunlarını saxla: [^2]

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

2. Öyrətmə datasetini (0, 1) miqyasına çevir: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```

4. İndi isə test datasetinin miqyasını dəyiş: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Zaman addımları ilə data yarat [^1]

SVR üçün verilən datanı `[batch, timesteps]` formatına çevirməyin lazımdır. İndi sən mövcud olan `train_data` və `test_data` datasetlərinin formasını yeni ölçüdə - zaman addımları ilə bölünmüş formaya çevirməyin lazımdır.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Bu nümunədə biz `timesteps = 5` parametrini istifadə edirik. Yəni verilən dataların ilk 4 zaman addımındakı datalar modelə giriş kimi istifadə olunacaq və nəticə 5-cü addım üçün data olacaq.

```python
timesteps=5
```
Öyrətmə datasını iç-içə yığılmış siyahıdan istifadə etməklə 2D tensor formasına çeviririk:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Test datasını 2D tensora çevirik:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Öyrətmə və test datalarından giriş və çıxış dəyərlərini seçirik:

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

### SVR tətbiq et [^1]

İndi SVR tətbiq etmək zamanıdır. Bu tətbiq barədə daha ətraflı məlumat üçün [bu sənədə](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) baxa bilərsiniz. Bizim tətbiq üçün bu addımları izləmək lazımdır:

  1. `SVR()` çağıraraq modeli təyin et və modelə bu hiperparametrləri ötür: kernel, gamma, c və epsilon
  2. `fit()` funksiyası ilə modeli öyrətmə datası üçün hazırla
  3. Proqnozları `predict()`  funksiyasını çağırmaqla əldə edə bilərsən

Biz artıq SVR modeli hazırladıq. Burada biz [RBF kerneli](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) istifadə etdik, gamma, C və epsilon hiperparametrlərinə 0.5, 10 və 0.05 dəyərlərini verdik.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Öyrətmə datasına modeli uyğunlaşdırmaq [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Modeldən proqnozlar əldə edin [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Sən SVR modeli yaratdın! Bizə onun dəqiqliyini ölçmək lazımdır.

### Modelin dəqiqliyini ölç [^1]

Yoxlama üçün biz birinci olaraq datanı əvvəlki miqyasına geri qaytarmalıyıq. Sonra, performansı yoxlamaq üçün biz əsl və proqnozlaşdırılan zaman seriyalarının qrafikini çəkəcəyik və MAPE nəticələrini konsola yazacağıq.

Proqnozlaşdırılmış və orijinal datanın miqyasını dəyiş:

```python
# proqnozların miqyasının dəyişdirilməsi
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# orijinal dəyərlərin miqyasının dəyişdirilməsi
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Modelin performansını öyrətmə və test datası ilə yoxla [^1]

Biz datasetdən zaman məlumatlarını qrafikin x xəttində göstərmək üçün götürmüşük. Nəzərə al ki, biz birinci `timesteps-1` dəyərlərini birinci çıxış dəyərlərini hesablamağa giriş kimi istifadə etmişik, yəni bundan sonrakı zaman dəyərləri əsas çıxış dəyərləri üçün istifadə olunacaq.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Öyrətmə datası üzərindən proqnozları qrafikdə çək:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![Öyrətmə datasının proqnozlar](../images/train-data-predict.png)

Öyrətmə datası üçün MAPE dəyərini çap et:

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Yoxlama datası üçün proqnozları qrafikdə çək:

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![Yoxlama datası proqnozları](../images/test-data-predict.png)

Yoxlama datası üçün MAPE dəyərini çap et:

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Sən yoxlama dataseti üçün çox yaxşı nəticə əldə etmisən!

### Modelin performansını bütün dataset üzərindən yoxla [^1]

```python
# yüklənmiş datanı numpy array-ə çevrilməsi
data = energy.copy().values

# miqyasın dəyişdirilməsi
data = scaler.transform(data)

# modelin giriş tələblərinə uyğun 2D tensora çevirmək
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# datadan giriş və çıxış dəyərlərinin seçilməsi
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4)
Y shape:  (26300, 1)
```

```python
# model proqnozları hesabla
Y_pred = model.predict(X).reshape(-1,1)

# miqyası geri qaytar və formasını dəyiş
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

![bütün data proqnozları](../images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```


🏆 Bu gözəl qrafiklər modelin yaxşı dəqiqlikdə olduğunu göstərir. Əla!

---

## 🚀 Məşğələ

- Model yaradarkən və data üzərində yoxlayarkən hiperparameterləri (gamma, C, epsilon) dəyişməyi yoxla və hansı dəyərlər çoxluğunun yoxlama datası ilə daha yaxşı nəticə əldə etdiyini gör. Bu hiperparametrlər barədə daha çox öyrənmək üçün [bu sənədə](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) baxa bilərsən.
- Model üçün fərqli kernel funksiyalarını yoxla və onların dataset üzərində performanslarını analiz et. [Bu sənəd]((https://scikit-learn.org/stable/modules/svm.html#kernel-functions)) çox faydalı ola bilər.
- `timesteps` üçün fərqli dəyərlər yoxla və modelin proqnozlarına diqqət et.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/52/?loc=az)

## Təkrarlayın və özünüz öyrənin

Bu dərs zaman seriyalarında proqnozlaşdırma üçün SVR modelinin tətbiqinə giriş idi. SVR haqqında daha çox oxumaq üçün [bu bloqa](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) baxa bilərsən. [scikit-learn-də bu texniki sənəd](https://scikit-learn.org/stable/modules/svm.html) SVM-lər barədə ümumi olaraq çox detallı məlumatlar verir. Əlavə olaraq [SVR-lar](https://scikit-learn.org/stable/modules/svm.html#regression) və fərqli [kernel funksiyalarının](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) başqa tətbiqləri və parametrləri barədə ətraflı məlumatı da əldə edə bilərsiniz.

## Tapşırıq

[Yeni SVR modeli](assignment.az.md)

## İstinadlar

[^1]: Bu bölmənin mətni, kodu və nəticələri[@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) tərəfindən töhfə verilib
[^2]: Bu bölmənin mətni, kodu və nəticələri [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)-dan götürülüb