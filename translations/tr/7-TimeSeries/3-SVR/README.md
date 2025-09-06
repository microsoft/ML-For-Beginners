<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-06T07:50:12+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "tr"
}
-->
# Destek Vektör Regresörü ile Zaman Serisi Tahmini

Önceki derste, ARIMA modelini kullanarak zaman serisi tahminleri yapmayı öğrendiniz. Şimdi, sürekli verileri tahmin etmek için kullanılan bir regresör modeli olan Destek Vektör Regresörü modeline bakacağız.

## [Ders Öncesi Testi](https://ff-quizzes.netlify.app/en/ml/) 

## Giriş

Bu derste, regresyon için [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) veya **SVR: Destek Vektör Regresörü** ile model oluşturmanın özel bir yolunu keşfedeceksiniz.

### Zaman Serisi Bağlamında SVR [^1]

SVR'nin zaman serisi tahminindeki önemini anlamadan önce bilmeniz gereken bazı önemli kavramlar şunlardır:

- **Regresyon:** Sürekli değerleri verilen bir dizi girdiden tahmin etmek için kullanılan denetimli öğrenme tekniği. Amaç, özellik uzayında maksimum veri noktası sayısına sahip bir eğri (veya çizgi) oluşturmaktır. Daha fazla bilgi için [buraya tıklayın](https://en.wikipedia.org/wiki/Regression_analysis).
- **Destek Vektör Makinesi (SVM):** Sınıflandırma, regresyon ve aykırı değer tespiti için kullanılan bir tür denetimli makine öğrenimi modeli. Model, özellik uzayında bir hiper düzlemdir; sınıflandırma durumunda bir sınır olarak, regresyon durumunda ise en iyi uyum çizgisi olarak işlev görür. SVM'de, genellikle veri setini daha yüksek boyutlu bir uzaya dönüştürmek için bir Çekirdek fonksiyonu kullanılır, böylece veriler kolayca ayrılabilir hale gelir. SVM'ler hakkında daha fazla bilgi için [buraya tıklayın](https://en.wikipedia.org/wiki/Support-vector_machine).
- **Destek Vektör Regresörü (SVR):** SVM'nin bir türü olup, maksimum veri noktası sayısına sahip en iyi uyum çizgisini (SVM durumunda bir hiper düzlem) bulmayı amaçlar.

### Neden SVR? [^1]

Son derste, zaman serisi verilerini tahmin etmek için çok başarılı bir istatistiksel doğrusal yöntem olan ARIMA'yı öğrendiniz. Ancak, birçok durumda zaman serisi verileri *doğrusal olmayan* özelliklere sahiptir ve bu doğrusal modellerle haritalanamaz. Bu gibi durumlarda, SVM'nin regresyon görevlerinde verilerdeki doğrusal olmayanlığı dikkate alma yeteneği, SVR'yi zaman serisi tahmininde başarılı kılar.

## Egzersiz - SVR modeli oluşturma

Veri hazırlama için ilk birkaç adım, [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) hakkındaki önceki derste olduğu ile aynıdır.

Bu dersteki [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) klasörünü açın ve [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb) dosyasını bulun.[^2]

1. Notebook'u çalıştırın ve gerekli kütüphaneleri içe aktarın: [^2]

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

2. `/data/energy.csv` dosyasından verileri bir Pandas veri çerçevesine yükleyin ve göz atın: [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Ocak 2012'den Aralık 2014'e kadar mevcut tüm enerji verilerini görselleştirin: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![tam veri](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Şimdi SVR modelimizi oluşturalım.

### Eğitim ve test veri setlerini oluşturma

Artık verileriniz yüklendiğine göre, bunları eğitim ve test setlerine ayırabilirsiniz. Daha sonra SVR için gerekli olan zaman adımı tabanlı bir veri seti oluşturmak için verileri yeniden şekillendireceksiniz. Modelinizi eğitim setinde eğiteceksiniz. Model eğitimi tamamlandıktan sonra, modelin doğruluğunu eğitim setinde, test setinde ve ardından genel performansı görmek için tam veri setinde değerlendireceksiniz. Test setinin, modelin gelecekteki zaman dilimlerinden bilgi edinmesini engellemek için eğitim setinden daha sonraki bir dönemi kapsadığından emin olmanız gerekir [^2] (bu duruma *Aşırı Uyum* denir).

1. Eğitim seti için 1 Eylül - 31 Ekim 2014 arasındaki iki aylık dönemi ayırın. Test seti, 1 Kasım - 31 Aralık 2014 arasındaki iki aylık dönemi içerecektir: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Farkları görselleştirin: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![eğitim ve test verileri](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Eğitim için verileri hazırlama

Şimdi, verilerinizi filtreleme ve ölçeklendirme işlemleri yaparak eğitime hazırlamanız gerekiyor. Veri setinizi yalnızca ihtiyaç duyduğunuz zaman dilimlerini ve sütunları içerecek şekilde filtreleyin ve verilerin 0,1 aralığında projeksiyonunu sağlamak için ölçeklendirme yapın.

1. Orijinal veri setini yalnızca belirtilen zaman dilimlerini ve yalnızca gerekli 'load' sütunu ile tarihi içerecek şekilde filtreleyin: [^2]

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
   
2. Eğitim verilerini (0, 1) aralığında ölçeklendirin: [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Şimdi test verilerini ölçeklendirin: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Zaman adımları ile veri oluşturma [^1]

SVR için, giriş verilerini `[batch, timesteps]` formuna dönüştürürsünüz. Bu nedenle, mevcut `train_data` ve `test_data` verilerini yeniden şekillendirerek zaman adımlarını ifade eden yeni bir boyut oluşturursunuz.

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Bu örnek için `timesteps = 5` alıyoruz. Yani, modelin girdileri ilk 4 zaman adımının verileri olacak ve çıktı 5. zaman adımının verileri olacaktır.

```python
timesteps=5
```

Eğitim verilerini iç içe liste anlayışı kullanarak 2D tensöre dönüştürme:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Test verilerini 2D tensöre dönüştürme:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Eğitim ve test verilerinden giriş ve çıkışları seçme:

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

### SVR'yi Uygulama [^1]

Şimdi SVR'yi uygulama zamanı. Bu uygulama hakkında daha fazla bilgi için [bu dokümantasyona](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) başvurabilirsiniz. Uygulamamız için şu adımları takip ediyoruz:

  1. `SVR()` çağırarak ve model hiperparametrelerini (kernel, gamma, c ve epsilon) geçirerek modeli tanımlayın
  2. `fit()` fonksiyonunu çağırarak modeli eğitim verilerine hazırlayın
  3. `predict()` fonksiyonunu çağırarak tahminler yapın

Şimdi bir SVR modeli oluşturuyoruz. Burada [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) kullanıyoruz ve hiperparametreleri gamma, C ve epsilon'u sırasıyla 0.5, 10 ve 0.05 olarak ayarlıyoruz.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Modeli eğitim verileri üzerinde eğitme [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Model tahminleri yapma [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

SVR'nizi oluşturdunuz! Şimdi bunu değerlendirmemiz gerekiyor.

### Modelinizi Değerlendirin [^1]

Değerlendirme için, önce verileri orijinal ölçeğimize geri ölçeklendireceğiz. Ardından, performansı kontrol etmek için orijinal ve tahmin edilen zaman serisi grafiğini çizeceğiz ve ayrıca MAPE sonucunu yazdıracağız.

Tahmin edilen ve orijinal çıktıyı ölçeklendirin:

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

#### Eğitim ve test verilerinde model performansını kontrol edin [^1]

Grafiğimizin x ekseninde göstermek için veri setinden zaman damgalarını çıkarıyoruz. İlk ```timesteps-1``` değerlerini ilk çıktı için giriş olarak kullandığımızı unutmayın, bu nedenle çıktının zaman damgaları bundan sonra başlayacaktır.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Eğitim verileri için tahminleri çizin:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![eğitim verisi tahmini](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Eğitim verileri için MAPE'yi yazdırın

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Test verileri için tahminleri çizin

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![test verisi tahmini](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Test verileri için MAPE'yi yazdırın

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Test veri setinde çok iyi bir sonuç elde ettiniz!

### Tam veri setinde model performansını kontrol edin [^1]

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

![tam veri tahmini](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Çok güzel grafikler, iyi bir doğruluğa sahip bir modeli gösteriyor. Tebrikler!

---

## 🚀Meydan Okuma

- Modeli oluştururken hiperparametreleri (gamma, C, epsilon) değiştirin ve test verilerinde hangi hiperparametre setinin en iyi sonuçları verdiğini değerlendirin. Bu hiperparametreler hakkında daha fazla bilgi için [buradaki](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel) dokümantasyona başvurabilirsiniz.
- Model için farklı çekirdek fonksiyonları kullanmayı deneyin ve veri setindeki performanslarını analiz edin. Yardımcı bir dokümantasyon [burada](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) bulunabilir.
- Modelin tahmin yapması için `timesteps` için farklı değerler kullanmayı deneyin.

## [Ders Sonrası Testi](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu ders, SVR'nin Zaman Serisi Tahmini için uygulanmasını tanıtmayı amaçladı. SVR hakkında daha fazla bilgi için [bu bloga](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/) başvurabilirsiniz. Bu [scikit-learn dokümantasyonu](https://scikit-learn.org/stable/modules/svm.html), genel olarak SVM'ler, [SVR'ler](https://scikit-learn.org/stable/modules/svm.html#regression) ve ayrıca kullanılabilecek farklı [çekirdek fonksiyonları](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) gibi diğer uygulama detayları hakkında daha kapsamlı bir açıklama sağlar.

## Ödev

[Yeni bir SVR modeli](assignment.md)

## Katkılar

[^1]: Bu bölümdeki metin, kod ve çıktı [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD) tarafından katkıda bulunulmuştur.
[^2]: Bu bölümdeki metin, kod ve çıktı [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA) dosyasından alınmıştır.

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.