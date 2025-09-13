<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T19:09:03+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "ms"
}
-->
# Ramalan Siri Masa dengan Support Vector Regressor

Dalam pelajaran sebelumnya, anda telah belajar cara menggunakan model ARIMA untuk membuat ramalan siri masa. Kini anda akan melihat model Support Vector Regressor, iaitu model regresi yang digunakan untuk meramalkan data berterusan.

## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/) 

## Pengenalan

Dalam pelajaran ini, anda akan meneroka cara khusus untuk membina model dengan [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) untuk regresi, atau **SVR: Support Vector Regressor**. 

### SVR dalam konteks siri masa [^1]

Sebelum memahami kepentingan SVR dalam ramalan siri masa, berikut adalah beberapa konsep penting yang perlu anda ketahui:

- **Regresi:** Teknik pembelajaran terarah untuk meramalkan nilai berterusan daripada set input yang diberikan. Idea utamanya adalah untuk memadankan lengkung (atau garis) dalam ruang ciri yang mempunyai bilangan titik data maksimum. [Klik di sini](https://en.wikipedia.org/wiki/Regression_analysis) untuk maklumat lanjut.
- **Support Vector Machine (SVM):** Jenis model pembelajaran mesin terarah yang digunakan untuk klasifikasi, regresi dan pengesanan pencilan. Model ini adalah hyperplane dalam ruang ciri, yang dalam kes klasifikasi bertindak sebagai sempadan, dan dalam kes regresi bertindak sebagai garis terbaik. Dalam SVM, fungsi Kernel biasanya digunakan untuk mengubah dataset ke ruang dengan bilangan dimensi yang lebih tinggi supaya ia mudah dipisahkan. [Klik di sini](https://en.wikipedia.org/wiki/Support-vector_machine) untuk maklumat lanjut tentang SVM.
- **Support Vector Regressor (SVR):** Jenis SVM, untuk mencari garis terbaik (yang dalam kes SVM adalah hyperplane) yang mempunyai bilangan titik data maksimum.

### Mengapa SVR? [^1]

Dalam pelajaran sebelumnya, anda telah belajar tentang ARIMA, yang merupakan kaedah linear statistik yang sangat berjaya untuk meramalkan data siri masa. Walau bagaimanapun, dalam banyak kes, data siri masa mempunyai *ketidaklinearan*, yang tidak dapat dipetakan oleh model linear. Dalam kes sebegini, keupayaan SVM untuk mempertimbangkan ketidaklinearan dalam data untuk tugas regresi menjadikan SVR berjaya dalam ramalan siri masa.

## Latihan - bina model SVR

Langkah-langkah awal untuk penyediaan data adalah sama seperti pelajaran sebelumnya tentang [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Buka folder [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) dalam pelajaran ini dan cari fail [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. Jalankan notebook dan import pustaka yang diperlukan:  [^2]

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

2. Muatkan data dari fail `/data/energy.csv` ke dalam dataframe Pandas dan lihat:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plot semua data tenaga yang tersedia dari Januari 2012 hingga Disember 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data penuh](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Sekarang, mari kita bina model SVR kita.

### Cipta dataset latihan dan ujian

Sekarang data anda telah dimuatkan, anda boleh memisahkannya kepada set latihan dan ujian. Kemudian anda akan mengubah bentuk data untuk mencipta dataset berdasarkan langkah masa yang diperlukan untuk SVR. Anda akan melatih model anda pada set latihan. Selepas model selesai dilatih, anda akan menilai ketepatannya pada set latihan, set ujian dan kemudian dataset penuh untuk melihat prestasi keseluruhan. Anda perlu memastikan bahawa set ujian merangkumi tempoh masa yang lebih lewat daripada set latihan untuk memastikan model tidak mendapat maklumat daripada tempoh masa akan datang [^2] (situasi yang dikenali sebagai *Overfitting*).

1. Peruntukkan tempoh dua bulan dari 1 September hingga 31 Oktober 2014 kepada set latihan. Set ujian akan merangkumi tempoh dua bulan dari 1 November hingga 31 Disember 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisasikan perbezaan: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data latihan dan ujian](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Sediakan data untuk latihan

Sekarang, anda perlu menyediakan data untuk latihan dengan melakukan penapisan dan penskalaan data anda. Tapis dataset anda untuk hanya memasukkan tempoh masa dan lajur yang diperlukan, serta penskalaan untuk memastikan data diproyeksikan dalam julat 0,1.

1. Tapis dataset asal untuk hanya memasukkan tempoh masa yang disebutkan bagi setiap set dan hanya memasukkan lajur 'load' yang diperlukan serta tarikh: [^2]

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
   
2. Skala data latihan untuk berada dalam julat (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Sekarang, skala data ujian: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Cipta data dengan langkah masa [^1]

Untuk SVR, anda mengubah data input supaya berbentuk `[batch, timesteps]`. Jadi, anda mengubah bentuk `train_data` dan `test_data` yang sedia ada supaya terdapat dimensi baharu yang merujuk kepada langkah masa. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Untuk contoh ini, kita ambil `timesteps = 5`. Jadi, input kepada model adalah data untuk 4 langkah masa pertama, dan output akan menjadi data untuk langkah masa ke-5.

```python
timesteps=5
```

Menukar data latihan kepada tensor 2D menggunakan list comprehension bersarang:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Menukar data ujian kepada tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Memilih input dan output daripada data latihan dan ujian:

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

### Laksanakan SVR [^1]

Sekarang, tiba masanya untuk melaksanakan SVR. Untuk membaca lebih lanjut tentang pelaksanaan ini, anda boleh merujuk kepada [dokumentasi ini](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Untuk pelaksanaan kita, kita ikuti langkah-langkah ini:

  1. Tentukan model dengan memanggil `SVR()` dan memasukkan hyperparameter model: kernel, gamma, c dan epsilon
  2. Sediakan model untuk data latihan dengan memanggil fungsi `fit()`
  3. Buat ramalan dengan memanggil fungsi `predict()`

Sekarang kita cipta model SVR. Di sini kita gunakan [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), dan tetapkan hyperparameter gamma, C dan epsilon sebagai 0.5, 10 dan 0.05 masing-masing.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Fit model pada data latihan [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Buat ramalan model [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Anda telah membina SVR anda! Sekarang kita perlu menilainya.

### Nilai model anda [^1]

Untuk penilaian, pertama kita akan skala semula data kepada skala asal kita. Kemudian, untuk memeriksa prestasi, kita akan plot siri masa asal dan yang diramalkan, serta mencetak hasil MAPE.

Skala semula output yang diramalkan dan asal:

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

#### Periksa prestasi model pada data latihan dan ujian [^1]

Kita ekstrak cap masa daripada dataset untuk ditunjukkan pada paksi-x plot kita. Perhatikan bahawa kita menggunakan ```timesteps-1``` nilai pertama sebagai input untuk output pertama, jadi cap masa untuk output akan bermula selepas itu.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plot ramalan untuk data latihan:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![ramalan data latihan](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Cetak MAPE untuk data latihan

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plot ramalan untuk data ujian

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![ramalan data ujian](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Cetak MAPE untuk data ujian

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Anda mendapat keputusan yang sangat baik pada dataset ujian!

### Periksa prestasi model pada dataset penuh [^1]

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

![ramalan data penuh](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Plot yang sangat bagus, menunjukkan model dengan ketepatan yang baik. Syabas!

---

## 🚀Cabaran

- Cuba ubah hyperparameter (gamma, C, epsilon) semasa mencipta model dan nilai pada data untuk melihat set hyperparameter mana yang memberikan keputusan terbaik pada data ujian. Untuk mengetahui lebih lanjut tentang hyperparameter ini, anda boleh merujuk kepada dokumen [di sini](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Cuba gunakan fungsi kernel yang berbeza untuk model dan analisis prestasi mereka pada dataset. Dokumen yang berguna boleh didapati [di sini](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Cuba gunakan nilai yang berbeza untuk `timesteps` untuk model melihat ke belakang untuk membuat ramalan.

## [Kuiz pasca-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Pelajaran ini bertujuan untuk memperkenalkan aplikasi SVR untuk Ramalan Siri Masa. Untuk membaca lebih lanjut tentang SVR, anda boleh merujuk kepada [blog ini](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). [Dokumentasi ini pada scikit-learn](https://scikit-learn.org/stable/modules/svm.html) menyediakan penjelasan yang lebih komprehensif tentang SVM secara umum, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) dan juga butiran pelaksanaan lain seperti [fungsi kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) yang berbeza yang boleh digunakan, dan parameternya.

## Tugasan

[Model SVR baharu](assignment.md)

## Kredit

[^1]: Teks, kod dan output dalam bahagian ini disumbangkan oleh [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Teks, kod dan output dalam bahagian ini diambil daripada [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.