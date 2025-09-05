<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-05T19:08:34+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "id"
}
-->
# Peramalan Deret Waktu dengan Support Vector Regressor

Pada pelajaran sebelumnya, Anda telah mempelajari cara menggunakan model ARIMA untuk membuat prediksi deret waktu. Sekarang Anda akan mempelajari model Support Vector Regressor, yaitu model regresi yang digunakan untuk memprediksi data kontinu.

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/) 

## Pendahuluan

Dalam pelajaran ini, Anda akan mempelajari cara spesifik untuk membangun model dengan [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) untuk regresi, atau **SVR: Support Vector Regressor**. 

### SVR dalam konteks deret waktu [^1]

Sebelum memahami pentingnya SVR dalam prediksi deret waktu, berikut adalah beberapa konsep penting yang perlu Anda ketahui:

- **Regresi:** Teknik pembelajaran terawasi untuk memprediksi nilai kontinu dari kumpulan input yang diberikan. Ide utamanya adalah menyesuaikan kurva (atau garis) di ruang fitur yang memiliki jumlah titik data maksimum. [Klik di sini](https://en.wikipedia.org/wiki/Regression_analysis) untuk informasi lebih lanjut.
- **Support Vector Machine (SVM):** Jenis model pembelajaran mesin terawasi yang digunakan untuk klasifikasi, regresi, dan deteksi outlier. Model ini adalah hyperplane di ruang fitur, yang dalam kasus klasifikasi bertindak sebagai batas, dan dalam kasus regresi bertindak sebagai garis terbaik. Dalam SVM, fungsi Kernel biasanya digunakan untuk mentransformasi dataset ke ruang dengan jumlah dimensi yang lebih tinggi, sehingga dapat dengan mudah dipisahkan. [Klik di sini](https://en.wikipedia.org/wiki/Support-vector_machine) untuk informasi lebih lanjut tentang SVM.
- **Support Vector Regressor (SVR):** Jenis SVM yang digunakan untuk menemukan garis terbaik (yang dalam kasus SVM adalah hyperplane) yang memiliki jumlah titik data maksimum.

### Mengapa SVR? [^1]

Pada pelajaran sebelumnya, Anda telah mempelajari tentang ARIMA, yang merupakan metode statistik linear yang sangat sukses untuk meramalkan data deret waktu. Namun, dalam banyak kasus, data deret waktu memiliki *non-linearitas*, yang tidak dapat dipetakan oleh model linear. Dalam kasus seperti itu, kemampuan SVM untuk mempertimbangkan non-linearitas dalam data untuk tugas regresi membuat SVR berhasil dalam peramalan deret waktu.

## Latihan - membangun model SVR

Langkah-langkah awal untuk persiapan data sama seperti pelajaran sebelumnya tentang [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Buka folder [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) dalam pelajaran ini dan temukan file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. Jalankan notebook dan impor pustaka yang diperlukan:  [^2]

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

2. Muat data dari file `/data/energy.csv` ke dalam dataframe Pandas dan lihat:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Plot semua data energi yang tersedia dari Januari 2012 hingga Desember 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data lengkap](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Sekarang, mari kita membangun model SVR kita.

### Membuat dataset pelatihan dan pengujian

Sekarang data Anda telah dimuat, Anda dapat memisahkannya menjadi set pelatihan dan pengujian. Kemudian Anda akan merubah bentuk data untuk membuat dataset berbasis langkah waktu yang akan diperlukan untuk SVR. Anda akan melatih model Anda pada set pelatihan. Setelah model selesai dilatih, Anda akan mengevaluasi akurasinya pada set pelatihan, set pengujian, dan kemudian dataset lengkap untuk melihat kinerja keseluruhan. Anda perlu memastikan bahwa set pengujian mencakup periode waktu yang lebih baru dari set pelatihan untuk memastikan bahwa model tidak mendapatkan informasi dari periode waktu di masa depan [^2] (situasi yang dikenal sebagai *Overfitting*).

1. Alokasikan periode dua bulan dari 1 September hingga 31 Oktober 2014 untuk set pelatihan. Set pengujian akan mencakup periode dua bulan dari 1 November hingga 31 Desember 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Visualisasikan perbedaannya: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![data pelatihan dan pengujian](../../../../7-TimeSeries/3-SVR/images/train-test.png)

### Mempersiapkan data untuk pelatihan

Sekarang, Anda perlu mempersiapkan data untuk pelatihan dengan melakukan penyaringan dan penskalaan data Anda. Saring dataset Anda untuk hanya menyertakan periode waktu dan kolom yang Anda butuhkan, serta penskalaan untuk memastikan data diproyeksikan dalam interval 0,1.

1. Saring dataset asli untuk hanya menyertakan periode waktu yang disebutkan sebelumnya per set dan hanya menyertakan kolom 'load' yang diperlukan serta tanggal: [^2]

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
   
2. Skala data pelatihan agar berada dalam rentang (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Sekarang, skala data pengujian: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Membuat data dengan langkah waktu [^1]

Untuk SVR, Anda mentransformasi data input menjadi bentuk `[batch, timesteps]`. Jadi, Anda merubah bentuk `train_data` dan `test_data` yang ada sehingga ada dimensi baru yang merujuk pada langkah waktu. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Untuk contoh ini, kita mengambil `timesteps = 5`. Jadi, input ke model adalah data untuk 4 langkah waktu pertama, dan outputnya adalah data untuk langkah waktu ke-5.

```python
timesteps=5
```

Mengubah data pelatihan menjadi tensor 2D menggunakan nested list comprehension:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Mengubah data pengujian menjadi tensor 2D:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Memilih input dan output dari data pelatihan dan pengujian:

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

### Mengimplementasikan SVR [^1]

Sekarang, saatnya mengimplementasikan SVR. Untuk membaca lebih lanjut tentang implementasi ini, Anda dapat merujuk ke [dokumentasi ini](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Untuk implementasi kita, kita mengikuti langkah-langkah berikut:

  1. Definisikan model dengan memanggil `SVR()` dan memasukkan hyperparameter model: kernel, gamma, c, dan epsilon
  2. Siapkan model untuk data pelatihan dengan memanggil fungsi `fit()`
  3. Lakukan prediksi dengan memanggil fungsi `predict()`

Sekarang kita membuat model SVR. Di sini kita menggunakan [kernel RBF](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), dan menetapkan hyperparameter gamma, C, dan epsilon masing-masing sebagai 0.5, 10, dan 0.05.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Melatih model pada data pelatihan [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Melakukan prediksi dengan model [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Anda telah membangun SVR Anda! Sekarang kita perlu mengevaluasinya.

### Mengevaluasi model Anda [^1]

Untuk evaluasi, pertama-tama kita akan mengembalikan skala data ke skala asli kita. Kemudian, untuk memeriksa kinerja, kita akan memplot grafik deret waktu asli dan prediksi, serta mencetak hasil MAPE.

Mengembalikan skala output prediksi dan asli:

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

#### Memeriksa kinerja model pada data pelatihan dan pengujian [^1]

Kita mengekstrak timestamp dari dataset untuk ditampilkan di sumbu x plot kita. Perhatikan bahwa kita menggunakan ```timesteps-1``` nilai pertama sebagai input untuk output pertama, sehingga timestamp untuk output akan dimulai setelah itu.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Plot prediksi untuk data pelatihan:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![prediksi data pelatihan](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Cetak MAPE untuk data pelatihan

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Plot prediksi untuk data pengujian

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![prediksi data pengujian](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Cetak MAPE untuk data pengujian

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Anda mendapatkan hasil yang sangat baik pada dataset pengujian!

### Memeriksa kinerja model pada dataset lengkap [^1]

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

![prediksi data lengkap](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```

🏆 Plot yang sangat bagus, menunjukkan model dengan akurasi yang baik. Kerja bagus!

---

## 🚀Tantangan

- Cobalah untuk mengubah hyperparameter (gamma, C, epsilon) saat membuat model dan evaluasi pada data untuk melihat set hyperparameter mana yang memberikan hasil terbaik pada data pengujian. Untuk mengetahui lebih lanjut tentang hyperparameter ini, Anda dapat merujuk ke dokumen [di sini](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Cobalah menggunakan fungsi kernel yang berbeda untuk model dan analisis kinerjanya pada dataset. Dokumen yang berguna dapat ditemukan [di sini](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Cobalah menggunakan nilai yang berbeda untuk `timesteps` agar model dapat melihat ke belakang untuk membuat prediksi.

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Pelajaran ini bertujuan untuk memperkenalkan aplikasi SVR untuk Peramalan Deret Waktu. Untuk membaca lebih lanjut tentang SVR, Anda dapat merujuk ke [blog ini](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). [Dokumentasi tentang scikit-learn](https://scikit-learn.org/stable/modules/svm.html) memberikan penjelasan yang lebih komprehensif tentang SVM secara umum, [SVR](https://scikit-learn.org/stable/modules/svm.html#regression) dan juga detail implementasi lainnya seperti [fungsi kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) yang dapat digunakan, dan parameter-parameter mereka.

## Tugas

[Model SVR baru](assignment.md)

## Kredit

[^1]: Teks, kode, dan output dalam bagian ini disumbangkan oleh [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Teks, kode, dan output dalam bagian ini diambil dari [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.