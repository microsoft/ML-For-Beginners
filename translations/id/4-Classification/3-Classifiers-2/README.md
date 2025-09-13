<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T19:56:49+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "id"
}
-->
# Pengklasifikasi Masakan 2

Dalam pelajaran klasifikasi kedua ini, Anda akan mengeksplorasi lebih banyak cara untuk mengklasifikasikan data numerik. Anda juga akan mempelajari dampak dari memilih satu pengklasifikasi dibandingkan yang lain.

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)

### Prasyarat

Kami mengasumsikan bahwa Anda telah menyelesaikan pelajaran sebelumnya dan memiliki dataset yang telah dibersihkan di folder `data` Anda dengan nama _cleaned_cuisines.csv_ di root folder 4 pelajaran ini.

### Persiapan

Kami telah memuat file _notebook.ipynb_ Anda dengan dataset yang telah dibersihkan dan telah membaginya menjadi dataframe X dan y, siap untuk proses pembangunan model.

## Peta klasifikasi

Sebelumnya, Anda telah mempelajari berbagai opsi yang tersedia untuk mengklasifikasikan data menggunakan lembar contekan dari Microsoft. Scikit-learn menawarkan lembar contekan serupa, tetapi lebih rinci, yang dapat membantu mempersempit pilihan estimator Anda (istilah lain untuk pengklasifikasi):

![Peta ML dari Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [kunjungi peta ini secara online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) dan klik jalurnya untuk membaca dokumentasi.

### Rencana

Peta ini sangat membantu setelah Anda memiliki pemahaman yang jelas tentang data Anda, karena Anda dapat 'berjalan' di sepanjang jalurnya untuk membuat keputusan:

- Kami memiliki >50 sampel
- Kami ingin memprediksi sebuah kategori
- Kami memiliki data yang diberi label
- Kami memiliki kurang dari 100K sampel
- ✨ Kami dapat memilih Linear SVC
- Jika itu tidak berhasil, karena kami memiliki data numerik
    - Kami dapat mencoba ✨ KNeighbors Classifier 
      - Jika itu tidak berhasil, coba ✨ SVC dan ✨ Ensemble Classifiers

Ini adalah jalur yang sangat membantu untuk diikuti.

## Latihan - membagi data

Mengikuti jalur ini, kita harus mulai dengan mengimpor beberapa pustaka yang diperlukan.

1. Impor pustaka yang diperlukan:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Bagi data pelatihan dan pengujian Anda:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Pengklasifikasi Linear SVC

Support-Vector Clustering (SVC) adalah bagian dari keluarga teknik ML Support-Vector Machines (pelajari lebih lanjut tentang ini di bawah). Dalam metode ini, Anda dapat memilih 'kernel' untuk menentukan cara mengelompokkan label. Parameter 'C' mengacu pada 'regularisasi' yang mengatur pengaruh parameter. Kernel dapat berupa [beberapa jenis](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); di sini kami mengaturnya ke 'linear' untuk memastikan bahwa kami menggunakan Linear SVC. Probabilitas secara default diatur ke 'false'; di sini kami mengaturnya ke 'true' untuk mendapatkan estimasi probabilitas. Kami mengatur random state ke '0' untuk mengacak data agar mendapatkan probabilitas.

### Latihan - menerapkan Linear SVC

Mulailah dengan membuat array pengklasifikasi. Anda akan menambahkan secara bertahap ke array ini saat kami menguji.

1. Mulailah dengan Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Latih model Anda menggunakan Linear SVC dan cetak laporan:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Hasilnya cukup baik:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Pengklasifikasi K-Neighbors

K-Neighbors adalah bagian dari keluarga metode ML "neighbors", yang dapat digunakan untuk pembelajaran terawasi dan tidak terawasi. Dalam metode ini, sejumlah titik yang telah ditentukan dibuat dan data dikumpulkan di sekitar titik-titik ini sehingga label yang digeneralisasi dapat diprediksi untuk data tersebut.

### Latihan - menerapkan pengklasifikasi K-Neighbors

Pengklasifikasi sebelumnya cukup baik dan bekerja dengan baik pada data, tetapi mungkin kita bisa mendapatkan akurasi yang lebih baik. Coba pengklasifikasi K-Neighbors.

1. Tambahkan baris ke array pengklasifikasi Anda (tambahkan koma setelah item Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Hasilnya sedikit lebih buruk:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Pelajari tentang [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Pengklasifikasi Support Vector

Pengklasifikasi Support-Vector adalah bagian dari keluarga [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) metode ML yang digunakan untuk tugas klasifikasi dan regresi. SVM "memetakan contoh pelatihan ke titik-titik di ruang" untuk memaksimalkan jarak antara dua kategori. Data berikutnya dipetakan ke ruang ini sehingga kategorinya dapat diprediksi.

### Latihan - menerapkan pengklasifikasi Support Vector

Mari coba mendapatkan akurasi yang sedikit lebih baik dengan pengklasifikasi Support Vector.

1. Tambahkan koma setelah item K-Neighbors, lalu tambahkan baris ini:

    ```python
    'SVC': SVC(),
    ```

    Hasilnya cukup baik!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Pelajari tentang [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Pengklasifikasi Ensemble

Mari ikuti jalur hingga akhir, meskipun pengujian sebelumnya cukup baik. Mari coba beberapa 'Pengklasifikasi Ensemble', khususnya Random Forest dan AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Hasilnya sangat baik, terutama untuk Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Pelajari tentang [Pengklasifikasi Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Metode Machine Learning ini "menggabungkan prediksi dari beberapa estimator dasar" untuk meningkatkan kualitas model. Dalam contoh kami, kami menggunakan Random Trees dan AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), metode rata-rata, membangun 'hutan' dari 'pohon keputusan' yang diinfuskan dengan elemen acak untuk menghindari overfitting. Parameter n_estimators diatur ke jumlah pohon.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) menyesuaikan pengklasifikasi ke dataset dan kemudian menyesuaikan salinan pengklasifikasi tersebut ke dataset yang sama. Metode ini berfokus pada bobot item yang diklasifikasikan secara salah dan menyesuaikan fit untuk pengklasifikasi berikutnya untuk memperbaiki.

---

## 🚀Tantangan

Setiap teknik ini memiliki sejumlah besar parameter yang dapat Anda sesuaikan. Teliti parameter default masing-masing dan pikirkan apa arti penyesuaian parameter ini untuk kualitas model.

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Ada banyak istilah teknis dalam pelajaran ini, jadi luangkan waktu untuk meninjau [daftar ini](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) istilah yang berguna!

## Tugas 

[Parameter play](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.