<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T19:51:03+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "id"
}
-->
# Pengelompokan Masakan 1

Dalam pelajaran ini, Anda akan menggunakan dataset yang telah Anda simpan dari pelajaran sebelumnya, yang berisi data seimbang dan bersih tentang berbagai jenis masakan.

Anda akan menggunakan dataset ini dengan berbagai pengelompokan untuk _memprediksi jenis masakan nasional berdasarkan kelompok bahan_. Sambil melakukannya, Anda akan mempelajari lebih lanjut tentang beberapa cara algoritma dapat digunakan untuk tugas klasifikasi.

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)
# Persiapan

Dengan asumsi Anda telah menyelesaikan [Pelajaran 1](../1-Introduction/README.md), pastikan file _cleaned_cuisines.csv_ ada di folder root `/data` untuk empat pelajaran ini.

## Latihan - memprediksi jenis masakan nasional

1. Bekerja di folder _notebook.ipynb_ pelajaran ini, impor file tersebut bersama dengan pustaka Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data terlihat seperti ini:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sekarang, impor beberapa pustaka lagi:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Pisahkan koordinat X dan y ke dalam dua dataframe untuk pelatihan. `cuisine` dapat menjadi dataframe label:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Data akan terlihat seperti ini:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Hapus kolom `Unnamed: 0` dan kolom `cuisine` dengan memanggil `drop()`. Simpan data lainnya sebagai fitur yang dapat dilatih:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Fitur Anda akan terlihat seperti ini:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sekarang Anda siap untuk melatih model Anda!

## Memilih pengelompokan

Setelah data Anda bersih dan siap untuk pelatihan, Anda harus memutuskan algoritma mana yang akan digunakan untuk tugas ini.

Scikit-learn mengelompokkan klasifikasi di bawah Pembelajaran Terawasi, dan dalam kategori tersebut Anda akan menemukan banyak cara untuk mengelompokkan. [Ragamnya](https://scikit-learn.org/stable/supervised_learning.html) cukup membingungkan pada pandangan pertama. Metode berikut semuanya mencakup teknik klasifikasi:

- Model Linear
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Metode Ensemble (Voting Classifier)
- Algoritma Multikelas dan multioutput (klasifikasi multikelas dan multilabel, klasifikasi multikelas-multioutput)

> Anda juga dapat menggunakan [jaringan saraf untuk mengelompokkan data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), tetapi itu di luar cakupan pelajaran ini.

### Pengelompokan mana yang harus dipilih?

Jadi, pengelompokan mana yang harus Anda pilih? Sering kali, mencoba beberapa dan mencari hasil yang baik adalah cara untuk menguji. Scikit-learn menawarkan [perbandingan berdampingan](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) pada dataset yang dibuat, membandingkan KNeighbors, SVC dua cara, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB, dan QuadraticDiscriminationAnalysis, menunjukkan hasil yang divisualisasikan:

![perbandingan pengelompokan](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafik dihasilkan dari dokumentasi Scikit-learn

> AutoML menyelesaikan masalah ini dengan menjalankan perbandingan ini di cloud, memungkinkan Anda memilih algoritma terbaik untuk data Anda. Coba [di sini](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Pendekatan yang lebih baik

Pendekatan yang lebih baik daripada menebak secara acak adalah mengikuti ide-ide pada [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) yang dapat diunduh ini. Di sini, kita menemukan bahwa, untuk masalah multikelas kita, kita memiliki beberapa pilihan:

![cheatsheet untuk masalah multikelas](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Bagian dari Algorithm Cheat Sheet Microsoft, merinci opsi klasifikasi multikelas

✅ Unduh cheat sheet ini, cetak, dan tempel di dinding Anda!

### Penalaran

Mari kita lihat apakah kita dapat menalar melalui pendekatan yang berbeda mengingat kendala yang kita miliki:

- **Jaringan saraf terlalu berat**. Mengingat dataset kita yang bersih tetapi minimal, dan fakta bahwa kita menjalankan pelatihan secara lokal melalui notebook, jaringan saraf terlalu berat untuk tugas ini.
- **Tidak menggunakan pengelompokan dua kelas**. Kita tidak menggunakan pengelompokan dua kelas, jadi itu mengesampingkan one-vs-all.
- **Decision tree atau logistic regression bisa digunakan**. Decision tree mungkin cocok, atau logistic regression untuk data multikelas.
- **Multiclass Boosted Decision Trees menyelesaikan masalah yang berbeda**. Multiclass boosted decision tree paling cocok untuk tugas nonparametrik, misalnya tugas yang dirancang untuk membangun peringkat, sehingga tidak berguna untuk kita.

### Menggunakan Scikit-learn 

Kita akan menggunakan Scikit-learn untuk menganalisis data kita. Namun, ada banyak cara untuk menggunakan logistic regression di Scikit-learn. Lihat [parameter yang dapat diteruskan](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Pada dasarnya ada dua parameter penting - `multi_class` dan `solver` - yang perlu kita tentukan, saat kita meminta Scikit-learn untuk melakukan logistic regression. Nilai `multi_class` menerapkan perilaku tertentu. Nilai solver adalah algoritma yang akan digunakan. Tidak semua solver dapat dipasangkan dengan semua nilai `multi_class`.

Menurut dokumentasi, dalam kasus multikelas, algoritma pelatihan:

- **Menggunakan skema one-vs-rest (OvR)**, jika opsi `multi_class` diatur ke `ovr`
- **Menggunakan cross-entropy loss**, jika opsi `multi_class` diatur ke `multinomial`. (Saat ini opsi `multinomial` hanya didukung oleh solver ‘lbfgs’, ‘sag’, ‘saga’, dan ‘newton-cg’.)"

> 🎓 'Skema' di sini bisa berupa 'ovr' (one-vs-rest) atau 'multinomial'. Karena logistic regression sebenarnya dirancang untuk mendukung klasifikasi biner, skema ini memungkinkan algoritma tersebut menangani tugas klasifikasi multikelas dengan lebih baik. [sumber](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' didefinisikan sebagai "algoritma yang digunakan dalam masalah optimasi". [sumber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn menawarkan tabel ini untuk menjelaskan bagaimana solver menangani tantangan yang berbeda yang disajikan oleh berbagai jenis struktur data:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Latihan - membagi data

Kita dapat fokus pada logistic regression untuk percobaan pelatihan pertama kita karena Anda baru saja mempelajari tentang hal ini dalam pelajaran sebelumnya.
Pisahkan data Anda menjadi kelompok pelatihan dan pengujian dengan memanggil `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Latihan - menerapkan logistic regression

Karena Anda menggunakan kasus multikelas, Anda perlu memilih _skema_ yang akan digunakan dan _solver_ yang akan diatur. Gunakan LogisticRegression dengan pengaturan multikelas dan solver **liblinear** untuk melatih.

1. Buat logistic regression dengan multi_class diatur ke `ovr` dan solver diatur ke `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Coba solver lain seperti `lbfgs`, yang sering diatur sebagai default
Gunakan fungsi Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) untuk meratakan data Anda jika diperlukan.
Akurasi model ini cukup baik, yaitu di atas **80%**!

1. Anda dapat melihat model ini beraksi dengan menguji satu baris data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Hasilnya dicetak:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Cobalah nomor baris yang berbeda dan periksa hasilnya.

1. Lebih mendalam, Anda dapat memeriksa akurasi prediksi ini:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Hasilnya dicetak - masakan India adalah tebakan terbaiknya, dengan probabilitas yang baik:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Bisakah Anda menjelaskan mengapa model ini cukup yakin bahwa ini adalah masakan India?

1. Dapatkan lebih banyak detail dengan mencetak laporan klasifikasi, seperti yang Anda lakukan dalam pelajaran regresi:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Tantangan

Dalam pelajaran ini, Anda menggunakan data yang telah dibersihkan untuk membangun model pembelajaran mesin yang dapat memprediksi jenis masakan berdasarkan serangkaian bahan. Luangkan waktu untuk membaca berbagai opsi yang disediakan Scikit-learn untuk mengklasifikasikan data. Pelajari lebih dalam konsep 'solver' untuk memahami apa yang terjadi di balik layar.

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Pelajari lebih dalam tentang matematika di balik regresi logistik dalam [pelajaran ini](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tugas 

[Pelajari tentang solver](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.