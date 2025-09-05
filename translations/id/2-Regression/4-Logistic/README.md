<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T18:47:51+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "id"
}
-->
# Regresi Logistik untuk Memprediksi Kategori

![Infografik regresi logistik vs regresi linear](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Pendahuluan

Dalam pelajaran terakhir tentang Regresi ini, salah satu teknik ML _klasik_ dasar, kita akan mempelajari Regresi Logistik. Anda dapat menggunakan teknik ini untuk menemukan pola guna memprediksi kategori biner. Apakah permen ini cokelat atau bukan? Apakah penyakit ini menular atau tidak? Apakah pelanggan ini akan memilih produk ini atau tidak?

Dalam pelajaran ini, Anda akan mempelajari:

- Perpustakaan baru untuk visualisasi data
- Teknik untuk regresi logistik

✅ Perdalam pemahaman Anda tentang bekerja dengan jenis regresi ini di [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Prasyarat

Setelah bekerja dengan data labu, kita sekarang cukup familiar untuk menyadari bahwa ada satu kategori biner yang dapat kita gunakan: `Color`.

Mari kita bangun model regresi logistik untuk memprediksi, berdasarkan beberapa variabel, _warna apa yang kemungkinan besar dimiliki oleh labu tertentu_ (oranye 🎃 atau putih 👻).

> Mengapa kita membahas klasifikasi biner dalam pelajaran tentang regresi? Hanya untuk kenyamanan linguistik, karena regresi logistik sebenarnya adalah [metode klasifikasi](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), meskipun berbasis linear. Pelajari cara lain untuk mengklasifikasikan data di kelompok pelajaran berikutnya.

## Tentukan Pertanyaan

Untuk tujuan kita, kita akan mengekspresikan ini sebagai biner: 'Putih' atau 'Bukan Putih'. Ada juga kategori 'striped' dalam dataset kita, tetapi jumlahnya sedikit, jadi kita tidak akan menggunakannya. Kategori ini juga akan hilang setelah kita menghapus nilai null dari dataset.

> 🎃 Fakta menyenangkan, kita kadang-kadang menyebut labu putih sebagai labu 'hantu'. Mereka tidak mudah diukir, jadi tidak sepopuler labu oranye, tetapi mereka terlihat keren! Jadi kita juga bisa merumuskan ulang pertanyaan kita sebagai: 'Hantu' atau 'Bukan Hantu'. 👻

## Tentang Regresi Logistik

Regresi logistik berbeda dari regresi linear, yang telah Anda pelajari sebelumnya, dalam beberapa cara penting.

[![ML untuk pemula - Memahami Regresi Logistik untuk Klasifikasi Pembelajaran Mesin](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML untuk pemula - Memahami Regresi Logistik untuk Klasifikasi Pembelajaran Mesin")

> 🎥 Klik gambar di atas untuk video singkat tentang regresi logistik.

### Klasifikasi Biner

Regresi logistik tidak menawarkan fitur yang sama seperti regresi linear. Regresi logistik memberikan prediksi tentang kategori biner ("putih atau bukan putih"), sedangkan regresi linear mampu memprediksi nilai kontinu, misalnya berdasarkan asal labu dan waktu panen, _berapa harga labu akan naik_.

![Model Klasifikasi Labu](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Klasifikasi Lainnya

Ada jenis regresi logistik lainnya, termasuk multinomial dan ordinal:

- **Multinomial**, yang melibatkan lebih dari satu kategori - "Oranye, Putih, dan Striped".
- **Ordinal**, yang melibatkan kategori yang terurut, berguna jika kita ingin mengurutkan hasil secara logis, seperti labu kita yang diurutkan berdasarkan sejumlah ukuran tertentu (mini, sm, med, lg, xl, xxl).

![Regresi multinomial vs ordinal](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Variabel TIDAK Harus Berkorelasi

Ingat bagaimana regresi linear bekerja lebih baik dengan variabel yang lebih berkorelasi? Regresi logistik adalah kebalikannya - variabel tidak harus sejajar. Ini cocok untuk data ini yang memiliki korelasi yang agak lemah.

### Anda Membutuhkan Banyak Data yang Bersih

Regresi logistik akan memberikan hasil yang lebih akurat jika Anda menggunakan lebih banyak data; dataset kecil kita tidak optimal untuk tugas ini, jadi ingatlah hal itu.

[![ML untuk pemula - Analisis dan Persiapan Data untuk Regresi Logistik](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML untuk pemula - Analisis dan Persiapan Data untuk Regresi Logistik")

✅ Pikirkan jenis data yang cocok untuk regresi logistik

## Latihan - rapikan data

Pertama, bersihkan data sedikit, hapus nilai null, dan pilih hanya beberapa kolom:

1. Tambahkan kode berikut:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Anda selalu dapat melihat sekilas dataframe baru Anda:

    ```python
    pumpkins.info
    ```

### Visualisasi - plot kategorikal

Sekarang Anda telah memuat [notebook awal](../../../../2-Regression/4-Logistic/notebook.ipynb) dengan data labu sekali lagi dan membersihkannya sehingga hanya menyisakan dataset yang berisi beberapa variabel, termasuk `Color`. Mari kita visualisasikan dataframe di notebook menggunakan pustaka yang berbeda: [Seaborn](https://seaborn.pydata.org/index.html), yang dibangun di atas Matplotlib yang kita gunakan sebelumnya.

Seaborn menawarkan beberapa cara menarik untuk memvisualisasikan data Anda. Misalnya, Anda dapat membandingkan distribusi data untuk setiap `Variety` dan `Color` dalam plot kategorikal.

1. Buat plot seperti itu dengan menggunakan fungsi `catplot`, menggunakan data labu kita `pumpkins`, dan menentukan pemetaan warna untuk setiap kategori labu (oranye atau putih):

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![Grid data yang divisualisasikan](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    Dengan mengamati data, Anda dapat melihat bagaimana data Color berhubungan dengan Variety.

    ✅ Berdasarkan plot kategorikal ini, eksplorasi menarik apa yang dapat Anda bayangkan?

### Pra-pemrosesan Data: Pengkodean Fitur dan Label

Dataset labu kita berisi nilai string untuk semua kolomnya. Bekerja dengan data kategorikal intuitif bagi manusia tetapi tidak bagi mesin. Algoritma pembelajaran mesin bekerja dengan baik dengan angka. Itulah mengapa pengkodean adalah langkah yang sangat penting dalam fase pra-pemrosesan data, karena memungkinkan kita mengubah data kategorikal menjadi data numerik, tanpa kehilangan informasi apa pun. Pengkodean yang baik menghasilkan model yang baik.

Untuk pengkodean fitur, ada dua jenis pengkode utama:

1. Pengkode ordinal: cocok untuk variabel ordinal, yaitu variabel kategorikal di mana datanya mengikuti urutan logis, seperti kolom `Item Size` dalam dataset kita. Pengkode ini membuat pemetaan sehingga setiap kategori diwakili oleh angka, yang merupakan urutan kategori dalam kolom.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Pengkode kategorikal: cocok untuk variabel nominal, yaitu variabel kategorikal di mana datanya tidak mengikuti urutan logis, seperti semua fitur selain `Item Size` dalam dataset kita. Ini adalah pengkodean one-hot, yang berarti bahwa setiap kategori diwakili oleh kolom biner: variabel yang dikodekan sama dengan 1 jika labu termasuk dalam Variety tersebut dan 0 jika tidak.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Kemudian, `ColumnTransformer` digunakan untuk menggabungkan beberapa pengkode ke dalam satu langkah dan menerapkannya ke kolom yang sesuai.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Di sisi lain, untuk mengkodekan label, kita menggunakan kelas `LabelEncoder` dari scikit-learn, yang merupakan kelas utilitas untuk membantu menormalkan label sehingga hanya berisi nilai antara 0 dan n_classes-1 (di sini, 0 dan 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Setelah kita mengkodekan fitur dan label, kita dapat menggabungkannya ke dalam dataframe baru `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Apa keuntungan menggunakan pengkode ordinal untuk kolom `Item Size`?

### Analisis Hubungan Antar Variabel

Sekarang kita telah memproses data kita, kita dapat menganalisis hubungan antara fitur dan label untuk mendapatkan gambaran seberapa baik model akan dapat memprediksi label berdasarkan fitur.
Cara terbaik untuk melakukan analisis semacam ini adalah dengan memplot data. Kita akan menggunakan kembali fungsi `catplot` dari Seaborn, untuk memvisualisasikan hubungan antara `Item Size`, `Variety`, dan `Color` dalam plot kategorikal. Untuk memplot data dengan lebih baik, kita akan menggunakan kolom `Item Size` yang telah dikodekan dan kolom `Variety` yang belum dikodekan.

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```
![Plot kategorikal data yang divisualisasikan](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Gunakan Plot Swarm

Karena Color adalah kategori biner (Putih atau Tidak), kategori ini membutuhkan '[pendekatan khusus](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) untuk visualisasi'. Ada cara lain untuk memvisualisasikan hubungan kategori ini dengan variabel lainnya.

Anda dapat memvisualisasikan variabel secara berdampingan dengan plot Seaborn.

1. Cobalah plot 'swarm' untuk menunjukkan distribusi nilai:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm data yang divisualisasikan](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Perhatikan**: kode di atas mungkin menghasilkan peringatan, karena seaborn gagal merepresentasikan sejumlah besar titik data ke dalam plot swarm. Solusi yang mungkin adalah mengurangi ukuran penanda, dengan menggunakan parameter 'size'. Namun, perlu diingat bahwa ini memengaruhi keterbacaan plot.

> **🧮 Tunjukkan Matematikanya**
>
> Regresi logistik bergantung pada konsep 'maximum likelihood' menggunakan [fungsi sigmoid](https://wikipedia.org/wiki/Sigmoid_function). Fungsi 'Sigmoid' pada plot terlihat seperti bentuk 'S'. Fungsi ini mengambil nilai dan memetakannya ke antara 0 dan 1. Kurvanya juga disebut 'kurva logistik'. Rumusnya terlihat seperti ini:
>
> ![fungsi logistik](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> di mana titik tengah sigmoid berada di titik 0 x, L adalah nilai maksimum kurva, dan k adalah kemiringan kurva. Jika hasil fungsi lebih dari 0.5, label yang dimaksud akan diberi kelas '1' dari pilihan biner. Jika tidak, akan diklasifikasikan sebagai '0'.

## Bangun Model Anda

Membangun model untuk menemukan klasifikasi biner ini ternyata cukup sederhana di Scikit-learn.

[![ML untuk pemula - Regresi Logistik untuk klasifikasi data](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML untuk pemula - Regresi Logistik untuk klasifikasi data")

> 🎥 Klik gambar di atas untuk video singkat tentang membangun model regresi linear

1. Pilih variabel yang ingin Anda gunakan dalam model klasifikasi Anda dan bagi set pelatihan dan pengujian dengan memanggil `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Sekarang Anda dapat melatih model Anda, dengan memanggil `fit()` dengan data pelatihan Anda, dan mencetak hasilnya:

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    Lihatlah skor model Anda. Tidak buruk, mengingat Anda hanya memiliki sekitar 1000 baris data:

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## Pemahaman Lebih Baik melalui Matriks Kebingungan

Meskipun Anda dapat mendapatkan laporan skor [istilah](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) dengan mencetak item di atas, Anda mungkin dapat memahami model Anda lebih mudah dengan menggunakan [matriks kebingungan](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) untuk membantu kita memahami bagaimana model bekerja.

> 🎓 '[Matriks kebingungan](https://wikipedia.org/wiki/Confusion_matrix)' (atau 'matriks kesalahan') adalah tabel yang mengekspresikan positif dan negatif sejati vs. salah model Anda, sehingga mengukur akurasi prediksi.

1. Untuk menggunakan matriks kebingungan, panggil `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Lihatlah matriks kebingungan model Anda:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Di Scikit-learn, baris (axis 0) adalah label aktual dan kolom (axis 1) adalah label yang diprediksi.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Apa yang terjadi di sini? Misalnya model kita diminta untuk mengklasifikasikan labu antara dua kategori biner, kategori 'putih' dan kategori 'bukan putih'.

- Jika model Anda memprediksi labu sebagai bukan putih dan sebenarnya termasuk kategori 'bukan putih', kita menyebutnya negatif sejati, ditunjukkan oleh angka kiri atas.
- Jika model Anda memprediksi labu sebagai putih dan sebenarnya termasuk kategori 'bukan putih', kita menyebutnya negatif palsu, ditunjukkan oleh angka kiri bawah.
- Jika model Anda memprediksi labu sebagai bukan putih dan sebenarnya termasuk kategori 'putih', kita menyebutnya positif palsu, ditunjukkan oleh angka kanan atas.
- Jika model Anda memprediksi labu sebagai putih dan sebenarnya termasuk kategori 'putih', kita menyebutnya positif sejati, ditunjukkan oleh angka kanan bawah.

Seperti yang mungkin Anda duga, lebih baik memiliki jumlah positif sejati dan negatif sejati yang lebih besar serta jumlah positif palsu dan negatif palsu yang lebih kecil, yang menunjukkan bahwa model bekerja lebih baik.
Bagaimana matriks kebingungan berhubungan dengan presisi dan recall? Ingat, laporan klasifikasi yang dicetak di atas menunjukkan presisi (0.85) dan recall (0.67).

Presisi = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Recall = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Q: Berdasarkan matriks kebingungan, bagaimana performa model? A: Tidak buruk; ada sejumlah besar true negatives tetapi juga beberapa false negatives.

Mari kita tinjau kembali istilah-istilah yang telah kita lihat sebelumnya dengan bantuan pemetaan TP/TN dan FP/FN dari matriks kebingungan:

🎓 Presisi: TP/(TP + FP) Fraksi dari instance yang relevan di antara instance yang diambil (misalnya, label mana yang diberi label dengan baik)

🎓 Recall: TP/(TP + FN) Fraksi dari instance yang relevan yang diambil, baik diberi label dengan baik atau tidak

🎓 f1-score: (2 * presisi * recall)/(presisi + recall) Rata-rata tertimbang dari presisi dan recall, dengan nilai terbaik adalah 1 dan terburuk adalah 0

🎓 Support: Jumlah kemunculan setiap label yang diambil

🎓 Akurasi: (TP + TN)/(TP + TN + FP + FN) Persentase label yang diprediksi dengan akurat untuk sebuah sampel.

🎓 Macro Avg: Perhitungan rata-rata metrik yang tidak berbobot untuk setiap label, tanpa memperhatikan ketidakseimbangan label.

🎓 Weighted Avg: Perhitungan rata-rata metrik untuk setiap label, dengan memperhatikan ketidakseimbangan label dengan memberi bobot berdasarkan support (jumlah instance yang benar untuk setiap label).

✅ Bisakah Anda memikirkan metrik mana yang harus diperhatikan jika Anda ingin model Anda mengurangi jumlah false negatives?

## Visualisasi kurva ROC dari model ini

[![ML untuk pemula - Menganalisis Performa Logistic Regression dengan Kurva ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML untuk pemula - Menganalisis Performa Logistic Regression dengan Kurva ROC")

> 🎥 Klik gambar di atas untuk video singkat tentang kurva ROC

Mari kita lakukan satu visualisasi lagi untuk melihat apa yang disebut 'kurva ROC':

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Menggunakan Matplotlib, plot [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) atau ROC dari model. Kurva ROC sering digunakan untuk mendapatkan gambaran keluaran dari sebuah classifier dalam hal true positives vs. false positives. "Kurva ROC biasanya menampilkan true positive rate pada sumbu Y, dan false positive rate pada sumbu X." Oleh karena itu, kemiringan kurva dan ruang antara garis tengah dan kurva menjadi penting: Anda menginginkan kurva yang cepat naik dan melewati garis. Dalam kasus kita, ada false positives di awal, dan kemudian garis naik dan melewati dengan baik:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Akhirnya, gunakan [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) dari Scikit-learn untuk menghitung 'Area Under the Curve' (AUC) yang sebenarnya:

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Hasilnya adalah `0.9749908725812341`. Mengingat bahwa AUC berkisar dari 0 hingga 1, Anda menginginkan skor yang besar, karena model yang 100% benar dalam prediksinya akan memiliki AUC sebesar 1; dalam kasus ini, modelnya _cukup baik_.

Dalam pelajaran klasifikasi di masa depan, Anda akan belajar bagaimana mengiterasi untuk meningkatkan skor model Anda. Tetapi untuk saat ini, selamat! Anda telah menyelesaikan pelajaran regresi ini!

---
## 🚀Tantangan

Masih banyak yang bisa dipelajari tentang logistic regression! Tetapi cara terbaik untuk belajar adalah dengan bereksperimen. Temukan dataset yang cocok untuk analisis jenis ini dan bangun model dengannya. Apa yang Anda pelajari? tip: coba [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) untuk dataset yang menarik.

## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Baca beberapa halaman pertama dari [makalah ini dari Stanford](https://web.stanford.edu/~jurafsky/slp3/5.pdf) tentang beberapa penggunaan praktis logistic regression. Pikirkan tentang tugas-tugas yang lebih cocok untuk salah satu jenis regresi yang telah kita pelajari sejauh ini. Apa yang akan bekerja paling baik?

## Tugas 

[Ulangi regresi ini](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.