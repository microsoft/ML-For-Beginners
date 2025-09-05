<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T18:48:36+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "ms"
}
-->
# Regresi Logistik untuk Meramal Kategori

![Infografik regresi logistik vs. regresi linear](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [Kuiz Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## Pengenalan

Dalam pelajaran terakhir mengenai Regresi, salah satu teknik asas _klasik_ ML, kita akan melihat Regresi Logistik. Anda boleh menggunakan teknik ini untuk mengenal pasti pola bagi meramal kategori binari. Adakah gula-gula ini coklat atau tidak? Adakah penyakit ini berjangkit atau tidak? Adakah pelanggan ini akan memilih produk ini atau tidak?

Dalam pelajaran ini, anda akan belajar:

- Perpustakaan baru untuk visualisasi data
- Teknik untuk regresi logistik

✅ Tingkatkan pemahaman anda tentang bekerja dengan jenis regresi ini dalam [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)

## Prasyarat

Setelah bekerja dengan data labu, kita kini cukup biasa dengannya untuk menyedari bahawa terdapat satu kategori binari yang boleh kita gunakan: `Color`.

Mari bina model regresi logistik untuk meramal, berdasarkan beberapa pemboleh ubah, _warna labu yang mungkin_ (oren 🎃 atau putih 👻).

> Mengapa kita bercakap tentang klasifikasi binari dalam pelajaran mengenai regresi? Hanya untuk kemudahan linguistik, kerana regresi logistik adalah [sebenarnya kaedah klasifikasi](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression), walaupun berasaskan linear. Ketahui cara lain untuk mengklasifikasikan data dalam kumpulan pelajaran seterusnya.

## Tentukan Soalan

Untuk tujuan kita, kita akan menyatakannya sebagai binari: 'Putih' atau 'Bukan Putih'. Terdapat juga kategori 'berjalur' dalam dataset kita tetapi terdapat sedikit contoh, jadi kita tidak akan menggunakannya. Ia akan hilang apabila kita membuang nilai null daripada dataset, bagaimanapun.

> 🎃 Fakta menarik, kita kadang-kadang memanggil labu putih sebagai labu 'hantu'. Ia tidak mudah untuk diukir, jadi ia tidak sepopular labu oren tetapi ia kelihatan menarik! Jadi kita juga boleh merumuskan semula soalan kita sebagai: 'Hantu' atau 'Bukan Hantu'. 👻

## Mengenai Regresi Logistik

Regresi logistik berbeza daripada regresi linear, yang telah anda pelajari sebelum ini, dalam beberapa cara penting.

[![ML untuk pemula - Memahami Regresi Logistik untuk Klasifikasi Pembelajaran Mesin](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "ML untuk pemula - Memahami Regresi Logistik untuk Klasifikasi Pembelajaran Mesin")

> 🎥 Klik imej di atas untuk video ringkas mengenai regresi logistik.

### Klasifikasi Binari

Regresi logistik tidak menawarkan ciri yang sama seperti regresi linear. Yang pertama menawarkan ramalan tentang kategori binari ("putih atau bukan putih") manakala yang kedua mampu meramal nilai berterusan, contohnya berdasarkan asal labu dan masa penuaian, _berapa banyak harganya akan meningkat_.

![Model Klasifikasi Labu](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

### Klasifikasi Lain

Terdapat jenis regresi logistik lain, termasuk multinomial dan ordinal:

- **Multinomial**, yang melibatkan lebih daripada satu kategori - "Oren, Putih, dan Berjalur".
- **Ordinal**, yang melibatkan kategori yang diatur, berguna jika kita ingin mengatur hasil kita secara logik, seperti labu kita yang diatur mengikut bilangan saiz terhingga (mini, sm, med, lg, xl, xxl).

![Regresi multinomial vs ordinal](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### Pemboleh ubah TIDAK Perlu Berkorelasi

Ingat bagaimana regresi linear berfungsi lebih baik dengan pemboleh ubah yang lebih berkorelasi? Regresi logistik adalah sebaliknya - pemboleh ubah tidak perlu sejajar. Ini berfungsi untuk data ini yang mempunyai korelasi yang agak lemah.

### Anda Memerlukan Banyak Data Bersih

Regresi logistik akan memberikan hasil yang lebih tepat jika anda menggunakan lebih banyak data; dataset kecil kita tidak optimum untuk tugas ini, jadi ingatlah perkara ini.

[![ML untuk pemula - Analisis dan Penyediaan Data untuk Regresi Logistik](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "ML untuk pemula - Analisis dan Penyediaan Data untuk Regresi Logistik")

> 🎥 Klik imej di atas untuk video ringkas mengenai penyediaan data untuk regresi linear

✅ Fikirkan jenis data yang sesuai untuk regresi logistik

## Latihan - kemas kini data

Mula-mula, bersihkan data sedikit, buang nilai null dan pilih hanya beberapa lajur:

1. Tambahkan kod berikut:

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    Anda sentiasa boleh melihat dataframe baru anda:

    ```python
    pumpkins.info
    ```

### Visualisasi - plot kategori

Pada masa ini anda telah memuatkan [notebook permulaan](../../../../2-Regression/4-Logistic/notebook.ipynb) dengan data labu sekali lagi dan membersihkannya untuk mengekalkan dataset yang mengandungi beberapa pemboleh ubah, termasuk `Color`. Mari visualisasikan dataframe dalam notebook menggunakan perpustakaan yang berbeza: [Seaborn](https://seaborn.pydata.org/index.html), yang dibina di atas Matplotlib yang kita gunakan sebelum ini.

Seaborn menawarkan beberapa cara menarik untuk memvisualisasikan data anda. Sebagai contoh, anda boleh membandingkan taburan data untuk setiap `Variety` dan `Color` dalam plot kategori.

1. Buat plot sedemikian dengan menggunakan fungsi `catplot`, menggunakan data labu kita `pumpkins`, dan menentukan pemetaan warna untuk setiap kategori labu (oren atau putih):

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

    Dengan memerhatikan data, anda boleh melihat bagaimana data Color berkaitan dengan Variety.

    ✅ Berdasarkan plot kategori ini, apakah beberapa penerokaan menarik yang boleh anda bayangkan?

### Pra-pemprosesan data: pengekodan ciri dan label
Dataset labu kita mengandungi nilai string untuk semua lajur. Bekerja dengan data kategori adalah intuitif untuk manusia tetapi tidak untuk mesin. Algoritma pembelajaran mesin berfungsi dengan baik dengan nombor. Itulah sebabnya pengekodan adalah langkah yang sangat penting dalam fasa pra-pemprosesan data, kerana ia membolehkan kita menukar data kategori kepada data berangka, tanpa kehilangan sebarang maklumat. Pengekodan yang baik membawa kepada pembinaan model yang baik.

Untuk pengekodan ciri terdapat dua jenis pengekod utama:

1. Pengekod ordinal: sesuai untuk pemboleh ubah ordinal, iaitu pemboleh ubah kategori di mana datanya mengikuti susunan logik, seperti lajur `Item Size` dalam dataset kita. Ia mencipta pemetaan supaya setiap kategori diwakili oleh nombor, yang merupakan susunan kategori dalam lajur.

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. Pengekod kategori: sesuai untuk pemboleh ubah nominal, iaitu pemboleh ubah kategori di mana datanya tidak mengikuti susunan logik, seperti semua ciri yang berbeza daripada `Item Size` dalam dataset kita. Ia adalah pengekodan satu-haba, yang bermaksud bahawa setiap kategori diwakili oleh lajur binari: pemboleh ubah yang dikodkan adalah sama dengan 1 jika labu tergolong dalam Variety itu dan 0 sebaliknya.

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
Kemudian, `ColumnTransformer` digunakan untuk menggabungkan beberapa pengekod ke dalam satu langkah dan menerapkannya pada lajur yang sesuai.

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
Sebaliknya, untuk mengekod label, kita menggunakan kelas `LabelEncoder` scikit-learn, yang merupakan kelas utiliti untuk membantu menormalkan label supaya ia hanya mengandungi nilai antara 0 dan n_classes-1 (di sini, 0 dan 1).

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
Setelah kita mengekod ciri dan label, kita boleh menggabungkannya ke dalam dataframe baru `encoded_pumpkins`.

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ Apakah kelebihan menggunakan pengekod ordinal untuk lajur `Item Size`?

### Analisis hubungan antara pemboleh ubah

Sekarang kita telah memproses data kita, kita boleh menganalisis hubungan antara ciri dan label untuk mendapatkan idea tentang sejauh mana model akan dapat meramal label berdasarkan ciri.
Cara terbaik untuk melakukan analisis jenis ini adalah dengan memplotkan data. Kita akan menggunakan semula fungsi `catplot` Seaborn, untuk memvisualisasikan hubungan antara `Item Size`, `Variety` dan `Color` dalam plot kategori. Untuk memplotkan data dengan lebih baik kita akan menggunakan lajur `Item Size` yang dikodkan dan lajur `Variety` yang tidak dikodkan.

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
![Catplot data yang divisualisasikan](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### Gunakan plot swarm

Oleh kerana Color adalah kategori binari (Putih atau Tidak), ia memerlukan 'pendekatan [khusus](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) untuk visualisasi'. Terdapat cara lain untuk memvisualisasikan hubungan kategori ini dengan pemboleh ubah lain.

Anda boleh memvisualisasikan pemboleh ubah secara bersebelahan dengan plot Seaborn.

1. Cuba plot 'swarm' untuk menunjukkan taburan nilai:

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![Swarm data yang divisualisasikan](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**Perhatian**: kod di atas mungkin menghasilkan amaran, kerana seaborn gagal mewakili sejumlah besar titik data ke dalam plot swarm. Penyelesaian yang mungkin adalah mengurangkan saiz penanda, dengan menggunakan parameter 'size'. Walau bagaimanapun, sedar bahawa ini mempengaruhi kebolehbacaan plot.

> **🧮 Tunjukkan Matematik**
>
> Regresi logistik bergantung pada konsep 'kebolehjadian maksimum' menggunakan [fungsi sigmoid](https://wikipedia.org/wiki/Sigmoid_function). Fungsi 'Sigmoid' pada plot kelihatan seperti bentuk 'S'. Ia mengambil nilai dan memetakan ke suatu tempat antara 0 dan 1. Lengkungnya juga dipanggil 'lengkung logistik'. Formula kelihatan seperti ini:
>
> ![fungsi logistik](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> di mana titik tengah sigmoid berada pada titik 0 x, L adalah nilai maksimum lengkung, dan k adalah kecuraman lengkung. Jika hasil fungsi lebih daripada 0.5, label yang dimaksudkan akan diberikan kelas '1' daripada pilihan binari. Jika tidak, ia akan diklasifikasikan sebagai '0'.

## Bina model anda

Membina model untuk mencari klasifikasi binari ini adalah agak mudah dalam Scikit-learn.

[![ML untuk pemula - Regresi Logistik untuk klasifikasi data](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "ML untuk pemula - Regresi Logistik untuk klasifikasi data")

> 🎥 Klik imej di atas untuk video ringkas mengenai membina model regresi linear

1. Pilih pemboleh ubah yang ingin anda gunakan dalam model klasifikasi anda dan bahagikan set latihan dan ujian dengan memanggil `train_test_split()`:

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. Kini anda boleh melatih model anda, dengan memanggil `fit()` dengan data latihan anda, dan cetak hasilnya:

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

    Lihat papan skor model anda. Ia tidak buruk, memandangkan anda hanya mempunyai kira-kira 1000 baris data:

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

## Pemahaman yang lebih baik melalui matriks kekeliruan

Walaupun anda boleh mendapatkan laporan papan skor [terma](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report) dengan mencetak item di atas, anda mungkin dapat memahami model anda dengan lebih mudah dengan menggunakan [matriks kekeliruan](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) untuk membantu kita memahami bagaimana model berfungsi.

> 🎓 '[Matriks kekeliruan](https://wikipedia.org/wiki/Confusion_matrix)' (atau 'matriks ralat') ialah jadual yang menyatakan positif dan negatif sebenar vs. palsu model anda, dengan itu mengukur ketepatan ramalan.

1. Untuk menggunakan matriks kekeliruan, panggil `confusion_matrix()`:

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    Lihat matriks kekeliruan model anda:

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Dalam Scikit-learn, baris matriks kekeliruan (paksi 0) adalah label sebenar dan lajur (paksi 1) adalah label yang diramal.

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

Apa yang berlaku di sini? Katakan model kita diminta untuk mengklasifikasikan labu antara dua kategori binari, kategori 'putih' dan kategori 'bukan putih'.

- Jika model anda meramal labu sebagai bukan putih dan ia tergolong dalam kategori 'bukan putih' dalam realiti kita memanggilnya negatif benar, ditunjukkan oleh nombor kiri atas.
- Jika model anda meramal labu sebagai putih dan ia tergolong dalam kategori 'bukan putih' dalam realiti kita memanggilnya negatif palsu, ditunjukkan oleh nombor kiri bawah. 
- Jika model anda meramal labu sebagai bukan putih dan ia tergolong dalam kategori 'putih' dalam realiti kita memanggilnya positif palsu, ditunjukkan oleh nombor kanan atas. 
- Jika model anda meramal labu sebagai putih dan ia tergolong dalam kategori 'putih' dalam realiti kita memanggilnya positif benar, ditunjukkan oleh nombor kanan bawah.

Seperti yang anda mungkin telah teka, adalah lebih baik untuk mempunyai bilangan positif benar dan negatif benar yang lebih besar dan bilangan positif palsu dan negatif palsu yang lebih rendah, yang menunjukkan bahawa model berfungsi dengan lebih baik.
Bagaimana matriks kekeliruan berkaitan dengan ketepatan dan ingatan? Ingat, laporan klasifikasi yang dicetak di atas menunjukkan ketepatan (0.85) dan ingatan (0.67).

Ketepatan = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

Ingatan = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ S: Berdasarkan matriks kekeliruan, bagaimana prestasi model? J: Tidak buruk; terdapat sejumlah besar negatif benar tetapi juga beberapa negatif palsu.

Mari kita ulang semula istilah yang kita lihat sebelum ini dengan bantuan pemetaan TP/TN dan FP/FN dalam matriks kekeliruan:

🎓 Ketepatan: TP/(TP + FP) Bahagian contoh yang relevan di antara contoh yang diambil (contohnya, label mana yang dilabel dengan baik)

🎓 Ingatan: TP/(TP + FN) Bahagian contoh yang relevan yang diambil, sama ada dilabel dengan baik atau tidak

🎓 Skor f1: (2 * ketepatan * ingatan)/(ketepatan + ingatan) Purata berwajaran antara ketepatan dan ingatan, dengan yang terbaik adalah 1 dan yang terburuk adalah 0

🎓 Sokongan: Bilangan kejadian bagi setiap label yang diambil

🎓 Ketepatan: (TP + TN)/(TP + TN + FP + FN) Peratusan label yang diramal dengan tepat untuk satu sampel.

🎓 Purata Makro: Pengiraan purata metrik tanpa berat bagi setiap label, tanpa mengambil kira ketidakseimbangan label.

🎓 Purata Berwajaran: Pengiraan purata metrik bagi setiap label, mengambil kira ketidakseimbangan label dengan menimbangnya berdasarkan sokongan mereka (bilangan contoh benar bagi setiap label).

✅ Bolehkah anda fikirkan metrik mana yang perlu diperhatikan jika anda mahu model anda mengurangkan bilangan negatif palsu?

## Visualisasi lengkung ROC model ini

[![ML untuk pemula - Menganalisis Prestasi Regresi Logistik dengan Lengkung ROC](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML untuk pemula - Menganalisis Prestasi Regresi Logistik dengan Lengkung ROC")


> 🎥 Klik imej di atas untuk video ringkas mengenai lengkung ROC

Mari kita lakukan satu lagi visualisasi untuk melihat apa yang dipanggil 'ROC' curve:

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

Menggunakan Matplotlib, plotkan [Receiving Operating Characteristic](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) atau ROC model. Lengkung ROC sering digunakan untuk mendapatkan pandangan tentang output pengklasifikasi dari segi positif benar vs. positif palsu. "Lengkung ROC biasanya memaparkan kadar positif benar pada paksi Y, dan kadar positif palsu pada paksi X." Oleh itu, kecuraman lengkung dan ruang antara garis tengah dan lengkung adalah penting: anda mahukan lengkung yang cepat naik dan melepasi garis. Dalam kes kita, terdapat positif palsu pada permulaan, dan kemudian garis naik dan melepasi dengan betul:

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

Akhirnya, gunakan API [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) Scikit-learn untuk mengira 'Area Under the Curve' (AUC) sebenar:

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
Hasilnya ialah `0.9749908725812341`. Memandangkan AUC berkisar dari 0 hingga 1, anda mahukan skor yang besar, kerana model yang 100% betul dalam ramalannya akan mempunyai AUC sebanyak 1; dalam kes ini, model ini _agak baik_.

Dalam pelajaran masa depan mengenai klasifikasi, anda akan belajar bagaimana untuk mengulangi proses bagi meningkatkan skor model anda. Tetapi buat masa ini, tahniah! Anda telah menyelesaikan pelajaran regresi ini!

---
## 🚀Cabaran

Masih banyak lagi yang boleh diterokai mengenai regresi logistik! Tetapi cara terbaik untuk belajar adalah dengan bereksperimen. Cari dataset yang sesuai untuk jenis analisis ini dan bina model dengannya. Apa yang anda pelajari? tip: cuba [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) untuk dataset yang menarik.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Baca beberapa halaman pertama [kertas kerja dari Stanford ini](https://web.stanford.edu/~jurafsky/slp3/5.pdf) mengenai beberapa kegunaan praktikal untuk regresi logistik. Fikirkan tentang tugas yang lebih sesuai untuk satu jenis regresi berbanding yang lain yang telah kita pelajari setakat ini. Apa yang akan berfungsi dengan baik?

## Tugasan 

[Ulangi regresi ini](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat penting, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.