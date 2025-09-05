<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T19:59:30+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "id"
}
-->
# Pengantar Klasifikasi

Dalam empat pelajaran ini, Anda akan menjelajahi salah satu fokus utama dari pembelajaran mesin klasik - _klasifikasi_. Kita akan menggunakan berbagai algoritma klasifikasi dengan dataset tentang semua masakan luar biasa dari Asia dan India. Semoga Anda lapar!

![hanya sejumput!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Rayakan masakan pan-Asia dalam pelajaran ini! Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

Klasifikasi adalah bentuk [pembelajaran terawasi](https://wikipedia.org/wiki/Supervised_learning) yang memiliki banyak kesamaan dengan teknik regresi. Jika pembelajaran mesin berfokus pada memprediksi nilai atau nama sesuatu menggunakan dataset, maka klasifikasi umumnya terbagi menjadi dua kelompok: _klasifikasi biner_ dan _klasifikasi multikelas_.

[![Pengantar klasifikasi](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Pengantar klasifikasi")

> 🎥 Klik gambar di atas untuk video: John Guttag dari MIT memperkenalkan klasifikasi

Ingat:

- **Regresi linear** membantu Anda memprediksi hubungan antara variabel dan membuat prediksi akurat tentang di mana titik data baru akan berada dalam hubungan dengan garis tersebut. Misalnya, Anda dapat memprediksi _berapa harga labu pada bulan September vs. Desember_.
- **Regresi logistik** membantu Anda menemukan "kategori biner": pada titik harga ini, _apakah labu ini berwarna oranye atau tidak-oranye_?

Klasifikasi menggunakan berbagai algoritma untuk menentukan cara lain dalam menentukan label atau kelas suatu titik data. Mari kita bekerja dengan data masakan ini untuk melihat apakah, dengan mengamati sekelompok bahan, kita dapat menentukan asal masakannya.

## [Kuis pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Pengantar

Klasifikasi adalah salah satu aktivitas mendasar bagi peneliti pembelajaran mesin dan ilmuwan data. Dari klasifikasi dasar nilai biner ("apakah email ini spam atau tidak?"), hingga klasifikasi dan segmentasi gambar yang kompleks menggunakan penglihatan komputer, selalu berguna untuk dapat mengelompokkan data ke dalam kelas dan mengajukan pertanyaan tentangnya.

Untuk menyatakan proses ini dengan cara yang lebih ilmiah, metode klasifikasi Anda menciptakan model prediktif yang memungkinkan Anda memetakan hubungan antara variabel input ke variabel output.

![klasifikasi biner vs. multikelas](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Masalah biner vs. multikelas untuk algoritma klasifikasi. Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

Sebelum memulai proses membersihkan data, memvisualisasikannya, dan mempersiapkannya untuk tugas ML kita, mari kita pelajari sedikit tentang berbagai cara pembelajaran mesin dapat digunakan untuk mengklasifikasikan data.

Berasal dari [statistik](https://wikipedia.org/wiki/Statistical_classification), klasifikasi menggunakan pembelajaran mesin klasik menggunakan fitur seperti `smoker`, `weight`, dan `age` untuk menentukan _kemungkinan mengembangkan penyakit X_. Sebagai teknik pembelajaran terawasi yang mirip dengan latihan regresi yang Anda lakukan sebelumnya, data Anda diberi label dan algoritma ML menggunakan label tersebut untuk mengklasifikasikan dan memprediksi kelas (atau 'fitur') dari dataset dan menetapkannya ke grup atau hasil.

✅ Luangkan waktu sejenak untuk membayangkan dataset tentang masakan. Apa yang dapat dijawab oleh model multikelas? Apa yang dapat dijawab oleh model biner? Bagaimana jika Anda ingin menentukan apakah suatu masakan kemungkinan menggunakan fenugreek? Bagaimana jika Anda ingin melihat apakah, dengan bahan-bahan seperti bintang adas, artichoke, kembang kol, dan lobak, Anda dapat membuat hidangan khas India?

[![Keranjang misteri yang gila](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Keranjang misteri yang gila")

> 🎥 Klik gambar di atas untuk video. Premis utama dari acara 'Chopped' adalah 'keranjang misteri' di mana koki harus membuat hidangan dari pilihan bahan acak. Tentunya model ML akan sangat membantu!

## Halo 'classifier'

Pertanyaan yang ingin kita ajukan dari dataset masakan ini sebenarnya adalah pertanyaan **multikelas**, karena kita memiliki beberapa kemungkinan masakan nasional untuk dikerjakan. Dengan sekelompok bahan, ke kelas mana data ini akan cocok?

Scikit-learn menawarkan beberapa algoritma berbeda untuk digunakan dalam mengklasifikasikan data, tergantung pada jenis masalah yang ingin Anda selesaikan. Dalam dua pelajaran berikutnya, Anda akan mempelajari beberapa algoritma ini.

## Latihan - bersihkan dan seimbangkan data Anda

Tugas pertama yang harus dilakukan, sebelum memulai proyek ini, adalah membersihkan dan **menyeimbangkan** data Anda untuk mendapatkan hasil yang lebih baik. Mulailah dengan file kosong _notebook.ipynb_ di root folder ini.

Hal pertama yang perlu diinstal adalah [imblearn](https://imbalanced-learn.org/stable/). Ini adalah paket Scikit-learn yang memungkinkan Anda menyeimbangkan data dengan lebih baik (Anda akan mempelajari lebih lanjut tentang tugas ini sebentar lagi).

1. Untuk menginstal `imblearn`, jalankan `pip install`, seperti ini:

    ```python
    pip install imblearn
    ```

1. Impor paket yang Anda perlukan untuk mengimpor data dan memvisualisasikannya, juga impor `SMOTE` dari `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Sekarang Anda siap untuk mengimpor data berikutnya.

1. Tugas berikutnya adalah mengimpor data:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Menggunakan `read_csv()` akan membaca konten file csv _cusines.csv_ dan menempatkannya dalam variabel `df`.

1. Periksa bentuk data:

    ```python
    df.head()
    ```

   Lima baris pertama terlihat seperti ini:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Dapatkan informasi tentang data ini dengan memanggil `info()`:

    ```python
    df.info()
    ```

    Output Anda menyerupai:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Latihan - mempelajari tentang masakan

Sekarang pekerjaan mulai menjadi lebih menarik. Mari kita temukan distribusi data, per masakan.

1. Plot data sebagai batang dengan memanggil `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribusi data masakan](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Ada sejumlah masakan yang terbatas, tetapi distribusi data tidak merata. Anda dapat memperbaikinya! Sebelum melakukannya, jelajahi sedikit lebih jauh.

1. Cari tahu berapa banyak data yang tersedia per masakan dan cetak:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    Outputnya terlihat seperti ini:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Menemukan bahan-bahan

Sekarang Anda dapat menggali lebih dalam ke data dan mempelajari apa saja bahan-bahan khas per masakan. Anda harus membersihkan data berulang yang menciptakan kebingungan antara masakan, jadi mari kita pelajari tentang masalah ini.

1. Buat fungsi `create_ingredient()` dalam Python untuk membuat dataframe bahan. Fungsi ini akan mulai dengan menghapus kolom yang tidak membantu dan menyortir bahan berdasarkan jumlahnya:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Sekarang Anda dapat menggunakan fungsi tersebut untuk mendapatkan gambaran tentang sepuluh bahan paling populer berdasarkan masakan.

1. Panggil `create_ingredient()` dan plot dengan memanggil `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Lakukan hal yang sama untuk data Jepang:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Sekarang untuk bahan-bahan Cina:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plot bahan-bahan India:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Akhirnya, plot bahan-bahan Korea:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Sekarang, hapus bahan-bahan yang paling umum yang menciptakan kebingungan antara masakan yang berbeda, dengan memanggil `drop()`:

   Semua orang menyukai nasi, bawang putih, dan jahe!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Seimbangkan dataset

Setelah Anda membersihkan data, gunakan [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Teknik Oversampling Minoritas Sintetis" - untuk menyeimbangkannya.

1. Panggil `fit_resample()`, strategi ini menghasilkan sampel baru melalui interpolasi.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Dengan menyeimbangkan data Anda, Anda akan mendapatkan hasil yang lebih baik saat mengklasifikasikannya. Pikirkan tentang klasifikasi biner. Jika sebagian besar data Anda adalah satu kelas, model ML akan lebih sering memprediksi kelas tersebut, hanya karena ada lebih banyak data untuk itu. Menyeimbangkan data mengambil data yang miring dan membantu menghilangkan ketidakseimbangan ini.

1. Sekarang Anda dapat memeriksa jumlah label per bahan:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Output Anda terlihat seperti ini:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Data ini sudah bersih, seimbang, dan sangat lezat!

1. Langkah terakhir adalah menyimpan data yang telah seimbang, termasuk label dan fitur, ke dalam dataframe baru yang dapat diekspor ke file:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Anda dapat melihat data sekali lagi menggunakan `transformed_df.head()` dan `transformed_df.info()`. Simpan salinan data ini untuk digunakan dalam pelajaran mendatang:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    CSV baru ini sekarang dapat ditemukan di folder data root.

---

## 🚀Tantangan

Kurikulum ini berisi beberapa dataset yang menarik. Telusuri folder `data` dan lihat apakah ada yang berisi dataset yang cocok untuk klasifikasi biner atau multikelas? Pertanyaan apa yang akan Anda ajukan dari dataset ini?

## [Kuis pasca-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Jelajahi API SMOTE. Untuk kasus penggunaan apa SMOTE paling cocok digunakan? Masalah apa yang dapat diselesaikannya?

## Tugas 

[Telusuri metode klasifikasi](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.