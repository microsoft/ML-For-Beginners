<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T19:59:55+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada klasifikasi

Dalam empat pelajaran ini, anda akan meneroka fokus asas pembelajaran mesin klasik - _klasifikasi_. Kita akan menggunakan pelbagai algoritma klasifikasi dengan dataset tentang semua masakan hebat dari Asia dan India. Semoga anda lapar!

![hanya secubit!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Raikan masakan pan-Asia dalam pelajaran ini! Imej oleh [Jen Looper](https://twitter.com/jenlooper)

Klasifikasi adalah satu bentuk [pembelajaran terkawal](https://wikipedia.org/wiki/Supervised_learning) yang mempunyai banyak persamaan dengan teknik regresi. Jika pembelajaran mesin berkaitan dengan meramalkan nilai atau nama sesuatu menggunakan dataset, maka klasifikasi biasanya terbahagi kepada dua kumpulan: _klasifikasi binari_ dan _klasifikasi pelbagai kelas_.

[![Pengenalan kepada klasifikasi](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Pengenalan kepada klasifikasi")

> 🎥 Klik imej di atas untuk video: John Guttag dari MIT memperkenalkan klasifikasi

Ingat:

- **Regresi linear** membantu anda meramalkan hubungan antara pemboleh ubah dan membuat ramalan tepat tentang di mana titik data baru akan berada dalam hubungan dengan garis tersebut. Sebagai contoh, anda boleh meramalkan _berapa harga labu pada bulan September berbanding Disember_.
- **Regresi logistik** membantu anda menemui "kategori binari": pada titik harga ini, _adakah labu ini berwarna oren atau tidak-oren_?

Klasifikasi menggunakan pelbagai algoritma untuk menentukan cara lain dalam menentukan label atau kelas titik data. Mari kita bekerja dengan data masakan ini untuk melihat sama ada, dengan memerhatikan sekumpulan bahan, kita boleh menentukan asal-usul masakannya.

## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Pengenalan

Klasifikasi adalah salah satu aktiviti asas bagi penyelidik pembelajaran mesin dan saintis data. Daripada klasifikasi asas nilai binari ("adakah e-mel ini spam atau tidak?"), kepada klasifikasi imej kompleks dan segmentasi menggunakan penglihatan komputer, sentiasa berguna untuk dapat menyusun data ke dalam kelas dan bertanya soalan mengenainya.

Untuk menyatakan proses ini dengan cara yang lebih saintifik, kaedah klasifikasi anda mencipta model ramalan yang membolehkan anda memetakan hubungan antara pemboleh ubah input kepada pemboleh ubah output.

![klasifikasi binari vs. pelbagai kelas](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Masalah binari vs. pelbagai kelas untuk algoritma klasifikasi. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

Sebelum memulakan proses membersihkan data, memvisualisasikannya, dan menyediakan data untuk tugas ML kita, mari kita belajar sedikit tentang pelbagai cara pembelajaran mesin boleh digunakan untuk mengklasifikasikan data.

Berasal daripada [statistik](https://wikipedia.org/wiki/Statistical_classification), klasifikasi menggunakan pembelajaran mesin klasik menggunakan ciri-ciri seperti `smoker`, `weight`, dan `age` untuk menentukan _kemungkinan menghidap penyakit X_. Sebagai teknik pembelajaran terkawal yang serupa dengan latihan regresi yang anda lakukan sebelum ini, data anda dilabelkan dan algoritma ML menggunakan label tersebut untuk mengklasifikasikan dan meramalkan kelas (atau 'ciri') dataset dan menetapkannya kepada kumpulan atau hasil.

✅ Luangkan masa untuk membayangkan dataset tentang masakan. Apakah yang boleh dijawab oleh model pelbagai kelas? Apakah yang boleh dijawab oleh model binari? Bagaimana jika anda ingin menentukan sama ada sesuatu masakan cenderung menggunakan fenugreek? Bagaimana jika anda ingin melihat sama ada, dengan kehadiran beg runcit penuh dengan bunga lawang, artichoke, kembang kol, dan lobak pedas, anda boleh mencipta hidangan India yang tipikal?

[![Bakul misteri yang gila](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Bakul misteri yang gila")

> 🎥 Klik imej di atas untuk video. Premis utama rancangan 'Chopped' adalah 'bakul misteri' di mana chef perlu membuat hidangan daripada pilihan bahan yang rawak. Pasti model ML akan membantu!

## Hello 'classifier'

Soalan yang ingin kita tanyakan kepada dataset masakan ini sebenarnya adalah soalan **pelbagai kelas**, kerana kita mempunyai beberapa kemungkinan masakan kebangsaan untuk diterokai. Berdasarkan sekumpulan bahan, kelas mana daripada banyak kelas ini yang sesuai dengan data?

Scikit-learn menawarkan beberapa algoritma berbeza untuk digunakan bagi mengklasifikasikan data, bergantung kepada jenis masalah yang ingin anda selesaikan. Dalam dua pelajaran seterusnya, anda akan belajar tentang beberapa algoritma ini.

## Latihan - bersihkan dan seimbangkan data anda

Tugas pertama sebelum memulakan projek ini adalah membersihkan dan **menyeimbangkan** data anda untuk mendapatkan hasil yang lebih baik. Mulakan dengan fail kosong _notebook.ipynb_ dalam root folder ini.

Perkara pertama yang perlu dipasang ialah [imblearn](https://imbalanced-learn.org/stable/). Ini adalah pakej Scikit-learn yang akan membolehkan anda menyeimbangkan data dengan lebih baik (anda akan belajar lebih lanjut tentang tugas ini sebentar lagi).

1. Untuk memasang `imblearn`, jalankan `pip install`, seperti berikut:

    ```python
    pip install imblearn
    ```

1. Import pakej yang anda perlukan untuk mengimport data anda dan memvisualisasikannya, juga import `SMOTE` daripada `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Sekarang anda sudah bersedia untuk membaca dan mengimport data seterusnya.

1. Tugas seterusnya adalah mengimport data:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Menggunakan `read_csv()` akan membaca kandungan fail csv _cusines.csv_ dan meletakkannya dalam pemboleh ubah `df`.

1. Periksa bentuk data:

    ```python
    df.head()
    ```

   Lima baris pertama kelihatan seperti ini:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Dapatkan maklumat tentang data ini dengan memanggil `info()`:

    ```python
    df.info()
    ```

    Output anda menyerupai:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Latihan - belajar tentang masakan

Sekarang kerja mula menjadi lebih menarik. Mari kita temui pengagihan data, mengikut masakan.

1. Plot data sebagai bar dengan memanggil `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![pengagihan data masakan](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Terdapat bilangan masakan yang terhad, tetapi pengagihan data tidak sekata. Anda boleh membetulkannya! Sebelum melakukannya, terokai sedikit lagi.

1. Ketahui berapa banyak data yang tersedia bagi setiap masakan dan cetak:

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

    Output kelihatan seperti ini:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Menemui bahan-bahan

Sekarang anda boleh menyelami data dengan lebih mendalam dan mengetahui apakah bahan-bahan tipikal bagi setiap masakan. Anda harus membersihkan data berulang yang mencipta kekeliruan antara masakan, jadi mari kita belajar tentang masalah ini.

1. Cipta fungsi `create_ingredient()` dalam Python untuk mencipta dataframe bahan. Fungsi ini akan bermula dengan membuang lajur yang tidak membantu dan menyusun bahan mengikut kiraannya:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Sekarang anda boleh menggunakan fungsi itu untuk mendapatkan idea tentang sepuluh bahan paling popular mengikut masakan.

1. Panggil `create_ingredient()` dan plot dengan memanggil `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Lakukan perkara yang sama untuk data Jepun:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![jepun](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Sekarang untuk bahan Cina:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![cina](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Plot bahan India:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![india](../../../../4-Classification/1-Introduction/images/indian.png)

1. Akhir sekali, plot bahan Korea:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korea](../../../../4-Classification/1-Introduction/images/korean.png)

1. Sekarang, buang bahan yang paling biasa yang mencipta kekeliruan antara masakan yang berbeza, dengan memanggil `drop()`:

   Semua orang suka nasi, bawang putih dan halia!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Seimbangkan dataset

Sekarang setelah anda membersihkan data, gunakan [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Teknik Pengambilan Sampel Lebihan Minoriti Sintetik" - untuk menyeimbangkannya.

1. Panggil `fit_resample()`, strategi ini menjana sampel baru melalui interpolasi.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Dengan menyeimbangkan data anda, anda akan mendapat hasil yang lebih baik semasa mengklasifikasikannya. Fikirkan tentang klasifikasi binari. Jika kebanyakan data anda adalah satu kelas, model ML akan meramalkan kelas itu lebih kerap, hanya kerana terdapat lebih banyak data untuknya. Menyeimbangkan data mengambil data yang berat sebelah dan membantu menghapuskan ketidakseimbangan ini.

1. Sekarang anda boleh memeriksa bilangan label bagi setiap bahan:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Output anda kelihatan seperti ini:

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

    Data kini bersih, seimbang, dan sangat lazat!

1. Langkah terakhir adalah menyimpan data yang seimbang, termasuk label dan ciri, ke dalam dataframe baru yang boleh dieksport ke fail:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Anda boleh melihat data sekali lagi menggunakan `transformed_df.head()` dan `transformed_df.info()`. Simpan salinan data ini untuk digunakan dalam pelajaran masa depan:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    CSV segar ini kini boleh didapati dalam folder data root.

---

## 🚀Cabaran

Kurikulum ini mengandungi beberapa dataset yang menarik. Terokai folder `data` dan lihat sama ada terdapat dataset yang sesuai untuk klasifikasi binari atau pelbagai kelas? Apakah soalan yang akan anda tanyakan kepada dataset ini?

## [Kuiz pasca-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Terokai API SMOTE. Apakah kes penggunaan yang paling sesuai untuknya? Apakah masalah yang diselesaikannya?

## Tugasan 

[Terokai kaedah klasifikasi](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.