<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T19:14:54+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada pengelompokan

Pengelompokan adalah sejenis [Pembelajaran Tanpa Pengawasan](https://wikipedia.org/wiki/Unsupervised_learning) yang mengandaikan bahawa dataset tidak berlabel atau inputnya tidak dipadankan dengan output yang telah ditentukan. Ia menggunakan pelbagai algoritma untuk menyusun data yang tidak berlabel dan menyediakan kumpulan berdasarkan corak yang dikenalpasti dalam data.

[![No One Like You oleh PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You oleh PSquare")

> 🎥 Klik imej di atas untuk video. Sambil anda belajar pembelajaran mesin dengan pengelompokan, nikmati beberapa lagu Dance Hall Nigeria - ini adalah lagu yang sangat popular dari tahun 2014 oleh PSquare.

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

### Pengenalan

[Pengelompokan](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) sangat berguna untuk penerokaan data. Mari kita lihat jika ia boleh membantu mengenal pasti trend dan corak dalam cara penonton Nigeria menikmati muzik.

✅ Luangkan masa untuk memikirkan kegunaan pengelompokan. Dalam kehidupan sebenar, pengelompokan berlaku apabila anda mempunyai timbunan pakaian dan perlu menyusun pakaian ahli keluarga anda 🧦👕👖🩲. Dalam sains data, pengelompokan berlaku apabila cuba menganalisis pilihan pengguna, atau menentukan ciri-ciri dataset yang tidak berlabel. Pengelompokan, dalam satu cara, membantu memahami kekacauan, seperti laci stokin.

[![Pengenalan kepada ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Pengenalan kepada Pengelompokan")

> 🎥 Klik imej di atas untuk video: John Guttag dari MIT memperkenalkan pengelompokan

Dalam persekitaran profesional, pengelompokan boleh digunakan untuk menentukan perkara seperti segmentasi pasaran, menentukan kumpulan umur yang membeli barangan tertentu, sebagai contoh. Kegunaan lain adalah pengesanan anomali, mungkin untuk mengesan penipuan daripada dataset transaksi kad kredit. Atau anda mungkin menggunakan pengelompokan untuk mengenal pasti tumor dalam sekumpulan imbasan perubatan.

✅ Luangkan masa untuk memikirkan bagaimana anda mungkin pernah menemui pengelompokan 'di alam nyata', dalam perbankan, e-dagang, atau perniagaan.

> 🎓 Menariknya, analisis pengelompokan berasal dari bidang Antropologi dan Psikologi pada tahun 1930-an. Bolehkah anda bayangkan bagaimana ia mungkin digunakan?

Sebagai alternatif, anda boleh menggunakannya untuk mengelompokkan hasil carian - seperti pautan membeli-belah, imej, atau ulasan, sebagai contoh. Pengelompokan berguna apabila anda mempunyai dataset yang besar yang ingin anda kurangkan dan pada dataset tersebut anda ingin melakukan analisis yang lebih terperinci, jadi teknik ini boleh digunakan untuk mempelajari data sebelum model lain dibina.

✅ Setelah data anda diatur dalam kelompok, anda memberikan Id kelompok, dan teknik ini boleh berguna apabila ingin menjaga privasi dataset; anda boleh merujuk kepada titik data dengan Id kelompoknya, dan bukannya data yang lebih mendedahkan. Bolehkah anda memikirkan sebab lain mengapa anda merujuk kepada Id kelompok dan bukannya elemen lain dalam kelompok untuk mengenal pasti data?

Perdalam pemahaman anda tentang teknik pengelompokan dalam [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Memulakan pengelompokan

[Scikit-learn menawarkan pelbagai kaedah](https://scikit-learn.org/stable/modules/clustering.html) untuk melakukan pengelompokan. Jenis yang anda pilih akan bergantung pada kes penggunaan anda. Menurut dokumentasi, setiap kaedah mempunyai pelbagai manfaat. Berikut adalah jadual ringkas kaedah yang disokong oleh Scikit-learn dan kes penggunaan yang sesuai:

| Nama kaedah                  | Kes penggunaan                                                        |
| :--------------------------- | :-------------------------------------------------------------------- |
| K-Means                      | tujuan umum, induktif                                                 |
| Affinity propagation         | banyak, kelompok tidak sekata, induktif                              |
| Mean-shift                   | banyak, kelompok tidak sekata, induktif                              |
| Spectral clustering          | sedikit, kelompok sekata, transduktif                                |
| Ward hierarchical clustering | banyak, kelompok terhad, transduktif                                 |
| Agglomerative clustering     | banyak, terhad, jarak bukan Euclidean, transduktif                   |
| DBSCAN                       | geometri tidak rata, kelompok tidak sekata, transduktif              |
| OPTICS                       | geometri tidak rata, kelompok tidak sekata dengan ketumpatan berubah, transduktif |
| Gaussian mixtures            | geometri rata, induktif                                              |
| BIRCH                        | dataset besar dengan outlier, induktif                               |

> 🎓 Cara kita mencipta kelompok banyak berkaitan dengan cara kita mengumpulkan titik data ke dalam kumpulan. Mari kita jelaskan beberapa istilah:
>
> 🎓 ['Transduktif' vs. 'Induktif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferens transduktif diperoleh daripada kes latihan yang diperhatikan yang memetakan kepada kes ujian tertentu. Inferens induktif diperoleh daripada kes latihan yang memetakan kepada peraturan umum yang hanya kemudian digunakan pada kes ujian.
> 
> Contoh: Bayangkan anda mempunyai dataset yang hanya sebahagiannya berlabel. Beberapa perkara adalah 'rekod', beberapa 'cd', dan beberapa kosong. Tugas anda adalah memberikan label untuk yang kosong. Jika anda memilih pendekatan induktif, anda akan melatih model mencari 'rekod' dan 'cd', dan menerapkan label tersebut pada data yang tidak berlabel. Pendekatan ini akan menghadapi kesukaran mengklasifikasikan perkara yang sebenarnya 'kaset'. Pendekatan transduktif, sebaliknya, menangani data yang tidak diketahui ini dengan lebih berkesan kerana ia berfungsi untuk mengelompokkan item serupa bersama-sama dan kemudian menerapkan label pada kumpulan. Dalam kes ini, kelompok mungkin mencerminkan 'benda muzik bulat' dan 'benda muzik segi empat'.
> 
> 🎓 ['Geometri tidak rata' vs. 'rata'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Berasal daripada istilah matematik, geometri tidak rata vs. rata merujuk kepada ukuran jarak antara titik dengan cara 'rata' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) atau 'tidak rata' (bukan Euclidean).
>
>'Rata' dalam konteks ini merujuk kepada geometri Euclidean (bahagian daripadanya diajar sebagai geometri 'dataran'), dan tidak rata merujuk kepada geometri bukan Euclidean. Apa kaitan geometri dengan pembelajaran mesin? Nah, sebagai dua bidang yang berakar dalam matematik, mesti ada cara umum untuk mengukur jarak antara titik dalam kelompok, dan itu boleh dilakukan dengan cara 'rata' atau 'tidak rata', bergantung pada sifat data. [Jarak Euclidean](https://wikipedia.org/wiki/Euclidean_distance) diukur sebagai panjang segmen garis antara dua titik. [Jarak bukan Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) diukur sepanjang lengkung. Jika data anda, yang divisualisasikan, nampaknya tidak wujud pada dataran, anda mungkin perlu menggunakan algoritma khusus untuk menanganinya.
>
![Infografik Geometri Rata vs Tidak Rata](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Jarak'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Kelompok ditentukan oleh matriks jaraknya, contohnya jarak antara titik. Jarak ini boleh diukur dengan beberapa cara. Kelompok Euclidean ditentukan oleh purata nilai titik, dan mengandungi 'centroid' atau titik tengah. Jarak diukur dengan jarak ke centroid tersebut. Jarak bukan Euclidean merujuk kepada 'clustroid', titik yang paling dekat dengan titik lain. Clustroid pula boleh ditentukan dengan pelbagai cara.
> 
> 🎓 ['Terhad'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Pengelompokan Terhad](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) memperkenalkan pembelajaran 'semi-supervised' ke dalam kaedah tanpa pengawasan ini. Hubungan antara titik ditandai sebagai 'tidak boleh dihubungkan' atau 'mesti dihubungkan' jadi beberapa peraturan dipaksa pada dataset.
>
>Contoh: Jika algoritma dilepaskan pada sekumpulan data yang tidak berlabel atau separa berlabel, kelompok yang dihasilkannya mungkin berkualiti rendah. Dalam contoh di atas, kelompok mungkin mengelompokkan 'benda muzik bulat' dan 'benda muzik segi empat' dan 'benda segi tiga' dan 'kuih'. Jika diberikan beberapa kekangan, atau peraturan untuk diikuti ("item mesti diperbuat daripada plastik", "item perlu dapat menghasilkan muzik") ini boleh membantu 'mengekang' algoritma untuk membuat pilihan yang lebih baik.
> 
> 🎓 'Ketumpatan'
> 
> Data yang 'berisik' dianggap 'padat'. Jarak antara titik dalam setiap kelompoknya mungkin terbukti, pada pemeriksaan, lebih atau kurang padat, atau 'sesak' dan oleh itu data ini perlu dianalisis dengan kaedah pengelompokan yang sesuai. [Artikel ini](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) menunjukkan perbezaan antara menggunakan pengelompokan K-Means vs. algoritma HDBSCAN untuk meneroka dataset berisik dengan ketumpatan kelompok yang tidak sekata.

## Algoritma pengelompokan

Terdapat lebih daripada 100 algoritma pengelompokan, dan penggunaannya bergantung pada sifat data yang ada. Mari kita bincangkan beberapa yang utama:

- **Pengelompokan hierarki**. Jika objek diklasifikasikan berdasarkan jaraknya dengan objek berdekatan, dan bukannya dengan objek yang lebih jauh, kelompok dibentuk berdasarkan jarak anggotanya ke dan dari objek lain. Pengelompokan agglomeratif Scikit-learn adalah hierarki.

   ![Infografik Pengelompokan Hierarki](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Pengelompokan centroid**. Algoritma popular ini memerlukan pilihan 'k', atau bilangan kelompok untuk dibentuk, selepas itu algoritma menentukan titik tengah kelompok dan mengumpulkan data di sekitar titik tersebut. [Pengelompokan K-means](https://wikipedia.org/wiki/K-means_clustering) adalah versi pengelompokan centroid yang popular. Pusat ditentukan oleh purata terdekat, maka namanya. Jarak kuadrat dari kelompok diminimumkan.

   ![Infografik Pengelompokan Centroid](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Pengelompokan berdasarkan pengedaran**. Berdasarkan pemodelan statistik, pengelompokan berdasarkan pengedaran berpusat pada menentukan kebarangkalian bahawa titik data tergolong dalam kelompok, dan menetapkannya dengan sewajarnya. Kaedah campuran Gaussian tergolong dalam jenis ini.

- **Pengelompokan berdasarkan ketumpatan**. Titik data ditetapkan kepada kelompok berdasarkan ketumpatannya, atau pengelompokan di sekeliling satu sama lain. Titik data yang jauh dari kumpulan dianggap sebagai outlier atau bunyi. DBSCAN, Mean-shift dan OPTICS tergolong dalam jenis pengelompokan ini.

- **Pengelompokan berdasarkan grid**. Untuk dataset multi-dimensi, grid dibuat dan data dibahagikan di antara sel grid, dengan itu mencipta kelompok.

## Latihan - kelompokkan data anda

Pengelompokan sebagai teknik sangat dibantu oleh visualisasi yang baik, jadi mari kita mulakan dengan memvisualisasikan data muzik kita. Latihan ini akan membantu kita memutuskan kaedah pengelompokan mana yang paling berkesan digunakan untuk sifat data ini.

1. Buka fail [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dalam folder ini.

1. Import pakej `Seaborn` untuk visualisasi data yang baik.

    ```python
    !pip install seaborn
    ```

1. Tambahkan data lagu dari [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Muatkan dataframe dengan beberapa data tentang lagu-lagu tersebut. Bersiaplah untuk meneroka data ini dengan mengimport perpustakaan dan memaparkan data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Periksa beberapa baris pertama data:

    |     | nama                     | album                        | artis               | genre_teratas_artis | tarikh_keluar | panjang | populariti | keboleh_menari | keakustikan | tenaga | instrumentalness | keliveness | kekuatan | keboleh_bersuara | tempo   | tanda_masa     |
    | --- | ------------------------ | ---------------------------- | ------------------- | ------------------- | ------------- | ------- | ---------- | -------------- | ----------- | ------ | ---------------- | ---------- | -------- | ---------------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b     | 2019          | 144000  | 48         | 0.666          | 0.851       | 0.42   | 0.534            | 0.11       | -6.699   | 0.0829           | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop             | 2020          | 89488   | 30         | 0.71           | 0.0822      | 0.683  | 0.000169         | 0.101      | -5.64    | 0.36             | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dapatkan maklumat tentang dataframe dengan memanggil `info()`:

    ```python
    df.info()
    ```

   Outputnya kelihatan seperti ini:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Periksa semula nilai null dengan memanggil `isnull()` dan pastikan jumlahnya adalah 0:

    ```python
    df.isnull().sum()
    ```

    Nampak baik:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Huraikan data:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Jika kita bekerja dengan pengelompokan, kaedah tanpa pengawasan yang tidak memerlukan data berlabel, mengapa kita menunjukkan data ini dengan label? Dalam fasa penerokaan data, ia berguna, tetapi ia tidak diperlukan untuk algoritma pengelompokan berfungsi. Anda juga boleh membuang tajuk lajur dan merujuk data mengikut nombor lajur.

Lihat nilai umum data. Perhatikan bahawa populariti boleh menjadi '0', yang menunjukkan lagu-lagu yang tidak mempunyai ranking. Mari kita buang nilai-nilai tersebut sebentar lagi.

1. Gunakan barplot untuk mengetahui genre yang paling popular:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Jika anda ingin melihat lebih banyak nilai teratas, ubah top `[:5]` kepada nilai yang lebih besar, atau buang untuk melihat semuanya.

Perhatikan, apabila genre teratas digambarkan sebagai 'Missing', itu bermaksud Spotify tidak mengklasifikasikannya, jadi mari kita buang data tersebut.

1. Buang data yang hilang dengan menapisnya keluar

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sekarang periksa semula genre:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Tiga genre teratas jelas mendominasi dataset ini. Mari kita fokus pada `afro dancehall`, `afropop`, dan `nigerian pop`, serta tapis dataset untuk membuang apa-apa dengan nilai populariti 0 (bermaksud ia tidak diklasifikasikan dengan populariti dalam dataset dan boleh dianggap sebagai gangguan untuk tujuan kita):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lakukan ujian pantas untuk melihat jika data berkorelasi dengan cara yang sangat kuat:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Satu-satunya korelasi kuat adalah antara `energy` dan `loudness`, yang tidak terlalu mengejutkan, memandangkan muzik yang kuat biasanya cukup bertenaga. Selain itu, korelasi agak lemah. Ia akan menarik untuk melihat apa yang algoritma pengelompokan dapat buat dengan data ini.

    > 🎓 Perhatikan bahawa korelasi tidak bermaksud sebab-akibat! Kita mempunyai bukti korelasi tetapi tiada bukti sebab-akibat. [Laman web yang menghiburkan](https://tylervigen.com/spurious-correlations) mempunyai beberapa visual yang menekankan perkara ini.

Adakah terdapat penumpuan dalam dataset ini sekitar populariti lagu dan kebolehmenariannya? Grid Facet menunjukkan terdapat lingkaran sepusat yang sejajar, tanpa mengira genre. Mungkinkah citarasa Nigeria berkumpul pada tahap kebolehmenarian tertentu untuk genre ini?

✅ Cuba titik data yang berbeza (energy, loudness, speechiness) dan lebih banyak atau genre muzik yang berbeza. Apa yang boleh anda temui? Lihat jadual `df.describe()` untuk melihat penyebaran umum titik data.

### Latihan - pengedaran data

Adakah tiga genre ini berbeza secara signifikan dalam persepsi kebolehmenarian mereka, berdasarkan populariti?

1. Periksa pengedaran data tiga genre teratas kita untuk populariti dan kebolehmenarian di sepanjang paksi x dan y yang diberikan.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Anda boleh menemui lingkaran sepusat di sekitar titik penumpuan umum, menunjukkan pengedaran titik.

    > 🎓 Perhatikan bahawa contoh ini menggunakan graf KDE (Kernel Density Estimate) yang mewakili data menggunakan lengkung ketumpatan kebarangkalian berterusan. Ini membolehkan kita mentafsir data apabila bekerja dengan pelbagai pengedaran.

    Secara umum, tiga genre ini sejajar secara longgar dari segi populariti dan kebolehmenarian. Menentukan kelompok dalam data yang sejajar secara longgar ini akan menjadi cabaran:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Buat plot taburan:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Plot taburan paksi yang sama menunjukkan corak penumpuan yang serupa

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Secara umum, untuk pengelompokan, anda boleh menggunakan plot taburan untuk menunjukkan kelompok data, jadi menguasai jenis visualisasi ini sangat berguna. Dalam pelajaran seterusnya, kita akan mengambil data yang telah ditapis ini dan menggunakan pengelompokan k-means untuk menemui kelompok dalam data ini yang kelihatan bertindih dengan cara yang menarik.

---

## 🚀Cabaran

Sebagai persediaan untuk pelajaran seterusnya, buat carta tentang pelbagai algoritma pengelompokan yang mungkin anda temui dan gunakan dalam persekitaran pengeluaran. Masalah jenis apa yang cuba diselesaikan oleh pengelompokan?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Sebelum anda menggunakan algoritma pengelompokan, seperti yang telah kita pelajari, adalah idea yang baik untuk memahami sifat dataset anda. Baca lebih lanjut tentang topik ini [di sini](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Artikel yang berguna ini](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) menerangkan cara pelbagai algoritma pengelompokan berfungsi, berdasarkan bentuk data yang berbeza.

## Tugasan

[Selidik visualisasi lain untuk pengelompokan](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.