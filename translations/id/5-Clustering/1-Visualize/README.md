<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T19:14:02+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "id"
}
-->
# Pengantar clustering

Clustering adalah jenis [Pembelajaran Tanpa Pengawasan](https://wikipedia.org/wiki/Unsupervised_learning) yang mengasumsikan bahwa dataset tidak memiliki label atau inputnya tidak dipasangkan dengan output yang telah ditentukan sebelumnya. Clustering menggunakan berbagai algoritma untuk memilah data yang tidak berlabel dan memberikan pengelompokan berdasarkan pola yang ditemukan dalam data tersebut.

[![No One Like You oleh PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You oleh PSquare")

> 🎥 Klik gambar di atas untuk menonton video. Sambil mempelajari machine learning dengan clustering, nikmati beberapa lagu Dance Hall Nigeria - ini adalah lagu yang sangat populer dari tahun 2014 oleh PSquare.

## [Kuis sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

### Pengantar

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) sangat berguna untuk eksplorasi data. Mari kita lihat apakah clustering dapat membantu menemukan tren dan pola dalam cara audiens Nigeria mengonsumsi musik.

✅ Luangkan waktu sejenak untuk memikirkan kegunaan clustering. Dalam kehidupan sehari-hari, clustering terjadi setiap kali Anda memiliki tumpukan cucian dan perlu memilah pakaian anggota keluarga 🧦👕👖🩲. Dalam data science, clustering terjadi saat mencoba menganalisis preferensi pengguna, atau menentukan karakteristik dari dataset yang tidak berlabel. Clustering, dalam beberapa hal, membantu memahami kekacauan, seperti laci kaus kaki.

[![Pengantar ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Pengantar Clustering")

> 🎥 Klik gambar di atas untuk menonton video: John Guttag dari MIT memperkenalkan clustering.

Dalam lingkungan profesional, clustering dapat digunakan untuk menentukan hal-hal seperti segmentasi pasar, misalnya menentukan kelompok usia mana yang membeli barang tertentu. Penggunaan lainnya adalah deteksi anomali, mungkin untuk mendeteksi penipuan dari dataset transaksi kartu kredit. Atau Anda mungkin menggunakan clustering untuk menentukan tumor dalam kumpulan pemindaian medis.

✅ Luangkan waktu sejenak untuk memikirkan bagaimana Anda mungkin pernah menemui clustering 'di dunia nyata', dalam konteks perbankan, e-commerce, atau bisnis.

> 🎓 Menariknya, analisis cluster berasal dari bidang Antropologi dan Psikologi pada tahun 1930-an. Bisakah Anda membayangkan bagaimana ini mungkin digunakan?

Sebagai alternatif, Anda dapat menggunakannya untuk mengelompokkan hasil pencarian - misalnya berdasarkan tautan belanja, gambar, atau ulasan. Clustering berguna ketika Anda memiliki dataset besar yang ingin Anda kurangi dan analisis lebih mendalam, sehingga teknik ini dapat digunakan untuk mempelajari data sebelum model lain dibangun.

✅ Setelah data Anda diorganisasi dalam cluster, Anda memberikan ID cluster, dan teknik ini dapat berguna untuk menjaga privasi dataset; Anda dapat merujuk pada titik data dengan ID cluster-nya, daripada dengan data yang lebih mengungkapkan. Bisakah Anda memikirkan alasan lain mengapa Anda merujuk pada ID cluster daripada elemen lain dari cluster untuk mengidentifikasinya?

Perdalam pemahaman Anda tentang teknik clustering dalam [modul pembelajaran ini](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Memulai dengan clustering

[Scikit-learn menawarkan berbagai metode](https://scikit-learn.org/stable/modules/clustering.html) untuk melakukan clustering. Jenis yang Anda pilih akan bergantung pada kasus penggunaan Anda. Menurut dokumentasi, setiap metode memiliki berbagai manfaat. Berikut adalah tabel sederhana dari metode yang didukung oleh Scikit-learn dan kasus penggunaannya yang sesuai:

| Nama metode                 | Kasus penggunaan                                                      |
| :-------------------------- | :-------------------------------------------------------------------- |
| K-Means                     | tujuan umum, induktif                                                |
| Affinity propagation        | banyak, cluster tidak merata, induktif                               |
| Mean-shift                  | banyak, cluster tidak merata, induktif                               |
| Spectral clustering         | sedikit, cluster merata, transduktif                                 |
| Ward hierarchical clustering| banyak, cluster terbatas, transduktif                                |
| Agglomerative clustering    | banyak, terbatas, jarak non-Euclidean, transduktif                   |
| DBSCAN                      | geometri tidak datar, cluster tidak merata, transduktif              |
| OPTICS                      | geometri tidak datar, cluster tidak merata dengan kepadatan variabel, transduktif |
| Gaussian mixtures           | geometri datar, induktif                                             |
| BIRCH                       | dataset besar dengan outlier, induktif                               |

> 🎓 Cara kita membuat cluster sangat berkaitan dengan cara kita mengelompokkan titik data ke dalam grup. Mari kita bahas beberapa istilah:
>
> 🎓 ['Transduktif' vs. 'Induktif'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Inferensi transduktif berasal dari kasus pelatihan yang diamati yang dipetakan ke kasus uji tertentu. Inferensi induktif berasal dari kasus pelatihan yang dipetakan ke aturan umum yang kemudian diterapkan pada kasus uji.
> 
> Contoh: Bayangkan Anda memiliki dataset yang hanya sebagian berlabel. Beberapa hal adalah 'rekaman', beberapa 'CD', dan beberapa kosong. Tugas Anda adalah memberikan label untuk yang kosong. Jika Anda memilih pendekatan induktif, Anda akan melatih model untuk mencari 'rekaman' dan 'CD', dan menerapkan label tersebut pada data yang tidak berlabel. Pendekatan ini akan kesulitan mengklasifikasikan hal-hal yang sebenarnya adalah 'kaset'. Pendekatan transduktif, di sisi lain, menangani data yang tidak diketahui ini lebih efektif karena bekerja untuk mengelompokkan item serupa bersama-sama dan kemudian menerapkan label ke grup. Dalam kasus ini, cluster mungkin mencerminkan 'benda musik bulat' dan 'benda musik persegi'.
> 
> 🎓 ['Geometri tidak datar' vs. 'datar'](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Berasal dari terminologi matematika, geometri tidak datar vs. datar mengacu pada pengukuran jarak antara titik-titik dengan metode geometris 'datar' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) atau 'tidak datar' (non-Euclidean).
>
>'Datar' dalam konteks ini mengacu pada geometri Euclidean (bagian dari geometri ini diajarkan sebagai geometri 'bidang'), dan tidak datar mengacu pada geometri non-Euclidean. Apa hubungannya geometri dengan machine learning? Nah, sebagai dua bidang yang berakar pada matematika, harus ada cara umum untuk mengukur jarak antara titik-titik dalam cluster, dan itu dapat dilakukan dengan cara 'datar' atau 'tidak datar', tergantung pada sifat data. [Jarak Euclidean](https://wikipedia.org/wiki/Euclidean_distance) diukur sebagai panjang segmen garis antara dua titik. [Jarak non-Euclidean](https://wikipedia.org/wiki/Non-Euclidean_geometry) diukur sepanjang kurva. Jika data Anda, saat divisualisasikan, tampaknya tidak ada di bidang, Anda mungkin perlu menggunakan algoritma khusus untuk menanganinya.
>
![Infografis Geometri Datar vs Tidak Datar](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Jarak'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Cluster didefinisikan oleh matriks jaraknya, misalnya jarak antara titik-titik. Jarak ini dapat diukur dengan beberapa cara. Cluster Euclidean didefinisikan oleh rata-rata nilai titik, dan memiliki 'centroid' atau titik pusat. Jarak diukur berdasarkan jarak ke centroid tersebut. Jarak non-Euclidean mengacu pada 'clustroid', titik yang paling dekat dengan titik lainnya. Clustroid pada gilirannya dapat didefinisikan dengan berbagai cara.
> 
> 🎓 ['Terbatas'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Clustering Terbatas](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) memperkenalkan pembelajaran 'semi-terawasi' ke dalam metode tanpa pengawasan ini. Hubungan antara titik-titik ditandai sebagai 'tidak boleh terhubung' atau 'harus terhubung' sehingga beberapa aturan dipaksakan pada dataset.
>
>Contoh: Jika algoritma dibiarkan bebas pada kumpulan data yang tidak berlabel atau semi-berlabel, cluster yang dihasilkannya mungkin berkualitas buruk. Dalam contoh di atas, cluster mungkin mengelompokkan 'benda musik bulat' dan 'benda musik persegi' dan 'benda segitiga' dan 'kue'. Jika diberikan beberapa batasan, atau aturan untuk diikuti ("item harus terbuat dari plastik", "item harus dapat menghasilkan musik") ini dapat membantu 'membatasi' algoritma untuk membuat pilihan yang lebih baik.
> 
> 🎓 'Kepadatan'
> 
> Data yang 'berisik' dianggap 'padat'. Jarak antara titik-titik dalam setiap cluster-nya mungkin, setelah diperiksa, lebih atau kurang padat, atau 'ramai' dan dengan demikian data ini perlu dianalisis dengan metode clustering yang sesuai. [Artikel ini](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) menunjukkan perbedaan antara menggunakan algoritma K-Means clustering vs. HDBSCAN untuk mengeksplorasi dataset yang berisik dengan kepadatan cluster yang tidak merata.

## Algoritma clustering

Ada lebih dari 100 algoritma clustering, dan penggunaannya bergantung pada sifat data yang ada. Mari kita bahas beberapa yang utama:

- **Clustering hierarkis**. Jika sebuah objek diklasifikasikan berdasarkan kedekatannya dengan objek terdekat, bukan dengan yang lebih jauh, cluster terbentuk berdasarkan jarak anggotanya ke dan dari objek lain. Agglomerative clustering dari Scikit-learn bersifat hierarkis.

   ![Infografis Clustering Hierarkis](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering centroid**. Algoritma populer ini membutuhkan pilihan 'k', atau jumlah cluster yang akan dibentuk, setelah itu algoritma menentukan titik pusat cluster dan mengumpulkan data di sekitar titik tersebut. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) adalah versi clustering centroid yang populer. Pusat ditentukan oleh rata-rata terdekat, sehingga dinamakan demikian. Jarak kuadrat dari cluster diminimalkan.

   ![Infografis Clustering Centroid](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Clustering berbasis distribusi**. Berdasarkan pemodelan statistik, clustering berbasis distribusi berpusat pada menentukan probabilitas bahwa suatu titik data termasuk dalam cluster, dan menetapkannya sesuai. Metode campuran Gaussian termasuk dalam jenis ini.

- **Clustering berbasis kepadatan**. Titik data ditetapkan ke cluster berdasarkan kepadatannya, atau pengelompokannya di sekitar satu sama lain. Titik data yang jauh dari grup dianggap sebagai outlier atau noise. DBSCAN, Mean-shift, dan OPTICS termasuk dalam jenis clustering ini.

- **Clustering berbasis grid**. Untuk dataset multi-dimensi, grid dibuat dan data dibagi di antara sel grid, sehingga membentuk cluster.

## Latihan - cluster data Anda

Clustering sebagai teknik sangat terbantu oleh visualisasi yang tepat, jadi mari kita mulai dengan memvisualisasikan data musik kita. Latihan ini akan membantu kita memutuskan metode clustering mana yang paling efektif digunakan untuk sifat data ini.

1. Buka file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) di folder ini.

1. Impor paket `Seaborn` untuk visualisasi data yang baik.

    ```python
    !pip install seaborn
    ```

1. Tambahkan data lagu dari [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Muat dataframe dengan beberapa data tentang lagu-lagu tersebut. Bersiaplah untuk mengeksplorasi data ini dengan mengimpor pustaka dan mencetak data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Periksa beberapa baris pertama data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dapatkan informasi tentang dataframe dengan memanggil `info()`:

    ```python
    df.info()
    ```

   Outputnya terlihat seperti ini:

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

1. Periksa ulang apakah ada nilai kosong dengan memanggil `isnull()` dan memastikan jumlahnya adalah 0:

    ```python
    df.isnull().sum()
    ```

    Terlihat baik:

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

1. Deskripsikan data:

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

> 🤔 Jika kita bekerja dengan clustering, metode tanpa pengawasan yang tidak memerlukan data berlabel, mengapa kita menunjukkan data ini dengan label? Pada fase eksplorasi data, label berguna, tetapi tidak diperlukan agar algoritma clustering dapat bekerja. Anda juga bisa menghapus header kolom dan merujuk data berdasarkan nomor kolom.

Lihat nilai umum dari data. Perhatikan bahwa popularitas bisa bernilai '0', yang menunjukkan lagu-lagu yang tidak memiliki peringkat. Mari kita hapus nilai-nilai tersebut sebentar lagi.

1. Gunakan barplot untuk mengetahui genre yang paling populer:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Jika Anda ingin melihat lebih banyak nilai teratas, ubah `[:5]` menjadi nilai yang lebih besar, atau hapus untuk melihat semuanya.

Catatan, ketika genre teratas dijelaskan sebagai 'Missing', itu berarti Spotify tidak mengklasifikasikannya, jadi mari kita hilangkan.

1. Hilangkan data yang hilang dengan memfilternya:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Sekarang periksa ulang genre:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Tiga genre teratas mendominasi dataset ini. Mari kita fokus pada `afro dancehall`, `afropop`, dan `nigerian pop`, serta memfilter dataset untuk menghapus apa pun dengan nilai popularitas 0 (yang berarti tidak diklasifikasikan dengan popularitas dalam dataset dan dapat dianggap sebagai noise untuk tujuan kita):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Lakukan tes cepat untuk melihat apakah data berkorelasi dengan cara yang sangat kuat:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Satu-satunya korelasi kuat adalah antara `energy` dan `loudness`, yang tidak terlalu mengejutkan, mengingat musik yang keras biasanya cukup energik. Selain itu, korelasi relatif lemah. Akan menarik untuk melihat apa yang dapat dilakukan algoritma clustering terhadap data ini.

    > 🎓 Perhatikan bahwa korelasi tidak menyiratkan sebab-akibat! Kita memiliki bukti korelasi tetapi tidak memiliki bukti sebab-akibat. [Sebuah situs web yang menghibur](https://tylervigen.com/spurious-correlations) memiliki beberapa visual yang menekankan poin ini.

Apakah ada konvergensi dalam dataset ini terkait popularitas lagu yang dirasakan dan danceability? Sebuah FacetGrid menunjukkan bahwa ada lingkaran konsentris yang sejajar, terlepas dari genre. Mungkinkah selera Nigeria berkumpul pada tingkat danceability tertentu untuk genre ini?  

✅ Coba titik data yang berbeda (energy, loudness, speechiness) dan lebih banyak atau genre musik yang berbeda. Apa yang dapat Anda temukan? Lihat tabel `df.describe()` untuk melihat penyebaran umum dari titik data.

### Latihan - distribusi data

Apakah ketiga genre ini berbeda secara signifikan dalam persepsi danceability mereka, berdasarkan popularitas?

1. Periksa distribusi data dari tiga genre teratas kita untuk popularitas dan danceability di sepanjang sumbu x dan y tertentu.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Anda dapat menemukan lingkaran konsentris di sekitar titik konvergensi umum, menunjukkan distribusi titik.

    > 🎓 Perhatikan bahwa contoh ini menggunakan grafik KDE (Kernel Density Estimate) yang mewakili data menggunakan kurva kepadatan probabilitas kontinu. Ini memungkinkan kita untuk menafsirkan data saat bekerja dengan beberapa distribusi.

    Secara umum, ketiga genre ini sejajar secara longgar dalam hal popularitas dan danceability. Menentukan cluster dalam data yang sejajar secara longgar ini akan menjadi tantangan:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Buat scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot dari sumbu yang sama menunjukkan pola konvergensi yang serupa

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Secara umum, untuk clustering, Anda dapat menggunakan scatterplot untuk menunjukkan cluster data, jadi menguasai jenis visualisasi ini sangat berguna. Dalam pelajaran berikutnya, kita akan menggunakan data yang telah difilter ini dan menggunakan clustering k-means untuk menemukan grup dalam data ini yang tampaknya tumpang tindih dengan cara yang menarik.

---

## 🚀Tantangan

Sebagai persiapan untuk pelajaran berikutnya, buatlah diagram tentang berbagai algoritma clustering yang mungkin Anda temukan dan gunakan dalam lingkungan produksi. Masalah apa yang coba diatasi oleh clustering?

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Sebelum Anda menerapkan algoritma clustering, seperti yang telah kita pelajari, ada baiknya memahami sifat dataset Anda. Baca lebih lanjut tentang topik ini [di sini](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Artikel yang bermanfaat ini](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) menjelaskan berbagai cara algoritma clustering berperilaku, mengingat bentuk data yang berbeda.

## Tugas

[Teliti visualisasi lain untuk clustering](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.