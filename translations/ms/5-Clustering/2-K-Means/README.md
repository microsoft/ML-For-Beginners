<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T19:18:30+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ms"
}
-->
# Pengelompokan K-Means

## [Kuiz Pra-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

Dalam pelajaran ini, anda akan belajar cara mencipta kelompok menggunakan Scikit-learn dan dataset muzik Nigeria yang telah anda import sebelum ini. Kita akan membincangkan asas K-Means untuk Pengelompokan. Ingatlah bahawa, seperti yang anda pelajari dalam pelajaran sebelumnya, terdapat banyak cara untuk bekerja dengan kelompok dan kaedah yang anda gunakan bergantung pada data anda. Kita akan mencuba K-Means kerana ia adalah teknik pengelompokan yang paling biasa. Mari kita mulakan!

Istilah yang akan anda pelajari:

- Skor Silhouette
- Kaedah Elbow
- Inertia
- Varians

## Pengenalan

[Pengelompokan K-Means](https://wikipedia.org/wiki/K-means_clustering) adalah kaedah yang berasal dari bidang pemprosesan isyarat. Ia digunakan untuk membahagikan dan mempartisi kumpulan data kepada 'k' kelompok menggunakan siri pemerhatian. Setiap pemerhatian berfungsi untuk mengelompokkan titik data yang diberikan kepada 'mean' terdekatnya, atau titik pusat kelompok.

Kelompok ini boleh divisualisasikan sebagai [diagram Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), yang merangkumi satu titik (atau 'benih') dan wilayah yang berkaitan dengannya.

![diagram voronoi](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infografik oleh [Jen Looper](https://twitter.com/jenlooper)

Proses pengelompokan K-Means [dijalankan dalam tiga langkah](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritma memilih k-nombor titik pusat dengan mengambil sampel daripada dataset. Selepas itu, ia berulang:
    1. Ia menetapkan setiap sampel kepada centroid terdekat.
    2. Ia mencipta centroid baru dengan mengambil nilai purata semua sampel yang ditetapkan kepada centroid sebelumnya.
    3. Kemudian, ia mengira perbezaan antara centroid baru dan lama dan mengulangi sehingga centroid stabil.

Satu kelemahan menggunakan K-Means adalah anda perlu menetapkan 'k', iaitu bilangan centroid. Nasib baik, 'kaedah elbow' membantu menganggarkan nilai permulaan yang baik untuk 'k'. Anda akan mencubanya sebentar lagi.

## Prasyarat

Anda akan bekerja dalam fail [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) pelajaran ini yang merangkumi import data dan pembersihan awal yang anda lakukan dalam pelajaran sebelumnya.

## Latihan - persediaan

Mulakan dengan melihat semula data lagu.

1. Cipta boxplot, panggil `boxplot()` untuk setiap lajur:

    ```python
    plt.figure(figsize=(20,20), dpi=200)
    
    plt.subplot(4,3,1)
    sns.boxplot(x = 'popularity', data = df)
    
    plt.subplot(4,3,2)
    sns.boxplot(x = 'acousticness', data = df)
    
    plt.subplot(4,3,3)
    sns.boxplot(x = 'energy', data = df)
    
    plt.subplot(4,3,4)
    sns.boxplot(x = 'instrumentalness', data = df)
    
    plt.subplot(4,3,5)
    sns.boxplot(x = 'liveness', data = df)
    
    plt.subplot(4,3,6)
    sns.boxplot(x = 'loudness', data = df)
    
    plt.subplot(4,3,7)
    sns.boxplot(x = 'speechiness', data = df)
    
    plt.subplot(4,3,8)
    sns.boxplot(x = 'tempo', data = df)
    
    plt.subplot(4,3,9)
    sns.boxplot(x = 'time_signature', data = df)
    
    plt.subplot(4,3,10)
    sns.boxplot(x = 'danceability', data = df)
    
    plt.subplot(4,3,11)
    sns.boxplot(x = 'length', data = df)
    
    plt.subplot(4,3,12)
    sns.boxplot(x = 'release_date', data = df)
    ```

    Data ini agak bising: dengan memerhatikan setiap lajur sebagai boxplot, anda boleh melihat nilai luar.

    ![nilai luar](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Anda boleh melalui dataset dan membuang nilai luar ini, tetapi itu akan menjadikan data agak minimum.

1. Buat masa ini, pilih lajur mana yang akan anda gunakan untuk latihan pengelompokan anda. Pilih yang mempunyai julat serupa dan kodkan lajur `artist_top_genre` sebagai data berangka:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Sekarang anda perlu memilih berapa banyak kelompok untuk disasarkan. Anda tahu terdapat 3 genre lagu yang kita ambil daripada dataset, jadi mari cuba 3:

    ```python
    from sklearn.cluster import KMeans
    
    nclusters = 3 
    seed = 0
    
    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)
    
    # Predict the cluster for each data point
    
    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Anda melihat array dicetak dengan kelompok yang diramalkan (0, 1, atau 2) untuk setiap baris dalam dataframe.

1. Gunakan array ini untuk mengira 'skor silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Skor Silhouette

Cari skor silhouette yang lebih dekat dengan 1. Skor ini berbeza dari -1 hingga 1, dan jika skor adalah 1, kelompok adalah padat dan terpisah dengan baik daripada kelompok lain. Nilai dekat 0 mewakili kelompok yang bertindih dengan sampel yang sangat dekat dengan sempadan keputusan kelompok jiran. [(Sumber)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Skor kita adalah **.53**, jadi berada di tengah-tengah. Ini menunjukkan bahawa data kita tidak begitu sesuai untuk jenis pengelompokan ini, tetapi mari kita teruskan.

### Latihan - bina model

1. Import `KMeans` dan mulakan proses pengelompokan.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Terdapat beberapa bahagian di sini yang memerlukan penjelasan.

    > 🎓 range: Ini adalah iterasi proses pengelompokan

    > 🎓 random_state: "Menentukan penjanaan nombor rawak untuk inisialisasi centroid." [Sumber](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "jumlah kuadrat dalam kelompok" mengukur jarak purata kuadrat semua titik dalam kelompok ke centroid kelompok. [Sumber](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inertia: Algoritma K-Means cuba memilih centroid untuk meminimumkan 'inertia', "ukuran sejauh mana kelompok adalah koheren secara dalaman." [Sumber](https://scikit-learn.org/stable/modules/clustering.html). Nilai ditambahkan kepada pembolehubah wcss pada setiap iterasi.

    > 🎓 k-means++: Dalam [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) anda boleh menggunakan pengoptimuman 'k-means++', yang "menginisialisasi centroid untuk menjadi (secara umum) jauh antara satu sama lain, menghasilkan keputusan yang mungkin lebih baik daripada inisialisasi rawak.

### Kaedah Elbow

Sebelumnya, anda mengandaikan bahawa, kerana anda telah menyasarkan 3 genre lagu, anda harus memilih 3 kelompok. Tetapi adakah itu benar?

1. Gunakan 'kaedah elbow' untuk memastikan.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Gunakan pembolehubah `wcss` yang anda bina dalam langkah sebelumnya untuk mencipta carta yang menunjukkan di mana 'bengkok' dalam elbow, yang menunjukkan bilangan kelompok optimum. Mungkin memang **3**!

    ![kaedah elbow](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Latihan - paparkan kelompok

1. Cuba proses sekali lagi, kali ini menetapkan tiga kelompok, dan paparkan kelompok sebagai scatterplot:

    ```python
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plt.scatter(df['popularity'],df['danceability'],c = labels)
    plt.xlabel('popularity')
    plt.ylabel('danceability')
    plt.show()
    ```

1. Periksa ketepatan model:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Ketepatan model ini tidak begitu baik, dan bentuk kelompok memberikan petunjuk mengapa.

    ![kelompok](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Data ini terlalu tidak seimbang, terlalu sedikit berkorelasi dan terdapat terlalu banyak varians antara nilai lajur untuk dikelompokkan dengan baik. Malah, kelompok yang terbentuk mungkin sangat dipengaruhi atau berat sebelah oleh tiga kategori genre yang kita tentukan di atas. Itu adalah proses pembelajaran!

    Dalam dokumentasi Scikit-learn, anda boleh melihat bahawa model seperti ini, dengan kelompok yang tidak begitu jelas, mempunyai masalah 'varians':

    ![model bermasalah](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografik dari Scikit-learn

## Varians

Varians ditakrifkan sebagai "purata perbezaan kuadrat dari Mean" [(Sumber)](https://www.mathsisfun.com/data/standard-deviation.html). Dalam konteks masalah pengelompokan ini, ia merujuk kepada data di mana nombor dalam dataset kita cenderung menyimpang terlalu banyak daripada mean.

✅ Ini adalah masa yang baik untuk memikirkan semua cara anda boleh membetulkan masalah ini. Ubah data sedikit lagi? Gunakan lajur yang berbeza? Gunakan algoritma yang berbeza? Petunjuk: Cuba [skala data anda](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) untuk menormalkannya dan uji lajur lain.

> Cuba '[kalkulator varians](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' ini untuk memahami konsep dengan lebih baik.

---

## 🚀Cabaran

Luangkan masa dengan notebook ini, ubah parameter. Bolehkah anda meningkatkan ketepatan model dengan membersihkan data lebih banyak (contohnya, membuang nilai luar)? Anda boleh menggunakan berat untuk memberikan lebih banyak berat kepada sampel data tertentu. Apa lagi yang boleh anda lakukan untuk mencipta kelompok yang lebih baik?

Petunjuk: Cuba skala data anda. Terdapat kod yang dikomen dalam notebook yang menambah penskalaan standard untuk menjadikan lajur data lebih serupa dari segi julat. Anda akan mendapati bahawa walaupun skor silhouette menurun, 'bengkok' dalam graf elbow menjadi lebih lancar. Ini kerana membiarkan data tidak berskala membolehkan data dengan kurang varians membawa lebih banyak berat. Baca lebih lanjut mengenai masalah ini [di sini](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Kuiz Pasca-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Lihat Simulator K-Means [seperti ini](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Anda boleh menggunakan alat ini untuk memvisualisasikan titik data sampel dan menentukan centroidnya. Anda boleh mengedit keacakan data, bilangan kelompok dan bilangan centroid. Adakah ini membantu anda mendapatkan idea tentang bagaimana data boleh dikelompokkan?

Juga, lihat [handout tentang K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) dari Stanford.

## Tugasan

[Cuba kaedah pengelompokan yang berbeza](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.