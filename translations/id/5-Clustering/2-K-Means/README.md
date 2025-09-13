<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T19:18:05+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "id"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Dalam pelajaran ini, Anda akan belajar cara membuat klaster menggunakan Scikit-learn dan dataset musik Nigeria yang telah Anda impor sebelumnya. Kita akan membahas dasar-dasar K-Means untuk Klasterisasi. Ingatlah bahwa, seperti yang Anda pelajari di pelajaran sebelumnya, ada banyak cara untuk bekerja dengan klaster, dan metode yang Anda gunakan bergantung pada data Anda. Kita akan mencoba K-Means karena ini adalah teknik klasterisasi yang paling umum. Mari kita mulai!

Istilah yang akan Anda pelajari:

- Skor Silhouette
- Metode Elbow
- Inersia
- Variansi

## Pengantar

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) adalah metode yang berasal dari domain pemrosesan sinyal. Metode ini digunakan untuk membagi dan mengelompokkan data ke dalam 'k' klaster menggunakan serangkaian observasi. Setiap observasi bekerja untuk mengelompokkan titik data tertentu ke 'mean' terdekatnya, atau titik pusat dari sebuah klaster.

Klaster dapat divisualisasikan sebagai [diagram Voronoi](https://wikipedia.org/wiki/Voronoi_diagram), yang mencakup sebuah titik (atau 'seed') dan wilayah yang sesuai dengannya.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

Proses klasterisasi K-Means [dijalankan dalam tiga langkah](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritma memilih sejumlah titik pusat (k) dengan mengambil sampel dari dataset. Setelah itu, algoritma akan berulang:
    1. Menugaskan setiap sampel ke centroid terdekat.
    2. Membuat centroid baru dengan mengambil nilai rata-rata dari semua sampel yang ditugaskan ke centroid sebelumnya.
    3. Kemudian, menghitung perbedaan antara centroid baru dan lama, dan mengulangi hingga centroid stabil.

Salah satu kelemahan menggunakan K-Means adalah Anda perlu menentukan 'k', yaitu jumlah centroid. Untungnya, 'metode elbow' membantu memperkirakan nilai awal yang baik untuk 'k'. Anda akan mencobanya sebentar lagi.

## Prasyarat

Anda akan bekerja di file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) pelajaran ini yang mencakup impor data dan pembersihan awal yang Anda lakukan di pelajaran sebelumnya.

## Latihan - persiapan

Mulailah dengan melihat kembali data lagu.

1. Buat boxplot dengan memanggil `boxplot()` untuk setiap kolom:

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

    Data ini cukup berisik: dengan mengamati setiap kolom sebagai boxplot, Anda dapat melihat outlier.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Anda bisa menelusuri dataset dan menghapus outlier ini, tetapi itu akan membuat data menjadi sangat minimal.

1. Untuk saat ini, pilih kolom mana yang akan Anda gunakan untuk latihan klasterisasi. Pilih kolom dengan rentang yang serupa dan encode kolom `artist_top_genre` sebagai data numerik:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Sekarang Anda perlu memilih berapa banyak klaster yang akan ditargetkan. Anda tahu ada 3 genre lagu yang kita ambil dari dataset, jadi mari kita coba 3:

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

Anda akan melihat array yang dicetak dengan klaster yang diprediksi (0, 1, atau 2) untuk setiap baris dalam dataframe.

1. Gunakan array ini untuk menghitung 'skor silhouette':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Skor Silhouette

Carilah skor silhouette yang mendekati 1. Skor ini bervariasi dari -1 hingga 1, dan jika skornya 1, klaster tersebut padat dan terpisah dengan baik dari klaster lainnya. Nilai mendekati 0 menunjukkan klaster yang saling tumpang tindih dengan sampel yang sangat dekat dengan batas keputusan klaster tetangga. [(Sumber)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Skor kita adalah **0.53**, jadi berada di tengah-tengah. Ini menunjukkan bahwa data kita tidak terlalu cocok untuk jenis klasterisasi ini, tetapi mari kita lanjutkan.

### Latihan - membangun model

1. Impor `KMeans` dan mulai proses klasterisasi.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Ada beberapa bagian di sini yang perlu dijelaskan.

    > 🎓 range: Ini adalah iterasi dari proses klasterisasi.

    > 🎓 random_state: "Menentukan pengacakan angka untuk inisialisasi centroid." [Sumber](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "jumlah kuadrat dalam klaster" mengukur jarak rata-rata kuadrat dari semua titik dalam klaster ke centroid klaster. [Sumber](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

    > 🎓 Inersia: Algoritma K-Means mencoba memilih centroid untuk meminimalkan 'inersia', "ukuran seberapa koheren klaster secara internal." [Sumber](https://scikit-learn.org/stable/modules/clustering.html). Nilai ini ditambahkan ke variabel wcss pada setiap iterasi.

    > 🎓 k-means++: Dalam [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means), Anda dapat menggunakan optimasi 'k-means++', yang "menginisialisasi centroid agar (umumnya) berjauhan satu sama lain, menghasilkan hasil yang mungkin lebih baik daripada inisialisasi acak."

### Metode Elbow

Sebelumnya, Anda menyimpulkan bahwa karena Anda menargetkan 3 genre lagu, Anda harus memilih 3 klaster. Tetapi apakah itu benar?

1. Gunakan 'metode elbow' untuk memastikan.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Gunakan variabel `wcss` yang Anda bangun di langkah sebelumnya untuk membuat grafik yang menunjukkan di mana 'tikungan' pada elbow berada, yang menunjukkan jumlah klaster yang optimal. Mungkin memang **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Latihan - menampilkan klaster

1. Coba prosesnya lagi, kali ini menetapkan tiga klaster, dan tampilkan klaster sebagai scatterplot:

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

1. Periksa akurasi model:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Akurasi model ini tidak terlalu baik, dan bentuk klaster memberikan petunjuk mengapa.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Data ini terlalu tidak seimbang, terlalu sedikit berkorelasi, dan terdapat terlalu banyak variansi antara nilai kolom untuk dapat dikelompokkan dengan baik. Faktanya, klaster yang terbentuk mungkin sangat dipengaruhi atau bias oleh tiga kategori genre yang kita definisikan di atas. Itu adalah proses pembelajaran!

    Dalam dokumentasi Scikit-learn, Anda dapat melihat bahwa model seperti ini, dengan klaster yang tidak terlalu terpisah dengan baik, memiliki masalah 'variansi':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infografik dari Scikit-learn

## Variansi

Variansi didefinisikan sebagai "rata-rata dari kuadrat perbedaan dari Mean" [(Sumber)](https://www.mathsisfun.com/data/standard-deviation.html). Dalam konteks masalah klasterisasi ini, variansi mengacu pada data di mana angka-angka dalam dataset cenderung menyimpang terlalu jauh dari mean.

✅ Ini adalah momen yang tepat untuk memikirkan semua cara yang dapat Anda lakukan untuk memperbaiki masalah ini. Mengubah data sedikit lebih banyak? Menggunakan kolom yang berbeda? Menggunakan algoritma yang berbeda? Petunjuk: Cobalah [menskalakan data Anda](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) untuk menormalkannya dan menguji kolom lainnya.

> Cobalah '[kalkulator variansi](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' untuk memahami konsep ini lebih lanjut.

---

## 🚀Tantangan

Habiskan waktu dengan notebook ini, mengubah parameter. Bisakah Anda meningkatkan akurasi model dengan membersihkan data lebih banyak (misalnya, menghapus outlier)? Anda dapat menggunakan bobot untuk memberikan lebih banyak bobot pada sampel data tertentu. Apa lagi yang dapat Anda lakukan untuk membuat klaster yang lebih baik?

Petunjuk: Cobalah menskalakan data Anda. Ada kode yang dikomentari di notebook yang menambahkan skala standar untuk membuat kolom data lebih mirip satu sama lain dalam hal rentang. Anda akan menemukan bahwa meskipun skor silhouette turun, 'tikungan' pada grafik elbow menjadi lebih halus. Hal ini terjadi karena membiarkan data tidak diskalakan memungkinkan data dengan variansi lebih kecil memiliki bobot lebih besar. Baca lebih lanjut tentang masalah ini [di sini](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Lihatlah Simulator K-Means [seperti ini](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Anda dapat menggunakan alat ini untuk memvisualisasikan titik data sampel dan menentukan centroidnya. Anda dapat mengedit tingkat keacakan data, jumlah klaster, dan jumlah centroid. Apakah ini membantu Anda mendapatkan gambaran tentang bagaimana data dapat dikelompokkan?

Selain itu, lihat [handout tentang K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) dari Stanford.

## Tugas

[Cobalah metode klasterisasi yang berbeda](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang timbul dari penggunaan terjemahan ini.