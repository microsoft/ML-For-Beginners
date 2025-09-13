<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T07:52:05+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "tr"
}
-->
# K-Means Kümeleme

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

Bu derste, daha önce içe aktardığınız Nijerya müzik veri setini kullanarak Scikit-learn ile nasıl kümeler oluşturacağınızı öğreneceksiniz. K-Means ile Kümeleme'nin temel konularını ele alacağız. Daha önceki derste öğrendiğiniz gibi, kümelerle çalışmanın birçok yolu vardır ve kullandığınız yöntem verinize bağlıdır. En yaygın kümeleme tekniği olduğu için K-Means yöntemini deneyeceğiz. Haydi başlayalım!

Öğreneceğiniz terimler:

- Silhouette skoru
- Dirsek yöntemi
- Atalet
- Varyans

## Giriş

[K-Means Kümeleme](https://wikipedia.org/wiki/K-means_clustering), sinyal işleme alanından türetilmiş bir yöntemdir. Verileri 'k' kümesine ayırmak ve gruplandırmak için bir dizi gözlem kullanılır. Her gözlem, bir veri noktasını en yakın 'ortalama'ya, yani bir kümenin merkez noktasına gruplandırmaya çalışır.

Kümeler, bir nokta (veya 'tohum') ve buna karşılık gelen bölgeyi içeren [Voronoi diyagramları](https://wikipedia.org/wiki/Voronoi_diagram) olarak görselleştirilebilir.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infographic by [Jen Looper](https://twitter.com/jenlooper)

K-Means kümeleme süreci [üç adımlı bir süreçte gerçekleştirilir](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algoritma, veri setinden örnekleme yaparak k sayıda merkez noktası seçer. Ardından şu döngüyü gerçekleştirir:
    1. Her örneği en yakın merkez noktaya atar.
    2. Önceki merkez noktalarına atanan tüm örneklerin ortalama değerini alarak yeni merkez noktaları oluşturur.
    3. Daha sonra yeni ve eski merkez noktaları arasındaki farkı hesaplar ve merkez noktalar sabitlenene kadar tekrar eder.

K-Means kullanmanın bir dezavantajı, 'k' yani merkez noktalarının sayısını belirlemeniz gerektiğidir. Neyse ki, 'dirsek yöntemi' 'k' için iyi bir başlangıç değeri tahmin etmenize yardımcı olur. Bunu birazdan deneyeceksiniz.

## Ön Koşul

Bu derste, önceki derste veri içe aktarma ve ön temizlik işlemlerini yaptığınız [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) dosyasında çalışacaksınız.

## Alıştırma - Hazırlık

Şarkı verilerine tekrar bir göz atarak başlayın.

1. Her sütun için `boxplot()` çağırarak bir kutu grafiği oluşturun:

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

    Bu veri biraz gürültülü: Her sütunu bir kutu grafiği olarak gözlemleyerek aykırı değerleri görebilirsiniz.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Veri setini gözden geçirip bu aykırı değerleri kaldırabilirsiniz, ancak bu veri setini oldukça minimal hale getirir.

1. Şimdi kümeleme alıştırmanız için hangi sütunları kullanacağınızı seçin. Benzer aralıklara sahip olanları seçin ve `artist_top_genre` sütununu sayısal veri olarak kodlayın:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Şimdi hedefleyeceğiniz küme sayısını seçmeniz gerekiyor. Veri setinden 3 şarkı türü ayırdığınızı biliyorsunuz, o yüzden 3'ü deneyelim:

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

Bir veri çerçevesinin her satırı için tahmin edilen kümeler (0, 1 veya 2) ile bir dizi çıktısı görüyorsunuz.

1. Bu diziyi kullanarak bir 'silhouette skoru' hesaplayın:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette Skoru

1'e yakın bir silhouette skoru arayın. Bu skor -1 ile 1 arasında değişir ve skor 1 ise küme yoğun ve diğer kümelerden iyi ayrılmıştır. 0'a yakın bir değer, komşu kümelerin karar sınırına çok yakın örneklerle örtüşen kümeleri temsil eder. [(Kaynak)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Skorumuz **.53**, yani ortada. Bu, verimizin bu tür bir kümeleme için pek uygun olmadığını gösteriyor, ancak devam edelim.

### Alıştırma - Model Oluşturma

1. `KMeans`'i içe aktarın ve kümeleme sürecini başlatın.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Burada açıklamaya değer birkaç bölüm var.

    > 🎓 range: Kümeleme sürecinin yinelemeleri

    > 🎓 random_state: "Merkez noktası başlatma için rastgele sayı üretimini belirler." [Kaynak](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "küme içi kareler toplamı", bir küme içindeki tüm noktaların küme merkezine olan ortalama kare mesafesini ölçer. [Kaynak](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Atalet: K-Means algoritmaları, 'ataleti' minimize etmek için merkez noktalarını seçmeye çalışır, "kümelerin ne kadar içsel olarak tutarlı olduğunu ölçen bir değer." [Kaynak](https://scikit-learn.org/stable/modules/clustering.html). Değer, her yinelemede wcss değişkenine eklenir.

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)'de 'k-means++' optimizasyonunu kullanabilirsiniz, bu "merkez noktalarını genellikle birbirinden uzak olacak şekilde başlatır, rastgele başlatmaya göre muhtemelen daha iyi sonuçlar verir."

### Dirsek Yöntemi

Daha önce, 3 şarkı türünü hedeflediğiniz için 3 küme seçmeniz gerektiğini varsaydınız. Ama gerçekten öyle mi?

1. Emin olmak için 'dirsek yöntemini' kullanın.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Önceki adımda oluşturduğunuz `wcss` değişkenini kullanarak, optimum küme sayısını gösteren 'dirsek' eğrisini içeren bir grafik oluşturun. Belki gerçekten **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Alıştırma - Kümeleri Görüntüleme

1. Süreci tekrar deneyin, bu sefer üç küme ayarlayın ve kümeleri bir saçılım grafiği olarak görüntüleyin:

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

1. Modelin doğruluğunu kontrol edin:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Bu modelin doğruluğu pek iyi değil ve kümelerin şekli nedenini anlamanız için ipucu veriyor.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Bu veri çok dengesiz, çok az korelasyonlu ve sütun değerleri arasında çok fazla varyans var, bu yüzden iyi bir şekilde kümelenemiyor. Aslında, oluşan kümeler muhtemelen yukarıda tanımladığımız üç tür kategorisi tarafından ağır şekilde etkileniyor veya çarpıtılıyor. Bu bir öğrenme süreciydi!

    Scikit-learn belgelerinde, bu model gibi, iyi ayrılmamış kümelerle bir modelin 'varyans' problemi olduğu görülebilir:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infographic from Scikit-learn

## Varyans

Varyans, "Ortalama'dan olan kare farkların ortalaması" olarak tanımlanır [(Kaynak)](https://www.mathsisfun.com/data/standard-deviation.html). Bu kümeleme problemi bağlamında, veri setimizin sayılarının ortalamadan biraz fazla sapma eğiliminde olduğunu ifade eder.

✅ Bu, bu sorunu düzeltmek için tüm yolları düşünmek için harika bir an. Veriyi biraz daha düzenlemek mi? Farklı sütunlar mı kullanmak? Farklı bir algoritma mı denemek? İpucu: Verinizi [ölçeklendirmeyi](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) deneyin, normalleştirin ve diğer sütunları test edin.

> Bu '[varyans hesaplayıcıyı](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' kullanarak kavramı biraz daha anlayabilirsiniz.

---

## 🚀Meydan Okuma

Bu notebook ile biraz zaman geçirin, parametreleri değiştirin. Veriyi daha fazla temizleyerek (örneğin aykırı değerleri kaldırarak) modelin doğruluğunu artırabilir misiniz? Belirli veri örneklerine daha fazla ağırlık vermek için ağırlıklar kullanabilirsiniz. Daha iyi kümeler oluşturmak için başka ne yapabilirsiniz?

İpucu: Verinizi ölçeklendirmeyi deneyin. Notebook'ta, veri sütunlarının aralık açısından birbirine daha yakın görünmesini sağlamak için standart ölçeklendirme ekleyen yorumlanmış kodlar var. Veriyi ölçeklendirilmemiş bırakmak, daha az varyansa sahip verilerin daha fazla ağırlık taşımasına izin verdiği için, silhouette skoru düşerken dirsek grafiğindeki 'kink' yumuşar. Bu sorun hakkında biraz daha okuyun [burada](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bir K-Means Simülatörüne [buradan](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) göz atın. Bu aracı kullanarak örnek veri noktalarını görselleştirebilir ve merkez noktalarını belirleyebilirsiniz. Verinin rastgeleliğini, küme sayılarını ve merkez noktası sayılarını düzenleyebilirsiniz. Bu, verinin nasıl gruplandırılabileceği hakkında bir fikir edinmenize yardımcı oluyor mu?

Ayrıca Stanford'dan [bu K-Means el kitabına](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) göz atın.

## Ödev

[Farklı kümeleme yöntemlerini deneyin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.