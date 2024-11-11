# K-Ortalama Klasterləşməsi

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/29/?loc=az)

Bu dərsdə siz Scikit-learn və Nigeriya musiqiləri data seti ilə klasterlərin yaradılması öyrənəcəksiniz. Biz Klasterləşmə üçünü K-Ortalamanın əsaslarından bəhs edəcəyik. Unutmayın ki, əvvəlki dərslərdən də sizə tanış olduğu kimi, klasterlərlə işləməyin müxtəlif yolları var və istifadə metodunuz datanızdan bilavasitə asılıdır. Ən yayğın klasterləşmə texnikası olduğu üçün K-Ortalamanı istifadə edəcəyik. Gəlin başlayaq!

Haqqında öyrənəcəyiniz anlayışlar:
- Siluet hesabı
- Dirsək üsulu
- Ətalət
- Fərqlilik

# Giriş

K-Ortalama Klasterləşməsi siqnal emalı sahəsindən törəmiş bir metoddur. Bir sıra müşahidələrdən istifadə edərək məlumat qruplarını 'k' klasterlərinə bölmək üçün istifadə olunur. Hər bir müşahidə ən yaxın “ortaya” və ya klasterin mərkəzi nöqtəsinə ən yaxın olan verilənlər nöqtəsini qruplaşdırmaq üçün işləyir.

Klasterlər [Voronoy diaqramları](https://wikipedia.org/wiki/Voronoi_diagram) kimi vizuallaşdırıla bilər. Bunlara nöqtə (və ya "toxum") və ona uyğun bölgə daxildir.

![voronoy diaqramı](../images/voronoi.png)

> [Jen Looper](https://twitter.com/jenlooper) tərəfindən çəkilmiş infoqrafik

1. Alqoritm datasetdən seçmə yolu ilə mərkəz nöqtələrinin k-sayda seçir. Bundan sonra, aşağıdakı addımları dövri olaraq təkrar edir:
     1. O, hər bir nümunəni ən yaxın mərkəzə təyin edir.
     2. Əvvəlki mərkəzlərə təyin edilmiş bütün nümunələrin orta qiymətini alaraq yeni mərkəzlər yaradır.
     3. Sonra yeni və köhnə mərkəzlər arasındakı fərqi hesablayır və mərkəzlər sabitləşənə qədər təkrarlanır.

K-Ortalama istifadə etməyin bir çatışmazlığı, 'k'-nı, yəni mərkəzlərin sayını qurmağınız lazım olacaq. Xoşbəxtlikdən "dirsək üsulu" "k" üçün yaxşı başlanğıc dəyərini tapmağa kömək edir. Bir dəqiqədən sonra sınayacaqsınız.

## İlkin şərt

Siz bu dərsin [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) faylında işləyəcəksiniz. Faylda son dərsdə etdiyiniz məlumatların idxalı və ilkin təmizləmə artıq mövcuddur.

## Məşq - hazırlıq

Mahnı məlumatlarına bir daha nəzər salmaqla başlayın.

1. Hər bir sütun üçün `boxplot()` çağıraraq boxplot yaradın:

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

    Bu məlumatlar bir az səs-küylüdür: hər sütunu bir boxplot şəklində müşahidə etməklə kənar göstəriciləri görə bilərsiniz.

    ![kənar göstəricilər](../images/boxplots.png)

Siz datasetə göz gəzdirə və oradan kənar göstəriciləri silə bilərsiniz, amma belə olduqda datalar olduqca minimal olacaqdır.

1. Hələlik klasterləşdirmə tapşırığınız üçün hansı sütunlardan istifadə edəcəyinizi seçin. Oxşar diapazonlu olanları seçin və `artist_top_genre` sütununu rəqəmsal data kimi kodlayın:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]

    y = df['artist_top_genre']

    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])

    y = le.transform(y)
    ```

1. İndi siz neçə klasteri hədəf alacağınızı seçməlisiniz. Datasetdən çıxardığımız 3 mahnı janrının olduğunu bilirsiniz, gəlin 3-ü də sınayaq:

    ```python
    from sklearn.cluster import KMeans

    nclusters = 3
    seed = 0

    km = KMeans(n_clusters=nclusters, random_state=seed)
    km.fit(X)

    # Hər bir nöqtə üçün klasteri təxmin edin

    y_cluster_kmeans = km.predict(X)
    y_cluster_kmeans
    ```

Siz data setdəki hər bir sətir üçün proqnozlaşdırılan klasterlərlə (0, 1 və ya 2) çap edilmiş massivi görürsünüz.

1. 'Siluet xalını' hesablamaq üçün bu massivdən istifadə edin:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Siluet xalı

Siluet hesabı 1-ə yaxın olanı axtarın. Bu xal -1-dən 1-ə qədər dəyişir və əgər xal 1-dirsə, deməli klaster sıx və digər çoxluqlardan yaxşı ayrılmışdır. 0-a yaxın dəyər isə qonşu klasterlərin qərar sərhədinə çox yaxın olan nümunələrlə üst-üstə düşən klasterləri təmsil edir. [(Mənbə)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Hesabımız **.53**-dür, yəni ortadadır. Bu, məlumatlarımızın bu tip klasterləşdirmə üçün o qədər də uyğun olmadığını göstərir, amma davam edək.

### Tapşırıq - model qurun

1. `KMeans` dəyərini daxil edin və klasterləşdirmə prosesinə başlayın.

    ```python
    from sklearn.cluster import KMeans
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    ```

    Burada izah tələb edən bir neçə məqam var.

     > 🎓 diapazon: Bunlar klasterləşmə prosesinin dövrləridir

     > 🎓 random_state: "Mərkəzi nöqtənin ilkin təyini üçün təsadüfi ədəd generasiyasını müəyyən edir." [Mənbə](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

     > 🎓 WCSS: "kvadratların çoxluqdaxili cəmi" çoxluqdakı bütün nöqtələrin klaster mərkəzinə olan orta məsafəsinin kvadratını ölçür. [Mənbə](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).

     > 🎓 Ətalət: K-Ortalama alqoritmləri “ətalət”-i, daha dəqiq desək “daxili çoxluqların əlaqəliliyinin ölçülməsini” minimuma endirmək üçün mərkəzləri seçməyə çalışır. [Mənbə](https://scikit-learn.org/stable/modules/clustering.html). Dəyər hər dövrdə wcss dəyişəninə əlavə olunur.

     > 🎓 k-means++: [Scikit-learn]-də(https://scikit-learn.org/stable/modules/clustering.html#k-means) "k-means++" optimallaşdırmasından istifadə edə bilərsiniz. Mərkəzlərin (ümumiyyətlə) bir-birindən uzaq olması, təsadüfi başlanğıcdan daha yaxşı nəticələrə gətirib çıxarır.

### Dirsək üsulu

Əvvəllər hesab edirdiniz ki, 3 mahnı janrını hədəflədiyiniz üçün 3 klaster seçməlisiniz. Amma bu həqiqətən də belədirmi?

1. Əmin olmaq üçün “dirsək üsulundan” istifadə edin.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(range(1, 11), wcss,marker='o',color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Dirsəkdəki 'əyilmə'-nin harada olduğunu göstərən diaqram yaratmaq üçün əvvəlki addımda qurduğunuz `wcss` dəyişənindən istifadə edin. Bu dəyişən optimal klaster sayını göstərir. Bəlkə də elə **3-dür**!

     ![dirsək üsulu](../images/elbow.png)

## Tapşırıq - klasterləri göstərin

1. Prosesi yenidən təkrarlayın, bu dəfə üç klaster təyin edin və klasterləri səpələnmə qrafiki kimi göstərin:

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

1. Modelin dəqiqliyini yoxlayın:

    ```python
    labels = kmeans.labels_

    correct_labels = sum(y == labels)

    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Bu modelin dəqiqliyi o qədər də yaxşı deyil və bunun niyəsi barədə klasterlərin forması sizə ipucu verir.

     ![klasterlər](../images/clusters.png)

     Bu məlumatlar çox balanssızdır, çox aşağı korrelyasiyalıdır və yaxşı qruplaşdırmaq üçün sütun dəyərləri arasında çoxlu fərq var. Əslində, formalaşan klasterlər, çox güman ki, yuxarıda müəyyən etdiyimiz üç janr kateqoriyasından çox təsirlənmiş və ya əyilmişdir. Bu, öyrənmə prosesi idi!

     Scikit-learn-in sənədlərində çox yaxşı demarkasiya edilməmiş çoxluqları olan bu kimi bir modelin "variasiya" problemi olduğunu görə bilərsiniz:

     ![problem modelləri](../images/problems.png)
     > Scikit-learn-dən infoqrafik

## Variasiya

Variasiya "Ortalamadan kvadrat fərqlərin ortası" [(Mənbə)](https://www.mathsisfun.com/data/standard-deviation.html) kimi müəyyən edilir. Bu klaster problemi kontekstində datasetimizin nömrələrinin orta qiymətdən bir qədər çox ayrılmağa meyilli olduğu məlumatlara aiddir.

✅ Hazırki problemi düzəltmək üçün mümkün olan bütün yollar haqqında düşünmək üçün əla vaxtdır. Məlumatları bir az daha dəyişək? Fərqli sütunlar istifadə olunur? Fərqli alqoritmdən istifadə edilir? İpucu: Normallaşdırmaq və digər sütunları yoxlamaq üçün [məlumatlarınızı miqyaslandırmağa](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) cəhd edin.

> Konsepti bir az daha yaxşı başa düşmək üçün '[variasiya kalkulyatorunu](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' yoxlayın.

---

## 🚀 Məşğələ

Parametrləri düzəltmək üçün bu notbuk faylında bir az vaxt keçirin. Məlumatları daha çox təmizləməklə (məsələn, kənar göstəriciləri silməklə) modelin dəqiqliyini artıra bilərsinizmi? Verilmiş məlumat nümunələrinə daha çox çəki vermək üçün çəkilərdən istifadə edə bilərsiniz. Daha yaxşı klasterlər yaratmaq üçün başqa nə edə bilərsiniz?

İpucu: Məlumatlarınızı ölçməyə çalışın. Notbukda məlumat sütunlarının diapazon baxımından bir-birinə daha çox bənzəməsi üçün standart miqyas əlavə edən şərhə alınmış kod var. Siz, siluet göstəricisi aşağı düşərkən dirsək qrafikindəki 'bilmə'-nin hamarlaşdığını tapa bilərsiniz. Bu, datanın miqyassız buraxılmasının daha az variasiya ilə daha çox çəki daşımasına imkan verməsinə görə baş verir. Bu problem haqqında bir az daha ətraflı [buradan](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226) oxuyun.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/30/?loc=az)

## Təkrarlayın və özünüz öyrənin

[Bu kimi](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) K-Ortalama Simulyatoruna nəzər salın. Nümunə məlumat nöqtələrini vizuallaşdırmaq və onların mərkəzlərini təyin etmək üçün bu alətdən istifadə edə bilərsiniz. Bundan əlavə, məlumatların təsadüfiliyini, klasterlərin və mərkəzlərin sayını dəyişə bilərsiniz. Bu, məlumatların necə qruplaşdırıla biləcəyi barədə fikir əldə etməyə kömək edirmi?

Həmçinin, Stenforddan [K-Ortalama haqqında olan bu materiala](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) nəzər salın.

## Tapşırıq

[Müxtəlif klasterləşdirmə üsullarını sınayın](assignment.az.md)