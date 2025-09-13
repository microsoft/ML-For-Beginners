<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T12:18:06+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "sr"
}
-->
# K-Means кластерисање

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

У овом лекцији ћете научити како да креирате кластере користећи Scikit-learn и нигеријски музички сет података који сте раније увезли. Покрићемо основе K-Means метода за кластерисање. Имајте на уму да, као што сте научили у претходној лекцији, постоји много начина за рад са кластерима, а метод који користите зависи од ваших података. Пробаћемо K-Means јер је то најчешћа техника кластерисања. Хајде да почнемо!

Термини које ћете научити:

- Силуетно оцењивање
- Елбоу метода
- Инерција
- Варијанса

## Увод

[K-Means кластерисање](https://wikipedia.org/wiki/K-means_clustering) је метода која потиче из области обраде сигнала. Користи се за поделу и груписање података у 'k' кластера користећи серију опсервација. Свака опсервација ради на груписању датих тачака података најближе њиховом 'средњем', или централној тачки кластера.

Кластери се могу визуализовати као [Воронојеви дијаграми](https://wikipedia.org/wiki/Voronoi_diagram), који укључују тачку (или 'семе') и њену одговарајућу регију.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> инфографика од [Jen Looper](https://twitter.com/jenlooper)

Процес K-Means кластерисања [извршава се у три корака](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Алгоритам бира k-број централних тачака узорковањем из скупа података. Након тога, понавља:
    1. Додељује сваки узорак најближем центроиду.
    2. Креира нове центроиде узимајући средњу вредност свих узорака додељених претходним центроидима.
    3. Затим израчунава разлику између нових и старих центроида и понавља док се центроиди не стабилизују.

Један недостатак коришћења K-Means је то што морате одредити 'k', односно број центроида. Срећом, 'елбоу метода' помаже у процени доброг почетног броја за 'k'. Ускоро ћете је испробати.

## Предуслов

Радићете у датотеци [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) која укључује увоз података и прелиминарно чишћење које сте урадили у претходној лекцији.

## Вежба - припрема

Почните тако што ћете поново погледати податке о песмама.

1. Креирајте boxplot, позивајући `boxplot()` за сваку колону:

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

    Ови подаци су мало бучни: посматрајући сваку колону као boxplot, можете видети одступања.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Могли бисте проћи кроз сет података и уклонити ова одступања, али то би учинило податке прилично минималним.

1. За сада, изаберите које колоне ћете користити за своју вежбу кластерисања. Одаберите оне са сличним опсегом и кодирајте колону `artist_top_genre` као нумеричке податке:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Сада треба да одредите колико кластера ћете циљати. Знате да постоје 3 жанра песама које смо издвојили из скупа података, па хајде да пробамо са 3:

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

Видећете низ исписан са предвиђеним кластерима (0, 1 или 2) за сваки ред у dataframe-у.

1. Користите овај низ да израчунате 'силуетну оцену':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Силуетна оцена

Тражите силуетну оцену ближу 1. Ова оцена варира од -1 до 1, и ако је оцена 1, кластер је густ и добро одвојен од других кластера. Вредност близу 0 представља преклапајуће кластере са узорцима веома близу граници одлуке суседних кластера. [(Извор)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Наша оцена је **.53**, што је негде у средини. Ово указује да наши подаци нису нарочито погодни за ову врсту кластерисања, али хајде да наставимо.

### Вежба - изградња модела

1. Увезите `KMeans` и започните процес кластерисања.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Постоји неколико делова овде који заслужују објашњење.

    > 🎓 range: Ово су итерације процеса кластерисања

    > 🎓 random_state: "Одређује генерисање случајних бројева за иницијализацију центроида." [Извор](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "сума квадрата унутар кластера" мери просечну квадратну удаљеност свих тачака унутар кластера до центроида кластера. [Извор](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Инерција: K-Means алгоритми покушавају да изаберу центроиде како би минимизовали 'инерцију', "меру колико су кластери унутрашње кохерентни." [Извор](https://scikit-learn.org/stable/modules/clustering.html). Вредност се додаје променљивој wcss на свакој итерацији.

    > 🎓 k-means++: У [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) можете користити 'k-means++' оптимизацију, која "иницијализује центроиде да буду (углавном) удаљени један од другог, што доводи до вероватно бољих резултата него случајна иницијализација."

### Елбоу метода

Раније сте претпоставили да, пошто сте циљали 3 жанра песама, треба да изаберете 3 кластера. Али да ли је то случај?

1. Користите 'елбоу методу' да се уверите.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Користите променљиву `wcss` коју сте изградили у претходном кораку да креирате графикон који показује где је 'прелом' у лакту, што указује на оптималан број кластера. Можда је то **заиста** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Вежба - приказ кластера

1. Поновите процес, овог пута постављајући три кластера, и прикажите кластере као scatterplot:

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

1. Проверите тачност модела:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Тачност овог модела није баш добра, а облик кластера вам даје наговештај зашто.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Ови подаци су превише неуравнотежени, премало корелисани и постоји превелика варијанса између вредности колона да би се добро кластерисали. У ствари, кластери који се формирају вероватно су снажно утицани или искривљени од стране три категорије жанра које смо горе дефинисали. То је био процес учења!

    У документацији Scikit-learn-а можете видети да модел попут овог, са кластерима који нису баш добро разграничени, има проблем 'варијансе':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Инфографика из Scikit-learn-а

## Варијанса

Варијанса је дефинисана као "просек квадратних разлика од средње вредности" [(Извор)](https://www.mathsisfun.com/data/standard-deviation.html). У контексту овог проблема кластерисања, она се односи на податке где бројеви нашег скупа података имају тенденцију да се превише удаљавају од средње вредности.

✅ Ово је одличан тренутак да размислите о свим начинима на које бисте могли да исправите овај проблем. Да ли да мало више прилагодите податке? Да ли да користите друге колоне? Да ли да користите другачији алгоритам? Савет: Пробајте [скалирање података](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) да их нормализујете и тестирајте друге колоне.

> Пробајте овај '[калкулатор варијансе](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' да боље разумете концепт.

---

## 🚀Изазов

Проведите мало времена са овим notebook-ом, прилагођавајући параметре. Можете ли побољшати тачност модела чишћењем података (на пример, уклањањем одступања)? Можете користити тежине да дате већу тежину одређеним узорцима података. Шта још можете урадити да креирате боље кластере?

Савет: Пробајте да скалирате своје податке. У notebook-у постоји коментарисан код који додаје стандардно скалирање како би колоне података више личиле једна на другу у смислу опсега. Приметићете да, иако силуетна оцена опада, 'прелом' у графикону лакта се изравнава. То је зато што остављање података нескалираним омогућава подацима са мањом варијансом да носе већу тежину. Прочитајте мало више о овом проблему [овде](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Квиз после предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Погледајте симулатор K-Means [као што је овај](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Можете користити овај алат за визуализацију узорака података и одређивање њихових центроида. Можете уређивати случајност података, број кластера и број центроида. Да ли вам ово помаже да добијете идеју о томе како се подаци могу груписати?

Такође, погледајте [овај материјал о K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) са Станфорда.

## Задатак

[Испробајте различите методе кластерисања](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.