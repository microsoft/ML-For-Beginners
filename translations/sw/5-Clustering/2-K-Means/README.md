<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T15:45:24+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "sw"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Katika somo hili, utajifunza jinsi ya kuunda makundi kwa kutumia Scikit-learn na seti ya data ya muziki wa Nigeria uliyoingiza awali. Tutazungumzia misingi ya K-Means kwa Clustering. Kumbuka kwamba, kama ulivyojifunza katika somo la awali, kuna njia nyingi za kufanya kazi na makundi, na mbinu unayotumia inategemea data yako. Tutajaribu K-Means kwa kuwa ni mbinu ya kawaida zaidi ya clustering. Twende kazi!

Maneno utakayojifunza:

- Silhouette scoring  
- Njia ya Elbow  
- Inertia  
- Variance  

## Utangulizi

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) ni mbinu inayotokana na uwanja wa usindikaji wa ishara. Inatumika kugawanya na kupanga vikundi vya data katika makundi 'k' kwa kutumia mfululizo wa uchunguzi. Kila uchunguzi hufanya kazi ya kuunganisha kipengele cha data kilicho karibu zaidi na 'mean' yake, au kituo cha kundi.

Makundi yanaweza kuonyeshwa kama [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram), ambazo zinajumuisha nukta (au 'mbegu') na eneo lake linalohusiana.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infographic na [Jen Looper](https://twitter.com/jenlooper)

Mchakato wa K-Means clustering [hufanyika kwa hatua tatu](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algorithimu huchagua idadi ya vituo vya k kwa kuchukua sampuli kutoka seti ya data. Baada ya hapo, inarudia:
    1. Inapanga kila sampuli kwa kituo kilicho karibu zaidi.
    2. Inaunda vituo vipya kwa kuchukua thamani ya wastani ya sampuli zote zilizopangwa kwa vituo vya awali.
    3. Kisha, inahesabu tofauti kati ya vituo vipya na vya zamani na kurudia hadi vituo viwe thabiti.

Changamoto moja ya kutumia K-Means ni kwamba unahitaji kuamua 'k', yaani idadi ya vituo. Kwa bahati nzuri, 'njia ya elbow' husaidia kukadiria thamani nzuri ya kuanzia kwa 'k'. Utajaribu muda si mrefu.

## Mahitaji ya awali

Utafanya kazi katika faili ya [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) ya somo hili ambayo inajumuisha uingizaji wa data na usafishaji wa awali uliofanya katika somo la mwisho.

## Zoezi - maandalizi

Anza kwa kuangalia tena data ya nyimbo.

1. Unda boxplot, ukitumia `boxplot()` kwa kila safu:

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

    Data hii ina kelele kidogo: kwa kuangalia kila safu kama boxplot, unaweza kuona outliers.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

Unaweza kupitia seti ya data na kuondoa outliers hizi, lakini hiyo itafanya data kuwa ndogo sana.

1. Kwa sasa, chagua safu ambazo utatumia kwa zoezi lako la clustering. Chagua zile zenye viwango vinavyofanana na encode safu ya `artist_top_genre` kama data ya nambari:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Sasa unahitaji kuchagua idadi ya makundi ya kulenga. Unajua kuna aina 3 za muziki ambazo tulizitambua kutoka seti ya data, kwa hivyo jaribu 3:

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

Unaona array iliyochapishwa na makundi yaliyotabiriwa (0, 1, au 2) kwa kila safu ya dataframe.

1. Tumia array hii kuhesabu 'silhouette score':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette score

Tafuta silhouette score iliyo karibu na 1. Alama hii inatofautiana kutoka -1 hadi 1, na ikiwa alama ni 1, kundi ni lenye msongamano na limetenganishwa vizuri na makundi mengine. Thamani karibu na 0 inawakilisha makundi yanayofuatana na sampuli zilizo karibu sana na mpaka wa maamuzi wa makundi jirani. [(Chanzo)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Alama yetu ni **.53**, kwa hivyo iko katikati. Hii inaonyesha kuwa data yetu haifai sana kwa aina hii ya clustering, lakini tuendelee.

### Zoezi - unda modeli

1. Ingiza `KMeans` na anza mchakato wa clustering.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Kuna sehemu chache hapa zinazostahili maelezo.

    > 🎓 range: Hizi ni marudio ya mchakato wa clustering  

    > 🎓 random_state: "Inaamua kizazi cha nambari za bahati nasibu kwa uanzishaji wa vituo." [Chanzo](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)  

    > 🎓 WCSS: "jumla ya mraba ndani ya kundi" hupima umbali wa wastani wa mraba wa pointi zote ndani ya kundi hadi kituo cha kundi. [Chanzo](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce).  

    > 🎓 Inertia: Algorithimu za K-Means hujaribu kuchagua vituo ili kupunguza 'inertia', "kipimo cha jinsi makundi yalivyo thabiti ndani." [Chanzo](https://scikit-learn.org/stable/modules/clustering.html). Thamani inaongezwa kwenye variable ya wcss kwa kila marudio.  

    > 🎓 k-means++: Katika [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) unaweza kutumia optimization ya 'k-means++', ambayo "inaanzisha vituo kuwa (kwa ujumla) mbali kutoka kwa kila mmoja, na kusababisha matokeo bora zaidi kuliko uanzishaji wa bahati nasibu."

### Njia ya Elbow

Hapo awali, ulidhani kwamba, kwa kuwa umelenga aina 3 za muziki, unapaswa kuchagua makundi 3. Lakini je, ni kweli?

1. Tumia 'njia ya elbow' kuhakikisha.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Tumia variable ya `wcss` uliyounda katika hatua ya awali kuunda chati inayoonyesha mahali ambapo 'kink' katika elbow iko, ambayo inaonyesha idadi bora ya makundi. Labda ni **3**!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Zoezi - onyesha makundi

1. Jaribu mchakato tena, wakati huu ukichagua makundi matatu, na onyesha makundi kama scatterplot:

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

1. Angalia usahihi wa modeli:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    Usahihi wa modeli hii si mzuri sana, na umbo la makundi linakupa dalili kwa nini.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    Data hii haina uwiano mzuri, haina uhusiano wa kutosha, na kuna tofauti kubwa sana kati ya thamani za safu ili kuunda makundi vizuri. Kwa kweli, makundi yanayoundwa huenda yameathiriwa sana au yamepotoshwa na aina tatu za muziki tulizotambua hapo juu. Hii ilikuwa mchakato wa kujifunza!

    Katika nyaraka za Scikit-learn, unaweza kuona kwamba modeli kama hii, yenye makundi yasiyo na mipaka dhahiri, ina tatizo la 'variance':

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infographic kutoka Scikit-learn

## Variance

Variance inafafanuliwa kama "wastani wa tofauti za mraba kutoka kwa Mean" [(Chanzo)](https://www.mathsisfun.com/data/standard-deviation.html). Katika muktadha wa tatizo hili la clustering, inahusu data ambayo nambari za seti yetu ya data zina mwelekeo wa kutofautiana sana kutoka kwa wastani.

✅ Huu ni wakati mzuri wa kufikiria njia zote unazoweza kutumia kurekebisha tatizo hili. Kuboresha data kidogo zaidi? Kutumia safu tofauti? Kutumia algorithimu tofauti? Kidokezo: Jaribu [kusawazisha data yako](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) ili kuifanya kuwa ya kawaida na ujaribu safu nyingine.

> Jaribu '[kikokotoo cha variance](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' ili kuelewa dhana hii zaidi.

---

## 🚀Changamoto

Tumia muda na notebook hii, ukibadilisha vigezo. Je, unaweza kuboresha usahihi wa modeli kwa kusafisha data zaidi (kwa mfano, kuondoa outliers)? Unaweza kutumia uzito ili kutoa uzito zaidi kwa sampuli fulani za data. Je, ni nini kingine unaweza kufanya ili kuunda makundi bora?

Kidokezo: Jaribu kusawazisha data yako. Kuna msimbo ulio na maoni katika notebook unaoongeza scaling ya kawaida ili kufanya safu za data zifanane zaidi kwa karibu katika suala la viwango. Utagundua kwamba ingawa silhouette score inashuka, 'kink' katika grafu ya elbow inakuwa laini. Hii ni kwa sababu kuacha data bila kusawazishwa kunaruhusu data yenye tofauti ndogo kuwa na uzito zaidi. Soma zaidi kuhusu tatizo hili [hapa](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujisomea

Angalia Simulator ya K-Means [kama hii](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). Unaweza kutumia zana hii kuonyesha pointi za data za sampuli na kuamua vituo vyake. Unaweza kuhariri nasibu ya data, idadi ya makundi na idadi ya vituo. Je, hii inakusaidia kupata wazo la jinsi data inaweza kugawanywa?

Pia, angalia [handout hii kuhusu K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) kutoka Stanford.

## Kazi

[Jaribu mbinu tofauti za clustering](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.