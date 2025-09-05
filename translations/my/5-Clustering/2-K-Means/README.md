<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T12:20:22+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "my"
}
-->
# K-Means clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

ဒီသင်ခန်းစာမှာ Scikit-learn နဲ့ နိုင်ဂျီးရီးယားဂီတဒေတာစနစ်ကို အသုံးပြုပြီး cluster တွေဖန်တီးနည်းကို သင်ယူပါမယ်။ K-Means Clustering ရဲ့ အခြေခံကို လေ့လာပါမယ်။ အရင်သင်ခန်းစာမှာ သင်ယူခဲ့သလို၊ cluster တွေကို အလုပ်လုပ်စေဖို့ နည်းလမ်းများစွာရှိပြီး သင့်ဒေတာပေါ်မူတည်ပြီး သင့်လျော်တဲ့နည်းလမ်းကို ရွေးချယ်ရပါမယ်။ K-Means က အများဆုံးအသုံးပြုတဲ့ clustering နည်းလမ်းဖြစ်တဲ့အတွက် အခုနည်းလမ်းကို စမ်းကြည့်ပါမယ်။ စလိုက်ကြစို့!

သင်လေ့လာရမယ့်အကြောင်းအရာများ:

- Silhouette scoring
- Elbow method
- Inertia
- Variance

## Introduction

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) က signal processing နယ်ပယ်ကနေ ဆင်းသက်လာတဲ့ နည်းလမ်းတစ်ခုဖြစ်ပါတယ်။ ဒါကို 'k' cluster တွေထဲမှာ observation တွေကို အသုံးပြုပြီး ဒေတာအုပ်စုတွေကို ခွဲခြားဖို့ အသုံးပြုပါတယ်။ observation တစ်ခုစီက cluster ရဲ့ center point (mean) နီးစပ်ဆုံးနေရာကို group ဖွဲ့ဖို့ အလုပ်လုပ်ပါတယ်။

ဒီ cluster တွေကို [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram) အနေနဲ့ visualization လုပ်နိုင်ပါတယ်။ Voronoi diagram တွေမှာ point (seed) နဲ့ region တွေပါဝင်ပါတယ်။

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> infographic by [Jen Looper](https://twitter.com/jenlooper)

K-Means clustering process [သုံးအဆင့်ဖြင့် အလုပ်လုပ်ပါတယ်](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. Algorithm က dataset ထဲကနေ k-number center points ကို ရွေးချယ်ပါတယ်။ အဲဒီနောက် loop လုပ်ပါတယ်:
    1. Sample တစ်ခုစီကို နီးစပ်ဆုံး centroid ကို assign လုပ်ပါတယ်။
    2. အရင် centroid တွေကို assign လုပ်ထားတဲ့ sample တွေကို mean value အနေနဲ့ အသစ်သော centroid တွေဖန်တီးပါတယ်။
    3. အဟောင်း centroid နဲ့ အသစ် centroid တွေကြားက အကွာအဝေးကိုတွက်ပြီး centroid တွေတည်ငြိမ်တဲ့အထိ ထပ်လုပ်ပါတယ်။

K-Means ရဲ့ အားနည်းချက်တစ်ခုက 'k' (centroid အရေအတွက်) ကို သတ်မှတ်ဖို့လိုပါတယ်။ ကံကောင်းစွာ 'elbow method' က 'k' ရဲ့ စတင်အရေအတွက်ကို ခန့်မှန်းဖို့ ကူညီပေးပါတယ်။ အခုလိုလုပ်ကြည့်ပါမယ်။

## Prerequisite

ဒီသင်ခန်းစာရဲ့ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) ဖိုင်မှာ အရင်သင်ခန်းစာမှာလုပ်ခဲ့တဲ့ data import နဲ့ preliminary cleaning ပါဝင်ပါတယ်။

## Exercise - preparation

သီချင်းဒေတာကို ပြန်ကြည့်ပါ။

1. Column တစ်ခုစီအတွက် `boxplot()` ကိုခေါ်ပြီး boxplot တစ်ခုဖန်တီးပါ:

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

    ဒီဒေတာက noise အနည်းငယ်ပါဝင်ပါတယ်။ Column တစ်ခုစီကို boxplot အနေနဲ့ကြည့်ပြီး outlier တွေကိုတွေ့နိုင်ပါတယ်။

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

ဒေတာထဲက outlier တွေကို ဖယ်ရှားနိုင်ပေမယ့် ဒေတာအရေအတွက်က နည်းသွားနိုင်ပါတယ်။

1. Clustering exercise အတွက် သုံးမယ့် column တွေကို ရွေးချယ်ပါ။ အတူတူ range ရှိတဲ့ column တွေကို ရွေးပြီး `artist_top_genre` column ကို numeric data အနေနဲ့ encode လုပ်ပါ:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Cluster အရေအတွက်ကို ရွေးချယ်ဖို့လိုပါတယ်။ Dataset ထဲက သီချင်းအမျိုးအစား 3 ခုရှိတာကိုသိပြီးသားဖြစ်တဲ့အတွက် 3 ကိုစမ်းကြည့်ပါ:

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

Dataframe ရဲ့ row တစ်ခုစီအတွက် (0, 1, 2) cluster တွေကို ခန့်မှန်းထားတဲ့ array ကို print ထုတ်ကြည့်နိုင်ပါတယ်။

1. ဒီ array ကို အသုံးပြုပြီး 'silhouette score' ကိုတွက်ပါ:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette score

Silhouette score 1 နီးစပ်တဲ့အရေအတွက်ကို ရှာပါ။ ဒီ score က -1 မှ 1 အထိရှိပြီး score 1 ဆို cluster က အထူးသီးသန့်ပြီး အခြား cluster တွေထဲက sample တွေကို ကောင်းစွာခွဲခြားထားပါတယ်။ 0 နီးစပ်တဲ့ value က cluster တွေ overlap ဖြစ်ပြီး sample တွေက decision boundary နီးစပ်နေတဲ့အခြေအနေကို ဖော်ပြပါတယ်။ [(Source)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

ကျွန်တော်တို့ရဲ့ score က **.53** ဖြစ်ပြီး အလယ်တန်းမှာရှိပါတယ်။ ဒါက ဒီ data က ဒီလို clustering နည်းလမ်းအတွက် သင့်လျော်မှုမရှိတာကို ဖော်ပြပါတယ်။ ဒါပေမယ့် ဆက်လုပ်ကြည့်ပါမယ်။

### Exercise - build a model

1. `KMeans` ကို import လုပ်ပြီး clustering process ကို စတင်ပါ။

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    ဒီမှာ အဓိကအချက်အချို့ကိုရှင်းပြပါမယ်။

    > 🎓 range: Clustering process ရဲ့ iteration တွေ

    > 🎓 random_state: "Centroid initialization အတွက် random number generation ကို သတ်မှတ်ပါတယ်။" [Source](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" က cluster centroid နဲ့ cluster ထဲမှာရှိတဲ့ point တွေကြားက squared average distance ကိုတိုင်းတာပါတယ်။ [Source](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce). 

    > 🎓 Inertia: K-Means algorithm တွေက 'inertia' ကို လျှော့ချဖို့ centroid တွေကို ရွေးချယ်ဖို့ကြိုးစားပါတယ်။ "Cluster တွေ internally coherent ဖြစ်မှုကိုတိုင်းတာတဲ့အချက်" [Source](https://scikit-learn.org/stable/modules/clustering.html). WCSS variable ထဲကို iteration တစ်ခုစီမှာ value ကိုထည့်သွင်းပါတယ်။

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) မှာ 'k-means++' optimization ကို အသုံးပြုနိုင်ပြီး "Centroid တွေကို အချင်းချင်းဝေးဝေးနေဖို့ initialize လုပ်ပြီး random initialization ထက်ပိုကောင်းတဲ့ရလဒ်ရဖို့ အကောင်းဆုံးဖြစ်နိုင်ပါတယ်။"

### Elbow method

အရင်က သီချင်းအမျိုးအစား 3 ခုကို target လုပ်ထားတဲ့အတွက် cluster 3 ခုကို ရွေးချယ်သင့်တယ်လို့ ခန့်မှန်းထားပါတယ်။ ဒါပေမယ့် အဲဒါမှန်ပါသလား?

1. 'elbow method' ကို အသုံးပြုပြီး သေချာစေပါ။

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    အရင်အဆင့်မှာ တည်ဆောက်ထားတဲ့ `wcss` variable ကို အသုံးပြုပြီး elbow graph ရဲ့ 'bend' ကိုဖော်ပြတဲ့ chart တစ်ခုဖန်တီးပါ။ ဒီနေရာက optimum cluster အရေအတွက်ကို ဖော်ပြပါတယ်။ 3 ဖြစ်နိုင်ပါတယ်!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Exercise - display the clusters

1. Process ကို ထပ်လုပ်ပြီး cluster 3 ခုကို သတ်မှတ်ပြီး scatterplot အနေနဲ့ cluster တွေကို ဖော်ပြပါ:

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

1. Model ရဲ့ accuracy ကိုစစ်ပါ:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    ဒီ model ရဲ့ accuracy က မကောင်းပါဘူး။ Cluster တွေ၏ shape က အကြောင်းပြချက်ကို ဖော်ပြပါတယ်။

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    ဒီ data က အလွန်မညီမျှမှုရှိပြီး column value တွေကြား variance များလွန်းတဲ့အတွက် cluster တွေကောင်းစွာဖွဲ့မရနိုင်ပါဘူး။ အမှန်တကယ် cluster တွေက အထက်မှာ သတ်မှတ်ထားတဲ့ genre category 3 ခုကြောင့် skew ဖြစ်နေပါတယ်။ ဒါက သင်ယူမှုဖြစ်ပါတယ်!

    Scikit-learn ရဲ့ documentation မှာ cluster တွေကောင်းစွာခွဲခြားမရတဲ့ model တစ်ခုက 'variance' ပြဿနာရှိတယ်လို့ ဖော်ပြထားပါတယ်။

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infographic from Scikit-learn

## Variance

Variance ကို "Mean နဲ့ အကွာအဝေး squared differences ရဲ့ အလယ်တန်း" အနေနဲ့ သတ်မှတ်ထားပါတယ် [(Source)](https://www.mathsisfun.com/data/standard-deviation.html). Clustering problem ရဲ့ အခြေအနေမှာ ဒေတာက mean နဲ့ အကွာအဝေးများလွန်းတာကို ဖော်ပြပါတယ်။

✅ ဒီအချိန်မှာ ဒီပြဿနာကို ဖြေရှင်းဖို့ နည်းလမ်းတွေကို စဉ်းစားဖို့ အချိန်ကောင်းတစ်ခုဖြစ်ပါတယ်။ ဒေတာကို နည်းနည်းပိုပြီး ပြင်ဆင်မလား? Column အခြားတစ်ခုကို အသုံးပြုမလား? Algorithm အခြားတစ်ခုကို စမ်းမလား? Hint: [ဒေတာကို scale လုပ်](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/)ပြီး normalize လုပ်ပြီး column အခြားတစ်ခုကို စမ်းကြည့်ပါ။

> '[variance calculator](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' ကို အသုံးပြုပြီး concept ကို နားလည်ပါ။

---

## 🚀Challenge

ဒီ notebook ကို အချိန်ယူပြီး parameter တွေကို ပြင်ဆင်ပါ။ ဒေတာကို ပိုပြီးသန့်စင် (outlier တွေဖယ်ရှားခြင်း) လုပ်ပြီး model ရဲ့ accuracy ကို တိုးမြှင့်နိုင်ပါသလား? Weight တွေကို အသုံးပြုပြီး data sample တစ်ခုချင်းစီကို ပိုအလေးထားနိုင်ပါတယ်။ Cluster တွေကို ပိုကောင်းစွာဖန်တီးဖို့ ဘာတွေလုပ်နိုင်မလဲ?

Hint: ဒေတာကို scale လုပ်ကြည့်ပါ။ Notebook ထဲမှာ commented code ရှိပြီး standard scaling ကိုထည့်သွင်းထားပါတယ်။ ဒါက data column တွေကို range အနေနဲ့ ပိုနီးစပ်စေပါတယ်။ Silhouette score က ကျသွားပေမယ့် elbow graph ရဲ့ 'kink' က smooth ဖြစ်သွားပါတယ်။ ဒေတာကို unscaled ထားရင် variance နည်းတဲ့ data က ပိုအလေးထားခံရပါတယ်။ ဒီပြဿနာအကြောင်းကို [ဒီမှာ](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226) ဖတ်ပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

K-Means Simulator [ဒီလို](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) တစ်ခုကို ကြည့်ပါ။ ဒီ tool ကို အသုံးပြုပြီး sample data point တွေကို visualization လုပ်ပြီး centroid တွေကို သတ်မှတ်နိုင်ပါတယ်။ ဒေတာရဲ့ randomness, cluster အရေအတွက်နဲ့ centroid အရေအတွက်ကို ပြင်ဆင်နိုင်ပါတယ်။ ဒေတာကို group ဖွဲ့နည်းကို နားလည်ဖို့ ကူညီပါသလား?

Stanford ရဲ့ [ဒီ K-Means handout](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) ကိုလည်း ကြည့်ပါ။

## Assignment

[Clustering နည်းလမ်းအခြားတစ်ခုကို စမ်းကြည့်ပါ](assignment.md)

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်ခြင်းတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါရှိနိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာရှိသော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအမှားများ သို့မဟုတ် အနားယူမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။