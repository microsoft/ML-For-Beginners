<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T06:58:34+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "pa"
}
-->
# ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਸਿੱਖੋਗੇ ਕਿ ਕਿਵੇਂ Scikit-learn ਅਤੇ ਨਾਈਜੀਰੀਆਈ ਮਿਊਜ਼ਿਕ ਡਾਟਾਸੈਟ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕਲੱਸਟਰ ਬਣਾਉਣੇ ਹਨ। ਅਸੀਂ ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ ਦੇ ਬੁਨਿਆਦੀ ਸਿਧਾਂਤਾਂ ਨੂੰ ਕਵਰ ਕਰਾਂਗੇ। ਯਾਦ ਰੱਖੋ ਕਿ, ਜਿਵੇਂ ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਸਿੱਖਿਆ ਸੀ, ਕਲੱਸਟਰਾਂ ਨਾਲ ਕੰਮ ਕਰਨ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ ਅਤੇ ਤੁਸੀਂ ਜੋ ਵਿਧੀ ਵਰਤਦੇ ਹੋ ਉਹ ਤੁਹਾਡੇ ਡਾਟਾ 'ਤੇ ਨਿਰਭਰ ਕਰਦੀ ਹੈ। ਅਸੀਂ ਕੇ-ਮੀਨਜ਼ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਾਂਗੇ ਕਿਉਂਕਿ ਇਹ ਸਭ ਤੋਂ ਆਮ ਕਲੱਸਟਰਿੰਗ ਤਕਨੀਕ ਹੈ। ਚਲੋ ਸ਼ੁਰੂ ਕਰੀਏ!

ਤੁਸੀਂ ਜਿਨ੍ਹਾਂ ਸ਼ਬਦਾਂ ਬਾਰੇ ਸਿੱਖੋਗੇ:

- ਸਿਲਹੂਟ ਸਕੋਰਿੰਗ
- ਐਲਬੋ ਵਿਧੀ
- ਇਨਰਸ਼ੀਆ
- ਵੈਰੀਅੰਸ

## ਪਰਿਚਯ

[ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ](https://wikipedia.org/wiki/K-means_clustering) ਸਿਗਨਲ ਪ੍ਰੋਸੈਸਿੰਗ ਦੇ ਖੇਤਰ ਤੋਂ ਲਿਆ ਗਿਆ ਇੱਕ ਤਰੀਕਾ ਹੈ। ਇਹ ਡਾਟਾ ਦੇ ਸਮੂਹਾਂ ਨੂੰ 'k' ਕਲੱਸਟਰਾਂ ਵਿੱਚ ਵੰਡਣ ਅਤੇ ਵਿਭਾਜਿਤ ਕਰਨ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ। ਹਰ ਇੱਕ ਅਧਿਐਨ ਇੱਕ ਦਿੱਤੇ ਡਾਟਾਪੌਇੰਟ ਨੂੰ ਉਸਦੇ ਸਭ ਤੋਂ ਨੇੜੇ 'mean', ਜਾਂ ਕਲੱਸਟਰ ਦੇ ਕੇਂਦਰੀ ਬਿੰਦੂ ਦੇ ਨੇੜੇ ਸਮੂਹਬੱਧ ਕਰਨ ਲਈ ਕੰਮ ਕਰਦਾ ਹੈ।

ਕਲੱਸਟਰਾਂ ਨੂੰ [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram) ਵਜੋਂ ਦਿਖਾਇਆ ਜਾ ਸਕਦਾ ਹੈ, ਜੋ ਇੱਕ ਬਿੰਦੂ (ਜਾਂ 'seed') ਅਤੇ ਇਸਦੇ ਸੰਬੰਧਿਤ ਖੇਤਰ ਨੂੰ ਸ਼ਾਮਲ ਕਰਦੇ ਹਨ।

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> ਇਨਫੋਗ੍ਰਾਫਿਕ [Jen Looper](https://twitter.com/jenlooper) ਦੁਆਰਾ

ਕੇ-ਮੀਨਜ਼ ਕਲੱਸਟਰਿੰਗ ਪ੍ਰਕਿਰਿਆ [ਤਿੰਨ-ਕਦਮ ਦੀ ਪ੍ਰਕਿਰਿਆ ਵਿੱਚ ਚਲਦੀ ਹੈ](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. ਐਲਗੋਰਿਦਮ ਡਾਟਾਸੈਟ ਤੋਂ ਨਮੂਨਾ ਲੈ ਕੇ k-ਗਿਣਤੀ ਕੇਂਦਰੀ ਬਿੰਦੂ ਚੁਣਦਾ ਹੈ। ਇਸ ਤੋਂ ਬਾਅਦ ਇਹ ਲੂਪ ਕਰਦਾ ਹੈ:
    1. ਇਹ ਹਰ ਨਮੂਨੇ ਨੂੰ ਸਭ ਤੋਂ ਨੇੜੇ ਸੈਂਟਰਾਇਡ ਨੂੰ ਅਲਾਟ ਕਰਦਾ ਹੈ।
    2. ਇਹ ਪਿਛਲੇ ਸੈਂਟਰਾਇਡਾਂ ਨੂੰ ਅਲਾਟ ਕੀਤੇ ਗਏ ਸਾਰੇ ਨਮੂਨਿਆਂ ਦੇ ਮੀਨ ਮੁੱਲ ਲੈ ਕੇ ਨਵੇਂ ਸੈਂਟਰਾਇਡ ਬਣਾਉਂਦਾ ਹੈ।
    3. ਫਿਰ, ਇਹ ਨਵੇਂ ਅਤੇ ਪੁਰਾਣੇ ਸੈਂਟਰਾਇਡਾਂ ਦੇ ਵਿਚਕਾਰ ਅੰਤਰ ਦੀ ਗਣਨਾ ਕਰਦਾ ਹੈ ਅਤੇ ਸੈਂਟਰਾਇਡਾਂ ਸਥਿਰ ਹੋਣ ਤੱਕ ਦੁਹਰਾਉਂਦਾ ਹੈ।

ਕੇ-ਮੀਨਜ਼ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦਾ ਇੱਕ ਨੁਕਸਾਨ ਇਹ ਹੈ ਕਿ ਤੁਹਾਨੂੰ 'k', ਯਾਨੀ ਸੈਂਟਰਾਇਡਾਂ ਦੀ ਗਿਣਤੀ ਸਥਾਪਿਤ ਕਰਨੀ ਪਵੇਗੀ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ 'ਐਲਬੋ ਵਿਧੀ' 'k' ਲਈ ਇੱਕ ਚੰਗੇ ਸ਼ੁਰੂਆਤੀ ਮੁੱਲ ਦਾ ਅਨੁਮਾਨ ਲਗਾਉਣ ਵਿੱਚ ਮਦਦ ਕਰਦੀ ਹੈ। ਤੁਸੀਂ ਇਸਨੂੰ ਇੱਕ ਮਿੰਟ ਵਿੱਚ ਅਜ਼ਮਾਓਗੇ।

## ਪੂਰਵ ਸ਼ਰਤ

ਤੁਸੀਂ ਇਸ ਪਾਠ ਦੇ [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) ਫਾਈਲ ਵਿੱਚ ਕੰਮ ਕਰੋਗੇ ਜਿਸ ਵਿੱਚ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਕੀਤੇ ਡਾਟਾ ਇੰਪੋਰਟ ਅਤੇ ਪ੍ਰਾਰੰਭਿਕ ਸਫਾਈ ਸ਼ਾਮਲ ਹੈ।

## ਅਭਿਆਸ - ਤਿਆਰੀ

ਗਾਣਿਆਂ ਦੇ ਡਾਟਾ ਨੂੰ ਮੁੜ ਦੇਖੋ।

1. ਹਰ ਕਾਲਮ ਲਈ `boxplot()` ਕਾਲ ਕਰਕੇ ਇੱਕ ਬਾਕਸਪਲਾਟ ਬਣਾਓ:

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

    ਇਹ ਡਾਟਾ ਥੋੜਾ ਸ਼ੋਰ ਵਾਲਾ ਹੈ: ਹਰ ਕਾਲਮ ਨੂੰ ਬਾਕਸਪਲਾਟ ਵਜੋਂ ਦੇਖ ਕੇ, ਤੁਸੀਂ ਆਉਟਲਾਇਰਜ਼ ਨੂੰ ਦੇਖ ਸਕਦੇ ਹੋ।

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

ਤੁਸੀਂ ਡਾਟਾਸੈਟ ਵਿੱਚੋਂ ਆਉਟਲਾਇਰਜ਼ ਨੂੰ ਹਟਾ ਸਕਦੇ ਹੋ, ਪਰ ਇਸ ਨਾਲ ਡਾਟਾ ਕਾਫ਼ੀ ਘੱਟ ਹੋ ਜਾਵੇਗਾ।

1. ਫਿਲਹਾਲ, ਉਹ ਕਾਲਮ ਚੁਣੋ ਜੋ ਤੁਸੀਂ ਆਪਣੇ ਕਲੱਸਟਰਿੰਗ ਅਭਿਆਸ ਲਈ ਵਰਤੋਗੇ। ਉਹਨਾਂ ਨੂੰ ਚੁਣੋ ਜਿਨ੍ਹਾਂ ਦੀ ਰੇਂਜ ਇੱਕ ਜੇਹੀ ਹੈ ਅਤੇ `artist_top_genre` ਕਾਲਮ ਨੂੰ ਨੰਬਰਿਕ ਡਾਟਾ ਵਜੋਂ ਐਨਕੋਡ ਕਰੋ:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. ਹੁਣ ਤੁਹਾਨੂੰ ਇਹ ਚੁਣਨਾ ਪਵੇਗਾ ਕਿ ਕਿੰਨੇ ਕਲੱਸਟਰ ਟਾਰਗਟ ਕਰਨੇ ਹਨ। ਤੁਹਾਨੂੰ ਪਤਾ ਹੈ ਕਿ ਡਾਟਾਸੈਟ ਵਿੱਚੋਂ ਅਸੀਂ 3 ਗਾਣਿਆਂ ਦੇ ਜਾਨਰ ਨਿਕਾਲੇ ਹਨ, ਤਾਂ ਆਓ 3 ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੀਏ:

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

ਤੁਹਾਨੂੰ ਇੱਕ ਐਰੇ ਪ੍ਰਿੰਟ ਕੀਤਾ ਹੋਇਆ ਦਿਖਾਈ ਦੇਵੇਗਾ ਜਿਸ ਵਿੱਚ ਡਾਟਾਫਰੇਮ ਦੀ ਹਰ ਪੰਗਤੀ ਲਈ ਅਨੁਮਾਨਿਤ ਕਲੱਸਟਰ (0, 1, ਜਾਂ 2) ਹਨ।

1. ਇਸ ਐਰੇ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ 'ਸਿਲਹੂਟ ਸਕੋਰ' ਦੀ ਗਣਨਾ ਕਰੋ:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## ਸਿਲਹੂਟ ਸਕੋਰ

ਸਿਲਹੂਟ ਸਕੋਰ 1 ਦੇ ਨੇੜੇ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ। ਇਹ ਸਕੋਰ -1 ਤੋਂ 1 ਤੱਕ ਵੱਖ-ਵੱਖ ਹੁੰਦਾ ਹੈ, ਅਤੇ ਜੇ ਸਕੋਰ 1 ਹੈ, ਤਾਂ ਕਲੱਸਟਰ ਸੰਘਣੇ ਅਤੇ ਹੋਰ ਕਲੱਸਟਰਾਂ ਤੋਂ ਚੰਗੀ ਤਰ੍ਹਾਂ ਵੱਖਰੇ ਹਨ। 0 ਦੇ ਨੇੜੇ ਮੁੱਲ ਉਹ ਕਲੱਸਟਰ ਦਰਸਾਉਂਦਾ ਹੈ ਜੋ ਇੱਕ ਦੂਜੇ ਦੇ ਨੇੜੇ ਹਨ ਅਤੇ ਜਿਨ੍ਹਾਂ ਦੇ ਨਮੂਨੇ ਫੈਸਲੇ ਦੀ ਸੀਮਾ ਦੇ ਨੇੜੇ ਹਨ। [(Source)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

ਸਾਡਾ ਸਕੋਰ **.53** ਹੈ, ਜੋ ਬਿਲਕੁਲ ਵਿਚਕਾਰ ਹੈ। ਇਹ ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਸਾਡਾ ਡਾਟਾ ਇਸ ਤਰ੍ਹਾਂ ਦੀ ਕਲੱਸਟਰਿੰਗ ਲਈ ਖਾਸ ਤੌਰ 'ਤੇ ਉਚਿਤ ਨਹੀਂ ਹੈ, ਪਰ ਚਲੋ ਅੱਗੇ ਵਧਦੇ ਹਾਂ।

### ਅਭਿਆਸ - ਮਾਡਲ ਬਣਾਉਣਾ

1. `KMeans` ਨੂੰ ਇੰਪੋਰਟ ਕਰੋ ਅਤੇ ਕਲੱਸਟਰਿੰਗ ਪ੍ਰਕਿਰਿਆ ਸ਼ੁਰੂ ਕਰੋ।

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    ਇੱਥੇ ਕੁਝ ਹਿੱਸੇ ਹਨ ਜੋ ਵਿਆਖਿਆ ਦੇ ਯੋਗ ਹਨ।

    > 🎓 ਰੇਂਜ: ਇਹ ਕਲੱਸਟਰਿੰਗ ਪ੍ਰਕਿਰਿਆ ਦੇ ਇਤਰਾਏਸ਼ਨ ਹਨ।

    > 🎓 random_state: "ਸੈਂਟਰਾਇਡ ਇਨੀਸ਼ੀਅਲਾਈਜ਼ੇਸ਼ਨ ਲਈ ਰੈਂਡਮ ਨੰਬਰ ਜਨਰੇਸ਼ਨ ਨੂੰ ਨਿਰਧਾਰਤ ਕਰਦਾ ਹੈ।" [Source](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "within-cluster sums of squares" ਕਲੱਸਟਰ ਸੈਂਟਰਾਇਡ ਦੇ ਨਾਲ ਕਲੱਸਟਰ ਦੇ ਅੰਦਰ ਸਾਰੇ ਬਿੰਦੂਆਂ ਦੀ ਵਰਤਮਾਨ ਦੂਰੀ ਨੂੰ ਮਾਪਦਾ ਹੈ। [Source](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 ਇਨਰਸ਼ੀਆ: ਕੇ-ਮੀਨਜ਼ ਐਲਗੋਰਿਦਮ 'ਇਨਰਸ਼ੀਆ' ਨੂੰ ਘਟਾਉਣ ਲਈ ਸੈਂਟਰਾਇਡ ਚੁਣਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦਾ ਹੈ, "ਇਹ ਮਾਪਦਾ ਹੈ ਕਿ ਕਲੱਸਟਰ ਅੰਦਰੂਨੀ ਤੌਰ 'ਤੇ ਕਿੰਨੇ ਸੰਗਠਿਤ ਹਨ।" [Source](https://scikit-learn.org/stable/modules/clustering.html)

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) ਵਿੱਚ ਤੁਸੀਂ 'k-means++' ਅਪਟਿਮਾਈਜ਼ੇਸ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰ ਸਕਦੇ ਹੋ, ਜੋ "ਸੈਂਟਰਾਇਡਾਂ ਨੂੰ ਇੱਕ ਦੂਜੇ ਤੋਂ ਦੂਰ ਇਨੀਸ਼ੀਅਲਾਈਜ਼ ਕਰਦਾ ਹੈ, ਜਿਸ ਨਾਲ ਸ਼ਾਇਦ ਬਿਹਤਰ ਨਤੀਜੇ ਮਿਲਦੇ ਹਨ।"

### ਐਲਬੋ ਵਿਧੀ

ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਅਨੁਮਾਨ ਲਗਾਇਆ ਕਿ, ਕਿਉਂਕਿ ਤੁਸੀਂ 3 ਗਾਣਿਆਂ ਦੇ ਜਾਨਰ ਟਾਰਗਟ ਕੀਤੇ ਹਨ, ਤੁਹਾਨੂੰ 3 ਕਲੱਸਟਰ ਚੁਣਣੇ ਚਾਹੀਦੇ ਹਨ। ਪਰ ਕੀ ਇਹ ਸਹੀ ਹੈ?

1. 'ਐਲਬੋ ਵਿਧੀ' ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇਹ ਪੱਕਾ ਕਰੋ।

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    ਪਿਛਲੇ ਕਦਮ ਵਿੱਚ ਬਣਾਏ `wcss` ਵੈਰੀਏਬਲ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਚਾਰਟ ਬਣਾਓ ਜੋ 'ਐਲਬੋ' ਵਿੱਚ ਮੋੜ ਦਰਸਾਉਂਦਾ ਹੈ, ਜੋ ਕਲੱਸਟਰਾਂ ਦੀ ਸੰਖਿਆ ਦਾ ਸਹੀ ਮੁੱਲ ਦਰਸਾਉਂਦਾ ਹੈ। ਸ਼ਾਇਦ ਇਹ **3** ਹੀ ਹੈ!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## ਅਭਿਆਸ - ਕਲੱਸਟਰਾਂ ਨੂੰ ਦਿਖਾਉਣਾ

1. ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਮੁੜ ਅਜ਼ਮਾਓ, ਇਸ ਵਾਰ ਤਿੰਨ ਕਲੱਸਟਰ ਸੈਟ ਕਰੋ, ਅਤੇ ਕਲੱਸਟਰਾਂ ਨੂੰ ਸਕੈਟਰਪਲਾਟ ਵਜੋਂ ਦਿਖਾਓ:

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

1. ਮਾਡਲ ਦੀ ਸ਼ੁੱਧਤਾ ਦੀ ਜਾਂਚ ਕਰੋ:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    ਇਸ ਮਾਡਲ ਦੀ ਸ਼ੁੱਧਤਾ ਬਹੁਤ ਚੰਗੀ ਨਹੀਂ ਹੈ, ਅਤੇ ਕਲੱਸਟਰਾਂ ਦੀ ਸ਼ਕਲ ਤੁਹਾਨੂੰ ਇਸਦਾ ਕਾਰਨ ਦਿਖਾਉਂਦੀ ਹੈ।

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    ਇਹ ਡਾਟਾ ਬਹੁਤ ਅਸਮਾਨ ਹੈ, ਬਹੁਤ ਘੱਟ ਸੰਬੰਧਿਤ ਹੈ ਅਤੇ ਕਾਲਮ ਮੁੱਲਾਂ ਵਿੱਚ ਬਹੁਤ ਜ਼ਿਆਦਾ ਵੈਰੀਅੰਸ ਹੈ ਜੋ ਚੰਗੀ ਤਰ੍ਹਾਂ ਕਲੱਸਟਰ ਨਹੀਂ ਕਰ ਸਕਦਾ। ਅਸਲ ਵਿੱਚ, ਜੋ ਕਲੱਸਟਰ ਬਣਦੇ ਹਨ ਉਹ ਸ਼ਾਇਦ ਉਪਰੋਕਤ ਤਿੰਨ ਜਾਨਰ ਸ਼੍ਰੇਣੀਆਂ ਦੁਆਰਾ ਭਾਰੀ ਪ੍ਰਭਾਵਿਤ ਜਾਂ ਝੁਕਾਏ ਗਏ ਹਨ। ਇਹ ਇੱਕ ਸਿੱਖਣ ਦੀ ਪ੍ਰਕਿਰਿਆ ਸੀ!

    Scikit-learn ਦੀ ਦਸਤਾਵੇਜ਼ੀ ਵਿੱਚ, ਤੁਸੀਂ ਦੇਖ ਸਕਦੇ ਹੋ ਕਿ ਇਸ ਤਰ੍ਹਾਂ ਦਾ ਮਾਡਲ, ਜਿਸ ਵਿੱਚ ਕਲੱਸਟਰ ਬਹੁਤ ਚੰਗੀ ਤਰ੍ਹਾਂ ਵੱਖਰੇ ਨਹੀਂ ਹਨ, 'ਵੈਰੀਅੰਸ' ਸਮੱਸਿਆ ਰੱਖਦਾ ਹੈ:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > ਇਨਫੋਗ੍ਰਾਫਿਕ Scikit-learn ਤੋਂ

## ਵੈਰੀਅੰਸ

ਵੈਰੀਅੰਸ ਨੂੰ "ਮੀਨ ਤੋਂ ਵਰਤਮਾਨ ਅੰਤਰਾਂ ਦੇ ਵਰਗਾਂ ਦਾ ਔਸਤ" ਵਜੋਂ ਪਰਿਭਾਸ਼ਿਤ ਕੀਤਾ ਜਾਂਦਾ ਹੈ [(Source)](https://www.mathsisfun.com/data/standard-deviation.html)। ਇਸ ਕਲੱਸਟਰਿੰਗ ਸਮੱਸਿਆ ਦੇ ਸੰਦਰਭ ਵਿੱਚ, ਇਹ ਦਰਸਾਉਂਦਾ ਹੈ ਕਿ ਸਾਡੇ ਡਾਟਾਸੈਟ ਦੇ ਨੰਬਰ ਮੀਨ ਤੋਂ ਕਾਫ਼ੀ ਵੱਖਰੇ ਹਨ।

✅ ਇਹ ਇੱਕ ਸ਼ਾਨਦਾਰ ਮੌਕਾ ਹੈ ਇਹ ਸੋਚਣ ਲਈ ਕਿ ਤੁਸੀਂ ਇਸ ਸਮੱਸਿਆ ਨੂੰ ਠੀਕ ਕਰਨ ਦੇ ਕਿੰਨੇ ਤਰੀਕੇ ਅਪਣਾਉ ਸਕਦੇ ਹੋ। ਡਾਟਾ ਨੂੰ ਹੋਰ ਠੀਕ ਕਰੋ? ਵੱਖਰੇ ਕਾਲਮ ਵਰਤੋ? ਵੱਖਰੀ ਐਲਗੋਰਿਦਮ ਦੀ ਵਰਤੋਂ ਕਰੋ? ਸੰਕੇਤ: ਆਪਣੇ ਡਾਟਾ ਨੂੰ [ਸਕੇਲਿੰਗ](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) ਕਰਕੇ ਨਾਰਮਲਾਈਜ਼ ਕਰੋ ਅਤੇ ਹੋਰ ਕਾਲਮਾਂ ਦੀ ਜਾਂਚ ਕਰੋ।

> ਇਸ '[ਵੈਰੀਅੰਸ ਕੈਲਕੂਲੇਟਰ](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਤਾਂ ਜੋ ਇਸ ਧਾਰਨਾ ਨੂੰ ਹੋਰ ਚੰਗੀ ਤਰ੍ਹਾਂ ਸਮਝ ਸਕੋ।

---

## 🚀ਚੁਣੌਤੀ

ਇਸ ਨੋਟਬੁੱਕ ਨਾਲ ਕੁਝ ਸਮਾਂ ਬਿਤਾਓ ਅਤੇ ਪੈਰਾਮੀਟਰਾਂ ਨੂੰ ਠੀਕ ਕਰੋ। ਕੀ ਤੁਸੀਂ ਡਾਟਾ ਨੂੰ ਹੋਰ ਸਾਫ਼ ਕਰਕੇ (ਜਿਵੇਂ ਕਿ ਆਉਟਲਾਇਰਜ਼ ਨੂੰ ਹਟਾ ਕੇ) ਮਾਡਲ ਦੀ ਸ਼ੁੱਧਤਾ ਵਿੱਚ ਸੁਧਾਰ ਕਰ ਸਕਦੇ ਹੋ? ਤੁਸੀਂ ਵਜ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕੁਝ ਡਾਟਾ ਨਮੂਨਿਆਂ ਨੂੰ ਹੋਰ ਵਧੇਰੇ ਵਜ਼ਨ ਦੇ ਸਕਦੇ ਹੋ। ਕੀ ਤੁਸੀਂ ਹੋਰ ਕੁਝ ਕਰ ਸਕਦੇ ਹੋ ਜੋ ਚੰਗੇ ਕਲੱਸਟਰ ਬਣਾਉਣ ਵਿੱਚ ਮਦਦ ਕਰੇ?

ਸੰਕੇਤ: ਆਪਣੇ ਡਾਟਾ ਨੂੰ ਸਕੇਲ ਕਰੋ। ਨੋਟਬੁੱਕ ਵਿੱਚ ਟਿੱਪਣੀ ਕੀਤੀ ਕੋਡ ਹੈ ਜੋ ਸਟੈਂਡਰਡ ਸਕੇਲਿੰਗ ਨੂੰ ਸ਼ਾਮਲ ਕਰਦਾ ਹੈ ਤਾਂ ਜੋ ਡਾਟਾ ਕਾਲਮ ਇੱਕ ਦੂਜੇ ਦੇ ਨੇੜੇ ਹੋਣ ਰੇਂਜ ਦੇ ਸੰਦਰਭ ਵਿੱਚ। ਤੁਸੀਂ ਪਾਓਗੇ ਕਿ ਜਦੋਂ ਸਿਲਹੂਟ ਸਕੋਰ ਘਟਦਾ ਹੈ, ਐਲਬੋ ਗ੍ਰਾਫ ਵਿੱਚ 'ਕਿੰਕ' ਸਮੂਥ ਹੋ ਜਾਂਦਾ ਹੈ। ਇਹ ਇਸ ਲਈ ਹੈ ਕਿਉਂਕਿ ਡਾਟਾ ਨੂੰ ਅਨਸਕੇਲ ਛੱਡਣ ਨਾਲ ਘੱਟ ਵੈਰੀਅੰਸ ਵਾਲੇ ਡਾਟਾ ਨੂੰ ਵਧੇਰੇ ਵਜ਼ਨ ਮਿਲਦਾ ਹੈ। ਇਸ ਸਮੱਸਿਆ ਬਾਰੇ ਹੋਰ ਪੜ੍ਹੋ [ਇੱਥੇ](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)।

## [ਪੋਸਟ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ-ਅਧਿਐਨ

ਕੇ-ਮੀਨਜ਼ ਸਿਮੂਲੇਟਰ [ਜਿਵੇਂ ਕਿ ਇਹ](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/) ਦੇਖੋ। ਤੁਸੀਂ ਇਸ ਟੂਲ ਦੀ ਵਰਤੋਂ ਨਮੂਨਾ ਡਾਟਾ ਬਿੰਦੂਆਂ ਨੂੰ ਦ੍ਰਿਸ਼ਮਾਨ ਕਰਨ ਅਤੇ ਇਸਦੇ ਸੈਂਟਰਾਇਡਾਂ ਨੂੰ ਨਿਰਧਾਰਤ ਕਰਨ ਲਈ ਕਰ ਸਕਦੇ ਹੋ। ਤੁਸੀਂ ਡਾਟਾ ਦੀ randomness, ਕਲੱਸਟਰਾਂ ਦੀ ਗਿਣਤੀ ਅਤੇ ਸੈਂਟਰਾਇਡਾਂ ਦੀ ਗਿਣਤੀ ਨੂੰ ਸੰਪਾਦਿਤ ਕਰ ਸਕਦੇ ਹੋ। ਕੀ ਇਸ ਨਾਲ ਤੁਹਾਨੂੰ ਇਹ ਸਮਝਣ ਵਿੱਚ ਮਦਦ ਮਿਲਦੀ ਹੈ ਕਿ ਡਾਟਾ ਨੂੰ ਕਿਵੇਂ ਸਮੂਹਬੱਧ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ?

ਇਸ ਤੋਂ ਇਲਾਵਾ, [Stanford](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) ਤੋਂ ਕੇ-ਮੀਨਜ਼ 'ਤੇ ਇਸ ਹੈਂਡਆਉਟ ਨੂੰ ਦੇਖੋ।

## ਅਸਾਈਨਮੈਂਟ

[ਵੱਖਰੀ ਕਲੱਸਟਰਿੰਗ ਵਿਧੀਆਂ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ](assignment.md)

---

**ਅਸਵੀਕਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਅਤ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚੀਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।