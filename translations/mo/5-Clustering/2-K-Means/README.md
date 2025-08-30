<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "085d571097d201810720df4cd379f8c2",
  "translation_date": "2025-08-29T21:05:55+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "mo"
}
-->
# K-Means 分群

## [課前測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/29/)

在本課程中，你將學習如何使用 Scikit-learn 和之前匯入的尼日利亞音樂數據集來建立分群。我們將介紹 K-Means 分群的基本概念。請記住，正如你在之前的課程中學到的，分群有許多不同的方法，選擇哪種方法取決於你的數據。我們將嘗試 K-Means，因為它是最常見的分群技術。讓我們開始吧！

你將學到的術語：

- Silhouette 分數
- 肘部法則
- 慣性
- 方差

## 介紹

[K-Means 分群](https://wikipedia.org/wiki/K-means_clustering) 是一種源自信號處理領域的方法。它用於將數據分成 "k" 個群組，並通過一系列觀察來進行分割。每次觀察都會將數據點分配到距離最近的 "均值"（即群組的中心點）。

這些群組可以用 [Voronoi 圖](https://wikipedia.org/wiki/Voronoi_diagram) 來視覺化，其中包括一個點（或 "種子"）及其對應的區域。

![voronoi 圖](../../../../translated_images/voronoi.1dc1613fb0439b9564615eca8df47a4bcd1ce06217e7e72325d2406ef2180795.mo.png)

> 信息圖由 [Jen Looper](https://twitter.com/jenlooper) 提供

K-Means 分群過程[分為三個步驟執行](https://scikit-learn.org/stable/modules/clustering.html#k-means)：

1. 演算法通過從數據集中抽樣選擇 k 個中心點。接著進行迴圈：
    1. 將每個樣本分配到最近的中心點。
    2. 通過計算分配到之前中心點的所有樣本的平均值來創建新的中心點。
    3. 計算新舊中心點之間的差異，並重複直到中心點穩定。

使用 K-Means 的一個缺點是需要確定 "k"，即中心點的數量。幸運的是，"肘部法則" 可以幫助估算 "k" 的良好起始值。你很快就會試試看。

## 前置條件

你將在本課程的 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) 文件中工作，其中包括你在上一課中完成的數據匯入和初步清理。

## 練習 - 準備工作

首先再次查看歌曲數據。

1. 為每一列調用 `boxplot()` 來創建箱型圖：

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

    這些數據有些雜亂：通過觀察每一列的箱型圖，你可以看到異常值。

    ![異常值](../../../../translated_images/boxplots.8228c29dabd0f29227dd38624231a175f411f1d8d4d7c012cb770e00e4fdf8b6.mo.png)

    你可以逐一檢查數據集並移除這些異常值，但這樣會使數據變得非常稀少。

1. 暫時選擇你將用於分群練習的列。選擇範圍相似的列，並將 `artist_top_genre` 列編碼為數值型數據：

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. 現在你需要選擇目標群組的數量。你知道數據集中有 3 個歌曲類型，因此我們嘗試 3：

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

你會看到一個陣列，列印出每行數據框的預測群組（0、1 或 2）。

1. 使用此陣列計算 "Silhouette 分數"：

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette 分數

尋找接近 1 的 Silhouette 分數。此分數範圍從 -1 到 1，如果分數為 1，則群組密集且與其他群組分離良好。接近 0 的值表示群組重疊，樣本非常接近相鄰群組的決策邊界。[(來源)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

我們的分數是 **0.53**，介於中間。這表明我們的數據並不特別適合這種分群方式，但我們繼續進行。

### 練習 - 建立模型

1. 匯入 `KMeans` 並開始分群過程。

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    這裡有幾個部分需要解釋。

    > 🎓 range：這是分群過程的迭代次數

    > 🎓 random_state："決定中心點初始化的隨機數生成。" [來源](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS："群組內平方和" 測量群組內所有點到群組中心點的平均平方距離。[來源](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 慣性：K-Means 演算法試圖選擇中心點以最小化 "慣性"，"慣性是群組內部一致性的度量。" [來源](https://scikit-learn.org/stable/modules/clustering.html)。該值在每次迭代中附加到 wcss 變數。

    > 🎓 k-means++：在 [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means) 中，你可以使用 "k-means++" 優化，這種方法初始化的中心點通常彼此距離較遠，可能比隨機初始化效果更好。

### 肘部法則

之前你推測，由於目標是 3 個歌曲類型，你應選擇 3 個群組。但真的是這樣嗎？

1. 使用 "肘部法則" 來確認。

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    使用你在前一步中建立的 `wcss` 變數來創建一個圖表，顯示肘部的 "彎曲" 處，這表明最佳群組數量。也許真的是 3！

    ![肘部法則](../../../../translated_images/elbow.72676169eed744ff03677e71334a16c6b8f751e9e716e3d7f40dd7cdef674cca.mo.png)

## 練習 - 顯示群組

1. 再次嘗試此過程，這次設置三個群組，並以散點圖顯示群組：

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

1. 檢查模型的準確性：

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    此模型的準確性不太好，群組的形狀給了你一些提示原因。

    ![群組](../../../../translated_images/clusters.b635354640d8e4fd4a49ef545495518e7be76172c97c13bd748f5b79f171f69a.mo.png)

    這些數據太不平衡，相關性太低，列值之間的方差太大，導致分群效果不佳。事實上，形成的群組可能受到我們之前定義的三個類型的影響或偏斜。這是一個學習過程！

    在 Scikit-learn 的文檔中，你可以看到像這樣的模型，群組分界不明顯，存在 "方差" 問題：

    ![問題模型](../../../../translated_images/problems.f7fb539ccd80608e1f35c319cf5e3ad1809faa3c08537aead8018c6b5ba2e33a.mo.png)
    > 信息圖來自 Scikit-learn

## 方差

方差被定義為 "與平均值的平方差的平均值" [(來源)](https://www.mathsisfun.com/data/standard-deviation.html)。在此分群問題的背景下，它指的是數據集中數值偏離平均值的程度。

✅ 這是一個很好的時機來思考如何解決這個問題。稍微調整數據？使用不同的列？使用不同的演算法？提示：嘗試[縮放數據](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/)以進行標準化並測試其他列。

> 試試這個 "[方差計算器](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)" 來更好地理解這個概念。

---

## 🚀挑戰

花些時間在這個 notebook 上，調整參數。你能否通過更多清理數據（例如移除異常值）來提高模型的準確性？你可以使用權重來給某些數據樣本更高的權重。還有什麼方法可以創建更好的群組？

提示：嘗試縮放數據。notebook 中有註解的程式碼，添加標準縮放以使數據列在範圍上更接近。你會發現，雖然 Silhouette 分數下降了，但肘部圖中的 "彎曲" 更平滑了。這是因為未縮放的數據允許方差較小的數據具有更大的影響力。閱讀更多相關問題[這裡](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)。

## [課後測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/30/)

## 回顧與自學

看看 K-Means 模擬器[例如這個](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/)。你可以使用這個工具來視覺化樣本數據點並確定其中心點。你可以編輯數據的隨機性、群組數量和中心點數量。這是否幫助你更好地理解數據如何分群？

另外，看看 [這份來自 Stanford 的 K-Means 手冊](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)。

## 作業

[嘗試不同的分群方法](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力確保翻譯的準確性，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵信息，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或誤釋不承擔責任。