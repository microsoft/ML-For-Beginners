<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "085d571097d201810720df4cd379f8c2",
  "translation_date": "2025-09-03T23:10:21+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ja"
}
-->
# K-Means クラスタリング

## [事前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/29/)

このレッスンでは、Scikit-learnと以前インポートしたナイジェリア音楽データセットを使用してクラスタを作成する方法を学びます。K-Meansを使用したクラスタリングの基本について説明します。前回のレッスンで学んだように、クラスタリングにはさまざまな方法があり、使用する方法はデータに依存します。ここでは、最も一般的なクラスタリング手法であるK-Meansを試してみましょう。それでは始めましょう！

このレッスンで学ぶ用語:

- シルエットスコア
- エルボー法
- 慣性 (Inertia)
- 分散 (Variance)

## はじめに

[K-Meansクラスタリング](https://wikipedia.org/wiki/K-means_clustering)は、信号処理の分野から派生した手法です。この手法は、データを「k」個のクラスタに分割するために使用され、観測値の一連の操作を通じて行われます。各観測値は、与えられたデータポイントを最も近い「平均値」またはクラスタの中心点にグループ化する役割を果たします。

クラスタは、[ボロノイ図](https://wikipedia.org/wiki/Voronoi_diagram)として視覚化することができます。これは、点（または「シード」）とその対応する領域を含みます。

![ボロノイ図](../../../../translated_images/voronoi.1dc1613fb0439b9564615eca8df47a4bcd1ce06217e7e72325d2406ef2180795.ja.png)

> インフォグラフィック: [Jen Looper](https://twitter.com/jenlooper)

K-Meansクラスタリングのプロセスは、[3つのステップで実行されます](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. アルゴリズムは、データセットからk個の中心点を選択します。その後、以下を繰り返します:
    1. 各サンプルを最も近いセントロイドに割り当てます。
    2. 前のセントロイドに割り当てられたすべてのサンプルの平均値を取ることで、新しいセントロイドを作成します。
    3. 新しいセントロイドと古いセントロイドの差を計算し、セントロイドが安定するまで繰り返します。

K-Meansの欠点の1つは、「k」、つまりセントロイドの数を事前に決定する必要があることです。ただし、「エルボー法」を使用すると、「k」の適切な初期値を推定するのに役立ちます。これをすぐに試してみましょう。

## 前提条件

このレッスンでは、前回のレッスンで行ったデータのインポートと初期クリーニングを含む[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb)ファイルを使用します。

## 演習 - 準備

まず、曲データをもう一度確認しましょう。

1. 各列に対して`boxplot()`を呼び出し、ボックスプロットを作成します:

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

    このデータは少しノイズが多いです。各列をボックスプロットとして観察することで、外れ値が見つかります。

    ![外れ値](../../../../translated_images/boxplots.8228c29dabd0f29227dd38624231a175f411f1d8d4d7c012cb770e00e4fdf8b6.ja.png)

データセットを確認してこれらの外れ値を削除することもできますが、それではデータが非常に少なくなってしまいます。

1. クラスタリング演習で使用する列を選択します。範囲が似ている列を選び、`artist_top_genre`列を数値データとしてエンコードします:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. 次に、ターゲットとするクラスタ数を選択する必要があります。このデータセットから抽出した3つの曲ジャンルがあることを知っているので、3を試してみましょう:

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

データフレームの各行に対して予測されたクラスタ（0, 1, または2）の配列が表示されます。

1. この配列を使用して「シルエットスコア」を計算します:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## シルエットスコア

シルエットスコアは1に近いほど良いです。このスコアは-1から1の範囲で変動し、スコアが1の場合、クラスタは密集しており、他のクラスタから明確に分離されています。0に近い値は、隣接するクラスタの境界に非常に近いサンプルがある重なり合ったクラスタを表します。[(出典)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

私たちのスコアは**0.53**で、ちょうど中間です。これは、このデータがこのタイプのクラスタリングに特に適していないことを示していますが、続けてみましょう。

### 演習 - モデルを構築する

1. `KMeans`をインポートし、クラスタリングプロセスを開始します。

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    ここにはいくつか説明が必要な部分があります。

    > 🎓 range: クラスタリングプロセスの反復回数

    > 🎓 random_state: セントロイドの初期化に使用される乱数生成を決定します。[出典](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: 「クラスタ内平方和」は、クラスタ内のすべてのポイントがクラスタのセントロイドに対して持つ平均平方距離を測定します。[出典](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 Inertia: K-Meansアルゴリズムは「慣性」を最小化するようにセントロイドを選択しようとします。これは、クラスタがどれだけ内部的に一貫しているかを測定するものです。[出典](https://scikit-learn.org/stable/modules/clustering.html)。この値は各反復でwcss変数に追加されます。

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)では、「k-means++」最適化を使用できます。これは、セントロイドを互いに（一般的に）遠く離れた位置に初期化し、ランダムな初期化よりも良い結果をもたらす可能性があります。

### エルボー法

以前、3つの曲ジャンルをターゲットにしているため、3つのクラスタを選択すべきだと推測しました。しかし、それは正しいのでしょうか？

1. 「エルボー法」を使用して確認してみましょう。

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    前のステップで作成した`wcss`変数を使用して、エルボーの「曲がり目」を示すチャートを作成します。これが最適なクラスタ数を示します。もしかすると、やはり**3**かもしれません！

    ![エルボー法](../../../../translated_images/elbow.72676169eed744ff03677e71334a16c6b8f751e9e716e3d7f40dd7cdef674cca.ja.png)

## 演習 - クラスタを表示する

1. 今回は3つのクラスタを設定してプロセスを再試行し、クラスタを散布図として表示します:

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

1. モデルの精度を確認します:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    このモデルの精度はあまり良くなく、クラスタの形状がその理由を示しています。

    ![クラスタ](../../../../translated_images/clusters.b635354640d8e4fd4a49ef545495518e7be76172c97c13bd748f5b79f171f69a.ja.png)

    このデータは不均衡で、相関が少なく、列の値の間に大きな分散があるため、うまくクラスタリングできません。実際、形成されたクラスタは、上記で定義した3つのジャンルカテゴリによって大きく影響を受けている可能性があります。これは学習プロセスの一環です！

    Scikit-learnのドキュメントでは、このようにクラスタがあまり明確に区別されていないモデルには「分散」の問題があると説明されています。

    ![問題のあるモデル](../../../../translated_images/problems.f7fb539ccd80608e1f35c319cf5e3ad1809faa3c08537aead8018c6b5ba2e33a.ja.png)
    > インフォグラフィック: Scikit-learn

## 分散

分散は、「平均からの差の二乗の平均」と定義されます[(出典)](https://www.mathsisfun.com/data/standard-deviation.html)。このクラスタリング問題の文脈では、データセットの数値が平均から少し離れすぎていることを指します。

✅ この問題を解決する方法を考える良いタイミングです。データをもう少し調整しますか？別の列を使用しますか？別のアルゴリズムを使用しますか？ヒント: データを正規化して他の列をテストするために[スケーリング](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/)を試してみてください。

> この「[分散計算機](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)」を試して、概念をもう少し理解してみてください。

---

## 🚀チャレンジ

このノートブックで時間をかけてパラメータを調整してみてください。データをさらにクリーニングすることで（例えば外れ値を削除するなど）、モデルの精度を向上させることができますか？特定のデータサンプルに重みを付けることもできます。他にどのような方法でより良いクラスタを作成できますか？

ヒント: データをスケーリングしてみてください。ノートブックには、データ列が範囲の点でより似たものになるように標準スケーリングを追加するコメント付きコードがあります。シルエットスコアは下がりますが、エルボーグラフの「曲がり」が滑らかになります。これは、データをスケーリングせずに放置すると、分散の少ないデータがより大きな重みを持つようになるためです。この問題についてもう少し[こちら](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)で読んでみてください。

## [事後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/30/)

## 復習と自己学習

K-Meansシミュレーター[こちら](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/)を見てみてください。このツールを使用して、サンプルデータポイントを視覚化し、そのセントロイドを決定できます。データのランダム性、クラスタ数、セントロイド数を編集できます。これにより、データがどのようにグループ化されるかのアイデアが得られますか？

また、スタンフォード大学の[このK-Meansに関するハンドアウト](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)も確認してみてください。

## 課題

[異なるクラスタリング手法を試してみましょう](assignment.md)

---

**免責事項**:  
この文書はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された文書を正式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当方は責任を負いません。