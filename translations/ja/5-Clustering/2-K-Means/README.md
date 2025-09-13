<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T09:31:24+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ja"
}
-->
# K-Meansクラスタリング

## [事前クイズ](https://ff-quizzes.netlify.app/en/ml/)

このレッスンでは、Scikit-learnと以前インポートしたナイジェリア音楽データセットを使用してクラスタを作成する方法を学びます。クラスタリングの基本としてK-Meansについて説明します。前のレッスンで学んだように、クラスタリングにはさまざまな方法があり、使用する方法はデータによって異なります。今回は最も一般的なクラスタリング手法であるK-Meansを試してみましょう。それでは始めましょう！

学ぶ用語:

- シルエットスコア
- エルボー法
- 慣性
- 分散

## はじめに

[K-Meansクラスタリング](https://wikipedia.org/wiki/K-means_clustering)は、信号処理の分野から派生した手法です。データを「k」個のクラスタに分割し、観測値を使用してグループ化します。各観測値は、クラスタの中心点である「平均」に最も近いデータポイントをグループ化する役割を果たします。

クラスタは[ボロノイ図](https://wikipedia.org/wiki/Voronoi_diagram)として視覚化できます。この図には点（または「種」）とその対応する領域が含まれます。

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> インフォグラフィック: [Jen Looper](https://twitter.com/jenlooper)

K-Meansクラスタリングプロセスは[3ステップで実行されます](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. アルゴリズムはデータセットからk個の中心点を選択します。その後、以下を繰り返します:
    1. 各サンプルを最も近い中心点に割り当てます。
    2. 前の中心点に割り当てられたすべてのサンプルの平均値を取ることで新しい中心点を作成します。
    3. 新旧の中心点の差を計算し、中心点が安定するまで繰り返します。

K-Meansの欠点の1つは、「k」、つまり中心点の数を設定する必要があることです。ただし、「エルボー法」を使用すると、kの良い初期値を推定するのに役立ちます。すぐに試してみましょう。

## 前提条件

このレッスンでは、前回のレッスンで行ったデータインポートと初期クリーニングを含む[_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb)ファイルを使用します。

## 演習 - 準備

まず、曲データをもう一度確認してください。

1. 各列に対して`boxplot()`を呼び出してボックスプロットを作成します:

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

    このデータは少しノイズが多いです。各列をボックスプロットとして観察することで、外れ値が見えます。

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

データセットを調べてこれらの外れ値を削除することもできますが、それではデータがかなり少なくなってしまいます。

1. 今回は、クラスタリング演習で使用する列を選択してください。範囲が似ている列を選び、`artist_top_genre`列を数値データとしてエンコードします:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. 次に、ターゲットとするクラスタ数を選択する必要があります。データセットから抽出した3つの曲ジャンルがあることを知っているので、3を試してみましょう:

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

データフレームの各行に対して予測されたクラスタ（0、1、または2）の配列が表示されます。

1. この配列を使用して「シルエットスコア」を計算します:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## シルエットスコア

シルエットスコアが1に近い値を目指してください。このスコアは-1から1の範囲で変動し、スコアが1の場合、クラスタは密集しており他のクラスタから十分に分離されています。0に近い値は、隣接するクラスタの境界付近に非常に近いサンプルがある重なり合ったクラスタを表します。[(出典)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

私たちのスコアは**0.53**で、ちょうど中間です。これは、このデータがこのタイプのクラスタリングに特に適していないことを示していますが、続けてみましょう。

### 演習 - モデルを構築する

1. `KMeans`をインポートしてクラスタリングプロセスを開始します。

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

    > 🎓 random_state: 「中心点の初期化のための乱数生成を決定します。」[出典](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: 「クラスタ内平方和」は、クラスタ内のすべての点がクラスタ中心点に対して持つ平均平方距離を測定します。[出典](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 慣性: K-Meansアルゴリズムは「クラスタが内部的にどれだけ一貫性があるか」を測定する慣性を最小化するように中心点を選択しようとします。[出典](https://scikit-learn.org/stable/modules/clustering.html)。この値は各反復でwcss変数に追加されます。

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)では、「k-means++」最適化を使用できます。これにより、中心点が互いに（一般的に）離れて初期化され、ランダムな初期化よりも良い結果が得られる可能性があります。

### エルボー法

以前、3つの曲ジャンルをターゲットにしているため、3つのクラスタを選択するべきだと推測しました。しかし、それは正しいのでしょうか？

1. エルボー法を使用して確認してください。

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    前のステップで構築した`wcss`変数を使用して、エルボーの「曲がり」を示すチャートを作成します。これが最適なクラスタ数を示します。もしかすると**3**が正解かもしれません！

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

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

1. モデルの精度を確認してください:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    このモデルの精度はあまり良くなく、クラスタの形状がその理由を示しています。

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    このデータは不均衡で、相関が少なく、列値間の分散が大きすぎてうまくクラスタリングできません。実際、形成されたクラスタは、上記で定義した3つのジャンルカテゴリによって大きく影響を受けたり偏ったりしている可能性があります。これは学習プロセスでした！

    Scikit-learnのドキュメントでは、このようにクラスタがあまり明確に区分されていないモデルには「分散」の問題があることが示されています。

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > インフォグラフィック: Scikit-learn

## 分散

分散は「平均からの平方差の平均」と定義されます[(出典)](https://www.mathsisfun.com/data/standard-deviation.html)。このクラスタリング問題の文脈では、データセットの数値が平均から少し離れすぎる傾向があることを指します。

✅ この問題を解決する方法を考える絶好の機会です。データをもう少し調整しますか？別の列を使用しますか？別のアルゴリズムを使用しますか？ヒント: データを[スケーリング](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/)して正規化し、他の列をテストしてみてください。

> この「[分散計算機](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)」を試して、概念をもう少し理解してください。

---

## 🚀チャレンジ

このノートブックで時間をかけてパラメータを調整してください。データをさらにクリーニングすることで（例えば外れ値を削除するなど）、モデルの精度を向上させることができますか？特定のデータサンプルに重みを付けることもできます。他にどのような方法でより良いクラスタを作成できますか？

ヒント: データをスケーリングしてみてください。ノートブックには、標準スケーリングを追加してデータ列を範囲の観点でより似たものにするコメント付きコードがあります。データをスケーリングせずに残すと、分散が少ないデータがより大きな重みを持つことになります。この問題についてもう少し読むには[こちら](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)をご覧ください。

## [事後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

[K-Meansシミュレーター](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/)を見てみてください。このツールを使用してサンプルデータポイントを視覚化し、その中心点を決定できます。データのランダム性、クラスタ数、中心点数を編集できます。これにより、データがどのようにグループ化されるかのアイデアが得られますか？

また、[スタンフォードのK-Meansに関するハンドアウト](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)も見てみてください。

## 課題

[異なるクラスタリング手法を試してみてください](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の利用に起因する誤解や誤認について、当方は一切の責任を負いません。