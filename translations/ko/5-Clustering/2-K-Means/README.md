<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-05T10:45:29+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "ko"
}
-->
# K-Means 클러스터링

## [강의 전 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

이번 강의에서는 Scikit-learn과 이전에 가져온 나이지리아 음악 데이터셋을 사용하여 클러스터를 만드는 방법을 배웁니다. K-Means를 사용한 클러스터링의 기본 개념을 다룰 것입니다. 이전 강의에서 배운 것처럼 클러스터를 다루는 방법은 여러 가지가 있으며, 사용하는 방법은 데이터에 따라 달라집니다. 이번에는 가장 일반적인 클러스터링 기법인 K-Means를 시도해 보겠습니다. 시작해볼까요?

이번 강의에서 배우게 될 용어:

- 실루엣 점수
- 엘보우 방법
- 관성
- 분산

## 소개

[K-Means 클러스터링](https://wikipedia.org/wiki/K-means_clustering)은 신호 처리 분야에서 유래한 방법으로, 데이터를 'k'개의 클러스터로 나누고 분할하는 데 사용됩니다. 각 관측값은 주어진 데이터 포인트를 가장 가까운 '평균값' 또는 클러스터의 중심점에 그룹화하는 역할을 합니다.

클러스터는 [보로노이 다이어그램](https://wikipedia.org/wiki/Voronoi_diagram)으로 시각화할 수 있으며, 여기에는 점(또는 '시드')과 해당 영역이 포함됩니다.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> 인포그래픽: [Jen Looper](https://twitter.com/jenlooper)

K-Means 클러스터링 과정은 [세 단계로 실행됩니다](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. 알고리즘이 데이터셋에서 k개의 중심점을 선택합니다. 이후 반복적으로 다음을 수행합니다:
    1. 각 샘플을 가장 가까운 중심점에 할당합니다.
    2. 이전 중심점에 할당된 모든 샘플의 평균값을 계산하여 새로운 중심점을 생성합니다.
    3. 새로운 중심점과 이전 중심점 간의 차이를 계산하고, 중심점이 안정화될 때까지 반복합니다.

K-Means의 단점 중 하나는 'k', 즉 중심점의 개수를 미리 설정해야 한다는 점입니다. 다행히도 '엘보우 방법'을 사용하면 'k'의 적절한 시작 값을 추정할 수 있습니다. 곧 이를 시도해볼 것입니다.

## 사전 준비

이번 강의에서는 이전 강의에서 데이터 가져오기와 초기 정리를 완료한 [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) 파일을 사용합니다.

## 실습 - 준비 단계

우선, 노래 데이터셋을 다시 살펴보세요.

1. 각 열에 대해 `boxplot()`을 호출하여 박스플롯을 생성하세요:

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

    이 데이터는 약간 노이즈가 있습니다. 각 열을 박스플롯으로 관찰하면 이상치(outlier)를 확인할 수 있습니다.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

데이터셋을 살펴보고 이러한 이상치를 제거할 수도 있지만, 그렇게 하면 데이터가 너무 축소될 수 있습니다.

1. 클러스터링 실습에 사용할 열을 선택하세요. 비슷한 범위를 가진 열을 선택하고 `artist_top_genre` 열을 숫자 데이터로 인코딩하세요:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. 이제 몇 개의 클러스터를 목표로 할지 선택해야 합니다. 데이터셋에서 3개의 노래 장르를 추출했으므로, 3개를 시도해봅시다:

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

데이터프레임의 각 행에 대해 예측된 클러스터(0, 1, 2)가 포함된 배열이 출력됩니다.

1. 이 배열을 사용하여 '실루엣 점수'를 계산하세요:

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## 실루엣 점수

실루엣 점수가 1에 가까운지 확인하세요. 이 점수는 -1에서 1 사이의 값을 가지며, 점수가 1에 가까울수록 클러스터가 밀집되어 있고 다른 클러스터와 잘 분리되어 있음을 나타냅니다. 0에 가까운 값은 클러스터가 겹쳐져 있고 샘플이 이웃 클러스터의 경계에 매우 가까이 있음을 나타냅니다. [(출처)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

우리의 점수는 **0.53**으로, 중간 정도입니다. 이는 데이터가 이 유형의 클러스터링에 특히 적합하지 않음을 나타내지만, 계속 진행해봅시다.

### 실습 - 모델 구축

1. `KMeans`를 가져와 클러스터링 프로세스를 시작하세요.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    여기에는 설명이 필요한 몇 가지 부분이 있습니다.

    > 🎓 range: 클러스터링 프로세스의 반복 횟수입니다.

    > 🎓 random_state: "중심점 초기화를 위한 난수 생성기를 결정합니다." [출처](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "클러스터 내 제곱합"으로, 클러스터 중심점에서 모든 점까지의 평균 제곱 거리를 측정합니다. [출처](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 관성(Inertia): K-Means 알고리즘은 '관성'을 최소화하도록 중심점을 선택하려고 시도합니다. 관성은 클러스터가 내부적으로 얼마나 일관성 있는지를 측정합니다. [출처](https://scikit-learn.org/stable/modules/clustering.html). 이 값은 각 반복에서 wcss 변수에 추가됩니다.

    > 🎓 k-means++: [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)에서는 'k-means++' 최적화를 사용할 수 있습니다. 이는 중심점이 서로 멀리 떨어지도록 초기화하여 무작위 초기화보다 더 나은 결과를 얻을 가능성이 높습니다.

### 엘보우 방법

이전에 3개의 노래 장르를 목표로 했으므로 3개의 클러스터를 선택해야 한다고 추측했습니다. 하지만 정말 그럴까요?

1. '엘보우 방법'을 사용하여 확인하세요.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    이전 단계에서 생성한 `wcss` 변수를 사용하여 '엘보우'의 꺾이는 지점을 나타내는 차트를 생성하세요. 이 지점이 최적의 클러스터 수를 나타냅니다. 아마도 **3개**일 것입니다!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## 실습 - 클러스터 표시

1. 이번에는 3개의 클러스터를 설정하고, 클러스터를 산점도로 표시하세요:

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

1. 모델의 정확도를 확인하세요:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    이 모델의 정확도는 그다지 좋지 않으며, 클러스터의 모양이 그 이유를 암시합니다.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    이 데이터는 너무 불균형하고, 상관관계가 적으며, 열 값 간의 분산이 너무 커서 잘 클러스터링되지 않습니다. 사실, 형성된 클러스터는 우리가 위에서 정의한 세 가지 장르 카테고리에 의해 크게 영향을 받거나 왜곡되었을 가능성이 높습니다. 이것은 학습 과정의 일부였습니다!

    Scikit-learn 문서에서는 이 모델처럼 클러스터가 잘 구분되지 않은 경우 '분산' 문제가 있다고 설명합니다:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > 인포그래픽: Scikit-learn

## 분산

분산은 "평균으로부터의 제곱 차이의 평균"으로 정의됩니다 [(출처)](https://www.mathsisfun.com/data/standard-deviation.html). 이 클러스터링 문제의 맥락에서, 이는 데이터셋의 숫자가 평균에서 너무 많이 벗어나는 경향이 있음을 나타냅니다.

✅ 이 문제를 해결할 수 있는 모든 방법을 생각해볼 좋은 기회입니다. 데이터를 조금 더 조정해볼까요? 다른 열을 사용해볼까요? 다른 알고리즘을 사용해볼까요? 힌트: 데이터를 [스케일링](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/)하여 정규화하고 다른 열을 테스트해보세요.

> '[분산 계산기](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)'를 사용하여 개념을 조금 더 이해해보세요.

---

## 🚀도전 과제

이 노트북을 사용하여 매개변수를 조정하는 데 시간을 투자해보세요. 데이터를 더 정리(예: 이상치 제거)하여 모델의 정확도를 개선할 수 있나요? 특정 데이터 샘플에 더 많은 가중치를 부여할 수 있습니다. 더 나은 클러스터를 생성하기 위해 무엇을 할 수 있을까요?

힌트: 데이터를 스케일링해보세요. 노트북에는 데이터 열이 범위 면에서 더 비슷해지도록 표준 스케일링을 추가하는 주석 처리된 코드가 있습니다. 데이터를 스케일링하지 않으면 분산이 적은 데이터가 더 큰 가중치를 가지게 됩니다. 이 문제에 대해 더 알아보려면 [여기](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226)를 읽어보세요.

## [강의 후 퀴즈](https://ff-quizzes.netlify.app/en/ml/)

## 복습 및 자기 학습

[K-Means 시뮬레이터](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/)를 살펴보세요. 이 도구를 사용하여 샘플 데이터 포인트를 시각화하고 중심점을 결정할 수 있습니다. 데이터의 무작위성, 클러스터 수, 중심점 수를 편집할 수 있습니다. 데이터가 어떻게 그룹화될 수 있는지 이해하는 데 도움이 되나요?

또한, Stanford의 [K-Means 핸드아웃](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)을 살펴보세요.

## 과제

[다른 클러스터링 방법 시도하기](assignment.md)

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서를 해당 언어로 작성된 상태에서 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.  