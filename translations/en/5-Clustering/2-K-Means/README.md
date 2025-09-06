<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7cdd17338d9bbd7e2171c2cd462eb081",
  "translation_date": "2025-09-06T10:50:53+00:00",
  "source_file": "5-Clustering/2-K-Means/README.md",
  "language_code": "en"
}
-->
# K-Means Clustering

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

In this lesson, you'll learn how to create clusters using Scikit-learn and the Nigerian music dataset you imported earlier. We'll cover the basics of K-Means for clustering. Remember, as you learned in the previous lesson, there are many ways to work with clusters, and the method you choose depends on your data. We'll try K-Means since it's the most common clustering technique. Let's dive in!

Key terms you'll learn:

- Silhouette scoring
- Elbow method
- Inertia
- Variance

## Introduction

[K-Means Clustering](https://wikipedia.org/wiki/K-means_clustering) is a technique from the field of signal processing. It is used to divide and group data into 'k' clusters based on a series of observations. Each observation works to group a given data point closest to its nearest 'mean,' or the center point of a cluster.

The clusters can be visualized as [Voronoi diagrams](https://wikipedia.org/wiki/Voronoi_diagram), which consist of a point (or 'seed') and its corresponding region.

![voronoi diagram](../../../../5-Clustering/2-K-Means/images/voronoi.png)

> Infographic by [Jen Looper](https://twitter.com/jenlooper)

The K-Means clustering process [follows a three-step procedure](https://scikit-learn.org/stable/modules/clustering.html#k-means):

1. The algorithm selects k-number of center points by sampling from the dataset. Then it loops:
    1. Assigns each sample to the nearest centroid.
    2. Creates new centroids by calculating the mean value of all samples assigned to the previous centroids.
    3. Calculates the difference between the new and old centroids and repeats until the centroids stabilize.

One limitation of K-Means is that you need to define 'k,' the number of centroids. Luckily, the 'elbow method' can help estimate a good starting value for 'k.' You'll try it shortly.

## Prerequisite

You'll work in this lesson's [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/2-K-Means/notebook.ipynb) file, which includes the data import and preliminary cleaning you completed in the previous lesson.

## Exercise - Preparation

Start by revisiting the songs dataset.

1. Create a boxplot by calling `boxplot()` for each column:

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

    This data is a bit noisy: by observing each column as a boxplot, you can identify outliers.

    ![outliers](../../../../5-Clustering/2-K-Means/images/boxplots.png)

You could go through the dataset and remove these outliers, but that would leave you with very minimal data.

1. For now, decide which columns to use for your clustering exercise. Choose ones with similar ranges and encode the `artist_top_genre` column as numeric data:

    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    X = df.loc[:, ('artist_top_genre','popularity','danceability','acousticness','loudness','energy')]
    
    y = df['artist_top_genre']
    
    X['artist_top_genre'] = le.fit_transform(X['artist_top_genre'])
    
    y = le.transform(y)
    ```

1. Next, determine how many clusters to target. You know there are 3 song genres in the dataset, so let's try 3:

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

You'll see an array printed out with predicted clusters (0, 1, or 2) for each row in the dataframe.

1. Use this array to calculate a 'silhouette score':

    ```python
    from sklearn import metrics
    score = metrics.silhouette_score(X, y_cluster_kmeans)
    score
    ```

## Silhouette Score

Aim for a silhouette score closer to 1. This score ranges from -1 to 1. A score of 1 indicates that the cluster is dense and well-separated from other clusters. A value near 0 suggests overlapping clusters with samples close to the decision boundary of neighboring clusters. [(Source)](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)

Our score is **0.53**, which is moderate. This suggests that our data isn't particularly well-suited for this type of clustering, but let's proceed.

### Exercise - Build a Model

1. Import `KMeans` and begin the clustering process.

    ```python
    from sklearn.cluster import KMeans
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    ```

    Here's an explanation of some key parts:

    > 🎓 range: These are the iterations of the clustering process.

    > 🎓 random_state: "Determines random number generation for centroid initialization." [Source](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

    > 🎓 WCSS: "Within-cluster sums of squares" measures the squared average distance of all points within a cluster to the cluster centroid. [Source](https://medium.com/@ODSC/unsupervised-learning-evaluating-clusters-bd47eed175ce)

    > 🎓 Inertia: K-Means algorithms aim to choose centroids that minimize 'inertia,' "a measure of how internally coherent clusters are." [Source](https://scikit-learn.org/stable/modules/clustering.html). The value is appended to the wcss variable during each iteration.

    > 🎓 k-means++: In [Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means), you can use the 'k-means++' optimization, which "initializes the centroids to be (generally) distant from each other, leading to likely better results than random initialization."

### Elbow Method

Earlier, you assumed that 3 clusters would be appropriate because of the 3 song genres. But is that correct?

1. Use the 'elbow method' to confirm.

    ```python
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1, 11), y=wcss, marker='o', color='red')
    plt.title('Elbow')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

    Use the `wcss` variable you built earlier to create a chart showing the 'bend' in the elbow, which indicates the optimal number of clusters. Perhaps it **is** 3!

    ![elbow method](../../../../5-Clustering/2-K-Means/images/elbow.png)

## Exercise - Display the Clusters

1. Repeat the process, this time setting three clusters, and display the clusters as a scatterplot:

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

1. Check the model's accuracy:

    ```python
    labels = kmeans.labels_
    
    correct_labels = sum(y == labels)
    
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    
    print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))
    ```

    The model's accuracy isn't great, and the shape of the clusters gives you a clue as to why.

    ![clusters](../../../../5-Clustering/2-K-Means/images/clusters.png)

    The data is too imbalanced, poorly correlated, and has too much variance between column values to cluster effectively. In fact, the clusters that form are likely heavily influenced or skewed by the three genre categories we defined earlier. This was a learning experience!

    According to Scikit-learn's documentation, a model like this one, with poorly defined clusters, has a 'variance' problem:

    ![problem models](../../../../5-Clustering/2-K-Means/images/problems.png)
    > Infographic from Scikit-learn

## Variance

Variance is defined as "the average of the squared differences from the Mean" [(Source)](https://www.mathsisfun.com/data/standard-deviation.html). In the context of this clustering problem, it means that the numbers in our dataset diverge too much from the mean.

✅ This is a good time to think about ways to address this issue. Should you tweak the data further? Use different columns? Try a different algorithm? Hint: Consider [scaling your data](https://www.mygreatlearning.com/blog/learning-data-science-with-k-means-clustering/) to normalize it and test other columns.

> Try this '[variance calculator](https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php)' to better understand the concept.

---

## 🚀Challenge

Spend some time with this notebook, tweaking parameters. Can you improve the model's accuracy by cleaning the data further (e.g., removing outliers)? You can use weights to give more importance to certain data samples. What else can you do to create better clusters?

Hint: Try scaling your data. There's commented code in the notebook that adds standard scaling to make the data columns more similar in range. You'll find that while the silhouette score decreases, the 'kink' in the elbow graph becomes smoother. This is because leaving the data unscaled allows data with less variance to have more influence. Read more about this issue [here](https://stats.stackexchange.com/questions/21222/are-mean-normalization-and-feature-scaling-needed-for-k-means-clustering/21226#21226).

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Check out a K-Means Simulator [like this one](https://user.ceng.metu.edu.tr/~akifakkus/courses/ceng574/k-means/). This tool lets you visualize sample data points and determine their centroids. You can adjust the data's randomness, number of clusters, and number of centroids. Does this help you better understand how data can be grouped?

Also, review [this handout on K-Means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html) from Stanford.

## Assignment

[Experiment with different clustering methods](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.