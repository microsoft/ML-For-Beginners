<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-06T10:50:03+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "en"
}
-->
# Introduction to clustering

Clustering is a type of [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) that assumes a dataset is unlabelled or that its inputs are not paired with predefined outputs. It uses various algorithms to analyze unlabeled data and group it based on patterns identified within the data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Click the image above for a video. While studying machine learning with clustering, enjoy some Nigerian Dance Hall tracks—this is a highly rated song from 2014 by PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) is incredibly useful for exploring data. Let's see if it can help uncover trends and patterns in how Nigerian audiences consume music.

✅ Take a moment to think about the applications of clustering. In everyday life, clustering happens when you sort a pile of laundry into family members' clothes 🧦👕👖🩲. In data science, clustering is used to analyze user preferences or identify characteristics in any unlabeled dataset. Clustering, in essence, helps bring order to chaos—like organizing a sock drawer.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Click the image above for a video: MIT's John Guttag introduces clustering.

In a professional context, clustering can be used for tasks like market segmentation—for example, identifying which age groups purchase specific items. It can also be used for anomaly detection, such as identifying fraud in a dataset of credit card transactions. Another application might be detecting tumors in medical scans.

✅ Take a moment to think about how you might have encountered clustering in real-world scenarios, such as in banking, e-commerce, or business.

> 🎓 Interestingly, cluster analysis originated in the fields of Anthropology and Psychology in the 1930s. Can you imagine how it might have been applied back then?

Alternatively, clustering can be used to group search results—for example, by shopping links, images, or reviews. It's particularly useful for large datasets that need to be reduced for more detailed analysis, making it a valuable tool for understanding data before building other models.

✅ Once your data is organized into clusters, you can assign it a cluster ID. This technique is useful for preserving a dataset's privacy, as you can refer to a data point by its cluster ID rather than by more identifiable information. Can you think of other reasons why you might use a cluster ID instead of specific elements of the cluster for identification?

Deepen your understanding of clustering techniques in this [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott).

## Getting started with clustering

[Scikit-learn offers a wide range](https://scikit-learn.org/stable/modules/clustering.html) of methods for clustering. The method you choose will depend on your specific use case. According to the documentation, each method has its own advantages. Here's a simplified table of the methods supported by Scikit-learn and their ideal use cases:

| Method name                  | Use case                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | General purpose, inductive                                             |
| Affinity propagation         | Many, uneven clusters, inductive                                       |
| Mean-shift                   | Many, uneven clusters, inductive                                       |
| Spectral clustering          | Few, even clusters, transductive                                       |
| Ward hierarchical clustering | Many, constrained clusters, transductive                               |
| Agglomerative clustering     | Many, constrained, non-Euclidean distances, transductive               |
| DBSCAN                       | Non-flat geometry, uneven clusters, transductive                       |
| OPTICS                       | Non-flat geometry, uneven clusters with variable density, transductive |
| Gaussian mixtures            | Flat geometry, inductive                                               |
| BIRCH                        | Large dataset with outliers, inductive                                 |

> 🎓 How we create clusters depends heavily on how we group data points together. Let's break down some key terms:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transductive inference is derived from observed training cases that map to specific test cases. Inductive inference is derived from training cases that map to general rules, which are then applied to test cases.
> 
> Example: Imagine you have a dataset that's only partially labeled. Some items are 'records,' some are 'CDs,' and others are blank. Your task is to label the blanks. Using an inductive approach, you'd train a model to identify 'records' and 'CDs' and apply those labels to the unlabeled data. This approach might struggle to classify items that are actually 'cassettes.' A transductive approach, however, groups similar items together and applies labels to the groups. In this case, clusters might represent 'round musical items' and 'square musical items.'
> 
> 🎓 ['Non-flat' vs. 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Derived from mathematical terminology, non-flat vs. flat geometry refers to how distances between points are measured—either 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) or 'non-flat' (non-Euclidean).
>
>'Flat' refers to Euclidean geometry (often taught as 'plane' geometry), while 'non-flat' refers to non-Euclidean geometry. In machine learning, these methods are used to measure distances between points in clusters. [Euclidean distances](https://wikipedia.org/wiki/Euclidean_distance) are measured as the length of a straight line between two points. [Non-Euclidean distances](https://wikipedia.org/wiki/Non-Euclidean_geometry) are measured along a curve. If your data doesn't exist on a plane when visualized, you may need a specialized algorithm to handle it.
>
![Flat vs Nonflat Geometry Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters are defined by their distance matrix, which measures the distances between points. Euclidean clusters are defined by the average of the point values and have a 'centroid' or center point. Distances are measured relative to this centroid. Non-Euclidean distances use 'clustroids,' the point closest to other points, which can be defined in various ways.
> 
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduces 'semi-supervised' learning into this unsupervised method. Relationships between points are flagged as 'cannot link' or 'must-link,' imposing rules on the dataset.
>
> Example: If an algorithm is applied to unlabelled or semi-labelled data, the resulting clusters may be of poor quality. For instance, clusters might group 'round musical items,' 'square musical items,' 'triangular items,' and 'cookies.' Adding constraints like "the item must be made of plastic" or "the item must produce music" can help the algorithm make better choices.
> 
> 🎓 'Density'
> 
> Data that is 'noisy' is considered 'dense.' The distances between points in its clusters may vary, requiring the use of appropriate clustering methods. [This article](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) compares K-Means clustering and HDBSCAN algorithms for analyzing noisy datasets with uneven cluster density.

## Clustering algorithms

There are over 100 clustering algorithms, and their application depends on the nature of the data. Let's explore some of the major ones:

- **Hierarchical clustering**. Objects are grouped based on their proximity to nearby objects rather than distant ones. Clusters are formed based on the distances between their members. Scikit-learn's agglomerative clustering is hierarchical.

   ![Hierarchical clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. This popular algorithm requires selecting 'k,' the number of clusters to form. The algorithm then determines the center point of each cluster and groups data around it. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) is a well-known example. The center is determined by the nearest mean, hence the name. The squared distance from the cluster is minimized.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribution-based clustering**. Based on statistical modeling, this method assigns data points to clusters based on the probability of their belonging. Gaussian mixture methods fall under this category.

- **Density-based clustering**. Data points are grouped based on their density or proximity to one another. Points far from the group are considered outliers or noise. DBSCAN, Mean-shift, and OPTICS are examples of this type.

- **Grid-based clustering**. For multi-dimensional datasets, a grid is created, and data is divided among the grid's cells, forming clusters.

## Exercise - cluster your data

Clustering is greatly enhanced by effective visualization, so let's start by visualizing our music data. This exercise will help us determine the most suitable clustering method for this dataset.

1. Open the [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) file in this folder.

1. Import the `Seaborn` package for better data visualization.

    ```python
    !pip install seaborn
    ```

1. Append the song data from [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Load a dataframe with song data. Prepare to explore this data by importing the libraries and displaying the data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Check the first few rows of data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Get some information about the dataframe by calling `info()`:

    ```python
    df.info()
    ```

   The output looks like this:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Double-check for null values by calling `isnull()` and verifying the sum is 0:

    ```python
    df.isnull().sum()
    ```

    Everything looks good:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Describe the data:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 If clustering is an unsupervised method that doesn't require labeled data, why are we showing this data with labels? During the data exploration phase, labels are helpful, but they aren't necessary for clustering algorithms to work. You could remove the column headers and refer to the data by column number instead.

Take a look at the general values in the data. Note that popularity can be '0', which indicates songs with no ranking. We'll remove those shortly.

1. Use a barplot to identify the most popular genres:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ If you'd like to see more top values, change the top `[:5]` to a larger value, or remove it to see everything.

When the top genre is listed as 'Missing', it means Spotify didn't classify it. Let's filter it out.

1. Remove missing data by filtering it out:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Now check the genres again:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. The top three genres dominate this dataset. Let's focus on `afro dancehall`, `afropop`, and `nigerian pop`. Additionally, filter the dataset to remove entries with a popularity value of 0 (indicating they weren't classified with popularity and can be considered noise for our purposes):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Perform a quick test to see if the data has any strong correlations:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    The only strong correlation is between `energy` and `loudness`, which isn't surprising since loud music is often energetic. Otherwise, the correlations are relatively weak. It'll be interesting to see what a clustering algorithm can uncover in this data.

    > 🎓 Remember, correlation does not imply causation! We have evidence of correlation but no proof of causation. An [amusing website](https://tylervigen.com/spurious-correlations) provides visuals that emphasize this point.

Is there any convergence in this dataset around a song's perceived popularity and danceability? A FacetGrid shows concentric circles aligning, regardless of genre. Could it be that Nigerian tastes converge at a certain level of danceability for this genre?

✅ Try different data points (energy, loudness, speechiness) and explore more or different musical genres. What can you discover? Refer to the `df.describe()` table to understand the general spread of the data points.

### Exercise - Data Distribution

Are these three genres significantly different in their perception of danceability based on popularity?

1. Examine the data distribution for popularity and danceability in our top three genres along a given x and y axis:

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    You can observe concentric circles around a general point of convergence, showing the distribution of points.

    > 🎓 This example uses a KDE (Kernel Density Estimate) graph, which represents the data using a continuous probability density curve. This helps interpret data when working with multiple distributions.

    In general, the three genres align loosely in terms of popularity and danceability. Identifying clusters in this loosely-aligned data will be challenging:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Create a scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    A scatterplot of the same axes shows a similar pattern of convergence:

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Scatterplots are useful for visualizing clusters of data, making them essential for clustering tasks. In the next lesson, we'll use k-means clustering to identify groups in this data that overlap in interesting ways.

---

## 🚀Challenge

To prepare for the next lesson, create a chart about the various clustering algorithms you might encounter and use in a production environment. What types of problems is clustering designed to solve?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Before applying clustering algorithms, it's important to understand the nature of your dataset. Learn more about this topic [here](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html).

[This helpful article](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) explains how different clustering algorithms behave with various data shapes.

## Assignment

[Research other visualizations for clustering](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.