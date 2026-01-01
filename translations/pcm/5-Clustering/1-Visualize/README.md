<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-11-18T19:06:15+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "pcm"
}
-->
# Introduction to clustering

Clustering na one kain [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) wey dey assume say dataset no get label or say e input no dey match with any predefined output. E dey use different algorithm to arrange data wey no get label and group dem based on pattern wey e see for di data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Click di image wey dey up for video. As you dey study machine learning with clustering, make you enjoy some Nigerian Dance Hall songs - dis na one correct song from 2014 by PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) dey very useful for data exploration. Make we see if e fit help us discover trends and pattern for how Nigerian people dey enjoy music.

✅ Take one minute think about how clustering dey useful. For real life, clustering dey happen anytime you get pile of clothes wey you wan sort out for your family members 🧦👕👖🩲. For data science, clustering dey happen when you dey try analyze user preference or determine di characteristics of any dataset wey no get label. Clustering dey help make sense of wahala, like sock drawer.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Click di image wey dey up for video: MIT's John Guttag dey explain clustering

For professional setting, clustering fit help determine things like market segmentation, like which age group dey buy which item. Another use na anomaly detection, maybe to catch fraud for dataset of credit card transactions. Or you fit use clustering to find tumor for batch of medical scans.

✅ Think small about how you don see clustering 'for di wild', maybe for banking, e-commerce, or business setting.

> 🎓 E dey interesting say cluster analysis start for Anthropology and Psychology for di 1930s. You fit imagine how dem take use am?

Another way you fit use am na to group search results - like shopping links, images, or reviews. Clustering dey useful when you get big dataset wey you wan reduce and perform more detailed analysis on top am, so di technique fit help you learn about di data before you build other models.

✅ Once you don organize your data inside clusters, you go give am cluster Id, and dis technique fit dey useful to protect di privacy of di dataset; you fit dey refer to data point by di cluster id instead of di more revealing identifiable data. You fit think of other reasons why you go prefer use cluster Id instead of other elements of di cluster to identify am?

Make you learn more about clustering techniques for dis [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Getting started with clustering

[Scikit-learn get plenty methods](https://scikit-learn.org/stable/modules/clustering.html) wey you fit use for clustering. Di one wey you go choose go depend on your use case. According to di documentation, each method get different benefits. Dis na simple table of di methods wey Scikit-learn support and di use case wey dem fit:

| Method name                  | Use case                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | general purpose, inductive                                             |
| Affinity propagation         | many, uneven clusters, inductive                                       |
| Mean-shift                   | many, uneven clusters, inductive                                       |
| Spectral clustering          | few, even clusters, transductive                                       |
| Ward hierarchical clustering | many, constrained clusters, transductive                               |
| Agglomerative clustering     | many, constrained, non Euclidean distances, transductive               |
| DBSCAN                       | non-flat geometry, uneven clusters, transductive                       |
| OPTICS                       | non-flat geometry, uneven clusters with variable density, transductive |
| Gaussian mixtures            | flat geometry, inductive                                               |
| BIRCH                        | large dataset with outliers, inductive                                 |

> 🎓 How we dey create clusters get plenty to do with how we dey gather di data points into groups. Make we break down some vocabulary:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Transductive inference dey come from training cases wey dem observe wey dey map to specific test cases. Inductive inference dey come from training cases wey dey map to general rules wey dem go later apply to test cases.
> 
> Example: Imagine say you get dataset wey no complete label. Some things dey labelled as 'records', some 'cds', and some dey blank. Your work na to give label to di blank ones. If you choose inductive approach, you go train model wey dey look for 'records' and 'cds', then apply di labels to di data wey no get label. Dis approach go struggle to classify things wey be 'cassettes'. Transductive approach go handle dis unknown data better as e dey group similar items together before e go give label to di group. For dis case, clusters fit show 'round musical things' and 'square musical things'.
> 
> 🎓 ['Non-flat' vs. 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Di term dey come from mathematics, non-flat vs. flat geometry dey talk about how we dey measure distance between points, either 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) or 'non-flat' (non-Euclidean) geometry.
>
>'Flat' for dis context mean Euclidean geometry (parts of am dey taught as 'plane' geometry), and non-flat mean non-Euclidean geometry. Wetin geometry get to do with machine learning? Well, as di two fields dey based on mathematics, we need common way to measure distance between points for clusters, and we fit do am in 'flat' or 'non-flat' way, depending on di nature of di data. [Euclidean distances](https://wikipedia.org/wiki/Euclidean_distance) dey measure di length of line segment between two points. [Non-Euclidean distances](https://wikipedia.org/wiki/Non-Euclidean_geometry) dey measure distance along curve. If your data, when you visualize am, no dey for plane, you go need special algorithm to handle am.
>
![Flat vs Nonflat Geometry Infographic](../../../../translated_images/flat-nonflat.d1c8c6e2a96110c1.pcm.png)
> Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Clusters dey defined by di distance matrix, e.g. di distance between points. Dis distance fit dey measured in different ways. Euclidean clusters dey defined by di average of di point values, and dem get 'centroid' or center point. Distance dey measured by di distance to di centroid. Non-Euclidean distances dey refer to 'clustroids', di point wey dey closest to other points. Clustroids fit dey defined in different ways.
> 
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) dey add 'semi-supervised' learning to dis unsupervised method. Di relationship between points dey flagged as 'cannot link' or 'must-link' so some rules go dey forced on di dataset.
>
>Example: If algorithm dey free to work on batch of data wey no get label or wey get small label, di clusters wey e go produce fit no make sense. For di example wey dey up, di clusters fit group 'round music things', 'square music things', 'triangular things', and 'cookies'. If you give am some constraints or rules ("di item must be made of plastic", "di item need fit produce music") e go help 'constrain' di algorithm to make better choices.
> 
> 🎓 'Density'
> 
> Data wey dey 'noisy' dey considered as 'dense'. Di distance between points for each cluster fit dey more or less dense, or 'crowded', and dis kind data need di correct clustering method. [Dis article](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) dey show di difference between using K-Means clustering vs. HDBSCAN algorithms to explore noisy dataset wey get uneven cluster density.

## Clustering algorithms

Plenty clustering algorithms dey, more than 100, and di one wey you go use depend on di nature of di data wey you get. Make we talk about some major ones:

- **Hierarchical clustering**. If object dey classified by how e near another object, instead of how far e dey, clusters go form based on di distance of di members to and from other objects. Scikit-learn agglomerative clustering na hierarchical.

   ![Hierarchical clustering Infographic](../../../../translated_images/hierarchical.bf59403aa43c8c47.pcm.png)
   > Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Dis popular algorithm dey require make you choose 'k', or di number of clusters wey you wan form, then di algorithm go find di center point of di cluster and gather data around di point. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) na popular version of centroid clustering. Di center dey determined by di nearest mean, na why dem call am di name. Di squared distance from di cluster dey minimized.

   ![Centroid clustering Infographic](../../../../translated_images/centroid.097fde836cf6c918.pcm.png)
   > Infographic by [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribution-based clustering**. Dis one dey based on statistical modeling, e dey focus on di probability say data point belong to cluster, then e go assign am. Gaussian mixture methods dey belong to dis type.

- **Density-based clustering**. Data points dey assigned to clusters based on di density, or how dem dey group around each other. Data points wey dey far from di group dey considered as outliers or noise. DBSCAN, Mean-shift and OPTICS dey belong to dis type of clustering.

- **Grid-based clustering**. For multi-dimensional datasets, grid go dey created and di data go dey divided among di grid cells, so clusters go dey formed.

## Exercise - cluster your data

Clustering dey work well when you fit visualize am well, so make we start by visualizing our music data. Dis exercise go help us decide which method of clustering go work best for di nature of dis data.

1. Open di [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) file wey dey dis folder.

1. Import di `Seaborn` package to help you visualize di data well.

    ```python
    !pip install seaborn
    ```

1. Add di song data from [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Load dataframe with some data about di songs. Prepare to explore di data by importing di libraries and dumping di data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Check di first few lines of di data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Make we check info about di dataframe, use `info()`:

    ```python
    df.info()
    ```

   Di output go look like dis:

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

1. Double-check say null values no dey, use `isnull()` and confirm say di sum na 0:

    ```python
    df.isnull().sum()
    ```

    E dey okay:

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

1. Describe di data:

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

> 🤔 If we dey work with clustering, one unsupervised method wey no need labeled data, why we dey show dis data with labels? For di data exploration phase, e dey useful, but e no dey necessary for di clustering algorithms to work. You fit even remove di column headers and refer to di data by column number.

Make we look di general values for di data. Note say popularity fit be '0', wey mean say di song no get ranking. Make we remove dem soon.

1. Use barplot to find di most popular genres:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../translated_images/popular.9c48d84b3386705f.pcm.png)

✅ If you wan see more top values, change di top `[:5]` to bigger value, or remove am to see all.

Note, when di top genre dey described as 'Missing', e mean say Spotify no classify am, so make we remove am.

1. Remove missing data by filtering am out

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Now check di genres again:

    ![most popular](../../../../translated_images/all-genres.1d56ef06cefbfcd6.pcm.png)

1. Di top three genres dey dominate dis dataset. Make we focus on `afro dancehall`, `afropop`, and `nigerian pop`, plus filter di dataset to remove anything wey get 0 popularity value (meaning e no dey classified with popularity for di dataset and fit be noise for our purpose):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Do quick test to see if di data dey correlate in any strong way:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../translated_images/correlation.a9356bb798f5eea5.pcm.png)

    Di only strong correlation na between `energy` and `loudness`, wey no too surprise, as loud music dey usually energetic. Otherwise, di correlations dey relatively weak. E go dey interesting to see wetin clustering algorithm fit do with dis data.

    > 🎓 Note say correlation no mean causation! We get proof of correlation but no proof of causation. One [funny website](https://tylervigen.com/spurious-correlations) get some visuals wey dey emphasize dis point.

E get any convergence for dis dataset around song popularity and danceability? FacetGrid dey show say concentric circles dey align, no matter di genre. E fit be say Nigerian taste dey converge for certain level of danceability for dis genre?

✅ Try different datapoints (energy, loudness, speechiness) and more or different musical genres. Wetin you fit discover? Check di `df.describe()` table to see di general spread of di data points.

### Exercise - data distribution

Di three genres dey different well well for di perception of their danceability, based on their popularity?

1. Check di top three genres data distribution for popularity and danceability along given x and y axis.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    You fit discover concentric circles around general point of convergence, wey dey show di distribution of points.

    > 🎓 Note say dis example dey use KDE (Kernel Density Estimate) graph wey dey represent di data using continuous probability density curve. Dis one dey help us interpret data when we dey work with multiple distributions.

    Generally, di three genres dey align small in terms of their popularity and danceability. To find clusters for dis loosely-aligned data go dey challenging:

    ![distribution](../../../../translated_images/distribution.9be11df42356ca95.pcm.png)

1. Create scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Scatterplot for di same axes dey show similar pattern of convergence

    ![Facetgrid](../../../../translated_images/facetgrid.9b2e65ce707eba1f.pcm.png)

Generally, for clustering, you fit use scatterplots to show clusters of data, so e good to sabi dis type of visualization well. For di next lesson, we go use dis filtered data and use k-means clustering to find groups for dis data wey dey overlap in interesting ways.

---

## 🚀Challenge

Prepare for di next lesson, make chart about di different clustering algorithms wey you fit discover and use for production environment. Wetin di clustering dey try solve?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Before you apply clustering algorithms, as we don learn, e good to understand di nature of your dataset. Read more about dis topic [here](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Dis helpful article](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) go show you di different ways wey clustering algorithms dey behave, based on different data shapes.

## Assignment

[Research other visualizations for clustering](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI transleshion service [Co-op Translator](https://github.com/Azure/co-op-translator) do di transleshion. Even as we dey try make am accurate, abeg make you sabi say automatik transleshion fit get mistake or no dey correct well. Di original dokyument wey dey for im native language na di one wey you go take as di main source. For important mata, e good make you use professional human transleshion. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis transleshion.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->