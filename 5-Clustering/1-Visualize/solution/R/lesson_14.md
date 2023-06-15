---
title: 'Introduction to clustering: Clean, prep and visualize your data'
output:
  html_document:
    df_print: paged
    theme: flatly
    highlight: breezedark
    toc: yes
    toc_float: yes
    code_download: yes
---

## **Nigerian Music scraped from Spotify - an analysis**

Clustering is a type of [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) that presumes that a dataset is unlabelled or that its inputs are not matched with predefined outputs. It uses various algorithms to sort through unlabeled data and provide groupings according to patterns it discerns in the data.

[**Pre-lecture quiz**](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/27/)

### **Introduction**

[Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) is very useful for data exploration. Let's see if it can help discover trends and patterns in the way Nigerian audiences consume music.

> ✅ Take a minute to think about the uses of clustering. In real life, clustering happens whenever you have a pile of laundry and need to sort out your family members' clothes 🧦👕👖🩲. In data science, clustering happens when trying to analyze a user's preferences, or determine the characteristics of any unlabeled dataset. Clustering, in a way, helps make sense of chaos, like a sock drawer.

In a professional setting, clustering can be used to determine things like market segmentation, determining what age groups buy what items, for example. Another use would be anomaly detection, perhaps to detect fraud from a dataset of credit card transactions. Or you might use clustering to determine tumors in a batch of medical scans.

✅ Think a minute about how you might have encountered clustering 'in the wild', in a banking, e-commerce, or business setting.

> 🎓 Interestingly, cluster analysis originated in the fields of Anthropology and Psychology in the 1930s. Can you imagine how it might have been used?

Alternately, you could use it for grouping search results - by shopping links, images, or reviews, for example. Clustering is useful when you have a large dataset that you want to reduce and on which you want to perform more granular analysis, so the technique can be used to learn about data before other models are constructed.

✅ Once your data is organized in clusters, you assign it a cluster Id, and this technique can be useful when preserving a dataset's privacy; you can instead refer to a data point by its cluster id, rather than by more revealing identifiable data. Can you think of other reasons why you'd refer to a cluster Id rather than other elements of the cluster to identify it?

### Getting started with clustering

> 🎓 How we create clusters has a lot to do with how we gather up the data points into groups. Let's unpack some vocabulary:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
>
> Transductive inference is derived from observed training cases that map to specific test cases. Inductive inference is derived from training cases that map to general rules which are only then applied to test cases.
>
> An example: Imagine you have a dataset that is only partially labelled. Some things are 'records', some 'cds', and some are blank. Your job is to provide labels for the blanks. If you choose an inductive approach, you'd train a model looking for 'records' and 'cds', and apply those labels to your unlabeled data. This approach will have trouble classifying things that are actually 'cassettes'. A transductive approach, on the other hand, handles this unknown data more effectively as it works to group similar items together and then applies a label to a group. In this case, clusters might reflect 'round musical things' and 'square musical things'.
>
> 🎓 ['Non-flat' vs. 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
>
> Derived from mathematical terminology, non-flat vs. flat geometry refers to the measure of distances between points by either 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) or 'non-flat' (non-Euclidean) geometrical methods.
>
> 'Flat' in this context refers to Euclidean geometry (parts of which are taught as 'plane' geometry), and non-flat refers to non-Euclidean geometry. What does geometry have to do with machine learning? Well, as two fields that are rooted in mathematics, there must be a common way to measure distances between points in clusters, and that can be done in a 'flat' or 'non-flat' way, depending on the nature of the data. [Euclidean distances](https://wikipedia.org/wiki/Euclidean_distance) are measured as the length of a line segment between two points. [Non-Euclidean distances](https://wikipedia.org/wiki/Non-Euclidean_geometry) are measured along a curve. If your data, visualized, seems to not exist on a plane, you might need to use a specialized algorithm to handle it.

![Infographic by Dasani Madipalli](../../images/flat-nonflat.png){width="500"}

> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
>
> Clusters are defined by their distance matrix, e.g. the distances between points. This distance can be measured a few ways. Euclidean clusters are defined by the average of the point values, and contain a 'centroid' or center point. Distances are thus measured by the distance to that centroid. Non-Euclidean distances refer to 'clustroids', the point closest to other points. Clustroids in turn can be defined in various ways.
>
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
>
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) introduces 'semi-supervised' learning into this unsupervised method. The relationships between points are flagged as 'cannot link' or 'must-link' so some rules are forced on the dataset.
>
> An example: If an algorithm is set free on a batch of unlabelled or semi-labelled data, the clusters it produces may be of poor quality. In the example above, the clusters might group 'round music things' and 'square music things' and 'triangular things' and 'cookies'. If given some constraints, or rules to follow ("the item must be made of plastic", "the item needs to be able to produce music") this can help 'constrain' the algorithm to make better choices.
>
> 🎓 'Density'
>
> Data that is 'noisy' is considered to be 'dense'. The distances between points in each of its clusters may prove, on examination, to be more or less dense, or 'crowded' and thus this data needs to be analyzed with the appropriate clustering method. [This article](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) demonstrates the difference between using K-Means clustering vs. HDBSCAN algorithms to explore a noisy dataset with uneven cluster density.

Deepen your understanding of clustering techniques in this [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

### **Clustering algorithms**

There are over 100 clustering algorithms, and their use depends on the nature of the data at hand. Let's discuss some of the major ones:

-   **Hierarchical clustering**. If an object is classified by its proximity to a nearby object, rather than to one farther away, clusters are formed based on their members' distance to and from other objects. Hierarchical clustering is characterized by repeatedly combining two clusters.

![Infographic by Dasani Madipalli](../../images/hierarchical.png){width="500"}

-   **Centroid clustering**. This popular algorithm requires the choice of 'k', or the number of clusters to form, after which the algorithm determines the center point of a cluster and gathers data around that point. [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) is a popular version of centroid clustering which separates a data set into pre-defined K groups. The center is determined by the nearest mean, thus the name. The squared distance from the cluster is minimized.![Infographic by Dasani Madipalli](../../images/centroid.png){width="500"}

-   **Distribution-based clustering**. Based in statistical modeling, distribution-based clustering centers on determining the probability that a data point belongs to a cluster, and assigning it accordingly. Gaussian mixture methods belong to this type.

-   **Density-based clustering**. Data points are assigned to clusters based on their density, or their grouping around each other. Data points far from the group are considered outliers or noise. DBSCAN, Mean-shift and OPTICS belong to this type of clustering.

-   **Grid-based clustering**. For multi-dimensional datasets, a grid is created and the data is divided amongst the grid's cells, thereby creating clusters.

The best way to learn about clustering is to try it for yourself, so that's what you'll do in this exercise.

We'll require some packages to knock-off this module. You can have them installed as: `install.packages(c('tidyverse', 'tidymodels', 'DataExplorer', 'summarytools', 'plotly', 'paletteer', 'corrplot', 'patchwork'))`

Alternatively, the script below checks whether you have the packages required to complete this module and installs them for you in case some are missing.

```{r}
suppressWarnings(if(!require("pacman")) install.packages("pacman"))

pacman::p_load('tidyverse', 'tidymodels', 'DataExplorer', 'summarytools', 'plotly', 'paletteer', 'corrplot', 'patchwork')
```

```{r setup}
knitr::opts_chunk$set(warning = F, message = F)

```

## Exercise - cluster your data

Clustering as a technique is greatly aided by proper visualization, so let's get started by visualizing our music data. This exercise will help us decide which of the methods of clustering we should most effectively use for the nature of this data.

Let's hit the ground running by importing the data.

```{r}
# Load the core tidyverse and make it available in your current R session
library(tidyverse)

# Import the data into a tibble
df <- read_csv(file = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/5-Clustering/data/nigerian-songs.csv")

# View the first 5 rows of the data set
df %>% 
  slice_head(n = 5)

```

Sometimes, we may want some little more information on our data. We can have a look at the `data` and `its structure` by using the [*glimpse()*](https://pillar.r-lib.org/reference/glimpse.html) function:

```{r}
# Glimpse into the data set
df %>% 
  glimpse()
```

Good job!💪

We can observe that `glimpse()` will give you the total number of rows (observations) and columns (variables), then, the first few entries of each variable in a row after the variable name. In addition, the *data type* of the variable is given immediately after each variable's name inside `< >`.

`DataExplorer::introduce()` can summarize this information neatly:

```{r DataExplorer}
# Describe basic information for our data
df %>% 
  introduce()

# A visual display of the same
df %>% 
  plot_intro()

```

Awesome! We have just learnt that our data has no missing values.

While we are at it, we can explore common central tendency statistics (e.g [mean](https://en.wikipedia.org/wiki/Arithmetic_mean) and [median](https://en.wikipedia.org/wiki/Median)) and measures of dispersion (e.g [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)) using `summarytools::descr()`

```{r summarytools}
# Describe common statistics
df %>% 
  descr(stats = "common")

```

Let's look at the general values of the data. Note that popularity can be `0`, which show songs that have no ranking. We'll remove those shortly.

> 🤔 If we are working with clustering, an unsupervised method that does not require labeled data, why are we showing this data with labels? In the data exploration phase, they come in handy, but they are not necessary for the clustering algorithms to work.

### 1. Explore popular genres

Let's go ahead and find out the most popular genres 🎶 by making a count of the instances it appears.

```{r count_genres}
# Popular genres
top_genres <- df %>% 
  count(artist_top_genre, sort = TRUE) %>% 
# Encode to categorical and reorder the according to count
  mutate(artist_top_genre = factor(artist_top_genre) %>% fct_inorder())

# Print the top genres
top_genres

```

That went well! They say a picture is worth a thousand rows of a data frame (actually nobody ever says that 😅). But you get the gist of it, right?

One way to visualize categorical data (character or factor variables) is using barplots. Let's make a barplot of the top 10 genres:

```{r bar_plot_genre}
# Change the default gray theme
theme_set(theme_light())

# Visualize popular genres
top_genres %>%
  slice(1:10) %>% 
  ggplot(mapping = aes(x = artist_top_genre, y = n,
                       fill = artist_top_genre)) +
  geom_col(alpha = 0.8) +
  paletteer::scale_fill_paletteer_d("rcartocolor::Vivid") +
  ggtitle("Top genres") +
  theme(plot.title = element_text(hjust = 0.5),
        # Rotates the X markers (so we can read them)
    axis.text.x = element_text(angle = 90))
```

Now it's way easier to identify that we have `missing` genres 🧐!

> A good visualisation will show you things that you did not expect, or raise new questions about the data - Hadley Wickham and Garrett Grolemund, [R For Data Science](https://r4ds.had.co.nz/introduction.html)

Note, when the top genre is described as `Missing`, that means that Spotify did not classify it, so let's get rid of it.

```{r remove_missing}
# Visualize popular genres
top_genres %>%
  filter(artist_top_genre != "Missing") %>% 
  slice(1:10) %>% 
  ggplot(mapping = aes(x = artist_top_genre, y = n,
                       fill = artist_top_genre)) +
  geom_col(alpha = 0.8) +
  paletteer::scale_fill_paletteer_d("rcartocolor::Vivid") +
  ggtitle("Top genres") +
  theme(plot.title = element_text(hjust = 0.5),
        # Rotates the X markers (so we can read them)
    axis.text.x = element_text(angle = 90))
```

From the little data exploration, we learn that the top three genres dominate this dataset. Let's concentrate on `afro dancehall`, `afropop`, and `nigerian pop`, additionally filter the dataset to remove anything with a 0 popularity value (meaning it was not classified with a popularity in the dataset and can be considered noise for our purposes):

```{r new_dataset}
nigerian_songs <- df %>% 
  # Concentrate on top 3 genres
  filter(artist_top_genre %in% c("afro dancehall", "afropop","nigerian pop")) %>% 
  # Remove unclassified observations
  filter(popularity != 0)



# Visualize popular genres
nigerian_songs %>%
  count(artist_top_genre) %>%
  ggplot(mapping = aes(x = artist_top_genre, y = n,
                       fill = artist_top_genre)) +
  geom_col(alpha = 0.8) +
  paletteer::scale_fill_paletteer_d("ggsci::category10_d3") +
  ggtitle("Top genres") +
  theme(plot.title = element_text(hjust = 0.5))
```

Let's see whether there is any apparent linear relationship among the numerical variables in our data set. This relationship is quantified mathematically by the [correlation statistic](https://en.wikipedia.org/wiki/Correlation).

The correlation statistic is a value between -1 and 1 that indicates the strength of a relationship. Values above 0 indicate a *positive* correlation (high values of one variable tend to coincide with high values of the other), while values below 0 indicate a *negative* correlation (high values of one variable tend to coincide with low values of the other).

```{r correlation}
# Narrow down to numeric variables and fid correlation
corr_mat <- nigerian_songs %>% 
  select(where(is.numeric)) %>% 
  cor()

# Visualize correlation matrix
corrplot(corr_mat, order = 'AOE', col = c('white', 'black'), bg = 'gold2')  
```

The data is not strongly correlated except between `energy` and `loudness`, which makes sense, given that loud music is usually pretty energetic. `Popularity` has a correspondence to `release date`, which also makes sense, as more recent songs are probably more popular. Length and energy seem to have a correlation too.

It will be interesting to see what a clustering algorithm can make of this data!

> 🎓 Note that correlation does not imply causation! We have proof of correlation but no proof of causation. An [amusing web site](https://tylervigen.com/spurious-correlations) has some visuals that emphasize this point.

### 2. Explore data distribution

Let's ask some more subtle questions. Are the genres significantly different in the perception of their danceability, based on their popularity? Let's examine our top three genres data distribution for popularity and danceability along a given x and y axis using [density plots](https://www.khanacademy.org/math/ap-statistics/density-curves-normal-distribution-ap/density-curves/v/density-curves).

```{r}
# Perform 2D kernel density estimation
density_estimate_2d <- nigerian_songs %>% 
  ggplot(mapping = aes(x = popularity, y = danceability, color = artist_top_genre)) +
  geom_density_2d(bins = 5, size = 1) +
  paletteer::scale_color_paletteer_d("RSkittleBrewer::wildberry") +
  xlim(-20, 80) +
  ylim(0, 1.2)

# Density plot based on the popularity
density_estimate_pop <- nigerian_songs %>% 
  ggplot(mapping = aes(x = popularity, fill = artist_top_genre, color = artist_top_genre)) +
  geom_density(size = 1, alpha = 0.5) +
  paletteer::scale_fill_paletteer_d("RSkittleBrewer::wildberry") +
  paletteer::scale_color_paletteer_d("RSkittleBrewer::wildberry") +
  theme(legend.position = "none")

# Density plot based on the danceability
density_estimate_dance <- nigerian_songs %>% 
  ggplot(mapping = aes(x = danceability, fill = artist_top_genre, color = artist_top_genre)) +
  geom_density(size = 1, alpha = 0.5) +
  paletteer::scale_fill_paletteer_d("RSkittleBrewer::wildberry") +
  paletteer::scale_color_paletteer_d("RSkittleBrewer::wildberry")


# Patch everything together
library(patchwork)
density_estimate_2d / (density_estimate_pop + density_estimate_dance)
```

We see that there are concentric circles that line up, regardless of genre. Could it be that Nigerian tastes converge at a certain level of danceability for this genre?

In general, the three genres align in terms of their popularity and danceability. Determining clusters in this loosely-aligned data will be a challenge. Let's see whether a scatter plot can support this.

```{r scatter_plot}
# A scatter plot of popularity and danceability
scatter_plot <- nigerian_songs %>% 
  ggplot(mapping = aes(x = popularity, y = danceability, color = artist_top_genre, shape = artist_top_genre)) +
  geom_point(size = 2, alpha = 0.8) +
  paletteer::scale_color_paletteer_d("futurevisions::mars")

# Add a touch of interactivity
ggplotly(scatter_plot)
```

A scatterplot of the same axes shows a similar pattern of convergence.

In general, for clustering, you can use scatterplots to show clusters of data, so mastering this type of visualization is very useful. In the next lesson, we will take this filtered data and use k-means clustering to discover groups in this data that see to overlap in interesting ways.

## **🚀 Challenge**

In preparation for the next lesson, make a chart about the various clustering algorithms you might discover and use in a production environment. What kinds of problems is the clustering trying to address?

## [**Post-lecture quiz**](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/28/)

## **Review & Self Study**

Before you apply clustering algorithms, as we have learned, it's a good idea to understand the nature of your dataset. Read more on this topic [here](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

Deepen your understanding of clustering techniques:

-   [Train and Evaluate Clustering Models using Tidymodels and friends](https://rpubs.com/eR_ic/clustering)

-   Bradley Boehmke & Brandon Greenwell, [*Hands-On Machine Learning with R*](https://bradleyboehmke.github.io/HOML/)*.*

## **Assignment**

[Research other visualizations for clustering](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/assignment.md)

## THANK YOU TO:

[Jen Looper](https://www.twitter.com/jenlooper) for creating the original Python version of this module ♥️

[`Dasani Madipalli`](https://twitter.com/dasani_decoded) for creating the amazing illustrations that make machine learning concepts more interpretable and easier to understand.

Happy Learning,

[Eric](https://twitter.com/ericntay), Gold Microsoft Learn Student Ambassador.
