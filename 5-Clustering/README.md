# Clustering models for machine learning

Clustering is a machine learning task where it looks to find objects that resemble one another and group these into groups called clusters.  What differs clustering from other approaches in machine learning, is that things happen automatically, in fact, it's fair to say it's the opposite of supervised learning. 

## Regional topic: clustering models for a Nigerian audience's musical taste 🎧

Nigeria's diverse audience has diverse musical tastes. Using data scraped from Spotify (inspired by [this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), let's look at some music popular in Nigeria. This dataset includes data about various songs' 'danceability' score, 'acousticness', loudness, 'speechiness', popularity and energy. It will be interesting to discover patterns in this data!

![A turntable](./images/turntable.jpg)

> Photo by <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> on <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In this series of lessons, you will discover new ways to analyze data using clustering techniques. Clustering is particularly useful when your dataset lacks labels. If it does have labels, then classification techniques such as those you learned in previous lessons might be more useful. But in cases where you are looking to group unlabelled data, clustering is a great way to discover patterns.

> There are useful low-code tools that can help you learn about working with clustering models. Try [Azure ML for this task](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lessons

1. [Introduction to clustering](1-Visualize/README.md)
2. [K-Means clustering](2-K-Means/README.md)

## Credits

These lessons were written with 🎶 by [Jen Looper](https://www.twitter.com/jenlooper) with helpful reviews by [Rishit Dagli](https://rishit_dagli) and [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

The [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) dataset was sourced from Kaggle as scraped from Spotify.

Useful K-Means examples that aided in creating this lesson include this [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), this [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), and this [hypothetical NGO example](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).
