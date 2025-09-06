<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-06T10:49:54+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "en"
}
-->
# Clustering models for machine learning

Clustering is a machine learning task that aims to identify objects that are similar to each other and group them into clusters. What sets clustering apart from other machine learning approaches is that it happens automatically. In fact, it’s fair to say it’s the opposite of supervised learning.

## Regional topic: clustering models for a Nigerian audience's musical taste 🎧

Nigeria’s diverse population has equally diverse musical preferences. Using data scraped from Spotify (inspired by [this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), let’s explore some music that’s popular in Nigeria. This dataset includes information about various songs, such as their 'danceability' score, 'acousticness', loudness, 'speechiness', popularity, and energy. It will be fascinating to uncover patterns in this data!

![A turntable](../../../5-Clustering/images/turntable.jpg)

> Photo by <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> on <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In this series of lessons, you’ll learn new ways to analyze data using clustering techniques. Clustering is especially useful when your dataset doesn’t have labels. If your dataset does have labels, classification techniques like the ones you learned in earlier lessons might be more appropriate. However, when you want to group unlabeled data, clustering is an excellent way to uncover patterns.

> There are helpful low-code tools available to assist you in working with clustering models. Consider using [Azure ML for this task](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott).

## Lessons

1. [Introduction to clustering](1-Visualize/README.md)
2. [K-Means clustering](2-K-Means/README.md)

## Credits

These lessons were created with 🎶 by [Jen Looper](https://www.twitter.com/jenlooper), with valuable reviews by [Rishit Dagli](https://rishit_dagli) and [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

The [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) dataset was sourced from Kaggle, based on data scraped from Spotify.

Useful K-Means examples that contributed to the development of this lesson include this [iris exploration](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), this [introductory notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), and this [hypothetical NGO example](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.