# 机器学习中的聚类模型

聚类（clustering）是一项机器学习任务，用于寻找类似对象并将他们分成不同的组（这些组称做“聚类”（cluster））。聚类与其它机器学习方法的不同之处在于聚类是自动进行的。事实上，我们可以说它是监督学习的对立面。

## 本节主题: 尼日利亚观众音乐品味的聚类模型🎧

尼日利亚多样化的观众有着多样化的音乐品味。使用从 Spotify 上抓取的数据（受到[本文](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)的启发），让我们看看尼日利亚流行的一些音乐。这个数据集包括关于各种歌曲的舞蹈性、声学、响度、言语、流行度和活力的分数。从这些数据中发现一些模式（pattern）会是很有趣的事情!

![A turntable](../images/turntable.jpg)

> <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 在 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> 上的照片

在本系列课程中，您将发现使用聚类技术分析数据的新方法。当数据集缺少标签的时候，聚类特别有用。如果它有标签，那么分类技术(比如您在前面的课程中所学的那些)可能会更有用。但是如果要对未标记的数据进行分组，聚类是发现模式的好方法。

> 这里有一些有用的低代码工具可以帮助您了解如何使用聚类模型。尝试 [Azure ML for this task](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 课程安排

1. [介绍聚类](../1-Visualize/translations/README.zh-cn.md)
2. [K-Means 聚类](../2-K-Means/translations/README.zh-cn.md)

## 致谢

这些课程由 Jen Looper 在 🎶 上撰写，并由 [Rishit Dagli](https://rishit_dagli) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 进行了有帮助的评审。

[尼日利亚歌曲数据集](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 来自 Kaggle 抓取的 Spotify 数据。

一些帮助创造了这节课程的 K-Means 例子包括:[虹膜探索(iris exploration)](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)，[介绍性的笔记(introductory notebook)](https://www.kaggle.com/prashant111/k-means-clustering-with-python)，和 [假设非政府组织的例子(hypothetical NGO example)](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

