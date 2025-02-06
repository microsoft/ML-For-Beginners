# 机器学习中的聚类模型

聚类是一种机器学习任务，旨在寻找相似的对象并将它们分组到称为簇的组中。聚类与机器学习中的其他方法不同，因为它是自动发生的，实际上可以说它是监督学习的反面。

## 地区专题：为尼日利亚观众的音乐品味设计的聚类模型 🎧

尼日利亚的观众有着多样化的音乐品味。使用从Spotify抓取的数据（灵感来源于[这篇文章](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)，让我们看看尼日利亚流行的一些音乐。这个数据集包括关于各种歌曲的“舞蹈性”得分、“声学性”、响度、“演讲性”、流行度和能量的数据。发现这些数据中的模式将会很有趣！

![唱盘](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.zh.jpg)

> 照片由<a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a>拍摄，发布于<a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在这一系列课程中，你将发现使用聚类技术分析数据的新方法。当你的数据集缺乏标签时，聚类特别有用。如果数据集有标签，那么你在之前课程中学到的分类技术可能会更有用。但在你想要对未标记的数据进行分组的情况下，聚类是发现模式的好方法。

> 有一些有用的低代码工具可以帮助你学习如何使用聚类模型。试试[Azure ML来完成这个任务](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 课程

1. [聚类简介](1-Visualize/README.md)
2. [K-Means聚类](2-K-Means/README.md)

## 致谢

这些课程由[Jen Looper](https://www.twitter.com/jenlooper)精心编写，并得到了[Rishit Dagli](https://rishit_dagli)和[Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)的有益评审。

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify)数据集来源于Kaggle，由Spotify抓取。

一些有用的K-Means示例对创建这节课提供了帮助，包括这个[iris探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)，这个[入门笔记](https://www.kaggle.com/prashant111/k-means-clustering-with-python)，以及这个[假设的NGO示例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

**免责声明**:
本文件使用基于机器的人工智能翻译服务进行翻译。虽然我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应将原文档视为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们不承担任何责任。