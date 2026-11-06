# 机器学习中的聚类模型

聚类是一项机器学习任务，它旨在寻找相似的对象并将它们归为称为簇的组。聚类与机器学习中其他方法的不同之处在于，过程是自动进行的，事实上，可以说它与监督学习正好相反。

## 区域主题：面向尼日利亚听众音乐口味的聚类模型 🎧

尼日利亚多样化的听众拥有多样的音乐口味。利用从 Spotify 抓取的数据（灵感来自[this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），我们来看一下尼日利亚流行的一些音乐。该数据集包含有关各种歌曲的“舞曲感”评分、“原声度”、响度、“语音性”、流行度和能量等数据。发现这些数据中的模式将很有趣！

![一个唱盘](../../../translated_images/zh-CN/turntable.f2b86b13c53302dc.webp)

> 照片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 于 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> 提供
  
在这一系列课程中，你将发现使用聚类技术分析数据的新方法。当你的数据集缺少标签时，聚类尤其有用。如果数据集有标签，那么像你在之前课程中学到的分类技术可能更有用。但在你希望对无标签数据进行分组的情况下，聚类是一种发现模式的绝佳方法。

> 有许多有用的低代码工具可以帮助你学习如何使用聚类模型。试试[用于此任务的 Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 课程

1. [聚类简介](1-Visualize/README.md)
2. [K-均值聚类](2-K-Means/README.md)

## 致谢

这些课程由🎶[Jen Looper](https://www.twitter.com/jenlooper) 编写，[Rishit Dagli](https://rishit_dagli/) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供有益的审阅。

[尼日利亚歌曲](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 数据集来源于 Kaggle，数据抓取自 Spotify。

帮助编写本课程的有用 K-均值示例包括这份[鸢尾花探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、这份[入门笔记本](https://www.kaggle.com/prashant111/k-means-clustering-with-python)和这份[假设非政府组织示例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->