<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T17:02:43+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "tw"
}
-->
# 機器學習中的分群模型

分群是一種機器學習任務，旨在尋找彼此相似的物件並將它們分組成稱為「群集」的群體。分群與機器學習中的其他方法不同之處在於，它是自動進行的，事實上，可以說它是監督式學習的反面。

## 區域主題：針對尼日利亞觀眾音樂品味的分群模型 🎧

尼日利亞的多元化觀眾擁有多樣化的音樂品味。使用從 Spotify 擷取的數據（靈感來自[這篇文章](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），讓我們來看看一些在尼日利亞流行的音樂。這個數據集包含了各種歌曲的「舞蹈性」分數、「聲學性」、音量、「語音性」、流行度和能量等數據。探索這些數據中的模式將會非常有趣！

![唱盤](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.tw.jpg)

> 照片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 提供，來自 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在這系列課程中，您將學習使用分群技術分析數據的新方法。當您的數據集缺乏標籤時，分群特別有用。如果數據集有標籤，那麼您在之前課程中學到的分類技術可能會更有用。但在需要對未標記數據進行分組的情況下，分群是一種發現模式的好方法。

> 有一些有用的低代碼工具可以幫助您學習如何使用分群模型。試試 [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) 來完成這項任務。

## 課程

1. [分群簡介](1-Visualize/README.md)
2. [K-Means 分群](2-K-Means/README.md)

## 致謝

這些課程由 [Jen Looper](https://www.twitter.com/jenlooper) 精心撰寫，並由 [Rishit Dagli](https://rishit_dagli) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供了有益的審核。

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 數據集來自 Kaggle，並從 Spotify 擷取。

有助於創建這些課程的有用 K-Means 示例包括這個[鳶尾花探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、這個[入門筆記本](https://www.kaggle.com/prashant111/k-means-clustering-with-python)，以及這個[假設的 NGO 示例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們致力於提供準確的翻譯，請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵資訊，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋不承擔責任。