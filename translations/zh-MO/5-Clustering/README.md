# 用於機器學習的分群模型

分群是機器學習中的一項任務，旨在尋找彼此相似的物件，並將它們分組成稱為叢集的群體。與機器學習中的其他方法不同的是，分群是自動進行的，事實上，可以說它是監督式學習的相反。

## 地區主題：為奈及利亞觀眾的音樂品味設計的分群模型 🎧

奈及利亞多元化的觀眾擁有多樣的音樂品味。使用從 Spotify 抓取的數據（靈感來自[本文](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），讓我們來看看在奈及利亞流行的一些音樂。此數據集包含各種歌曲的「舞蹈性」分數、「原聲性」、響度、「語音性」、人氣和能量等數據。發掘此數據中的模式將會很有趣！

![一個唱盤](../../../translated_images/zh-MO/turntable.f2b86b13c53302dc.webp)

> 照片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 拍攝，出自 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在這系列課程中，你將發現使用分群技術分析數據的新方法。當數據集缺少標籤時，分群特別有用。如果有標籤，則像你之前課程中學過的分類技術可能更為適合。但在你想要分組無標籤數據的情況下，分群是一種絕佳的模式發現方式。

> 有一些有用的低代碼工具，能幫助你學習如何處理分群模型。嘗試用這個任務的 [Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) 吧

## 課程

1. [分群導論](1-Visualize/README.md)
2. [K-均值分群](2-K-Means/README.md)

## 致謝

這些課程由 [Jen Looper](https://www.twitter.com/jenlooper) 撰寫🎶，並由 [Rishit Dagli](https://rishit_dagli/) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供寶貴的評審意見。

[奈及利亞歌曲](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify)數據集來自 Kaggle，為從 Spotify 抓取的數據。

協助撰寫本課程的有用 K-均值範例包括這個 [鳶尾花探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、這個[入門筆記本](https://www.kaggle.com/prashant111/k-means-clustering-with-python) 以及這個 [假設的非政府組織範例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->