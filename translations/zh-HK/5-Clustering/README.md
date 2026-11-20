# 機器學習的群集模型

群集是一種機器學習任務，它旨在尋找彼此相似的物件，並將它們分組成所謂的群集。群集與其他機器學習方法的不同之處在於，所有過程是自動進行的，事實上，可以說它是監督式學習的相反。

## 地區主題：針對尼日利亞聽眾音樂品味的群集模型 🎧

尼日利亞多元化的聽眾群體擁有多樣化的音樂品味。我們用從 Spotify 擷取的數據（靈感來自[this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），來看看在尼日利亞受歡迎的一些音樂。這個資料集包含了各種歌曲的「適合跳舞度」分數、「原聲度」、響度、「語音性」、受歡迎度和能量。探索這些數據中的模式會很有趣！

![A turntable](../../../translated_images/zh-HK/turntable.f2b86b13c53302dc.webp)

> 圖片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 拍攝，來源於 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在這系列課程中，您將會發現使用群集技術分析數據的新方法。當您的資料集缺乏標籤時，群集特別有用。如果資料集中有標籤，那麼像之前課程中學過的分類技術可能會更有用。但在尋找群組無標籤資料的情況下，群集是一個很棒的方式來發現模式。

> 有一些有用的低程式碼工具可以幫助您學習如何使用群集模型。試試 [適用於此任務的 Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 課程

1. [群集入門](1-Visualize/README.md)
2. [K-Means 群集](2-K-Means/README.md)

## 版權聲明

這些課程由 [Jen Looper](https://www.twitter.com/jenlooper) 撰寫並作曲 🎶，並由 [Rishit Dagli](https://rishit_dagli/) 及 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供寶貴的審閱意見。

[尼日利亞歌曲](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 資料集來自於 Kaggle，資料是從 Spotify 擷取而得。

有助於創建本課程的有用 K-Means 範例包括這個 [鳶尾花資料集探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、這個 [入門筆記本](https://www.kaggle.com/prashant111/k-means-clustering-with-python)，以及這個 [假設性的非政府組織範例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->