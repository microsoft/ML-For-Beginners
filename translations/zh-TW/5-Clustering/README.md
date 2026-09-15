# 用於機器學習的群集模型

群集是一種機器學習任務，它旨在尋找彼此相似的物件並將它們分組為稱為群集的群組。群集與機器學習中的其他方法不同之處在於一切自動發生，事實上，可以說它是監督式學習的相反。

## 地區主題：針對奈及利亞觀眾音樂品味的群集模型 🎧

奈及利亞多元的觀眾擁有多元的音樂品味。利用從 Spotify 擷取的資料（靈感來自[this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)），讓我們看看在奈及利亞流行的一些音樂。此資料集包含關於各首歌曲的「舞蹈性」分數、「聲學性」、響度、「語音性」、流行度和能量的數據。發掘這些數據中的模式將會很有趣！

![一個唱盤](../../../translated_images/zh-TW/turntable.f2b86b13c53302dc.webp)

> 照片由 <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> 拍攝，來自 <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
在這系列課程中，你將發現利用群集技術分析數據的新方法。群集特別適用於資料集中缺少標籤的情況。如果有標籤，則如同你在先前課程中學到的分類技術可能會更有用。但在尋找將無標籤資料分組的情況下，群集是一種發現模式的極佳方法。

> 有許多實用的低程式碼工具可以幫助你學習如何使用群集模型。試試這個任務用的[Azure ML](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## 課程

1. [群集介紹](1-Visualize/README.md)
2. [K-均值群集](2-K-Means/README.md)

## 參考資料

這些課程是由 🎶 [Jen Looper](https://www.twitter.com/jenlooper) 撰寫，並由 [Rishit Dagli](https://rishit_dagli/) 和 [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) 提供寶貴的評論。

[奈及利亞歌曲](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) 資料集取自 Kaggle，資料是從 Spotify 擷取的。

協助本課程製作的有用 K-均值範例包括這個[鳶尾花探索](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering)、這個[入門筆記本](https://www.kaggle.com/prashant111/k-means-clustering-with-python)以及這個[假想 NGO 案例](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->