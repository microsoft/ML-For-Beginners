<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-03T18:27:17+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "tw"
}
-->
# 強化學習簡介

強化學習（RL）被視為與監督學習和非監督學習並列的基本機器學習範式之一。RL 的核心在於決策：做出正確的決策，或者至少從中學習。

想像你有一個模擬環境，例如股票市場。如果你施加某項規定，會發生什麼？它會產生正面還是負面的影響？如果發生負面影響，你需要接受這種_負面強化_，從中學習並改變方向。如果是正面結果，你需要基於這種_正面強化_進一步發展。

![彼得與狼](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.tw.png)

> 彼得和他的朋友們需要逃離飢餓的狼！圖片由 [Jen Looper](https://twitter.com/jenlooper) 提供

## 區域主題：彼得與狼（俄羅斯）

[彼得與狼](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) 是由俄羅斯作曲家 [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) 創作的一部音樂童話故事。這是一個關於年輕的先鋒彼得的故事，他勇敢地走出家門，來到森林空地追逐狼。在本節中，我們將訓練機器學習算法來幫助彼得：

- **探索**周圍環境並建立最佳導航地圖
- **學習**如何使用滑板並保持平衡，以便更快地移動。

[![彼得與狼](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 點擊上方圖片收聽 Prokofiev 的《彼得與狼》

## 強化學習

在之前的章節中，你已經看到兩個機器學習問題的例子：

- **監督學習**，我們擁有建議問題解決方案的數據集。[分類](../4-Classification/README.md) 和 [回歸](../2-Regression/README.md) 是監督學習任務。
- **非監督學習**，我們沒有標記的訓練數據。非監督學習的主要例子是 [聚類](../5-Clustering/README.md)。

在本節中，我們將介紹一種不需要標記訓練數據的新型學習問題。有幾種類型的此類問題：

- **[半監督學習](https://wikipedia.org/wiki/Semi-supervised_learning)**，我們擁有大量未標記的數據，可以用來預訓練模型。
- **[強化學習](https://wikipedia.org/wiki/Reinforcement_learning)**，代理通過在某些模擬環境中進行實驗來學習如何行為。

### 示例 - 電腦遊戲

假設你想教電腦玩遊戲，例如象棋或 [超級瑪利歐](https://wikipedia.org/wiki/Super_Mario)。為了讓電腦玩遊戲，我們需要它在每個遊戲狀態下預測應該採取的行動。雖然這看起來像是一個分類問題，但事實並非如此——因為我們沒有包含狀態和相應行動的數據集。雖然我們可能擁有一些數據，例如現有的象棋比賽或玩家玩超級瑪利歐的錄像，但這些數據可能不足以涵蓋足夠多的可能狀態。

與其尋找現有的遊戲數據，**強化學習**（RL）的核心思想是讓電腦多次玩遊戲並觀察結果。因此，要應用強化學習，我們需要兩樣東西：

- **一個環境**和**一個模擬器**，允許我們多次玩遊戲。這個模擬器會定義所有的遊戲規則以及可能的狀態和行動。

- **一個獎勵函數**，告訴我們在每次行動或遊戲中表現得如何。

其他類型的機器學習與 RL 的主要區別在於，RL 通常直到遊戲結束才知道我們是贏還是輸。因此，我們無法判斷某個單獨的行動是否是好的——我們只有在遊戲結束時才會收到獎勵。而我們的目標是設計算法，讓我們能夠在不確定的條件下訓練模型。我們將學習一種名為 **Q-learning** 的 RL 算法。

## 課程

1. [強化學習與 Q-Learning 簡介](1-QLearning/README.md)
2. [使用 gym 模擬環境](2-Gym/README.md)

## 致謝

《強化學習簡介》由 [Dmitry Soshnikov](http://soshnikov.com) 用 ♥️ 編寫

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們致力於提供準確的翻譯，請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵資訊，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或誤釋不承擔責任。