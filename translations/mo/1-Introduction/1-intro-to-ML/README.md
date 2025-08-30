<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "73e9a7245aa57f00cd413ffd22c0ccb6",
  "translation_date": "2025-08-29T21:29:01+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "mo"
}
-->
# 機器學習簡介

## [課前測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![初學者的機器學習 - 機器學習入門](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "初學者的機器學習 - 機器學習入門")

> 🎥 點擊上方圖片觀看本課程的簡短介紹影片。

歡迎來到這門針對初學者的經典機器學習課程！無論您是完全新手，還是有經驗的機器學習從業者希望重新學習某些領域，我們都很高興您加入我們！我們希望為您的機器學習研究創造一個友好的起點，並樂於評估、回應和採納您的[反饋](https://github.com/microsoft/ML-For-Beginners/discussions)。

[![機器學習簡介](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "機器學習簡介")

> 🎥 點擊上方圖片觀看影片：麻省理工學院的 John Guttag 介紹機器學習

---
## 開始學習機器學習

在開始學習本課程之前，您需要先設置好您的電腦，準備在本地運行筆記本。

- **使用以下影片配置您的電腦**。點擊以下連結學習[如何安裝 Python](https://youtu.be/CXZYvNRIAKM)以及[設置文本編輯器](https://youtu.be/EU8eayHWoZg)進行開發。
- **學習 Python**。建議您對[Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott)有基本的了解，這是一種對數據科學家非常有用的編程語言，我們在本課程中會使用它。
- **學習 Node.js 和 JavaScript**。我們在本課程中偶爾會使用 JavaScript 來構建網頁應用，因此您需要安裝 [node](https://nodejs.org) 和 [npm](https://www.npmjs.com/)，以及[Visual Studio Code](https://code.visualstudio.com/)來進行 Python 和 JavaScript 的開發。
- **創建 GitHub 帳戶**。既然您在 [GitHub](https://github.com) 找到了我們，您可能已經有帳戶了，但如果沒有，請創建一個帳戶，然後 fork 本課程以供自己使用。（也可以給我們點個星星 😊）
- **探索 Scikit-learn**。熟悉 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)，這是一組我們在課程中引用的機器學習庫。

---
## 什麼是機器學習？

「機器學習」這個詞是當今最流行且最常用的術語之一。如果您對技術有一定的了解，無論您從事哪個領域，都有很大的可能性至少聽過一次這個詞。然而，機器學習的運作機制對大多數人來說仍然是個謎。對於機器學習初學者來說，這個主題有時可能會讓人感到不知所措。因此，了解機器學習的真正含義並通過實際例子逐步學習它是非常重要的。

---
## 熱潮曲線

![機器學習熱潮曲線](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.mo.png)

> Google 趨勢顯示了「機器學習」這個詞最近的熱潮曲線

---
## 神秘的宇宙

我們生活在一個充滿迷人謎團的宇宙中。像史蒂芬·霍金、阿爾伯特·愛因斯坦等偉大的科學家都將畢生精力投入到尋找有意義的信息，以揭示我們周圍世界的奧秘。這是人類學習的本質：一個孩子隨著成長逐年學習新事物，揭示他們世界的結構。

---
## 孩子的大腦

孩子的大腦和感官感知周圍環境的事實，並逐漸學習生活中隱藏的模式，幫助孩子制定邏輯規則來識別已學習的模式。人類大腦的學習過程使人類成為世界上最複雜的生物。通過不斷學習隱藏的模式並在這些模式上進行創新，我們能夠在一生中不斷提升自己。這種學習能力和進化能力與一個名為[大腦可塑性](https://www.simplypsychology.org/brain-plasticity.html)的概念有關。表面上，我們可以將人類大腦的學習過程與機器學習的概念進行一些激勵性的類比。

---
## 人類大腦

[人類大腦](https://www.livescience.com/29365-human-brain.html)從現實世界中感知事物，處理感知到的信息，做出理性決策，並根據情況執行某些行動。這就是我們所說的智能行為。當我們將智能行為過程的模擬編程到機器中時，這就被稱為人工智能（AI）。

---
## 一些術語

雖然這些術語可能會混淆，但機器學習（ML）是人工智能的一個重要子集。**機器學習專注於使用專門的算法從感知到的數據中挖掘有意義的信息並發現隱藏的模式，以支持理性決策過程**。

---
## AI、ML、深度學習

![AI、ML、深度學習、數據科學](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.mo.png)

> 一張展示 AI、ML、深度學習和數據科學之間關係的圖表。由 [Jen Looper](https://twitter.com/jenlooper) 製作，靈感來自[這張圖表](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## 涵蓋的概念

在本課程中，我們將僅涵蓋機器學習的核心概念，這些是初學者必須了解的。我們主要使用 Scikit-learn 來教授所謂的「經典機器學習」，這是一個許多學生用來學習基礎知識的優秀庫。要理解人工智能或深度學習的更廣泛概念，扎實的機器學習基礎知識是不可或缺的，因此我們希望在此提供這些知識。

---
## 在本課程中您將學到：

- 機器學習的核心概念
- 機器學習的歷史
- 機器學習與公平性
- 回歸機器學習技術
- 分類機器學習技術
- 聚類機器學習技術
- 自然語言處理機器學習技術
- 時間序列預測機器學習技術
- 強化學習
- 機器學習的實際應用

---
## 我們不會涵蓋的內容

- 深度學習
- 神經網絡
- 人工智能

為了提供更好的學習體驗，我們將避免涉及神經網絡的複雜性、「深度學習」——使用神經網絡構建多層模型——以及人工智能，這些內容我們會在其他課程中討論。我們還將提供即將推出的數據科學課程，專注於這個更大領域的相關方面。

---
## 為什麼要學習機器學習？

從系統的角度來看，機器學習被定義為創建能夠從數據中學習隱藏模式以幫助做出智能決策的自動化系統。

這種動機在某種程度上受到人類大腦如何根據外界感知到的數據學習某些事物的啟發。

✅ 想一想，為什麼企業會選擇使用機器學習策略，而不是創建基於硬編碼規則的引擎？

---
## 機器學習的應用

機器學習的應用如今幾乎無處不在，就像我們社會中流動的數據一樣，這些數據由智能手機、連接設備和其他系統生成。考慮到最先進的機器學習算法的巨大潛力，研究人員一直在探索它們解決多維度和多學科的現實問題的能力，並取得了非常積極的成果。

---
## 應用機器學習的例子

**機器學習有許多用途**：

- 從患者的病史或報告中預測疾病的可能性。
- 利用天氣數據預測天氣事件。
- 理解文本的情感。
- 檢測假新聞以阻止宣傳的傳播。

金融、經濟、地球科學、太空探索、生物醫學工程、認知科學，甚至人文領域都已經採用機器學習來解決其領域中繁重的數據處理問題。

---
## 結論

機器學習通過從現實世界或生成的數據中發現有意義的洞察來自動化模式發現的過程。它已經在商業、健康和金融應用等領域證明了自己的巨大價值。

在不久的將來，了解機器學習的基礎知識將成為任何領域人士的必備技能，因為它的廣泛採用。

---
# 🚀 挑戰

使用紙筆或像 [Excalidraw](https://excalidraw.com/) 這樣的在線應用程序，繪製您對 AI、ML、深度學習和數據科學之間差異的理解。添加一些每種技術擅長解決的問題的想法。

# [課後測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# 回顧與自學

要了解如何在雲端中使用機器學習算法，請參考此[學習路徑](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott)。

參加一個關於機器學習基礎的[學習路徑](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott)。

---
# 作業

[開始學習](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。