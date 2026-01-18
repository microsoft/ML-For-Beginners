<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0a6f4476a4f3934a4aa47c1bf47158bc",
  "translation_date": "2026-01-16T09:20:33+00:00",
  "source_file": "README.md",
  "language_code": "hk"
}
-->
[![GitHub license](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)  
[![GitHub contributors](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)  
[![GitHub issues](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)  
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)  
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)  

[![GitHub watchers](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)  
[![GitHub forks](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)  
[![GitHub stars](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)  

### 🌐 多語言支援

#### 透過 GitHub Action 支援（自動化並永遠更新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->  
[Arabic](../ar/README.md) | [Bengali](../bn/README.md) | [Bulgarian](../bg/README.md) | [Burmese (Myanmar)](../my/README.md) | [Chinese (Simplified)](../zh/README.md) | [Chinese (Traditional, Hong Kong)](./README.md) | [Chinese (Traditional, Macau)](../mo/README.md) | [Chinese (Traditional, Taiwan)](../tw/README.md) | [Croatian](../hr/README.md) | [Czech](../cs/README.md) | [Danish](../da/README.md) | [Dutch](../nl/README.md) | [Estonian](../et/README.md) | [Finnish](../fi/README.md) | [French](../fr/README.md) | [German](../de/README.md) | [Greek](../el/README.md) | [Hebrew](../he/README.md) | [Hindi](../hi/README.md) | [Hungarian](../hu/README.md) | [Indonesian](../id/README.md) | [Italian](../it/README.md) | [Japanese](../ja/README.md) | [Kannada](../kn/README.md) | [Korean](../ko/README.md) | [Lithuanian](../lt/README.md) | [Malay](../ms/README.md) | [Malayalam](../ml/README.md) | [Marathi](../mr/README.md) | [Nepali](../ne/README.md) | [Nigerian Pidgin](../pcm/README.md) | [Norwegian](../no/README.md) | [Persian (Farsi)](../fa/README.md) | [Polish](../pl/README.md) | [Portuguese (Brazil)](../br/README.md) | [Portuguese (Portugal)](../pt/README.md) | [Punjabi (Gurmukhi)](../pa/README.md) | [Romanian](../ro/README.md) | [Russian](../ru/README.md) | [Serbian (Cyrillic)](../sr/README.md) | [Slovak](../sk/README.md) | [Slovenian](../sl/README.md) | [Spanish](../es/README.md) | [Swahili](../sw/README.md) | [Swedish](../sv/README.md) | [Tagalog (Filipino)](../tl/README.md) | [Tamil](../ta/README.md) | [Telugu](../te/README.md) | [Thai](../th/README.md) | [Turkish](../tr/README.md) | [Ukrainian](../uk/README.md) | [Urdu](../ur/README.md) | [Vietnamese](../vi/README.md)  

> **想本地克隆？**  

> 此倉庫包含超過 50 種語言翻譯，會顯著增加下載大小。若想不帶翻譯文件克隆，請使用稀疏結帳：  
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
> 這樣你就能以更快的下載速度獲得完成課程所需的所有內容。  
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->  

#### 加入我們的社群

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們正在舉辦 Discord 的 AI 學習系列，了解詳情並於 2025 年 9 月 18 日至 30 日加入我們，詳情請見 [Learn with AI Series](https://aka.ms/learnwithai/discord)。你將獲得使用 GitHub Copilot 於資料科學的技巧與秘訣。

![Learn with AI series](../../../../translated_images/hk/3.9b58fd8d6c373c20.webp)

# 機器學習入門 - 課程大綱

> 🌍 透過世界各地文化來環遊世界，認識機器學習 🌍

微軟的 Cloud Advocates 很高興提供一個為期 12 週、共 26 節課的 **機器學習** 課程大綱。在這個課程中，你會學習到有時稱作 **經典機器學習** 的內容，主要使用 Scikit-learn 這個函式庫，並避開深度學習（深度學習則包含在我們的 [AI for Beginners 課程](https://aka.ms/ai4beginners) 中）。你也可以搭配我們的 ['Data Science for Beginners' 課程](https://aka.ms/ds4beginners) 一起學習！

與我們一同遊歷世界，將這些經典技術應用於世界各地的資料。每節課包含課前與課後測驗、書面教學、解答、習題等內容。我們採用以專案為導向的教學法，透過實作學習，確保新技能牢牢掌握。

**✍️ 衷心感謝作者團隊**Jen Looper、Stephen Howell、Francesca Lazzeri、Tomomi Imura、Cassie Breviu、Dmitry Soshnikov、Chris Noring、Anirban Mukherjee、Ornella Altunyan、Ruth Yakubu 及 Amy Boyd

**🎨 亦感謝插畫師**Tomomi Imura、Dasani Madipalli 和 Jen Looper

**🙏 特別感謝 🙏 微軟學生大使作者、審閱者及內容貢獻者**，尤其是 Rishit Dagli、Muhammad Sakib Khan Inan、Rohan Raj、Alexandru Petrescu、Abhishek Jaiswal、Nawrin Tabassum、Ioan Samuila 及 Snigdha Agarwal

**🤩 額外感謝微軟學生大使 Eric Wanjau、Jasleen Sondhi 和 Vidushi Gupta 貢獻我們的 R 課程！**

# 開始

請按照以下步驟操作：  
1. **Fork 倉庫**：點擊本頁右上角的「Fork」按鈕。  
2. **克隆倉庫**：`git clone https://github.com/microsoft/ML-For-Beginners.git`

> [你可以在我們的 Microsoft Learn 集合中找到本課程的所有附加資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要協助嗎？** 請參考我們的 [疑難排解指南](TROUBLESHOOTING.md)，其中涵蓋安裝、配置及執行課程常見問題的解決方案。

**[學生](https://aka.ms/student-page)**，使用此課程請將整個倉庫 fork 至你自己的 GitHub 帳戶，並自行完成練習或組成小組：

- 從課前小測開始。  
- 閱讀課程並完成活動，於每個知識點靜下心反思。  
- 嘗試理解課程內容並自行建立專案，而非直接執行解答程式碼；不過，每個專案導向課程的 `/solution` 資料夾中仍有對應的解答程式碼。  
- 完成課後小測。  
- 完成挑戰。  
- 完成作業。  
- 完成一組課程後，請造訪 [討論區](https://github.com/microsoft/ML-For-Beginners/discussions) 並以「大聲學習」的方式填寫相應的 PAT 打分表。PAT（進度評估工具）是你填寫以自行學習進程的評分標準。你也可以對其他人的 PAT 作出回應，和大家一起進步。

> 若想進一步學習，我們推薦以下 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 模組與學習路徑。  

**教師**，我們也在 [for-teachers.md](for-teachers.md) 提供了如何使用本課程的一些建議。

---

## 視頻示範

部分課程有短片教學，可在課程內嵌連結找到，或在 [Microsoft Developer YouTube 頻道的 ML for Beginners 播放清單](https://aka.ms/ml-beginners-videos) 觀看，點擊下方圖片即可。

[![ML for beginners banner](../../../../translated_images/hk/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 團隊介紹

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**GIF 動畫由** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal) 製作

> 🎥 點擊上方圖片，觀看本專案及創作者的影片介紹！

---

## 教學方法

我們建立本課程時秉持兩大教學原則：確保它是以**專案導向**為基礎，並包含**頻繁的測驗**。此外，本課程圍繞著一個共同的**主題**，以凝聚整體內容。

確保內容與專案對齊，使學生更投入學習並增強觀念的記憶。此外，上課前的低壓力測驗有助訂立學習目標，課後的第二次測驗則加強記憶。課程設計靈活且有趣，可全程參加或選擇部分學習。專案從小而簡單開始，12 週期末逐漸提升難度。本課程並包含關於機器學習實際應用的後記，可以作為額外學習學分或討論基礎。

> 請參考我們的 [行為守則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯指南](TRANSLATIONS.md)、與 [疑難排解](TROUBLESHOOTING.md)，我們歡迎你的建設性回饋！

## 每節課包括

- 選配速寫筆記  
- 選配輔助視頻  
- 視頻示範（部分課程）  
- [課前暖身測驗](https://ff-quizzes.netlify.app/en/ml/)  
- 書面課程內容  
- 專案課程包含逐步指引建置專案  
- 知識檢測  
- 挑戰題  
- 補充閱讀  
- 作業  
- [課後測驗](https://ff-quizzes.netlify.app/en/ml/)  

> **關於語言的備註**：本課程主要使用 Python 撰寫，但也有許多課程提供 R 版本。若要完成 R 課程，請到 `/solution` 資料夾尋找 R 課程，文件帶有 .rmd 副檔名，代表 **R Markdown** 文件。R Markdown 是一種結合 `程式碼區塊`（可包含 R 或其他語言）與 `YAML 標頭`（用於格式化輸出，如 PDF）於 `Markdown 文件`中的檔案格式。此架構非常適合數據科學創作，因可同時包含程式碼、輸出結果及書寫的想法。R Markdown 也可以輸出為 PDF、HTML 或 Word 等格式。
> **有關小測的說明**：所有小測均收錄於 [Quiz App folder](../../quiz-app)，共52個小測，每個含三題。它們會從課程中連結，但小測應用程式可本地執行；請依 `quiz-app` 資料夾中的指示在本地端架設或部署至 Azure。

| Lesson Number |                             主題                              |                   課程分類                   | 學習目標                                                                                                             |                                                              相關課程                                                               |                        作者                        |
| :-----------: | :------------------------------------------------------------: | :------------------------------------------: | ------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------: |
|      01       |                機器學習簡介                |      [Introduction](1-Introduction/README.md)       | 了解機器學習的基本概念                                                                                       |                                             [Lesson](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                       |
|      02       |                機器學習的歷史                 |      [Introduction](1-Introduction/README.md)       | 了解此領域的歷史                                                                                              |                                            [Lesson](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen and Amy                      |
|      03       |                 公平性與機器學習                  |      [Introduction](1-Introduction/README.md)       | 學生應考量建立與應用機器學習模型時，公平性的哲學議題                                                         |                                              [Lesson](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|      04       |                機器學習技術                 |      [Introduction](1-Introduction/README.md)       | 機器學習研究者使用哪種技術來建立模型？                                                                       |                                          [Lesson](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris and Jen                     |
|      05       |                   回歸簡介                   |        [Regression](2-Regression/README.md)         | 開始使用 Python 和 Scikit-learn 建立回歸模型                                                                   |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|      06       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 視覺化與清理資料以準備機器學習                                                                                |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|      07       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建立線性與多項式回歸模型                                                                                      |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen and Dmitry • Eric Wanjau       |
|      08       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建立邏輯回歸模型                                                                                              |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|      09       |                          Web App 🔌                          |           [Web App](3-Web-App/README.md)            | 建立一個網頁應用程式以使用你訓練的模型                                                                        |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|      10       |                 分類簡介                 |    [Classification](4-Classification/README.md)     | 清理、準備及視覺化資料；分類入門                                                                              | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen and Cassie • Eric Wanjau |
|      11       |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 介紹分類器                                                                                                   | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen and Cassie • Eric Wanjau |
|      12       |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                   | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen and Cassie • Eric Wanjau |
|      13       |             美味的亞洲和印度料理 🍜             |    [Classification](4-Classification/README.md)     | 使用你的模型建立推薦網頁應用程式                                                                              |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|      14       |                   叢集簡介                   |        [Clustering](5-Clustering/README.md)         | 清理、準備及視覺化資料；叢集入門                                                                              |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|      15       |              探索奈及利亞音樂品味 🎧              |        [Clustering](5-Clustering/README.md)         | 探索 K-均值叢集法                                                                                             |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|      16       |        自然語言處理簡介 ☕️         |   [Natural language processing](6-NLP/README.md)    | 透過建立簡單機器人學習 NLP 基礎                                                                              |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|      17       |                      常見的 NLP 工作 ☕️                      |   [Natural language processing](6-NLP/README.md)    | 透過理解處理語言結構時所需的常見任務來深化 NLP 知識                                                          |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|      18       |             翻譯與情感分析 ♥️              |   [Natural language processing](6-NLP/README.md)    | 使用 Jane Austen 文本進行翻譯與情感分析                                                                       |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|      19       |                  歐洲浪漫飯店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 使用飯店評論進行情感分析 1                                                                                     |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|      20       |                  歐洲浪漫飯店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 使用飯店評論進行情感分析 2                                                                                     |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|      21       |            時間序列預測簡介             |        [Time series](7-TimeSeries/README.md)        | 時間序列預測入門                                                                                              |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|      22       | ⚡️ 世界電力消耗 ⚡️ - 使用 ARIMA 進行時間序列預測 |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 進行時間序列預測                                                                                   |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|      23       |  ⚡️ 世界電力消耗 ⚡️ - 使用 SVR 進行時間序列預測  |        [Time series](7-TimeSeries/README.md)        | 使用支持向量回歸 (SVR) 進行時間序列預測                                                                       |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|      24       |             強化學習簡介             | [Reinforcement learning](8-Reinforcement/README.md) | Q-Learning 強化學習入門                                                                                         |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|      25       |                 幫彼得避開狼！🐺                  | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                   |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  後記   |            現實世界中的機器學習情境與應用            |      [ML in the Wild](9-Real-World/README.md)       | 古典機器學習有趣且啟發性的真實世界應用                                                                        |                                             [Lesson](9-Real-World/1-Applications/README.md)                                              |                         團隊                         |
|  後記   |            使用 RAI 儀表板進行機器學習模型除錯          |      [ML in the Wild](9-Real-World/README.md)       | 使用 Responsible AI 儀表板元件來進行機器學習模型除錯                                                          |                                             [Lesson](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [在我們的 Microsoft Learn 集合中找到此課程的所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線存取

你可以使用 [Docsify](https://docsify.js.org/#/) 離線瀏覽此說明文件。將本 repo 分叉，於本地機器安裝 [Docsify](https://docsify.js.org/#/quickstart)，接著在本 repo 根目錄輸入 `docsify serve`。網站會在本地主機的 3000 埠上提供服務：`localhost:3000`。

## PDF

可在此處找到課程 PDF 並含有連結 [here](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)。

## 🎒 其他課程

我們團隊還有其他課程！快來看看：

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)

---

### Azure / Edge / MCP / Agents
[![AZD for Beginners](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI for Beginners](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![MCP for Beginners](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![AI Agents for Beginners](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### 生成式 AI 系列
[![Generative AI for Beginners](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generative AI (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generative AI (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generative AI (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---
 
### 核心學習
[![ML for Beginners](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Data Science for Beginners](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![AI for Beginners](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Cybersecurity for Beginners](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Web Dev for Beginners](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT for Beginners](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR Development for Beginners](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Copilot 系列
[![Copilot for AI Paired Programming](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot for C#/.NET](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot Adventure](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## 獲得協助

如果你遇上困難或對構建 AI 應用程式有任何疑問，歡迎加入 MCP 中的學習者和經驗豐富的開發者討論。這是個支持性的社區，大家樂於提問，並自由分享知識。

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

如果你在構建中有產品反饋或遇到錯誤，請訪問：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件乃使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件以其原文語言版本為準參考。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->