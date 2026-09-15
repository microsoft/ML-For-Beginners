[![GitHub 授權](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub 貢獻者](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub 問題追蹤](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub 拉取請求](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![歡迎拉取請求](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub 觀察者](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub 分支](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub 星標](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 多語言支援

#### 透過 GitHub Action 支援（自動且持續更新）

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[阿拉伯語](../ar/README.md) | [孟加拉語](../bn/README.md) | [保加利亞語](../bg/README.md) | [緬甸語 (Myanmar)](../my/README.md) | [中文（簡體）](../zh-CN/README.md) | [中文（繁體，香港）](./README.md) | [中文（繁體，澳門）](../zh-MO/README.md) | [中文（繁體，台灣）](../zh-TW/README.md) | [克羅地亞語](../hr/README.md) | [捷克語](../cs/README.md) | [丹麥語](../da/README.md) | [荷蘭語](../nl/README.md) | [愛沙尼亞語](../et/README.md) | [芬蘭語](../fi/README.md) | [法語](../fr/README.md) | [德語](../de/README.md) | [希臘語](../el/README.md) | [希伯來語](../he/README.md) | [印地語](../hi/README.md) | [匈牙利語](../hu/README.md) | [印尼語](../id/README.md) | [意大利語](../it/README.md) | [日語](../ja/README.md) | [卡納達語](../kn/README.md) | [高棉語](../km/README.md) | [韓語](../ko/README.md) | [立陶宛語](../lt/README.md) | [馬來語](../ms/README.md) | [馬拉雅拉姆語](../ml/README.md) | [馬拉地語](../mr/README.md) | [尼泊爾語](../ne/README.md) | [奈及利亞皮欽語](../pcm/README.md) | [挪威語](../no/README.md) | [波斯語（法爾西）](../fa/README.md) | [波蘭語](../pl/README.md) | [葡萄牙語（巴西）](../pt-BR/README.md) | [葡萄牙語（葡萄牙）](../pt-PT/README.md) | [旁遮普語 (Gurmukhi)](../pa/README.md) | [羅馬尼亞語](../ro/README.md) | [俄語](../ru/README.md) | [塞爾維亞語（西里爾字母）](../sr/README.md) | [斯洛伐克語](../sk/README.md) | [斯洛文尼亞語](../sl/README.md) | [西班牙語](../es/README.md) | [斯瓦希里語](../sw/README.md) | [瑞典語](../sv/README.md) | [他加祿語（菲律賓語）](../tl/README.md) | [泰米爾語](../ta/README.md) | [泰盧固語](../te/README.md) | [泰語](../th/README.md) | [土耳其語](../tr/README.md) | [烏克蘭語](../uk/README.md) | [烏爾都語](../ur/README.md) | [越南語](../vi/README.md)

> **想要在本機克隆？**
>
> 此儲存庫包含超過 50 種語言的翻譯，會顯著增加下載大小。如需不包含翻譯的克隆，請使用稀疏簽出：
>
> **Bash / macOS / Linux：**
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
>
> **CMD（Windows）：**
> ```cmd
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone "/*" "!translations" "!translated_images"
> ```
>
> 這樣可以用更快的下載速度取得完成課程所需的所有內容。
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### 加入我們的社群

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

我們正在進行 Discord 的「與 AI 一同學習」系列，詳情與加入請見 [Learn with AI Series](https://aka.ms/learnwithai/discord)（2025 年 9 月 18 日至 30 日）。你將學習到使用 GitHub Copilot 於數據科學的技巧和秘訣。

![與 AI 一同學習系列](../../translated_images/zh-HK/3.9b58fd8d6c373c20.webp)

# 初學者機器學習課程大綱

> 🌍 透過世界各地文化的視角，環遊世界探索機器學習 🌍

微軟的雲端倡導者很高興提供一個為期 12 週、共 26 課的課程，專注於<strong>機器學習</strong>。本課程教你有時稱為<strong>經典機器學習</strong>的內容，主要使用 Scikit-learn 函式庫，避免深度學習，本課題已於我們的[初學者 AI 課程](https://aka.ms/ai4beginners)涵蓋。建議同時參考我們的[初學者數據科學課程](https://aka.ms/ds4beginners)。

與我們一同環遊世界，應用這些經典技術分析來自世界各地的數據。每節課包含課前與課後小測驗、書面指引、解決方案、作業等。我們以專案為基礎的教學法讓你一邊建構一邊學習，是學習新技能的良好方法。

**✍️ 衷心感謝我們的作者** Jen Looper、Stephen Howell、Francesca Lazzeri、Tomomi Imura、Cassie Breviu、Dmitry Soshnikov、Chris Noring、Anirban Mukherjee、Ornella Altunyan、Ruth Yakubu 與 Amy Boyd

**🎨 也感謝我們的插畫師** Tomomi Imura、Dasani Madipalli 與 Jen Looper

**🙏 特別感謝我們的 Microsoft 學生大使作者、審稿人與內容貢獻者**，包括 Rishit Dagli、Muhammad Sakib Khan Inan、Rohan Raj、Alexandru Petrescu、Abhishek Jaiswal、Nawrin Tabassum、Ioan Samuila 與 Snigdha Agarwal

**🤩 額外感謝 Microsoft 學生大使 Eric Wanjau、Jasleen Sondhi 與 Vidushi Gupta 提供我們的 R 課程！**

# 入門指南

請遵循以下步驟：
1. <strong>分叉本儲存庫</strong>：點擊本頁右上角的「Fork」按鈕。
2. <strong>克隆儲存庫</strong>： `git clone https://github.com/microsoft/ML-For-Beginners.git`

> 💡 **快速入門提示：** 想不用在本機安裝 Python 在瀏覽器中快速開始？使用 [GitHub Codespaces](https://github.com/features/codespaces) 為你的分叉建立雲端開發環境。點擊綠色 **Code** 選單，選擇 **Codespaces**，然後建立 codespace；依需要在其中安裝每節課所需的依賴。

> [在我們的 Microsoft Learn 集合中找到本課程所有額外資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **需要幫助？** 請參考我們的 [故障排除指南](TROUBLESHOOTING.md)，協助解決安裝、設定和執行課程的常見問題。


**[學員](https://aka.ms/student-page)** 如要使用本課程，請將整個儲存庫分叉到你的 GitHub 帳號，並自行或組隊完成練習：

- 從課前小測驗開始。
- 閱讀課程內容並完成活動，於每次知識檢測時暫停並反思。
- 嘗試理解課程內容自行創建專案，而非直接使用解決方案代碼；不過解決方案代碼可在每個專案課程的 `/solution` 資料夾中找到。
- 完成課後小測驗。
- 完成挑戰題。
- 完成作業。
- 完成一組課程後，請參加 [討論區](https://github.com/microsoft/ML-For-Beginners/discussions)，並透過填寫適用的 PAT 評量表「大聲學習」。PAT 為進度評估工具，是你填寫以推進學習的評量表，也可對其他人的 PAT 給予回饋，一同學習。

> 如需進一步學習，建議參考這些 [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) 模組與學習路徑。

<strong>教師們</strong>，我們提供了[使用本課程的建議](for-teachers.md)。

---

## 影片導覽

部分課程有提供短片形式的說明。你可以在課程內嵌中觀看這些影片，也可於 [Microsoft Developer YouTube 頻道的 ML for Beginners 播放清單](https://aka.ms/ml-beginners-videos)中點擊下方圖片觀看。

[![ML for beginners 橫幅](../../translated_images/zh-HK/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## 團隊介紹

[![宣傳影片](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

<strong>動圖由</strong> [Mohit Jaisal](https://linkedin.com/in/mohitjaisal) 製作

> 🎥 點擊上方圖片觀看關於本專案及創作者的影片！

---

## 教學法

我們在設計本課程時選擇了兩大教學準則：確保內容是動手做的<strong>專案導向</strong>，且包含<strong>頻繁的測驗</strong>。此外，本課程具有一致的<strong>主題</strong>以增強連貫性。

透過確保內容與專案對齊，提升學生的學習投入並加強概念記憶。課前低壓力測驗設置學習意向，課後測驗確保知識留存。本課程設計靈活且有趣，可整體或分段學習。專案由淺入深，於 12 週周期結尾達到高度複雜度。本課程亦包含對機器學習實際應用的補充說明，可作為額外學分或討論基礎。

> 查看我們的[行為準則](CODE_OF_CONDUCT.md)、[貢獻指南](CONTRIBUTING.md)、[翻譯指南](..)與[故障排除](TROUBLESHOOTING.md)政策。我們歡迎您的建設性回饋！

## 每節課包含

- 選擇性草圖筆記
- 選擇性補充影片
- 影片導覽（部分課程）
- [課前暖身測驗](https://ff-quizzes.netlify.app/en/ml/)
- 書面課程內容
- 專案導向課程附逐步操作指南
- 知識檢測
- 挑戰題
- 補充閱讀資料
- 作業
- [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

> <strong>關於語言說明</strong>：這些課程主要使用 Python 撰寫，但許多也提供 R 語言版本。要完成 R 課程，請前往 `/solution` 資料夾尋找帶有 .rmd 副檔名的 R 課程檔案。R Markdown 檔案是一種將 R（或其他語言）`程式碼區塊` 和 `YAML 標頭`（用於指引格式輸出如 PDF）嵌入於 `Markdown 文件` 的範例撰寫框架，可同時編寫程式碼、輸出與說明文字，是資料科學的理想工具。R Markdown 還可輸出為 PDF、HTML 或 Word 等格式。

> <strong>關於測驗提醒</strong>：所有測驗皆位於 [Quiz App 資料夾](../../quiz-app)，共 52 個測驗，每個測驗有三個問題。測驗從課程中連結，但也可在本機執行；請按照 `quiz-app` 資料夾中的指示，在本機或 Azure 部署。

| 課程編號 |                             主題                              |                   課程群組                   | 學習目標                                                                                                                     |                                                              關聯課程                                                               |                        作者                        |
| :-----------: | :------------------------------------------------------------: | :-------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------: |
|      01       |                機器學習導論                |      [Introduction](1-Introduction/README.md)       | 學習機器學習背後的基本概念                                                                                |                                             [Lesson](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                       |
|      02       |                機器學習的歷史                 |      [Introduction](1-Introduction/README.md)       | 瞭解這個領域背後的歷史                                                                                         |                                            [Lesson](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen and Amy                      |
|      03       |                 公平性與機器學習                  |      [Introduction](1-Introduction/README.md)       | 建立和應用機器學習模型時，學生應該考慮的公平性相關重要哲學議題是什麼？ |                                              [Lesson](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                        |
|      04       |                機器學習的技術                 |      [Introduction](1-Introduction/README.md)       | 機器學習研究人員用什麼技術來建立機器學習模型？                                                                       |                                          [Lesson](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris and Jen                     |
|      05       |                   回歸模型導論                   |        [Regression](2-Regression/README.md)         | 開始使用 Python 和 Scikit-learn 來建立回歸模型                                                                  |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|      06       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 視覺化及清理資料以準備機器學習                                                                                  |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|      07       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建立線性及多項式回歸模型                                                                                   |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen and Dmitry • Eric Wanjau       |
|      08       |                北美南瓜價格 🎃                |        [Regression](2-Regression/README.md)         | 建立邏輯回歸模型                                                                                               |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|      09       |                          一個網頁應用 🔌                          |           [Web App](3-Web-App/README.md)            | 建立網頁應用來使用你訓練的模型                                                                                       |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|      10       |                 分類導論                 |    [Classification](4-Classification/README.md)     | 清理、預處理與視覺化資料；分類導論                                                            | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen and Cassie • Eric Wanjau |
|      11       |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 分類器導論                                                                                                     | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen and Cassie • Eric Wanjau |
|      12       |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 更多分類器                                                                                                                | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen and Cassie • Eric Wanjau |
|      13       |             美味的亞洲與印度料理 🍜             |    [Classification](4-Classification/README.md)     | 使用你的模型建立推薦網頁應用                                                                                    |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|      14       |                   叢集導論                   |        [Clustering](5-Clustering/README.md)         | 清理、預處理與視覺化資料；叢集導論                                                                |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|      15       |              探索奈及利亞音樂喜好 🎧              |        [Clustering](5-Clustering/README.md)         | 探索 K-均值叢集方法                                                                                           |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|      16       |        自然語言處理導論 ☕️         |   [Natural language processing](6-NLP/README.md)    | 透過建立簡單的 Bot 學習 NLP 基礎                                                                             |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|      17       |                      常見的 NLP 任務 ☕️                      |   [Natural language processing](6-NLP/README.md)    | 深入理解處理語言結構時所需的常見任務                          |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|      18       |             翻譯與情感分析 ♥️              |   [Natural language processing](6-NLP/README.md)    | 使用 Jane Austen 進行翻譯及情感分析                                                                             |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|      19       |                  歐洲浪漫飯店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 使用飯店評論進行情感分析 1                                                                                         |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|      20       |                  歐洲浪漫飯店 ♥️                  |   [Natural language processing](6-NLP/README.md)    | 使用飯店評論進行情感分析 2                                                                                         |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|      21       |            時間序列預測導論             |        [Time series](7-TimeSeries/README.md)        | 時間序列預測導論                                                                                         |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|      22       | ⚡️ 世界電力使用量 ⚡️ - 使用 ARIMA 的時間序列預測 |        [Time series](7-TimeSeries/README.md)        | 使用 ARIMA 進行時間序列預測                                                                                              |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|      23       |  ⚡️ 世界電力使用量 ⚡️ - 使用 SVR 的時間序列預測  |        [Time series](7-TimeSeries/README.md)        | 使用支持向量回歸進行時間序列預測                                                                           |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|      24       |             強化學習導論             | [Reinforcement learning](8-Reinforcement/README.md) | 透過 Q-Learning 瞭解強化學習                                                                          |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|      25       |                 幫助 Peter 避免狼咬! 🐺                  | [Reinforcement learning](8-Reinforcement/README.md) | 強化學習 Gym                                                                                                      |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  附錄   |            現實世界的機器學習場景與應用            |      [ML in the Wild](9-Real-World/README.md)       | 傳統機器學習的有趣且具啟發性的現實世界應用                                                               |                                             [Lesson](9-Real-World/1-Applications/README.md)                                              |                         Team                         |
|  附錄   |            使用 RAI 儀表板進行機器學習模型除錯            |      [ML in the Wild](9-Real-World/README.md)       | 使用負責任 AI 儀表板組件進行機器學習模型除錯                                                              |                                             [Lesson](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                     Ruth Yakubu                      |

> [在我們的 Microsoft Learn 集合中找到此課程的所有附加資源](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## 離線存取

你可以使用 [Docsify](https://docsify.js.org/#/) 離線執行本文件。複製此倉庫，於本機安裝 [Docsify](https://docsify.js.org/#/quickstart)，然後在此倉庫根目錄輸入 `docsify serve`。網站將於本地主機的 3000 埠口提供服務：`localhost:3000`。

## PDF

在[此處](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf)找到課程大綱的 PDF，附帶連結。


## 🎒 其他課程

我們團隊製作了其他課程！請查看：

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)
[![LangChain for Beginners](https://img.shields.io/badge/LangChain%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://github.com/microsoft/langchain-for-beginners?WT.mc_id=m365-94501-dwahlin)
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

## 尋求協助

如果你在學習機器學習或構建 AI 應用時遇到困難或有疑問，別擔心 — 我們有支援。

你可以加入與其他學習者及開發者的討論，提問並分享你的想法。

- 加入社群提問並與他人一同學習
- 討論機器學習概念與專案想法
- 從經驗豐富的開發者獲得指導

支援性的社群是成長技能與快速解決問題的好方法。

[Microsoft Foundry Discord 社群](https://discord.gg/nTYy5BXMWG)

如果你發現錯誤、問題或有改進建議，也可以在此倉庫開設<strong>問題</strong>來回報。

欲提供產品回饋或搜尋現有社群貼文，請造訪開發者論壇：

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

## 額外學習建議

- 每課後複習筆記本以加深理解。
- 自行練習實作演算法。
- 利用所學概念探索現實數據集。

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->