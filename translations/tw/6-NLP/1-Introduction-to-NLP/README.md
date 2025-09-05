<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T10:02:07+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "tw"
}
-->
# 自然語言處理簡介

本課程涵蓋了*自然語言處理*（NLP）這一*計算語言學*子領域的簡史及重要概念。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 簡介

NLP（自然語言處理）是機器學習應用最廣泛的領域之一，並且已被用於生產軟體中。

✅ 你能想到每天使用的哪些軟體可能內嵌了 NLP 嗎？例如你經常使用的文字處理程式或手機應用程式？

你將學習以下內容：

- **語言的概念**：語言是如何發展的，以及主要的研究領域。
- **定義與概念**：你還將學習計算機如何處理文本的定義與概念，包括解析、語法以及名詞和動詞的識別。本課程中有一些編碼任務，並介紹了一些重要概念，這些概念將在後續課程中進一步學習如何編碼實現。

## 計算語言學

計算語言學是一個研究領域，經過數十年的發展，研究計算機如何與語言互動，甚至理解、翻譯和交流語言。自然語言處理（NLP）是一個相關領域，專注於計算機如何處理“自然”或人類語言。

### 範例 - 手機語音輸入

如果你曾經使用手機語音輸入代替打字，或者向虛擬助理提問，那麼你的語音已被轉換為文本形式，然後被處理或*解析*成你所說的語言。檢測到的關鍵詞隨後被處理成手機或助理可以理解並執行的格式。

![理解](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)  
> 真正的語言理解非常困難！圖片來源：[Jen Looper](https://twitter.com/jenlooper)

### 這項技術是如何實現的？

這是因為有人編寫了一個計算機程式來實現這一功能。幾十年前，一些科幻作家預測人們將主要通過語音與計算機交流，而計算機將始終準確理解人們的意思。不幸的是，這個問題比許多人想像的要困難得多。儘管今天對這個問題的理解已經大大加深，但在實現“完美”的自然語言處理方面仍然面臨重大挑戰，尤其是在理解句子含義時。當涉及到理解幽默或檢測句子中的諷刺情緒時，這是一個特別困難的問題。

此時，你可能會回想起學校課堂上老師講解句子語法部分的情景。在某些國家，學生會專門學習語法和語言學，而在許多國家，這些主題則作為語言學習的一部分：例如在小學學習母語（學習閱讀和寫作），以及在中學學習第二語言。如果你無法熟練區分名詞、動詞或副詞、形容詞，不用擔心！

如果你對於區分*一般現在時*和*現在進行時*感到困難，你並不孤單。這對許多人來說都是一個挑戰，即使是母語使用者。好消息是，計算機非常擅長應用形式化規則，你將學習如何編寫程式來像人類一樣*解析*句子。更大的挑戰是理解句子的*含義*和*情感*，這將在後續課程中進一步探討。

## 先修知識

本課程的主要先修條件是能夠閱讀並理解本課程的語言。課程中沒有數學問題或方程需要解決。雖然原作者以英文撰寫了本課程，但它也被翻譯成其他語言，因此你可能正在閱讀翻譯版本。課程中有一些例子使用了多種語言（用於比較不同語言的語法規則）。這些例子*未被翻譯*，但解釋性文本已被翻譯，因此應該能夠理解其含義。

在編碼任務中，你將使用 Python，並且範例使用的是 Python 3.8。

在本節中，你將需要並使用以下工具：

- **Python 3 基礎**：理解 Python 3 程式語言，本課程使用輸入、迴圈、文件讀取和陣列。
- **Visual Studio Code + 擴展**：我們將使用 Visual Studio Code 及其 Python 擴展。你也可以選擇使用其他 Python IDE。
- **TextBlob**：[TextBlob](https://github.com/sloria/TextBlob) 是一個簡化的 Python 文本處理庫。按照 TextBlob 網站上的說明將其安裝到你的系統中（同時安裝語料庫，如下所示）：

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：你可以直接在 VS Code 環境中運行 Python。查看 [文檔](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) 以獲取更多資訊。

## 與機器對話

讓計算機理解人類語言的歷史可以追溯到數十年前，最早考慮自然語言處理的科學家之一是*艾倫·圖靈*。

### “圖靈測試”

當圖靈在 1950 年代研究*人工智慧*時，他考慮了一種對話測試，該測試讓人類和計算機進行對話（通過打字通信），如果對話中的人類無法確定自己是在與另一個人類還是計算機交流，那麼是否可以說計算機在“思考”？

### 靈感來源 - “模仿遊戲”

這個想法來自一種叫做*模仿遊戲*的派對遊戲。在遊戲中，一名提問者獨自待在一個房間裡，試圖判斷另一個房間中的兩個人分別是男性還是女性。提問者可以發送紙條，並試圖設計問題，通過書面回答來揭示神秘人的性別。當然，另一個房間中的玩家會試圖誤導提問者，通過回答問題來混淆提問者，同時表現得像是在誠實回答。

### 開發 Eliza

1960 年代，麻省理工學院的科學家*約瑟夫·魏森鮑姆*開發了[*Eliza*](https://wikipedia.org/wiki/ELIZA)，一個模擬心理治療師的計算機程式。Eliza 會向人類提問，並給人一種理解其回答的假象。然而，雖然 Eliza 能夠解析句子並識別某些語法結構和關鍵詞以生成合理的回答，但它並不能真正*理解*句子。如果 Eliza 收到一個格式為“**我很**<u>難過</u>”的句子，它可能會重新排列並替換句子中的詞語，形成回應“你**有多長時間**<u>難過</u>”。

這給人一種 Eliza 理解了陳述並提出了後續問題的印象，但實際上，它只是改變了時態並添加了一些詞語。如果 Eliza 無法識別關鍵詞以生成回應，它會給出一個隨機的回應，這可能適用於許多不同的陳述。例如，如果用戶輸入“**你是**一輛<u>自行車</u>”，它可能會回應“我**有多長時間**是一輛<u>自行車</u>？”，而不是給出更合理的回答。

[![與 Eliza 對話](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "與 Eliza 對話")

> 🎥 點擊上方圖片觀看原始 ELIZA 程式的相關影片

> 注意：如果你有 ACM 帳戶，可以閱讀 1966 年發表的 [Eliza 原始描述](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)。或者，參考 [維基百科](https://wikipedia.org/wiki/ELIZA) 了解 Eliza。

## 練習 - 編寫一個基礎對話機器人

一個對話機器人（如 Eliza）是一個能夠引導用戶輸入並看似理解並智能回應的程式。與 Eliza 不同，我們的機器人不會有多條規則來模仿智能對話。相反，我們的機器人只有一個功能，即通過隨機回應來保持對話進行，這些回應幾乎適用於任何簡單對話。

### 計劃

構建對話機器人的步驟：

1. 打印指導用戶如何與機器人互動的說明。
2. 啟動一個迴圈：
   1. 接受用戶輸入。
   2. 如果用戶要求退出，則退出。
   3. 處理用戶輸入並確定回應（在本例中，回應是從可能的通用回應列表中隨機選擇）。
   4. 打印回應。
3. 返回步驟 2。

### 構建機器人

接下來，我們將創建機器人。首先定義一些短語。

1. 使用以下隨機回應在 Python 中自行創建此機器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    以下是一些示例輸出（用戶輸入以 `>` 開頭）：

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    任務的一個可能解決方案在[這裡](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ 停下來思考

    1. 你認為這些隨機回應能“欺騙”某人以為機器人真的理解他們嗎？
    2. 機器人需要哪些功能才能更有效？
    3. 如果機器人真的能“理解”句子的含義，它是否需要“記住”對話中前幾句的含義？

---

## 🚀挑戰

選擇上述“停下來思考”中的一個元素，嘗試用程式碼實現它，或者用偽代碼在紙上寫出解決方案。

在下一課中，你將學習其他解析自然語言和機器學習的方法。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

參考以下資源進行進一步閱讀。

### 參考資料

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 作業 

[尋找一個機器人](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。