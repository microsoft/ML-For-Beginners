<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-08-29T22:31:31+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "mo"
}
-->
# 自然語言處理簡介

本課程涵蓋了*自然語言處理*（NLP）這一*計算語言學*子領域的簡史及其重要概念。

## [課前測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## 簡介

NLP 是機器學習應用最廣泛的領域之一，並已被用於生產軟體中。

✅ 你能想到每天使用的哪些軟體可能內嵌了 NLP 嗎？例如你經常使用的文字處理程式或手機應用程式？

你將學習到：

- **語言的概念**：語言是如何發展的，以及主要的研究領域。
- **定義與概念**：你還將學習有關電腦如何處理文本的定義與概念，包括解析、語法以及辨識名詞和動詞。本課程中有一些編碼任務，並介紹了幾個重要概念，這些概念你將在後續課程中學習如何編碼。

## 計算語言學

計算語言學是一個研究和開發領域，已有數十年的歷史，研究如何讓電腦處理、理解、翻譯甚至與語言進行交流。自然語言處理（NLP）是一個相關領域，專注於電腦如何處理“自然”或人類語言。

### 範例 - 手機語音輸入

如果你曾經用語音輸入代替打字，或者向虛擬助理提問，那麼你的語音已被轉換為文本形式，然後被處理或*解析*成你所說的語言。檢測到的關鍵詞隨後被轉換為手機或助理可以理解並執行的格式。

![理解](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.mo.png)  
> 真正的語言理解非常困難！圖片來源：[Jen Looper](https://twitter.com/jenlooper)

### 這項技術是如何實現的？

這是因為有人編寫了一個電腦程式來實現這一功能。幾十年前，一些科幻作家預測人們將主要通過語音與電腦交流，而電腦將始終準確理解人們的意思。不幸的是，這個問題比許多人想像的要困難得多。儘管今天對這個問題的理解已經大大提高，但在實現“完美”的自然語言處理方面仍然面臨重大挑戰，特別是在理解句子的含義時。當涉及到理解幽默或檢測句子中的諷刺情緒時，這是一個特別困難的問題。

此時，你可能會回想起學校課堂上老師講解句子語法部分的情景。在一些國家，學生會專門學習語法和語言學，而在其他國家，這些主題則作為學習語言的一部分：在小學學習母語（學習閱讀和寫作），在中學或高中可能學習第二語言。如果你不擅長區分名詞和動詞，或者副詞和形容詞，不用擔心！

如果你對區分*一般現在式*和*現在進行式*感到困難，你並不孤單。這對許多人來說都是一個挑戰，即使是母語使用者。好消息是，電腦非常擅長應用形式化規則，你將學習編寫程式來像人類一樣*解析*句子。更大的挑戰是理解句子的*含義*和*情感*，這是你稍後將探討的內容。

## 先修知識

本課程的主要先修條件是能夠閱讀和理解本課程的語言。沒有數學問題或方程需要解決。雖然原作者用英文撰寫了本課程，但它也被翻譯成其他語言，因此你可能正在閱讀翻譯版本。有些例子使用了多種語言（用於比較不同語言的語法規則）。這些例子*不會*被翻譯，但解釋性文字會被翻譯，因此意義應該是清楚的。

對於編碼任務，你將使用 Python，範例使用的是 Python 3.8。

在本節中，你將需要並使用：

- **Python 3 理解能力**：理解 Python 3 程式語言，本課程使用輸入、迴圈、文件讀取、陣列。
- **Visual Studio Code + 擴展**：我們將使用 Visual Studio Code 及其 Python 擴展。你也可以使用你選擇的 Python IDE。
- **TextBlob**：[TextBlob](https://github.com/sloria/TextBlob) 是一個簡化的 Python 文本處理庫。按照 TextBlob 網站上的指示將其安裝到你的系統中（同時安裝語料庫，如下所示）：

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：你可以直接在 VS Code 環境中運行 Python。查看 [文檔](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) 了解更多資訊。

## 與機器對話

讓電腦理解人類語言的歷史可以追溯到幾十年前，最早考慮自然語言處理的科學家之一是*艾倫·圖靈*。

### “圖靈測試”

當圖靈在 1950 年代研究*人工智慧*時，他考慮了一種對話測試，讓人類和電腦進行對話（通過打字通信），如果對話中的人類無法確定自己是在與另一個人還是電腦對話，那麼是否可以說電腦在“思考”？

### 靈感來源 - “模仿遊戲”

這個想法來自一種派對遊戲，稱為*模仿遊戲*，遊戲中一名提問者獨自待在一個房間裡，試圖判斷另一個房間中的兩個人分別是男性還是女性。提問者可以發送紙條，並試圖設計問題，通過書面回答來揭示神秘人的性別。當然，另一個房間中的玩家試圖通過回答問題來誤導或混淆提問者，同時也要表現得像是在誠實回答。

### 開發 Eliza

1960 年代，麻省理工學院的科學家*約瑟夫·魏岑鮑姆*開發了 [*Eliza*](https://wikipedia.org/wiki/ELIZA)，一個電腦“治療師”，它會向人類提問，並給人一種理解其回答的假象。然而，儘管 Eliza 能夠解析句子並識別某些語法結構和關鍵詞以給出合理的回答，但它並不能說是*理解*句子。如果 Eliza 被輸入一個格式為“**我很**<u>難過</u>”的句子，它可能會重新排列並替換句子中的單詞，形成回應“你**已經**<u>難過</u>多久了”。

這給人一種 Eliza 理解了陳述並提出了後續問題的印象，而實際上，它只是改變了時態並添加了一些單詞。如果 Eliza 無法識別出它有回應的關鍵詞，它會給出一個隨機回應，這可能適用於許多不同的陳述。例如，如果用戶輸入“**你是**一個<u>自行車</u>”，它可能會回應“我**已經**是一個<u>自行車</u>多久了？”，而不是給出更合理的回應。

[![與 Eliza 對話](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "與 Eliza 對話")

> 🎥 點擊上方圖片觀看關於原始 ELIZA 程式的影片

> 注意：如果你有 ACM 帳戶，可以閱讀 1966 年發表的 [Eliza 原始描述](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)。或者，參考 [wikipedia](https://wikipedia.org/wiki/ELIZA) 上的 Eliza 介紹。

## 練習 - 編寫一個基本的對話機器人

一個對話機器人，如 Eliza，是一個能引導用戶輸入並看似能理解和智能回應的程式。與 Eliza 不同，我們的機器人不會有多條規則來模仿智能對話。相反，我們的機器人只有一個功能，即通過隨機回應來保持對話進行，這些回應可能適用於幾乎任何瑣碎的對話。

### 計劃

構建對話機器人的步驟：

1. 打印指示，告知用戶如何與機器人互動
2. 開始一個迴圈
   1. 接收用戶輸入
   2. 如果用戶要求退出，則退出
   3. 處理用戶輸入並確定回應（在本例中，回應是從可能的通用回應列表中隨機選擇的）
   4. 打印回應
3. 返回步驟 2

### 構建機器人

接下來我們來創建機器人。首先定義一些短語。

1. 使用以下隨機回應在 Python 中自行創建此機器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    以下是一些範例輸出（用戶輸入以 `>` 開頭）：

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

    1. 你認為隨機回應能“欺騙”某人讓他們以為機器人真的理解了他們嗎？
    2. 機器人需要哪些功能才能更有效？
    3. 如果機器人真的能“理解”句子的含義，它是否需要“記住”對話中前幾句的含義？

---

## 🚀挑戰

選擇上述“停下來思考”中的一個元素，嘗試用程式碼實現它，或者用偽代碼在紙上寫出解決方案。

在下一課中，你將學習其他解析自然語言和機器學習的方法。

## [課後測驗](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## 回顧與自學

查看以下參考資料，作為進一步學習的機會。

### 參考資料

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 作業 

[尋找一個機器人](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵信息，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋不承擔責任。