<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T09:37:29+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "hk"
}
-->
# 自然語言處理簡介

這節課涵蓋了*自然語言處理*（NLP）這一*計算語言學*的子領域的簡短歷史和重要概念。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 簡介

NLP（自然語言處理）是機器學習應用最廣泛的領域之一，並已被用於生產軟件中。

✅ 你能想到每天使用的軟件中可能嵌入了某些 NLP 技術嗎？例如你經常使用的文字處理程序或手機應用程式？

你將學習以下內容：

- **語言的概念**：語言是如何發展的，以及主要的研究領域。
- **定義和概念**：你還將學習有關計算機如何處理文本的定義和概念，包括解析、語法以及識別名詞和動詞。本課程中有一些編程任務，並引入了幾個重要概念，這些概念你將在後續課程中學習如何編程。

## 計算語言學

計算語言學是一個研究和開發領域，已有數十年的歷史，研究計算機如何與語言合作，甚至理解、翻譯和與語言交流。自然語言處理（NLP）是一個相關領域，專注於計算機如何處理“自然”或人類語言。

### 示例 - 手機語音輸入

如果你曾經用語音輸入代替打字，或者向虛擬助手提問，那麼你的語音已被轉換為文本形式，然後被處理或*解析*成你所說的語言。檢測到的關鍵詞隨後被處理成手機或助手可以理解並執行的格式。

![理解](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> 真正的語言理解非常困難！圖片來源：[Jen Looper](https://twitter.com/jenlooper)

### 這項技術是如何實現的？

這是因為有人編寫了一個計算機程序來完成這項工作。幾十年前，一些科幻作家預測人們將主要通過語音與計算機交流，而計算機將始終準確理解人們的意思。不幸的是，這個問題比許多人想像的要困難得多。雖然今天對這個問題的理解已經大大提高，但在實現“完美”的自然語言處理以理解句子的含義方面仍然存在重大挑戰。尤其是在理解幽默或檢測句子中的情感（如諷刺）時，這是一個特別困難的問題。

此時，你可能會回憶起學校課堂上老師講解句子語法部分的情景。在某些國家，學生會專門學習語法和語言學，但在許多國家，這些主題是作為學習語言的一部分來教授的：要麼是在小學學習母語（學習閱讀和寫作），要麼是在中學或高中學習第二語言。如果你不擅長區分名詞和動詞或副詞和形容詞，不用擔心！

如果你對*一般現在時*和*現在進行時*的區別感到困惑，你並不孤單。這對許多人來說都是一個挑戰，即使是母語使用者。好消息是，計算機非常擅長應用正式規則，你將學習編寫代碼來像人類一樣*解析*句子。更大的挑戰是理解句子的*含義*和*情感*，這是你稍後將探討的內容。

## 先決條件

本課程的主要先決條件是能夠閱讀和理解本課程的語言。沒有數學問題或方程需要解決。雖然原作者用英語編寫了本課程，但它也被翻譯成其他語言，因此你可能正在閱讀翻譯版本。有些示例使用了多種不同語言（用於比較不同語言的語法規則）。這些示例*未被翻譯*，但解釋性文本已被翻譯，因此含義應該是清晰的。

在編程任務中，你將使用 Python，示例使用的是 Python 3.8。

在本節中，你需要並使用以下工具：

- **Python 3 理解能力**：理解 Python 3 編程語言，本課程使用輸入、循環、文件讀取和數組。
- **Visual Studio Code + 擴展**：我們將使用 Visual Studio Code 及其 Python 擴展。你也可以使用自己選擇的 Python IDE。
- **TextBlob**：[TextBlob](https://github.com/sloria/TextBlob) 是一個簡化的 Python 文本處理庫。按照 TextBlob 網站上的說明將其安裝到你的系統上（同時安裝語料庫，如下所示）：

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 提示：你可以直接在 VS Code 環境中運行 Python。查看[文檔](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott)了解更多信息。

## 與機器交流

讓計算機理解人類語言的歷史可以追溯到幾十年前，最早考慮自然語言處理的科學家之一是*艾倫·圖靈*。

### “圖靈測試”

當圖靈在1950年代研究*人工智能*時，他考慮是否可以通過一個對話測試來分辨人類和計算機（通過打字通信），其中參與對話的人類無法確定自己是在與另一個人類還是計算機交流。

如果在一定時間的對話後，人類無法判斷回答是來自計算機還是人類，那麼是否可以說計算機在*思考*？

### 靈感來源 - “模仿遊戲”

這個想法來自一個叫做*模仿遊戲*的派對遊戲，遊戲中一名審問者獨自待在一個房間裡，試圖判斷另一個房間裡的兩個人分別是男性還是女性。審問者可以發送便條，並必須設法提出一些問題，通過書面回答來揭示神秘人物的性別。當然，另一個房間裡的玩家會試圖通過回答問題來誤導或混淆審問者，同時也要給出看似誠實的回答。

### 開發 Eliza

在1960年代，一位麻省理工學院的科學家*約瑟夫·魏岑鮑姆*開發了[*Eliza*](https://wikipedia.org/wiki/ELIZA)，一個計算機“治療師”，它會向人類提問並給出看似理解的回答。然而，雖然 Eliza 可以解析句子並識別某些語法結構和關鍵詞以給出合理的回答，但它並不能說是*理解*句子。如果 Eliza 被呈現一個格式為“**我很** <u>難過</u>”的句子，它可能會重新排列並替換句子中的詞語，形成回應“你**有多長時間** <u>難過</u>”。

這給人一種 Eliza 理解了陳述並提出了後續問題的印象，而實際上它只是改變了時態並添加了一些詞語。如果 Eliza 無法識別它有回應的關鍵詞，它會給出一個隨機回應，這可能適用於許多不同的陳述。例如，如果用戶寫道“**你是**一個<u>自行車</u>”，它可能會回應“我**有多長時間**是一個<u>自行車</u>？”而不是更合理的回應。

[![與 Eliza 聊天](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "與 Eliza 聊天")

> 🎥 點擊上方圖片觀看原始 ELIZA 程式的視頻

> 注意：如果你有 ACM 帳戶，可以閱讀1966年發表的[Eliza 原始描述](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract)。或者，請在[wikipedia](https://wikipedia.org/wiki/ELIZA)上了解 Eliza。

## 練習 - 編寫一個基本的對話機器人

一個對話機器人，如 Eliza，是一個能夠引導用戶輸入並看似理解和智能回應的程序。與 Eliza 不同，我們的機器人不會有多條規則使其看似能進行智能對話。相反，我們的機器人只有一個功能，即通過隨機回應來保持對話，這些回應可能適用於幾乎任何簡單的對話。

### 計劃

構建對話機器人的步驟：

1. 打印指示，告知用戶如何與機器人互動
2. 開始循環
   1. 接受用戶輸入
   2. 如果用戶要求退出，則退出
   3. 處理用戶輸入並確定回應（在本例中，回應是從可能的通用回應列表中隨機選擇）
   4. 打印回應
3. 返回步驟2

### 構建機器人

接下來讓我們創建機器人。我們將從定義一些短語開始。

1. 使用以下隨機回應在 Python 中自己創建這個機器人：

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    以下是一些示例輸出（用戶輸入以`>`開頭）：

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

    1. 你認為隨機回應能否“欺騙”某人，使其認為機器人真的理解了他們？
    2. 機器人需要哪些功能才能更有效？
    3. 如果機器人真的能“理解”句子的含義，它是否需要“記住”對話中前幾句的含義？

---

## 🚀挑戰

選擇上述“停下來思考”中的一個元素，嘗試用代碼實現它，或者用紙筆寫出解決方案的偽代碼。

在下一節課中，你將學習其他解析自然語言和機器學習的方法。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

查看以下參考資料作為進一步學習的機會。

### 參考資料

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## 作業 

[尋找一個機器人](assignment.md)

---

**免責聲明**：  
本文件已使用人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始語言的文件應被視為權威來源。對於重要資訊，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋概不負責。