<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T10:02:42+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "tw"
}
-->
# 翻譯與情感分析使用機器學習

在之前的課程中，你學會了如何使用 `TextBlob` 建立一個基本的聊天機器人。`TextBlob` 是一個內嵌機器學習技術的函式庫，用於執行基本的自然語言處理（NLP）任務，例如名詞片語提取。計算語言學中的另一個重要挑戰是準確地將句子從一種語言翻譯成另一種語言。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

翻譯是一個非常困難的問題，因為世界上有數千種語言，每種語言的語法規則可能截然不同。一種方法是將一種語言（例如英語）的正式語法規則轉換為一種不依賴語言的結構，然後再將其轉換回另一種語言。這種方法的步驟如下：

1. **識別**：將輸入語言中的單詞標記為名詞、動詞等。
2. **創建翻譯**：按照目標語言的格式，直接翻譯每個單詞。

### 範例句子：從英語到愛爾蘭語

在「英語」中，句子 _I feel happy_ 包含三個單詞，順序為：

- **主詞**（I）
- **動詞**（feel）
- **形容詞**（happy）

然而，在「愛爾蘭語」中，這句話的語法結構非常不同——像「快樂」或「悲傷」這樣的情感是以「在你身上」的形式表達的。

英語短語 `I feel happy` 翻譯成愛爾蘭語是 `Tá athas orm`。*字面*翻譯是 `Happy is upon me`。

一位愛爾蘭語使用者翻譯成英語時會說 `I feel happy`，而不是 `Happy is upon me`，因為他們理解句子的含義，即使單詞和句子結構不同。

愛爾蘭語句子的正式順序是：

- **動詞**（Tá 或 is）
- **形容詞**（athas，或 happy）
- **主詞**（orm，或 upon me）

## 翻譯

一個簡單的翻譯程式可能只翻譯單詞，忽略句子結構。

✅ 如果你曾經在成年後學習第二（或第三、更多）語言，你可能會先用母語思考，然後在腦海中逐字翻譯成第二語言，最後說出翻譯結果。這與簡單的翻譯程式的工作方式類似。要達到流利程度，重要的是超越這個階段！

簡單的翻譯會導致糟糕（有時甚至是搞笑）的誤譯：`I feel happy` 字面翻譯成愛爾蘭語是 `Mise bhraitheann athas`。這字面意思是 `me feel happy`，但這不是一個有效的愛爾蘭語句子。即使英語和愛爾蘭語是兩個相鄰島嶼上的語言，它們的語法結構仍然非常不同。

> 你可以觀看一些關於愛爾蘭語言傳統的影片，例如 [這個](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### 機器學習方法

到目前為止，你已經學習了基於正式規則的自然語言處理方法。另一種方法是忽略單詞的含義，而是*使用機器學習來檢測模式*。如果你擁有大量的文本（*語料庫*）或雙語文本（*語料*），這種方法在翻譯中可能會奏效。

例如，考慮《傲慢與偏見》這本書，這是 Jane Austen 在 1813 年寫的一本著名英語小說。如果你參考這本書的英語版本和其*法語*的人類翻譯版本，你可以發現一些短語在兩種語言中是*習語化*的翻譯。你將在稍後進行這樣的練習。

例如，當英語短語 `I have no money` 被字面翻譯成法語時，可能會變成 `Je n'ai pas de monnaie`。「Monnaie」是一個棘手的法語「假同源詞」，因為「money」和「monnaie」並不是同義詞。一個更好的翻譯是 `Je n'ai pas d'argent`，因為它更好地傳達了「我沒有錢」的意思（而不是「零錢」，這是「monnaie」的意思）。

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> 圖片由 [Jen Looper](https://twitter.com/jenlooper) 提供

如果一個機器學習模型擁有足夠多的人類翻譯文本來建立模型，它可以通過識別先前由雙語專家翻譯的文本中的常見模式來提高翻譯的準確性。

### 練習 - 翻譯

你可以使用 `TextBlob` 翻譯句子。試試 **《傲慢與偏見》** 的著名開場白：

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` 的翻譯效果相當不錯：「C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!」。

事實上，可以說 `TextBlob` 的翻譯比 1932 年由 V. Leconte 和 Ch. Pressoir 翻譯的法語版本更為精確：

「C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles。」

在這種情況下，基於機器學習的翻譯比人類翻譯更好，因為後者為了「清晰」而不必要地加入了原作者未提及的內容。

> 這裡發生了什麼？為什麼 `TextBlob` 的翻譯如此出色？事實上，它背後使用的是 Google 翻譯，一種能夠解析數百萬短語並預測最佳翻譯的高級人工智慧。這裡沒有任何手動操作，並且使用 `blob.translate` 時需要網路連線。

✅ 試試更多句子。哪種翻譯更好，機器學習還是人類翻譯？在哪些情況下？

## 情感分析

機器學習在情感分析方面也表現得非常出色。一種非機器學習的方法是識別「正面」和「負面」的單詞和短語。然後，給定一段新文本，計算正面、負面和中性單詞的總值，以確定整體情感。

這種方法很容易被欺騙，正如你可能在 Marvin 任務中看到的那樣——句子 `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` 是一個帶有諷刺意味的負面情感句子，但簡單的演算法會將「great」、「wonderful」、「glad」識別為正面，而將「waste」、「lost」和「dark」識別為負面。這些矛盾的單詞會影響整體情感的判斷。

✅ 停下來想一想，作為人類說話者，我們是如何表達諷刺的。語調的變化起著重要作用。試著用不同的語氣說「Well, that film was awesome」，看看你的聲音如何傳達不同的含義。

### 機器學習方法

機器學習的方法是手動收集正面和負面的文本——例如推文、電影評論，或者任何包含評分*和*書面意見的內容。然後，將 NLP 技術應用於這些意見和評分，從而發現模式（例如，正面的電影評論中「Oscar worthy」這個短語出現的頻率比負面評論高，而正面的餐廳評論中「gourmet」出現的頻率比「disgusting」高）。

> ⚖️ **範例**：如果你在一位政治家的辦公室工作，並且有一項新法律正在辯論，選民可能會寫信到辦公室表達支持或反對這項新法律。假設你的任務是閱讀這些信件並將它們分為兩類：*支持*和*反對*。如果信件數量很多，你可能會因為無法全部閱讀而感到不知所措。如果有一個機器人能幫你閱讀所有信件，理解它們並告訴你每封信應該歸入哪一類，那該多好！
> 
> 一種實現方法是使用機器學習。你可以用部分*反對*信件和部分*支持*信件來訓練模型。模型會傾向於將某些短語和單詞與反對方或支持方聯繫起來，*但它不會理解任何內容*，只會識別某些單詞和模式更可能出現在反對或支持的信件中。你可以用一些未用於訓練的信件進行測試，看看模型的結論是否與你的結論一致。然後，一旦你對模型的準確性感到滿意，就可以處理未來的信件，而無需逐一閱讀。

✅ 這個過程是否與你在之前的課程中使用的過程類似？

## 練習 - 情感句子

情感以 *極性*（polarity）來衡量，範圍是 -1 到 1，-1 表示最負面，1 表示最正面。情感還以 0 到 1 的分數衡量客觀性（0）和主觀性（1）。

再看看 Jane Austen 的《傲慢與偏見》。該文本可在 [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) 上獲得。以下範例展示了一個簡短的程式，該程式分析了書中第一句和最後一句的情感，並顯示其情感極性和主觀性/客觀性分數。

你應該使用上述的 `TextBlob` 函式庫來確定 `sentiment`（你不需要自己編寫情感計算器）來完成以下任務。

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

你會看到以下輸出：

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## 挑戰 - 檢查情感極性

你的任務是使用情感極性來判斷《傲慢與偏見》中是否有更多絕對正面的句子，而不是絕對負面的句子。對於此任務，你可以假設極性分數為 1 或 -1 的句子分別是絕對正面或絕對負面。

**步驟：**

1. 從 [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) 下載《傲慢與偏見》的副本作為 .txt 文件。刪除文件開頭和結尾的元數據，只保留原始文本。
2. 在 Python 中打開該文件並將內容提取為字串。
3. 使用該書的字串創建一個 TextBlob。
4. 在迴圈中分析書中的每個句子：
   1. 如果極性為 1 或 -1，將該句子存儲在正面或負面的訊息陣列或列表中。
5. 最後，分別列出所有正面句子和負面句子，並顯示每類句子的數量。

這裡是一個範例[解決方案](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb)。

✅ 知識檢查

1. 情感是基於句子中使用的單詞，但程式碼是否*理解*這些單詞？
2. 你認為情感極性準確嗎？換句話說，你是否*同意*這些分數？
   1. 特別是，你是否同意以下句子的絕對**正面**極性？
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. 以下三個句子被評為絕對正面情感，但仔細閱讀後，它們並不是正面句子。為什麼情感分析認為它們是正面句子？
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. 你是否同意以下句子的絕對**負面**極性？
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ 任何 Jane Austen 的愛好者都會明白，她經常在書中批判英國攝政時期社會中更荒謬的方面。《傲慢與偏見》的主角 Elizabeth Bennett 是一位敏銳的社會觀察者（就像作者本人一樣），她的語言經常充滿微妙的含義。甚至故事中的愛情對象 Mr. Darcy 也注意到 Elizabeth 的俏皮和戲謔的語言使用：「我與你相識的時間足夠長，知道你偶爾會表達一些實際上並非你真實想法的觀點，並從中獲得極大的樂趣。」

---

## 🚀挑戰

你能讓 Marvin 更加出色，通過從用戶輸入中提取其他特徵嗎？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學
有許多方法可以從文本中提取情感。想想可能使用這項技術的商業應用。再想想它可能出錯的情況。深入了解一些能夠分析情感的高級企業級系統，例如 [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott)。測試上面的一些《傲慢與偏見》的句子，看看它是否能檢測出細微差別。

## 作業

[詩意的自由](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。