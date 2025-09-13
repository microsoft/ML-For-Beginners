<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T10:03:27+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "tw"
}
-->
# 使用酒店評論進行情感分析

現在您已經詳細探索了數據集，是時候篩選欄位並對數據集使用 NLP 技術，以獲得關於酒店的新見解。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

### 篩選與情感分析操作

正如您可能已經注意到的，數據集中存在一些問題。一些欄位充滿了無用的信息，其他欄位似乎不正確。即使它們是正確的，也不清楚它們是如何計算的，並且無法通過自己的計算獨立驗證答案。

## 練習：進一步處理數據

稍微清理一下數據。添加一些稍後會有用的欄位，修改其他欄位中的值，並完全刪除某些欄位。

1. 初步欄位處理

   1. 刪除 `lat` 和 `lng`

   2. 將 `Hotel_Address` 的值替換為以下值（如果地址包含城市和國家的名稱，則將其更改為僅顯示城市和國家）。

      以下是數據集中唯一的城市和國家：

      阿姆斯特丹，荷蘭  
      巴塞隆納，西班牙  
      倫敦，英國  
      米蘭，義大利  
      巴黎，法國  
      維也納，奧地利  

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      現在您可以查詢國家級數據：

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. 處理酒店元評論欄位

   1. 刪除 `Additional_Number_of_Scoring`

   2. 用實際在數據集中該酒店的評論總數替換 `Total_Number_of_Reviews`

   3. 用我們自己計算的分數替換 `Average_Score`

      ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. 處理評論欄位

   1. 刪除 `Review_Total_Negative_Word_Counts`、`Review_Total_Positive_Word_Counts`、`Review_Date` 和 `days_since_review`

   2. 保留 `Reviewer_Score`、`Negative_Review` 和 `Positive_Review` 不變

   3. 暫時保留 `Tags`

      - 我們會在下一部分對標籤進行一些額外的篩選操作，然後再刪除標籤

4. 處理評論者欄位

   1. 刪除 `Total_Number_of_Reviews_Reviewer_Has_Given`

   2. 保留 `Reviewer_Nationality`

### 標籤欄位

`Tag` 欄位是一個問題，因為它是一個以文本形式存儲在欄位中的列表。不幸的是，該欄位中的子部分順序和數量並不總是相同。由於數據集中有 515,000 行和 1427 家酒店，每個評論者可以選擇的選項略有不同，因此人類很難識別出需要關注的正確短語。這正是 NLP 的用武之地。您可以掃描文本，找到最常見的短語並對其進行計數。

不幸的是，我們對單詞不感興趣，而是對多詞短語（例如 *商務旅行*）感興趣。在如此大量的數據（6762646 個單詞）上運行多詞頻率分佈算法可能需要花費大量時間，但如果不查看數據，這似乎是必要的開銷。這時，探索性數據分析就派上用場了，因為您已經看過一些標籤樣本，例如 `[' 商務旅行 ', ' 獨自旅行者 ', ' 單人房 ', ' 住了 5 晚 ', ' 從移動設備提交 ']`，您可以開始思考是否有可能大幅減少需要處理的數據量。幸運的是，這是可能的——但首先您需要遵循一些步驟來確定感興趣的標籤。

### 篩選標籤

請記住，數據集的目標是添加情感和欄位，幫助您選擇最佳酒店（無論是為自己還是為委託您製作酒店推薦機器人的客戶）。您需要問自己這些標籤在最終數據集中是否有用。以下是一種解釋（如果您出於其他原因需要數據集，不同的標籤可能會保留/刪除）：

1. 旅行類型是相關的，應保留
2. 客戶群類型很重要，應保留
3. 客人入住的房間、套房或工作室類型無關緊要（所有酒店基本上都有相同的房間）
4. 提交評論的設備無關緊要
5. 評論者入住的晚數*可能*相關，如果您認為更長的入住時間意味著他們更喜歡酒店，但這有點牽強，可能無關緊要

總結來說，**保留兩種類型的標籤，刪除其他標籤**。

首先，在標籤格式更好之前，您不想對標籤進行計數，這意味著需要刪除方括號和引號。您可以通過多種方式完成此操作，但您需要最快的方法，因為處理大量數據可能需要很長時間。幸運的是，pandas 提供了一種簡單的方法來完成這些步驟。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

每個標籤變成類似於：`商務旅行, 獨自旅行者, 單人房, 住了 5 晚, 從移動設備提交`。

接下來，我們發現了一個問題。一些評論（或行）有 5 個欄位，一些有 3 個，一些有 6 個。這是數據集創建方式的結果，很難修復。您希望獲得每個短語的頻率計數，但它們在每條評論中的順序不同，因此計數可能會出錯，酒店可能無法獲得應得的標籤。

相反，您將利用這種不同的順序，因為每個標籤是多詞的，但也用逗號分隔！最簡單的方法是創建 6 個臨時欄位，將每個標籤插入到對應於其順序的欄位中。然後，您可以將這 6 個欄位合併為一個大欄位，並對結果欄位運行 `value_counts()` 方法。打印出來後，您會看到有 2428 個唯一標籤。以下是其中的一小部分樣本：

| 標籤                              | 計數   |
| --------------------------------- | ------ |
| 休閒旅行                         | 417778 |
| 從移動設備提交                   | 307640 |
| 情侶                             | 252294 |
| 住了 1 晚                        | 193645 |
| 住了 2 晚                        | 133937 |
| 獨自旅行者                       | 108545 |
| 住了 3 晚                        | 95821  |
| 商務旅行                         | 82939  |
| 團體                             | 65392  |
| 帶小孩的家庭                     | 61015  |
| 住了 4 晚                        | 47817  |
| 雙人房                           | 35207  |
| 標準雙人房                       | 32248  |
| 高級雙人房                       | 31393  |
| 帶大孩的家庭                     | 26349  |
| 豪華雙人房                       | 24823  |
| 雙人或雙床房                     | 22393  |
| 住了 5 晚                        | 20845  |
| 標準雙人或雙床房                 | 17483  |
| 經典雙人房                       | 16989  |
| 高級雙人或雙床房                 | 13570  |
| 2 間房                           | 12393  |

一些常見的標籤如 `從移動設備提交` 對我們沒有用，因此在計算短語出現次數之前刪除它們可能是一個明智的選擇，但這是一個非常快速的操作，您可以將它們保留在內並忽略它們。

### 刪除入住時長標籤

刪除這些標籤是第一步，它稍微減少了需要考慮的標籤總數。請注意，您並未從數據集中刪除它們，只是選擇不將它們作為評論數據集中要計數/保留的值。

| 入住時長       | 計數   |
| -------------- | ------ |
| 住了 1 晚     | 193645 |
| 住了 2 晚     | 133937 |
| 住了 3 晚     | 95821  |
| 住了 4 晚     | 47817  |
| 住了 5 晚     | 20845  |
| 住了 6 晚     | 9776   |
| 住了 7 晚     | 7399   |
| 住了 8 晚     | 2502   |
| 住了 9 晚     | 1293   |
| ...            | ...    |

有各種各樣的房間、套房、工作室、公寓等。它們的含義大致相同，對您來說並不重要，因此從考慮中刪除它們。

| 房間類型                     | 計數   |
| ---------------------------- | ------ |
| 雙人房                       | 35207  |
| 標準雙人房                   | 32248  |
| 高級雙人房                   | 31393  |
| 豪華雙人房                   | 24823  |
| 雙人或雙床房                 | 22393  |
| 標準雙人或雙床房             | 17483  |
| 經典雙人房                   | 16989  |
| 高級雙人或雙床房             | 13570  |

最後，這是令人愉快的（因為幾乎不需要處理），您將剩下以下**有用的**標籤：

| 標籤                                           | 計數   |
| --------------------------------------------- | ------ |
| 休閒旅行                                     | 417778 |
| 情侶                                         | 252294 |
| 獨自旅行者                                   | 108545 |
| 商務旅行                                     | 82939  |
| 團體（與朋友旅行者合併）                     | 67535  |
| 帶小孩的家庭                                 | 61015  |
| 帶大孩的家庭                                 | 26349  |
| 帶寵物                                       | 1405   |

您可以認為 `與朋友旅行者` 與 `團體` 基本相同，將兩者合併是合理的，如上所示。識別正確標籤的代碼位於 [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)。

最後一步是為每個這些標籤創建新欄位。然後，對於每條評論行，如果 `Tag` 欄位與新欄位之一匹配，則添加 1，否則添加 0。最終結果將是計算有多少評論者選擇這家酒店（總體上）作為商務旅行、休閒旅行或帶寵物入住的選擇，這在推薦酒店時是有用的信息。

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### 保存文件

最後，將當前數據集保存為新名稱。

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 情感分析操作

在最後一部分中，您將對評論欄位應用情感分析，並將結果保存到數據集中。

## 練習：加載並保存篩選後的數據

請注意，現在您加載的是上一部分保存的篩選後數據集，而**不是**原始數據集。

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### 刪除停用詞

如果您對負面和正面評論欄位運行情感分析，可能需要很長時間。在一台性能強大的測試筆記本電腦上進行測試時，根據使用的情感分析庫不同，耗時約 12-14 分鐘。這是一段（相對）較長的時間，因此值得研究是否可以加快速度。

刪除停用詞（即不改變句子情感的常見英文字詞）是第一步。通過刪除它們，情感分析應該會運行得更快，但不會降低準確性（因為停用詞不影響情感，但會減慢分析速度）。

最長的負面評論有 395 個單詞，但刪除停用詞後，只有 195 個單詞。

刪除停用詞也是一個快速操作，在測試設備上，從 2 個評論欄位的 515,000 行中刪除停用詞耗時 3.3 秒。具體時間可能因您的設備 CPU 速度、RAM、是否有 SSD 等因素略有不同。操作相對較短，這意味著如果它能改善情感分析時間，那麼值得執行。

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### 執行情感分析

現在，您應該計算負面和正面評論欄位的情感分析，並將結果存儲在 2 個新欄位中。情感測試將與評論者對同一評論的評分進行比較。例如，如果情感分析認為負面評論的情感為 1（極度正面情感），正面評論的情感也為 1，但評論者給酒店的評分是最低分，那麼要麼評論文本與評分不匹配，要麼情感分析器無法正確識別情感。您應該預期某些情感分數完全錯誤，這通常是可以解釋的，例如評論可能極具諷刺意味：“當然，我**喜歡**住在一個沒有暖氣的房間裡”，情感分析器可能認為這是正面情感，但人類閱讀時會知道這是諷刺。
NLTK 提供了不同的情感分析器供學習使用，您可以替換它們，看看情感分析的準確性是否有所不同。這裡使用的是 VADER 情感分析。

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

在您的程式中，當您準備計算情感時，可以將其應用到每條評論，如下所示：

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

在我的電腦上這大約需要 120 秒，但每台電腦的時間可能會有所不同。如果您想列印結果並檢查情感是否與評論相符：

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

在挑戰中使用該檔案之前，最後要做的事情就是保存它！您還應該考慮重新排列所有新欄位，使其更易於操作（對人類來說，這是一個外觀上的改變）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

您應該執行[分析筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)的完整程式碼（在您執行[篩選筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)以生成 Hotel_Reviews_Filtered.csv 檔案之後）。

回顧一下，步驟如下：

1. 原始數據集檔案 **Hotel_Reviews.csv** 在上一課中使用[探索筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)進行了探索。
2. **Hotel_Reviews.csv** 通過[篩選筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)進行篩選，生成 **Hotel_Reviews_Filtered.csv**。
3. **Hotel_Reviews_Filtered.csv** 通過[情感分析筆記本](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)進行處理，生成 **Hotel_Reviews_NLP.csv**。
4. 在下面的 NLP 挑戰中使用 **Hotel_Reviews_NLP.csv**。

### 結論

當您開始時，您擁有一個包含欄位和數據的數據集，但並非所有數據都可以被驗證或使用。您已經探索了數據，篩選出不需要的部分，將標籤轉換為有用的內容，計算了自己的平均值，添加了一些情感欄位，並希望學到了一些關於處理自然文本的有趣知識。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 挑戰

現在您已經對數據集進行了情感分析，看看您是否可以使用本課程中學到的策略（例如聚類）來找出情感模式。

## 回顧與自學

參加[這個 Learn 模組](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)，學習更多內容，並使用不同的工具來探索文本中的情感。

## 作業

[嘗試一個不同的數據集](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對於因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。