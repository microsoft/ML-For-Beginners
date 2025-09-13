<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T10:01:00+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "tw"
}
-->
# 使用酒店評論進行情感分析 - 數據處理

在本節中，您將使用前幾課中的技術對一個大型數據集進行一些探索性數據分析。當您對各列的實用性有了良好的理解後，您將學習：

- 如何刪除不必要的列
- 如何基於現有列計算一些新數據
- 如何保存處理後的數據集以用於最終挑戰

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

### 簡介

到目前為止，您已經學習了文本數據與數值數據的不同之處。如果文本是由人類書寫或口述的，則可以通過分析來發現模式、頻率、情感和含義。本課將帶您進入一個真實的數據集，並面臨一個真實的挑戰：**[歐洲515K酒店評論數據](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**，該數據集包含[CC0: 公共領域許可證](https://creativecommons.org/publicdomain/zero/1.0/)。數據來自Booking.com的公共來源，由Jiashen Liu創建。

### 準備工作

您需要：

* 能夠使用Python 3運行.ipynb筆記本
* pandas
* NLTK，[您需要在本地安裝](https://www.nltk.org/install.html)
* 可從Kaggle下載的數據集[歐洲515K酒店評論數據](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解壓後約230 MB。將其下載到與這些NLP課程相關的根目錄`/data`文件夾中。

## 探索性數據分析

本挑戰假設您正在使用情感分析和客人評論分數構建一個酒店推薦機器人。您將使用的數據集包括6個城市中1493家不同酒店的評論。

使用Python、酒店評論數據集和NLTK的情感分析，您可以找出：

* 評論中最常用的詞語和短語是什麼？
* 描述酒店的官方*標籤*是否與評論分數相關（例如，某酒店的*家庭帶小孩*評論是否比*單人旅客*更負面，這可能表明該酒店更適合*單人旅客*）？
* NLTK的情感分數是否與酒店評論者的數字分數“吻合”？

#### 數據集

讓我們探索您已下載並本地保存的數據集。可以在VS Code或Excel等編輯器中打開該文件。

數據集的標題如下：

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

以下是按類別分組的方式，可能更容易檢查：
##### 酒店相關列

* `Hotel_Name`, `Hotel_Address`, `lat`（緯度）, `lng`（經度）
  * 使用*lat*和*lng*，您可以使用Python繪製一張地圖，顯示酒店位置（可能根據正面和負面評論進行顏色編碼）
  * Hotel_Address對我們來說似乎沒有明顯的用途，我們可能會用國家名稱替代，以便更容易排序和搜索

**酒店元評論列**

* `Average_Score`
  * 根據數據集創建者的說法，此列是*酒店的平均分數，基於過去一年內的最新評論計算*。這似乎是一種不尋常的計算方式，但由於數據是抓取的，我們暫時接受這一點。
  
  ✅ 根據此數據中的其他列，您能想到另一種計算平均分數的方法嗎？

* `Total_Number_of_Reviews`
  * 該酒店收到的評論總數——目前尚不清楚（需要編寫一些代碼）這是否指數據集中的評論。
* `Additional_Number_of_Scoring`
  * 這表示評論者給出了分數，但未撰寫正面或負面評論。

**評論相關列**

- `Reviewer_Score`
  - 這是一個數值，最多有1位小數，範圍在2.5到10之間
  - 未解釋為什麼最低分數是2.5
- `Negative_Review`
  - 如果評論者未撰寫任何內容，此字段將顯示“**No Negative**”
  - 請注意，評論者可能會在負面評論列中撰寫正面評論（例如，“這家酒店沒有任何不好的地方”）
- `Review_Total_Negative_Word_Counts`
  - 負面詞彙數量越多，分數越低（不考慮情感分析）
- `Positive_Review`
  - 如果評論者未撰寫任何內容，此字段將顯示“**No Positive**”
  - 請注意，評論者可能會在正面評論列中撰寫負面評論（例如，“這家酒店完全沒有任何好的地方”）
- `Review_Total_Positive_Word_Counts`
  - 正面詞彙數量越多，分數越高（不考慮情感分析）
- `Review_Date` 和 `days_since_review`
  - 可以對評論應用新鮮度或陳舊度的衡量（例如，由於酒店管理變更、翻修完成或新增設施等原因，較舊的評論可能不如較新的評論準確）
- `Tags`
  - 這些是評論者可能選擇用來描述他們身份的簡短描述（例如，單人或家庭）、房間類型、停留時間以及評論提交方式。
  - 不幸的是，使用這些標籤存在問題，請參閱下面討論其實用性的部分。

**評論者相關列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 這可能是推薦模型中的一個因素，例如，如果您能確定撰寫數百條評論的高產評論者更可能給出負面而非正面評論。然而，任何特定評論的評論者並未用唯一代碼標識，因此無法將其與一組評論聯繫起來。有30位評論者撰寫了100條或更多評論，但很難看出這如何幫助推薦模型。
- `Reviewer_Nationality`
  - 有些人可能認為某些國籍的人更可能給出正面或負面評論，這是基於國家傾向的假設。在模型中構建這樣的經驗之談需要謹慎。這些是國家（有時是種族）刻板印象，每位評論者都是基於自身經歷撰寫評論的個體。評論可能受到多種因素的影響，例如他們之前的酒店住宿經歷、旅行距離以及個人性格。認為評論分數是由國籍決定的很難站得住腳。

##### 示例

| 平均分數 | 總評論數 | 評論者分數 | 負面評論                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 正面評論                 | 標籤                                                                                      |
| -------- | -------- | -------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| 7.8      | 1945     | 2.5      | 這目前不是一家酒店，而是一個建築工地。我在長途旅行後休息並在房間工作時，從早到晚都被不可接受的施工噪音折磨。人們整天在隔壁房間用鑿岩機工作。我要求更換房間，但沒有安靜的房間可用。更糟糕的是，我被多收了費。我在晚上退房，因為我需要趕早班飛機，並收到了一張合適的賬單。一天後，酒店在未經我同意的情況下再次收取了超出預訂價格的費用。這是一個可怕的地方。不要折磨自己，千萬別預訂這裡。 | 沒有任何好的地方 | 商務旅行，夫妻，標準雙人房，停留2晚 |

如您所見，這位客人在這家酒店的住宿並不愉快。該酒店的平均分數為7.8，擁有1945條評論，但這位評論者給出了2.5分，並寫了115個字描述他們的負面經歷。如果他們在Positive_Review列中什麼都沒寫，您可能會推測沒有任何正面的地方，但他們還是寫了7個字的警告。如果我們僅僅計算字數而不是字詞的含義或情感，我們可能會對評論者的意圖產生偏差。奇怪的是，他們的2.5分令人困惑，因為如果這家酒店的住宿如此糟糕，為什麼還要給任何分數？仔細研究數據集，您會發現最低可能分數是2.5，而不是0。最高可能分數是10。

##### 標籤

如上所述，乍一看，使用`Tags`來分類數據似乎很有意義。不幸的是，這些標籤並未標準化，這意味著在某家酒店中，選項可能是*單人房*、*雙床房*和*雙人房*，但在另一家酒店中，則是*豪華單人房*、*經典大床房*和*行政特大床房*。這些可能是相同的房型，但變化如此之多，以至於選擇變得困難：

1. 嘗試將所有術語更改為單一標準，這非常困難，因為在每種情況下不清楚轉換路徑（例如，*經典單人房*映射到*單人房*，但*帶庭院花園或城市景觀的高級大床房*則更難映射）

2. 我們可以採用NLP方法，測量某些術語（如*單人*、*商務旅客*或*帶小孩的家庭*）在每家酒店中的出現頻率，並將其納入推薦模型中。

標籤通常（但不總是）是一個包含5到6個逗號分隔值的單一字段，對應於*旅行類型*、*客人類型*、*房間類型*、*停留天數*以及*提交評論的設備類型*。然而，由於某些評論者未填寫每個字段（可能留空一個），值並不總是按相同順序排列。

例如，考慮*群體類型*。在`Tags`列中，此字段有1025種唯一可能性，不幸的是，其中只有部分提到群體（有些是房間類型等）。如果您僅篩選提到家庭的標籤，結果包含許多*家庭房*類型的結果。如果您包括*with*這個詞，即計算*帶有*的家庭值，結果會更好，在515,000條結果中，有超過80,000條包含“帶小孩的家庭”或“帶大孩的家庭”這樣的短語。

這意味著標籤列對我們來說並非完全無用，但需要一些工作才能使其變得有用。

##### 酒店平均分數

數據集中有一些奇怪或不一致的地方，我無法完全理解，但在此列出以便您在構建模型時注意。如果您能弄清楚，請在討論區告訴我們！

數據集包含以下與平均分數和評論數相關的列：

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

數據集中評論數最多的酒店是*Britannia International Hotel Canary Wharf*，擁有515,000條評論中的4789條。但如果我們查看該酒店的`Total_Number_of_Reviews`值，則為9086。您可能會推測有更多未包含評論的分數，因此我們可能需要加上`Additional_Number_of_Scoring`列的值。該值為2682，將其加到4789得到7471，仍然比`Total_Number_of_Reviews`少1615。

如果您查看`Average_Score`列，您可能會推測它是數據集中評論的平均值，但Kaggle的描述是“*酒店的平均分數，基於過去一年內的最新評論計算*”。這似乎並不那麼有用，但我們可以根據數據集中的評論分數計算我們自己的平均值。以同一家酒店為例，給出的平均酒店分數是7.1，但數據集中計算的平均評論者分數是6.8。這很接近，但不是相同的值，我們只能猜測`Additional_Number_of_Scoring`評論中的分數將平均值提高到7.1。不幸的是，由於無法測試或證明這一假設，難以使用或信任`Average_Score`、`Additional_Number_of_Scoring`和`Total_Number_of_Reviews`，因為它們基於或引用了我們沒有的數據。

更複雜的是，評論數第二多的酒店的計算平均分數是8.12，而數據集中的`Average_Score`是8.1。這是否正確是一個巧合，還是第一家酒店存在不一致？

考慮到這些酒店可能是異常值，也許大多數值是匹配的（但某些值由於某些原因不匹配），我們將在接下來編寫一個簡短的程序來探索數據集中的值，並確定這些值的正確用法（或不使用）。
> 🚨 注意事項
>
> 在處理這個數據集時，您將撰寫程式碼來從文本中計算某些內容，而無需親自閱讀或分析文本。這正是自然語言處理（NLP）的核心——在不需要人類介入的情況下解讀意義或情感。然而，有可能您會讀到一些負面評論。我建議您不要這麼做，因為您並不需要這樣做。有些評論可能很無聊，或者是與酒店無關的負面評論，例如「天氣不好」，這是酒店甚至任何人都無法控制的事情。但有些評論的確存在陰暗面。有時候，負面評論可能帶有種族歧視、性別歧視或年齡歧視的內容。這很不幸，但在從公共網站抓取的數據集中是可以預期的。一些評論者可能會留下讓您感到反感、不適或不安的評論。最好讓程式碼來衡量情感，而不是親自閱讀這些評論並讓自己感到不快。話雖如此，撰寫這類評論的人是少數，但他們的確存在。
## 練習 - 資料探索
### 載入資料

視覺化檢視資料已經足夠，現在你需要撰寫一些程式碼來獲取答案！本節使用 pandas 函式庫。你的第一個任務是確保你能載入並讀取 CSV 資料。pandas 函式庫提供快速的 CSV 載入器，結果會像之前的課程一樣存放在 dataframe 中。我們要載入的 CSV 檔案有超過五十萬行，但只有 17 個欄位。pandas 提供了許多強大的方法來與 dataframe 互動，包括能對每一行進行操作的能力。

從這裡開始，課程中會有程式碼片段以及一些程式碼的解釋，並討論結果的意義。請使用附帶的 _notebook.ipynb_ 來撰寫你的程式碼。

讓我們從載入你將使用的資料檔案開始：

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

現在資料已載入，我們可以對其進行一些操作。請將這段程式碼放在程式的頂部，以便進行下一部分。

## 探索資料

在這個案例中，資料已經是*乾淨的*，這意味著它已準備好使用，並且不包含可能會讓只接受英文字符的演算法出錯的其他語言字符。

✅ 你可能需要處理一些需要初步格式化的資料，才能應用 NLP 技術，但這次不需要。如果需要，你會如何處理非英文字符？

花點時間確保資料載入後，你可以使用程式碼來探索它。很容易想要專注於 `Negative_Review` 和 `Positive_Review` 欄位。這些欄位充滿了自然文本，供你的 NLP 演算法處理。但等等！在進行 NLP 和情感分析之前，你應該遵循以下程式碼，確認資料集中提供的值是否與你使用 pandas 計算的值相符。

## Dataframe 操作

本課程的第一個任務是撰寫一些程式碼來檢查以下斷言是否正確（不更改資料框）。

> 像許多程式設計任務一樣，完成這些任務的方法有很多，但好的建議是以最簡單、最容易的方式完成，尤其是當你未來回顧這段程式碼時更容易理解。對於 dataframe，有一個全面的 API，通常可以有效地完成你想要的操作。

將以下問題視為程式設計任務，嘗試在不查看解答的情況下回答它們。

1. 輸出你剛載入的資料框的*形狀*（形狀是行數和列數）
2. 計算評論者國籍的頻率：
   1. `Reviewer_Nationality` 欄位中有多少個不同的值？它們是什麼？
   2. 資料集中最常見的評論者國籍是什麼（輸出國家和評論數量）？
   3. 接下來最常見的前 10 個國籍及其頻率分布是什麼？
3. 對於前 10 個最常見的評論者國籍，每個國籍最常被評論的酒店是什麼？
4. 資料集中每家酒店有多少評論（酒店的評論頻率分布）？
5. 雖然資料集中每家酒店都有一個 `Average_Score` 欄位，但你也可以計算平均分數（計算資料集中每家酒店所有評論者分數的平均值）。新增一個名為 `Calc_Average_Score` 的新欄位到你的 dataframe，該欄位包含計算出的平均分數。
6. 是否有酒店的 `Average_Score` 和 `Calc_Average_Score`（四捨五入到小數點後一位）相同？
   1. 嘗試撰寫一個 Python 函數，該函數接受一個 Series（行）作為參數並比較這些值，當值不相等時輸出一條訊息。然後使用 `.apply()` 方法處理每一行。
7. 計算並輸出 `Negative_Review` 欄位值為 "No Negative" 的行數
8. 計算並輸出 `Positive_Review` 欄位值為 "No Positive" 的行數
9. 計算並輸出 `Positive_Review` 欄位值為 "No Positive" **且** `Negative_Review` 欄位值為 "No Negative" 的行數

### 程式碼解答

1. 輸出你剛載入的資料框的*形狀*（形狀是行數和列數）

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. 計算評論者國籍的頻率：

   1. `Reviewer_Nationality` 欄位中有多少個不同的值？它們是什麼？
   2. 資料集中最常見的評論者國籍是什麼（輸出國家和評論數量）？

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. 接下來最常見的前 10 個國籍及其頻率分布是什麼？

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. 對於前 10 個最常見的評論者國籍，每個國籍最常被評論的酒店是什麼？

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. 資料集中每家酒店有多少評論（酒店的評論頻率分布）？

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   你可能會注意到*資料集中計算的*結果與 `Total_Number_of_Reviews` 的值不匹配。尚不清楚資料集中該值是否代表酒店的總評論數，但並非所有評論都被抓取，或者是其他計算方式。由於這種不明確性，`Total_Number_of_Reviews` 不會在模型中使用。

5. 雖然資料集中每家酒店都有一個 `Average_Score` 欄位，但你也可以計算平均分數（計算資料集中每家酒店所有評論者分數的平均值）。新增一個名為 `Calc_Average_Score` 的新欄位到你的 dataframe，該欄位包含計算出的平均分數。輸出欄位 `Hotel_Name`、`Average_Score` 和 `Calc_Average_Score`。

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   你可能也會好奇 `Average_Score` 的值以及為什麼它有時與計算出的平均分數不同。由於我們無法知道為什麼有些值匹配，但其他值有差異，在這種情況下，最安全的做法是使用我們擁有的評論分數自行計算平均分數。儘管如此，差異通常非常小，以下是資料集中平均分數與計算出的平均分數差異最大的酒店：

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   只有 1 家酒店的分數差異超過 1，這意味著我們可能可以忽略差異並使用計算出的平均分數。

6. 計算並輸出 `Negative_Review` 欄位值為 "No Negative" 的行數

7. 計算並輸出 `Positive_Review` 欄位值為 "No Positive" 的行數

8. 計算並輸出 `Positive_Review` 欄位值為 "No Positive" **且** `Negative_Review` 欄位值為 "No Negative" 的行數

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## 另一種方法

另一種方法是不用 Lambdas，並使用 sum 來計算行數：

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   你可能注意到有 127 行的 `Negative_Review` 和 `Positive_Review` 欄位分別為 "No Negative" 和 "No Positive"。這意味著評論者給酒店一個數字分數，但拒絕撰寫正面或負面的評論。幸運的是，這是一個很小的行數（127 行中的 515738 行，或 0.02%），因此它可能不會在任何特定方向上影響我們的模型或結果，但你可能不會預期到評論資料集中會有沒有評論的行，因此值得探索資料以發現這樣的行。

現在你已經探索了資料集，在下一課中你將篩選資料並進行一些情感分析。

---
## 🚀挑戰

本課程展示了，如同我們在之前的課程中看到的，了解你的資料及其特性在進行操作之前是多麼重要。特別是基於文本的資料，需要仔細檢查。深入挖掘各種以文本為主的資料集，看看你是否能發現可能引入偏差或情感偏斜的地方。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

參加 [這個 NLP 學習路徑](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)，探索在構建語音和文本模型時可以嘗試的工具。

## 作業

[NLTK](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。