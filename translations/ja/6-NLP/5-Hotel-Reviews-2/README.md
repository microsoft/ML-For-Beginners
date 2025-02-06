# ホテルレビューによる感情分析

データセットを詳細に調査したので、次は列をフィルタリングし、NLP技術を使ってホテルに関する新しい洞察を得ましょう。
## [事前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### フィルタリングと感情分析の操作

おそらく気づいたと思いますが、このデータセットにはいくつか問題があります。一部の列には無意味な情報が含まれており、他の列は正しくないように見えます。正しいとしても、どのように計算されたのか不明であり、自分の計算で独自に検証することはできません。

## 演習: もう少しデータ処理

データをもう少しきれいにしましょう。後で役立つ列を追加し、他の列の値を変更し、特定の列を完全に削除します。

1. 初期の列処理

   1. `lat` と `lng` を削除

   2. `Hotel_Address` の値を以下の値に置き換える（住所に都市名と国名が含まれている場合は、都市名と国名のみに変更）。

      データセットに含まれている都市と国は次の通りです：

      アムステルダム、オランダ

      バルセロナ、スペイン

      ロンドン、イギリス

      ミラノ、イタリア

      パリ、フランス

      ウィーン、オーストリア 

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

      これで国レベルのデータをクエリできます：

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | ホテル住所               | ホテル名 |
      | :--------------------- | :--------: |
      | アムステルダム、オランダ |    105     |
      | バルセロナ、スペイン       |    211     |
      | ロンドン、イギリス         |    400     |
      | ミラノ、イタリア           |    162     |
      | パリ、フランス             |    458     |
      | ウィーン、オーストリア     |    158     |

2. ホテルメタレビュー列の処理

  1. `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` を自分で計算したスコアで置き換え

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. レビュー列の処理

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` を削除し、タグの興味を確認するためのいくつかのステップに従う必要があります。

### タグのフィルタリング

データセットの目的は、感情と列を追加して、最適なホテルを選ぶのに役立つことです（自分自身やホテル推薦ボットを作成するクライアントのため）。最終データセットでタグが役立つかどうかを自問する必要があります。以下は一つの解釈です（他の理由でデータセットが必要な場合、異なるタグが選択に含まれるかもしれません）：

1. 旅行の種類は関連しており、保持すべき
2. ゲストグループの種類は重要で、保持すべき
3. ゲストが滞在した部屋、スイート、スタジオの種類は無関係（すべてのホテルに基本的に同じ部屋がある）
4. レビューが提出されたデバイスは無関係
5. レビュアーが滞在した夜数は、ホテルを気に入っている可能性があるため関連するかもしれないが、おそらく無関係

要約すると、**2種類のタグを保持し、他のタグを削除する**。

まず、タグがより良い形式になるまでカウントしたくないので、角括弧と引用符を削除する必要があります。これにはいくつかの方法がありますが、最速の方法を選びたいです。幸い、pandasにはこれらのステップを簡単に行う方法があります。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

各タグは次のようになります: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` 列が新しい列のいずれかと一致する場合は1を追加し、一致しない場合は0を追加します。最終結果は、ビジネス対レジャーのために、またはペットを連れて行くために、どれだけのレビュアーがこのホテルを選んだか（集計で）をカウントするものであり、これはホテルを推薦する際に有用な情報です。

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

### ファイルの保存

最後に、現在のデータセットを新しい名前で保存します。

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 感情分析の操作

この最後のセクションでは、レビュー列に感情分析を適用し、結果をデータセットに保存します。

## 演習: フィルタリングされたデータの読み込みと保存

注意すべき点は、今は前のセクションで保存されたフィルタリングされたデータセットを読み込んでいることです。**元のデータセットではありません**。

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

### ストップワードの削除

ネガティブおよびポジティブレビュー列で感情分析を実行すると、時間がかかることがあります。高速なCPUを持つ強力なテストラップトップでテストしたところ、使用する感情ライブラリによって12〜14分かかりました。それは（比較的）長い時間なので、スピードアップできるかどうかを調査する価値があります。

ストップワード、つまり文の感情を変えない一般的な英語の単語を削除することが最初のステップです。これらを削除することで、感情分析はより速く実行されるはずですが、精度は低下しません（ストップワードは感情に影響を与えませんが、分析を遅くします）。

最も長いネガティブレビューは395単語でしたが、ストップワードを削除した後は195単語になりました。

ストップワードの削除も高速な操作であり、515,000行の2つのレビュー列からストップワードを削除するのに3.3秒かかりました。デバイスのCPU速度、RAM、SSDの有無、その他の要因により、若干の時間の違いがあるかもしれません。この操作が感情分析の時間を改善するならば、それは価値があります。

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

### 感情分析の実行

次に、ネガティブおよびポジティブレビュー列の感情分析を計算し、その結果を2つの新しい列に保存する必要があります。同じレビューに対するレビュアーのスコアと比較して感情をテストします。たとえば、感情分析がネガティブレビューの感情を1（非常にポジティブな感情）と判断し、ポジティブレビューの感情も1と判断したが、レビュアーがホテルに最低のスコアを与えた場合、レビューのテキストがスコアと一致していないか、感情分析が感情を正しく認識できなかった可能性があります。一部の感情スコアが完全に間違っていることを期待するべきであり、その理由は説明可能であることがよくあります。たとえば、レビューが非常に皮肉である場合、「もちろん、暖房のない部屋で寝るのが大好きでした」といった場合、感情分析はそれがポジティブな感情であると考えますが、人間が読むとそれが皮肉であることがわかります。

NLTKは学習に使用できるさまざまな感情分析ツールを提供しており、それらを代替して感情がより正確かどうかを確認できます。ここではVADER感情分析が使用されています。

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

プログラムの後半で感情を計算する準備ができたら、各レビューに次のように適用できます：

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

これは私のコンピュータで約120秒かかりますが、各コンピュータで異なります。結果を印刷して感情がレビューと一致するか確認したい場合：

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

ファイルを使用する前に最後に行うことは、それを保存することです！また、新しい列をすべて再配置して、作業しやすくすることも検討してください（人間にとっては、これは見た目の変更です）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)の全コードを実行する必要があります（[フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)を実行してHotel_Reviews_Filtered.csvファイルを生成した後）。

手順を振り返ると：

1. 元のデータセットファイル **Hotel_Reviews.csv** は、[エクスプローラーノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)で前のレッスンで調査されました
2. Hotel_Reviews.csv は [フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) によってフィルタリングされ、**Hotel_Reviews_Filtered.csv** になります
3. Hotel_Reviews_Filtered.csv は [感情分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) によって処理され、**Hotel_Reviews_NLP.csv** になります
4. 以下のNLPチャレンジで Hotel_Reviews_NLP.csv を使用します

### 結論

最初は、列とデータが含まれているデータセットがありましたが、そのすべてを検証したり使用したりすることはできませんでした。データを調査し、不要なものをフィルタリングし、タグを有用なものに変換し、独自の平均値を計算し、いくつかの感情列を追加し、自然言語処理について興味深いことを学びました。

## [事後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## チャレンジ

感情のためにデータセットを分析したので、このカリキュラムで学んだ戦略（クラスター分析など）を使用して、感情に関するパターンを特定できるか試してみてください。

## 復習と自己学習

[このLearnモジュール](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)を取って、テキストの感情を探索するためのさまざまなツールを使用してみてください。
## 課題

[別のデータセットを試してみてください](assignment.md)

**免責事項**:
この文書は機械ベースのAI翻訳サービスを使用して翻訳されています。正確さを期すよう努めておりますが、自動翻訳には誤りや不正確さが含まれる可能性があることをご理解ください。原文はその言語での公式な文書とみなされるべきです。重要な情報については、専門の人間による翻訳をお勧めします。この翻訳の使用により生じた誤解や誤訳について、当社は一切の責任を負いません。