<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:55:38+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ja"
}
-->
# ホテルレビューによる感情分析

データセットを詳細に調査した後、列をフィルタリングし、NLP技術を使用してホテルに関する新しい洞察を得る時が来ました。
## [講義前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### フィルタリングと感情分析の操作

おそらく気づいたと思いますが、このデータセットにはいくつか問題があります。一部の列には役に立たない情報が含まれており、他の列は正確ではないように見えます。仮に正確だとしても、それらがどのように計算されたのか不明であり、自分自身の計算で独立して検証することができません。

## 演習: もう少しデータ処理

データをもう少しクリーンアップします。後で役立つ列を追加し、他の列の値を変更し、特定の列を完全に削除します。

1. 初期の列処理

   1. `lat` と `lng` を削除する

   2. `Hotel_Address` の値を以下の値に置き換える（住所に都市名と国名が含まれている場合、それを都市名と国名だけに変更する）。

      データセットに含まれる都市と国は以下の通りです：

      アムステルダム, オランダ

      バルセロナ, スペイン

      ロンドン, イギリス

      ミラノ, イタリア

      パリ, フランス

      ウィーン, オーストリア 

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

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | アムステルダム, オランダ |    105     |
      | バルセロナ, スペイン       |    211     |
      | ロンドン, イギリス         |    400     |
      | ミラノ, イタリア           |    162     |
      | パリ, フランス             |    458     |
      | ウィーン, オーストリア     |    158     |

2. ホテルメタレビュー列の処理

  1. `Additional_Number_of_Scoring` を削除する

  1. `Total_Number_of_Reviews` をそのホテルに実際に含まれているレビューの総数に置き換える

  1. `Average_Score` を独自に計算したスコアに置き換える

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. レビュー列の処理

   1. `Review_Total_Negative_Word_Counts`、`Review_Total_Positive_Word_Counts`、`Review_Date`、`days_since_review` を削除する

   2. `Reviewer_Score`、`Negative_Review`、`Positive_Review` はそのまま保持する
     
   3. `Tags` は一時的に保持する

     - 次のセクションでタグに対して追加のフィルタリング操作を行い、その後タグを削除します

4. レビュアー列の処理

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` を削除する
  
  2. `Reviewer_Nationality` を保持する

### タグ列

`Tag` 列は問題があります。この列にはリスト（テキスト形式）が格納されていますが、順序やサブセクションの数が常に同じではありません。この列には 515,000 行、1427 のホテルがあり、レビュアーが選択できるオプションが少しずつ異なるため、人間が正しいフレーズを特定するのは困難です。ここで NLP が役立ちます。テキストをスキャンして最も一般的なフレーズを見つけ、それらをカウントすることができます。

残念ながら、単語ではなく複数語のフレーズ（例: *Business trip*）に興味があるため、6762646 語のデータに対して複数語の頻度分布アルゴリズムを実行するのは非常に時間がかかる可能性があります。しかし、データを見ずにそれが必要な作業だと判断するのは早計です。ここで探索的データ分析が役立ちます。例えば、`[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` のようなタグのサンプルを見た場合、処理を大幅に削減できる可能性があるかどうかを検討することができます。幸いにも、それは可能ですが、まずいくつかのステップを踏んで興味のあるタグを確認する必要があります。

### タグのフィルタリング

データセットの目的は、感情や列を追加して最適なホテルを選ぶ手助けをすることです（自分自身のため、またはホテル推薦ボットを作るよう依頼されたクライアントのため）。タグが最終的なデータセットで役立つかどうかを自問する必要があります。以下は一つの解釈です（他の目的でデータセットが必要な場合、異なるタグが選択される可能性があります）：

1. 旅行の種類は関連性があり、保持するべき
2. ゲストグループの種類は重要であり、保持するべき
3. ゲストが滞在した部屋、スイート、スタジオの種類は無関係（すべてのホテルには基本的に同じ部屋があります）
4. レビューが送信されたデバイスは無関係
5. レビュアーが滞在した夜数は、ホテルを気に入った可能性があると関連付ける場合は関連性があるかもしれませんが、可能性は低く、おそらく無関係

要約すると、**2種類のタグを保持し、他を削除します**。

まず、タグをカウントする前に、より良い形式にする必要があります。つまり、角括弧や引用符を削除する必要があります。これを行う方法はいくつかありますが、最速の方法を選びたいです。大量のデータを処理するのに時間がかかる可能性があるためです。幸いにも、pandas にはこれらのステップを簡単に実行する方法があります。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

各タグは次のようになります：`Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`。

次に問題が発生します。一部のレビュー（行）には 5 列、一部には 3 列、一部には 6 列があります。これはデータセットの作成方法の結果であり、修正が困難です。各フレーズの頻度をカウントしたいのですが、レビューごとに順序が異なるため、カウントが正確でない可能性があり、ホテルが本来得るべきタグを割り当てられない可能性があります。

代わりに、この異なる順序を利用します。各タグは複数語で構成されていますが、コンマで区切られています！最も簡単な方法は、各タグをその順序に対応する列に挿入して一時的な 6 列を作成することです。その後、6 列を1つの大きな列に統合し、`value_counts()` メソッドを実行します。その結果を印刷すると、2428 のユニークなタグがあったことがわかります。以下はその一部です：

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

`Submitted from a mobile device` のような一般的なタグは役に立たないため、フレーズの出現回数をカウントする前に削除するのが賢明かもしれませんが、非常に高速な操作なのでそのまま残して無視することもできます。

### 滞在期間タグの削除

これらのタグを削除するのがステップ1です。これにより、考慮すべきタグの総数がわずかに減少します。データセットから削除するのではなく、レビューのデータセットで値としてカウント/保持することを選択しないだけです。

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

部屋、スイート、スタジオ、アパートメントなどの種類は非常に多様です。それらはほぼ同じ意味を持ち、関連性がないため、考慮から除外します。

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

最後に、これは喜ばしいことですが（ほとんど処理を必要としなかったため）、以下の**有用な**タグが残ります：

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

`Travellers with friends` は `Group` とほぼ同じであると主張することができ、それを上記のように統合するのは妥当です。正しいタグを特定するコードは [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) にあります。

最終ステップは、これらのタグごとに新しい列を作成することです。そして、各レビュー行について、`Tag` 列が新しい列のいずれかに一致する場合は 1 を追加し、一致しない場合は 0 を追加します。最終結果として、例えばビジネス vs レジャー、またはペットを連れて行くためにこのホテルを選んだレビュアーの数（集計）が得られます。これはホテルを推薦する際に有用な情報です。

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

最後に、現在の状態のデータセットを新しい名前で保存します。

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## 感情分析の操作

この最終セクションでは、レビュー列に感情分析を適用し、結果をデータセットに保存します。

## 演習: フィルタリングされたデータの読み込みと保存

ここでは、前のセクションで保存したフィルタリング済みデータセットを読み込むことに注意してください。**元のデータセットではありません**。

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

ネガティブレビューとポジティブレビュー列で感情分析を実行すると、時間がかかる可能性があります。高速なCPUを搭載したテスト用ノートパソコンでテストしたところ、使用した感情分析ライブラリによって12〜14分かかりました。これは（比較的）長い時間なので、速度を向上させる方法を調査する価値があります。

ストップワード（文の感情を変えない一般的な英単語）を削除するのが最初のステップです。これらを削除することで、感情分析の速度が向上する可能性がありますが、精度が低下することはありません（ストップワードは感情に影響を与えませんが、分析を遅くします）。

最も長いネガティブレビューは395語でしたが、ストップワードを削除した後は195語になりました。

ストップワードの削除も高速な操作であり、515,000行の2つのレビュー列からストップワードを削除するのにテストデバイスでは3.3秒かかりました。デバイスのCPU速度、RAM、SSDの有無、その他の要因によって、若干の時間差が生じる可能性があります。この操作が感情分析の時間を改善するのであれば、実行する価値があります。

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
レビューのネガティブおよびポジティブな列に対して感情分析を計算し、その結果を新しい2つの列に保存してください。感情のテストは、同じレビューに対するレビュアーのスコアと比較することになります。例えば、ネガティブレビューの感情が1（非常にポジティブな感情）で、ポジティブレビューの感情も1だとします。しかし、レビュアーがホテルに最低スコアを付けていた場合、レビューのテキストがスコアと一致していないか、感情分析ツールが正しく感情を認識できなかった可能性があります。いくつかの感情スコアが完全に間違っていることを予想してください。その理由が説明可能な場合もあります。例えば、レビューが非常に皮肉的である場合、「もちろん暖房のない部屋で寝るのが大好きでした」といったレビューがポジティブな感情と認識されることがありますが、人間が読めばそれが皮肉であることがわかります。

NLTKは学習用にさまざまな感情分析ツールを提供しており、それらを置き換えて感情がより正確かどうかを確認することができます。ここではVADER感情分析が使用されています。

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

プログラム内で感情を計算する準備ができたら、以下のように各レビューに適用できます。

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

私のコンピュータでは約120秒かかりますが、コンピュータによって異なります。結果を印刷して感情がレビューと一致しているか確認したい場合は以下を使用してください。

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

チャレンジで使用する前にファイルで最後に行うべきことは、保存することです！また、新しい列をすべて並べ替えて、扱いやすいようにすることを検討してください（人間にとっては見た目の変更です）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)全体を実行してください（[フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)を実行してHotel_Reviews_Filtered.csvファイルを生成した後）。

手順を振り返ると以下の通りです：

1. 元のデータセットファイル **Hotel_Reviews.csv** は、前のレッスンで[エクスプローラーノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)を使用して調査されました。
2. Hotel_Reviews.csv は[フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)によってフィルタリングされ、**Hotel_Reviews_Filtered.csv** が生成されました。
3. Hotel_Reviews_Filtered.csv は[感情分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)によって処理され、**Hotel_Reviews_NLP.csv** が生成されました。
4. NLPチャレンジで**Hotel_Reviews_NLP.csv**を使用してください。

### 結論

開始時点では、列とデータがあるデータセットを持っていましたが、そのすべてが検証または使用できるわけではありませんでした。データを調査し、必要のないものをフィルタリングし、タグを有用なものに変換し、独自の平均値を計算し、感情列を追加し、自然言語テキストを処理することについて興味深いことを学んだはずです。

## [講義後のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## チャレンジ

データセットの感情を分析したら、このカリキュラムで学んだ戦略（例えばクラスタリング）を使用して、感情に関するパターンを特定できるか試してみてください。

## 復習と自己学習

[このLearnモジュール](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)を受講して、さらに学び、異なるツールを使用してテキストの感情を探索してください。

## 課題

[別のデータセットを試してみる](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。