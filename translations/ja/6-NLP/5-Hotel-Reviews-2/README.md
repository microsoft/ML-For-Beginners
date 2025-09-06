<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T09:43:24+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ja"
}
-->
# ホテルレビューを用いた感情分析

データセットを詳細に調査した後は、列をフィルタリングし、NLP技術を使用してホテルに関する新たな洞察を得る時です。

## [事前講義クイズ](https://ff-quizzes.netlify.app/en/ml/)

### フィルタリングと感情分析の操作

おそらく気づいたと思いますが、このデータセットにはいくつかの問題があります。一部の列には無意味な情報が含まれており、他の列は正確性に疑問があります。仮に正確だとしても、それがどのように計算されたのか不明であり、自分自身の計算で独立して検証することができません。

## 演習: もう少しデータを処理する

データをもう少しクリーンにしましょう。後で役立つ列を追加し、他の列の値を変更し、特定の列を完全に削除します。

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
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. ホテルのメタレビュー列を処理する

   1. `Additional_Number_of_Scoring` を削除する

   2. `Total_Number_of_Reviews` を、そのホテルに実際に含まれるレビューの総数に置き換える

   3. `Average_Score` を独自に計算したスコアに置き換える

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. レビュー列を処理する

   1. `Review_Total_Negative_Word_Counts`、`Review_Total_Positive_Word_Counts`、`Review_Date`、`days_since_review` を削除する

   2. `Reviewer_Score`、`Negative_Review`、`Positive_Review` はそのまま保持する

   3. `Tags` は一時的に保持する

      - 次のセクションでタグに対して追加のフィルタリング操作を行い、その後タグを削除します

4. レビュアー列を処理する

   1. `Total_Number_of_Reviews_Reviewer_Has_Given` を削除する

   2. `Reviewer_Nationality` を保持する

### タグ列

`Tag` 列は問題があります。この列にはリスト（テキスト形式）が格納されており、順序やサブセクションの数が常に同じではありません。このため、515,000行、1,427のホテルがあり、それぞれがレビュアーに異なる選択肢を提供しているため、人間が興味のある正しいフレーズを特定するのは困難です。ここでNLPが役立ちます。テキストをスキャンして最も一般的なフレーズを見つけ、それらをカウントすることができます。

残念ながら、単語単位ではなく、複数単語のフレーズ（例：*Business trip*）に興味があります。このようなデータ（6,762,646語）に対して複数単語の頻度分布アルゴリズムを実行すると、非常に多くの時間がかかる可能性があります。しかし、データを見ずにそれが必要なコストだと判断するのは早計です。ここで探索的データ分析が役立ちます。例えば、`[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` のようなタグのサンプルを見た場合、処理を大幅に削減できる可能性があるかどうかを考え始めることができます。幸いにも、それは可能です。ただし、まずは興味のあるタグを確認するためにいくつかのステップを踏む必要があります。

### タグのフィルタリング

このデータセットの目的は、感情や列を追加して、最適なホテルを選ぶのに役立てることです（自分自身のため、またはホテル推薦ボットを作るよう依頼されたクライアントのため）。タグが最終的なデータセットで有用かどうかを自問する必要があります。以下は一つの解釈です（他の目的でデータセットが必要な場合、異なるタグが選択に含まれる可能性があります）：

1. 旅行の種類は関連性があり、保持すべき
2. ゲストグループの種類は重要であり、保持すべき
3. ゲストが宿泊した部屋、スイート、スタジオの種類は無関係（どのホテルも基本的に同じ部屋を持っている）
4. レビューが送信されたデバイスは無関係
5. 宿泊日数は、長期滞在がホテルを気に入ったことを示す場合に関連性があるかもしれないが、可能性は低く、おそらく無関係

要約すると、**2種類のタグを保持し、他を削除する**ということです。

まず、タグをカウントする前に、より良い形式にする必要があります。つまり、角括弧や引用符を削除する必要があります。これを行う方法はいくつかありますが、データ量が多いため、最速の方法を選びたいところです。幸いにも、pandasにはこれらのステップを簡単に実行する方法があります。

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

各タグは次のようになります：`Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`。

次に問題が発生します。一部のレビュー（行）には5列、他には3列、6列のものもあります。これはデータセットの作成方法によるもので、修正が困難です。各フレーズの頻度をカウントしたいのですが、レビューごとに順序が異なるため、カウントが正確でない可能性があり、ホテルが本来得るべきタグを割り当てられない場合があります。

しかし、この異なる順序を逆に利用することができます。各タグは複数単語で構成されていますが、カンマで区切られています！最も簡単な方法は、6つの一時的な列を作成し、各タグをその順序に対応する列に挿入することです。その後、6つの列を1つの大きな列にマージし、`value_counts()` メソッドを実行します。これを出力すると、2,428個のユニークなタグがあることがわかります。以下はその一部です：

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

`Submitted from a mobile device` のような一般的なタグの一部は役に立たないため、フレーズの出現回数をカウントする前に削除するのが賢明かもしれませんが、操作が非常に高速であるため、それらを残して無視することもできます。

### 宿泊日数タグの削除

これらのタグを削除するのが最初のステップです。これにより、考慮すべきタグの総数が若干減少します。データセットから削除するのではなく、レビューのデータセットでカウント/保持する値として考慮しないだけです。

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

部屋、スイート、スタジオ、アパートメントなどの種類は非常に多様です。これらはほぼ同じ意味を持ち、関連性がないため、考慮から除外します。

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

最後に、これは非常に簡単な処理で（ほとんど処理を必要としませんでした）、以下の**有用な**タグが残ります：

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

`Travellers with friends` は `Group` とほぼ同じ意味であると考えられるため、上記のように2つを統合するのが妥当です。正しいタグを特定するコードは[Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)にあります。

最終ステップは、これらのタグごとに新しい列を作成することです。そして、各レビュー行について、`Tag` 列が新しい列のいずれかに一致する場合は1を追加し、一致しない場合は0を追加します。その結果、例えばビジネス目的で選ばれたホテルの数や、レジャー目的で選ばれたホテルの数などを集計することができ、ホテルを推薦する際に有用な情報となります。

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

この最終セクションでは、レビュー列に感情分析を適用し、その結果をデータセットに保存します。

## 演習: フィルタリングされたデータの読み込みと保存

ここでは、元のデータセットではなく、前のセクションで保存したフィルタリング済みのデータセットを読み込みます。

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

ネガティブおよびポジティブレビュー列で感情分析を実行すると、時間がかかる可能性があります。高速なCPUを搭載したテスト用ノートPCでテストしたところ、使用する感情分析ライブラリによって12～14分かかりました。これは（比較的）長い時間なので、速度を向上させる方法を検討する価値があります。

最初のステップとして、ストップワード（英語の一般的な単語で、文の感情に影響を与えないもの）を削除します。これにより、感情分析の速度が向上するはずですが、精度が低下することはありません（ストップワードは感情に影響を与えませんが、分析を遅くします）。

最も長いネガティブレビューは395語でしたが、ストップワードを削除すると195語になりました。

ストップワードの削除も高速な操作であり、515,000行の2つのレビュー列からストップワードを削除するのに3.3秒かかりました。デバイスのCPU速度、RAM、SSDの有無、その他の要因によって若干の違いがあるかもしれませんが、操作が短時間で済むため、感情分析の時間が改善されるのであれば実行する価値があります。

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

次に、ネガティブおよびポジティブレビュー列の感情分析を計算し、その結果を2つの新しい列に保存します。感情のテストは、同じレビューに対するレビュアースコアと比較することです。例えば、感情分析がネガティブレビューの感情を1（非常にポジティブな感情）と判断し、ポジティブレビューの感情も1と判断したが、レビュアーがホテルに最低スコアを付けた場合、レビューのテキストがスコアと一致していないか、感情分析ツールが感情を正しく認識できなかった可能性があります。一部の感情スコアが完全に間違っていることを予想すべきであり、それはしばしば説明可能です。例えば、レビューが非常に皮肉的である場合（「もちろん、暖房のない部屋で寝るのが大好きでした」）は、感情分析ツールがそれをポジティブな感情と判断するかもしれませんが、人間が読めば皮肉であることがわかります。
NLTKは、さまざまな感情分析ツールを提供しており、それらを置き換えて感情分析の精度が向上するかどうかを確認することができます。ここではVADER感情分析を使用しています。

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

プログラム内で感情を計算する準備が整ったら、以下のように各レビューに適用することができます。

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

これには私のコンピュータで約120秒かかりますが、コンピュータによって異なります。結果を印刷して、感情がレビューと一致しているか確認したい場合は以下を実行してください。

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

チャレンジでファイルを使用する前に最後に行うべきことは、ファイルを保存することです！また、新しい列をすべて並べ替えて、人間が扱いやすいようにすることを検討してください（見た目の変更です）。

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)のコード全体を実行する必要があります（[フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)を実行してHotel_Reviews_Filtered.csvファイルを生成した後）。

手順を振り返ると以下の通りです：

1. 元のデータセットファイル **Hotel_Reviews.csv** は、前のレッスンで[エクスプローラーノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)を使用して調査されました。
2. Hotel_Reviews.csv は[フィルタリングノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)によってフィルタリングされ、**Hotel_Reviews_Filtered.csv** が生成されます。
3. Hotel_Reviews_Filtered.csv は[感情分析ノートブック](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)によって処理され、**Hotel_Reviews_NLP.csv** が生成されます。
4. 以下のNLPチャレンジでHotel_Reviews_NLP.csvを使用します。

### 結論

最初は列とデータが含まれたデータセットを持っていましたが、そのすべてが検証または使用できるわけではありませんでした。データを調査し、不要な部分をフィルタリングし、タグを有用なものに変換し、独自の平均値を計算し、感情に関する列を追加しました。そして、自然言語テキストを処理することについて興味深いことを学べたのではないでしょうか。

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## チャレンジ

感情分析が完了したデータセットを使用して、このカリキュラムで学んだ戦略（例えばクラスタリング）を活用し、感情に関するパターンを特定してみてください。

## 復習と自己学習

[このLearnモジュール](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott)を受講して、さらに学び、異なるツールを使用してテキストの感情を探求してください。

## 課題

[別のデータセットを試してみる](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期すよう努めておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文を公式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当方は一切の責任を負いません。