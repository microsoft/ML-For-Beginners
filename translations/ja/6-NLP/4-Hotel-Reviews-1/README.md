# ホテルレビューによる感情分析 - データの処理

このセクションでは、前のレッスンで学んだ技術を使って、大規模なデータセットの探索的データ分析を行います。各列の有用性を十分に理解した後、次のことを学びます:

- 不要な列の削除方法
- 既存の列を基に新しいデータを計算する方法
- 最終チャレンジで使用するために結果のデータセットを保存する方法

## [事前レクチャークイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### はじめに

これまで、テキストデータが数値データとは全く異なるものであることを学びました。人間が書いたり話したりしたテキストは、パターンや頻度、感情や意味を見つけるために分析することができます。このレッスンでは、実際のデータセットと実際のチャレンジに取り組みます: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** で、[CC0: Public Domainライセンス](https://creativecommons.org/publicdomain/zero/1.0/)が含まれています。このデータはBooking.comから公開ソースからスクレイピングされました。データセットの作成者はJiashen Liuです。

### 準備

必要なもの:

* Python 3を使用して.ipynbノートブックを実行できる能力
* pandas
* NLTK、[ローカルにインストールする必要があります](https://www.nltk.org/install.html)
* Kaggleで入手可能なデータセット [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解凍すると約230MBです。これをこれらのNLPレッスンに関連するルート `/data` フォルダーにダウンロードしてください。

## 探索的データ分析

このチャレンジでは、感情分析とゲストレビューのスコアを使用してホテル推薦ボットを構築することを前提としています。使用するデータセットには、6つの都市にある1493の異なるホテルのレビューが含まれています。

Python、ホテルレビューのデータセット、およびNLTKの感情分析を使用して次のことがわかります:

* レビューで最も頻繁に使用される単語やフレーズは何か？
* ホテルを説明する公式の *タグ* はレビューのスコアと相関しているか？（例えば、*Family with young children* のレビューが *Solo traveller* よりもネガティブなレビューが多い場合、そのホテルは *Solo travellers* に向いているかもしれません）
* NLTKの感情スコアはホテルレビューの数値スコアと一致するか？

#### データセット

ダウンロードしてローカルに保存したデータセットを探索してみましょう。VS CodeやExcelのようなエディタでファイルを開いてみてください。

データセットのヘッダーは次の通りです:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

ここでは、検査しやすいようにグループ化しています:
##### ホテル列

* `Hotel_Name`, `Hotel_Address`, `lat` (緯度), `lng` (経度)
  * *lat* と *lng* を使用して、Pythonでホテルの場所を示す地図をプロットできます（ネガティブレビューとポジティブレビューの色分けをすることも可能です）
  * Hotel_Address は明らかに有用ではないので、国に置き換えてソートや検索を容易にする予定です

**ホテルメタレビュー列**

* `Average_Score`
  * データセット作成者によると、この列は「過去1年の最新コメントに基づいて計算されたホテルの平均スコア」です。この方法でスコアを計算するのは珍しいですが、今のところそのまま受け入れることにします。
  
  ✅ 他の列に基づいて、平均スコアを計算する別の方法を考えられますか？

* `Total_Number_of_Reviews`
  * このホテルが受け取ったレビューの総数です。このデータセットのレビューに関するものかどうかは（コードを書かずに）明確ではありません。
* `Additional_Number_of_Scoring`
  * これはレビューのスコアが与えられたが、レビュアーによってポジティブまたはネガティブなレビューが書かれなかったことを意味します

**レビュー列**

- `Reviewer_Score`
  - 最小値と最大値の間で小数点以下1桁までの数値です
  - なぜ2.5が最低スコアなのかは説明されていません
- `Negative_Review`
  - レビュアーが何も書かなかった場合、このフィールドには「**No Negative**」と表示されます
  - レビュアーがネガティブレビュー欄にポジティブレビューを書くこともあります（例：「このホテルには悪いところが何もありません」）
- `Review_Total_Negative_Word_Counts`
  - ネガティブな単語数が多いほど、スコアは低くなります（感情をチェックしない場合）
- `Positive_Review`
  - レビュアーが何も書かなかった場合、このフィールドには「**No Positive**」と表示されます
  - レビュアーがポジティブレビュー欄にネガティブレビューを書くこともあります（例：「このホテルには全く良いところがありません」）
- `Review_Total_Positive_Word_Counts`
  - ポジティブな単語数が多いほど、スコアは高くなります（感情をチェックしない場合）
- `Review_Date` と `days_since_review`
  - レビューに新鮮さや古さの指標を適用することができます（古いレビューは、新しいレビューほど正確でないかもしれません。ホテルの管理が変わったり、改装が行われたり、プールが追加されたりするため）
- `Tags`
  - これはレビュアーが選択できる短い記述子で、ゲストの種類（例：一人旅や家族）、部屋の種類、滞在期間、レビューが提出されたデバイスの種類を示します。
  - ただし、これらのタグを使用するのは問題がある場合があります。以下のセクションでその有用性について説明します

**レビュアー列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 推奨モデルの要素になるかもしれません。例えば、数百のレビューを持つレビューアーがポジティブよりもネガティブなレビューを残す可能性が高いと判断できる場合。しかし、特定のレビューのレビュアーは一意のコードで識別されないため、一連のレビューにリンクすることはできません。100以上のレビューを持つレビュアーが30人いますが、これが推奨モデルにどのように役立つかは明確ではありません。
- `Reviewer_Nationality`
  - 一部の人々は、特定の国籍がポジティブまたはネガティブなレビューを残す傾向があると考えるかもしれませんが、これはモデルにそのような逸話的な見解を組み込む際には注意が必要です。これらは国や時には人種のステレオタイプであり、各レビュアーは彼らの経験に基づいてレビューを書いた個人です。彼らの国籍がレビューのスコアの理由であると考えるのは正当化するのが難しいです。

##### 例

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

このゲストは、このホテルでの滞在に満足していなかったことがわかります。ホテルは7.8の良い平均スコアと1945のレビューを持っていますが、このレビュアーは2.5を与え、滞在がいかにネガティブだったかを115語で書いています。ポジティブレビュー欄に何も書いていなければ、ポジティブな点がなかったと推測できますが、7語の警告を書いています。単語の数だけを数える代わりに、単語の意味や感情を分析しないと、レビュアーの意図が歪んでしまうかもしれません。奇妙なことに、2.5のスコアは混乱を招きます。なぜなら、そのホテル滞在がそれほど悪かったなら、なぜポイントを与えるのでしょうか？データセットを詳しく調査すると、最低スコアが2.5であり、0ではないことがわかります。最高スコアは10です。

##### タグ

前述のように、`Tags` を使用してデータを分類するアイデアは一見理にかなっているように見えます。しかし、これらのタグは標準化されていないため、あるホテルでは *Single room*、*Twin room*、*Double room* というオプションがあり、次のホテルでは *Deluxe Single Room*、*Classic Queen Room*、*Executive King Room* というオプションがあります。これらは同じものかもしれませんが、バリエーションが非常に多いため、選択肢は次のようになります:

1. すべての用語を単一の標準に変更する試み、これは非常に困難です。なぜなら、各ケースでの変換パスが明確ではないからです（例：*Classic single room* を *Single room* にマッピングするが、*Superior Queen Room with Courtyard Garden or City View* はマッピングが難しい）
1. NLPアプローチを取り、各ホテルに適用される *Solo*、*Business Traveller*、*Family with young kids* などの用語の頻度を測定し、それを推奨に組み込む

タグは通常（必ずしもそうではありませんが）、*Type of trip*、*Type of guests*、*Type of room*、*Number of nights*、*Type of device review was submitted on* に一致する5〜6のコンマ区切りの値のリストを含む単一のフィールドです。しかし、一部のレビュアーは各フィールドを埋めないことがあるため（空欄のままにすることがある）、値は常に同じ順序にはありません。

例えば、*Type of group* を取り上げます。このフィールドには `Tags` 列に1025の一意の可能性がありますが、そのうちの一部だけがグループに関連しています（いくつかは部屋の種類など）。*Family* に関連するものだけをフィルタリングすると、多くの *Family room* タイプの結果が含まれます。*with* という用語を含めると、*Family with* の値を数えると、結果は良くなり、515,000の結果のうち80,000以上が「Family with young children」または「Family with older children」を含んでいます。

これは、タグ列が完全に無用ではないが、役立つようにするには作業が必要であることを意味します。

##### ホテルの平均スコア

データセットにはいくつかの奇妙な点や不一致がありますが、モデルを構築する際にそれらに気付いているように、ここに示しています。もし解決方法がわかったら、ディスカッションセクションで教えてください！

データセットには、平均スコアとレビュー数に関連する次の列があります:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

このデータセットで最もレビューが多いホテルは *Britannia International Hotel Canary Wharf* で、515,000のレビューのうち4789件のレビューがあります。しかし、このホテルの `Total_Number_of_Reviews` の値を見ると、9086です。多くのスコアがレビューなしであると推測できますので、`Additional_Number_of_Scoring` 列の値を追加する必要があります。その値は2682で、4789に追加すると7471になりますが、`Total_Number_of_Reviews` の値よりも1615少ないです。

`Average_Score` 列を取ると、データセット内のレビューの平均であると推測できますが、Kaggleの説明には「*過去1年の最新コメントに基づいて計算されたホテルの平均スコア*」とあります。それはあまり役立たないように見えますが、データセット内のレビューのスコアに基づいて独自の平均を計算できます。同じホテルを例にとると、平均ホテルスコアは7.1とされていますが、データセット内のレビュアースコアの平均は6.8です。これは近いですが同じ値ではなく、`Additional_Number_of_Scoring` のレビューで与えられたスコアが平均を7.1に引き上げたと推測できますが、その主張をテストまたは証明する方法がないため、`Average_Score`、`Additional_Number_of_Scoring`、および `Total_Number_of_Reviews` の値を使用または信頼するのは難しいです。

さらに複雑なのは、レビュー数が2番目に多いホテルの計算された平均スコアが8.12で、データセットの `Average_Score` は8.1です。この正しいスコアは偶然の一致でしょうか、それとも最初のホテルの不一致でしょうか？

これらのホテルが外れ値である可能性があり、ほとんどの値が一致する（ただし、いくつかは何らかの理由で一致しない）ことを前提に、次にデータセットの値を探索し、値の正しい使用（または非使用）を決定するための短いプログラムを書きます。

> 🚨 注意点
>
> このデータセットを使用する際には、テキストを自分で読んだり分析したりすることなく、テキストから何かを計算するコードを書きます。これがNLPの本質であり、人間がそれを行わずに意味や感情を解釈することです。しかし、ネガティブなレビューを読む可能性があります。必要ないのであれば、読まないようにしましょう。中には「天気が良くなかった」など、ホテルや誰にもコントロールできないことを理由にした馬鹿げたネガティブなレビューもありますが、一部のレビューには人種差別、性差別、年齢差別が含まれていることもあります。これは残念ですが、公開されたウェブサイトからスクレイピングされたデータセットでは予想されることです。一部のレビュアーは、あなたが不快に感じたり、気分を害したりするようなレビューを残します。コードで感情を測定する方が、実際に読んで気分を害するよりも良いでしょう。とはいえ、そのようなことを書く人は少数ですが、それでも存在します。

## 演習 - データ探索
### データの読み込み

視覚的にデータを調べるのはここまでにして、コードを書いていくつかの答えを見つけましょう！このセクションでは、pandasライブラリを使用します。最初のタスクは、CSVデータを読み込んで表示できることを確認することです。pandasライブラリには高速なCSVローダーがあり、結果は前のレッスンのようにデータフレームに配置されます。読み込むCSVには50万行以上ありますが、列は17列だけです。pandasは、データ
rows have column `Positive_Review` values of "No Positive" 9. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ### Code answers 1. Print out the *shape* of the data frame you have just loaded (the shape is the number of rows and columns) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calculate the frequency count for reviewer nationalities: 1. How many distinct values are there for the column `Reviewer_Nationality` and what are they? 2. What reviewer nationality is the most common in the dataset (print country and number of reviews)? ```python
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
   ``` 3. What are the next top 10 most frequently found nationalities, and their frequency count? ```python
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
      ``` 3. What was the most frequently reviewed hotel for each of the top 10 most reviewer nationalities? ```python
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
   ``` 4. How many reviews are there per hotel (frequency count of hotel) in the dataset? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | You may notice that the *counted in the dataset* results do not match the value in `Total_Number_of_Reviews`. It is unclear if this value in the dataset represented the total number of reviews the hotel had, but not all were scraped, or some other calculation. `Total_Number_of_Reviews` is not used in the model because of this unclarity. 5. While there is an `Average_Score` column for each hotel in the dataset, you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel). Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. Print out the columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`. ```python
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
   ``` You may also wonder about the `Average_Score` value and why it is sometimes different from the calculated average score. As we can't know why some of the values match, but others have a difference, it's safest in this case to use the review scores that we have to calculate the average ourselves. That said, the differences are usually very small, here are the hotels with the greatest deviation from the dataset average and the calculated average: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | With only 1 hotel having a difference of score greater than 1, it means we can probably ignore the difference and use the calculated average score. 6. Calculate and print out how many rows have column `Negative_Review` values of "No Negative" 7. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" 8. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ```python
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
   ``` ## Another way Another way count items without Lambdas, and use sum to count the rows: ```python
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
   ``` You may have noticed that there are 127 rows that have both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review` respectively. That means that the reviewer gave the hotel a numerical score, but declined to write either a positive or negative review. Luckily this is a small amount of rows (127 out of 515738, or 0.02%), so it probably won't skew our model or results in any particular direction, but you might not have expected a data set of reviews to have rows with no reviews, so it's worth exploring the data to discover rows like this. Now that you have explored the dataset, in the next lesson you will filter the data and add some sentiment analysis. --- ## 🚀Challenge This lesson demonstrates, as we saw in previous lessons, how critically important it is to understand your data and its foibles before performing operations on it. Text-based data, in particular, bears careful scrutiny. Dig through various text-heavy datasets and see if you can discover areas that could introduce bias or skewed sentiment into a model. ## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Review & Self Study Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models. ## Assignment [NLTK](assignment.md)

**免責事項**:
この文書は機械ベースのAI翻訳サービスを使用して翻訳されています。正確さを期していますが、自動翻訳には誤りや不正確さが含まれる可能性があることをご理解ください。権威ある情報源としては、元の言語で書かれた原文を考慮すべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤訳について、当社は一切の責任を負いかねます。