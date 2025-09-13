<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T09:40:57+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ja"
}
-->
# ホテルレビューによる感情分析 - データの処理

このセクションでは、前のレッスンで学んだ技術を使って、大規模なデータセットの探索的データ分析を行います。各列の有用性を十分に理解した後、以下を学びます：

- 不要な列を削除する方法
- 既存の列を基に新しいデータを計算する方法
- 最終的な課題で使用するために結果のデータセットを保存する方法

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

### はじめに

これまでに、テキストデータが数値データとは大きく異なることを学びました。人間が書いたり話したりしたテキストは、パターンや頻度、感情、意味を分析することができます。このレッスンでは、実際のデータセットと課題に取り組みます：**[ヨーロッパの515Kホテルレビューのデータ](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**。このデータセットは[CC0: パブリックドメインライセンス](https://creativecommons.org/publicdomain/zero/1.0/)のもとで提供されており、Booking.comから公開情報をスクレイピングして作成されました。データセットの作成者はJiashen Liuです。

### 準備

必要なもの：

* Python 3で.ipynbノートブックを実行する能力
* pandas
* NLTK、[ローカルにインストールしてください](https://www.nltk.org/install.html)
* Kaggleからダウンロード可能なデータセット [ヨーロッパの515Kホテルレビューのデータ](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)。解凍後約230MBです。このNLPレッスンに関連するルート`/data`フォルダに保存してください。

## 探索的データ分析

この課題では、感情分析とゲストレビューのスコアを使用してホテル推薦ボットを構築することを想定しています。使用するデータセットには、6つの都市にある1493の異なるホテルのレビューが含まれています。

Python、ホテルレビューのデータセット、そしてNLTKの感情分析を使用して以下を調べることができます：

* レビューで最も頻繁に使用される単語やフレーズは何か？
* ホテルを説明する公式の*タグ*はレビューのスコアと関連しているか？（例えば、*若い子供連れの家族*のレビューが*一人旅*よりもネガティブな場合、そのホテルは*一人旅*に適している可能性がある）
* NLTKの感情スコアはホテルレビューの数値スコアと一致しているか？

#### データセット

ダウンロードしてローカルに保存したデータセットを探索してみましょう。VS CodeやExcelのようなエディタでファイルを開いてください。

データセットのヘッダーは以下の通りです：

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

以下のようにグループ化すると、より簡単に確認できます：
##### ホテル関連の列

* `Hotel_Name`, `Hotel_Address`, `lat` (緯度), `lng` (経度)
  * *lat*と*lng*を使用して、Pythonでホテルの位置を示す地図をプロットすることができます（例えば、ネガティブレビューとポジティブレビューを色分けする）
  * Hotel_Addressはあまり有用ではないため、国名に置き換えてソートや検索を簡単にする予定です

**ホテルのメタレビュー関連の列**

* `Average_Score`
  * データセット作成者によると、この列は「ホテルの平均スコアで、過去1年間の最新コメントに基づいて計算されたもの」です。この計算方法は少し特殊ですが、現時点ではそのまま受け入れることにします。
  
  ✅ このデータの他の列を基に、別の方法で平均スコアを計算する方法を考えられますか？

* `Total_Number_of_Reviews`
  * このホテルが受け取ったレビューの総数 - この値がデータセット内のレビューを指しているかどうかはコードを書かないと明確ではありません。
* `Additional_Number_of_Scoring`
  * レビューのスコアが付けられたが、ポジティブまたはネガティブなレビューが書かれていない場合を意味します

**レビュー関連の列**

- `Reviewer_Score`
  - 小数点以下1桁までの数値で、最小値2.5から最大値10の間
  - なぜ2.5が最低スコアなのかは説明されていません
- `Negative_Review`
  - レビューが書かれていない場合、このフィールドには「**No Negative**」と記載されます
  - ネガティブレビューの列にポジティブな内容を書くレビューもあります（例：「このホテルには悪いところがありません」）
- `Review_Total_Negative_Word_Counts`
  - ネガティブな単語数が多いほど、スコアが低い傾向があります（感情分析を行わない場合）
- `Positive_Review`
  - レビューが書かれていない場合、このフィールドには「**No Positive**」と記載されます
  - ポジティブレビューの列にネガティブな内容を書くレビューもあります（例：「このホテルには良いところが全くありません」）
- `Review_Total_Positive_Word_Counts`
  - ポジティブな単語数が多いほど、スコアが高い傾向があります（感情分析を行わない場合）
- `Review_Date`と`days_since_review`
  - レビューの新鮮さや古さを測る指標として使用できます（古いレビューは、ホテルの管理が変わったり、改装が行われたり、プールが追加されたりして、正確性が低い可能性があります）
- `Tags`
  - レビューアが選択する短い記述で、ゲストのタイプ（例：一人旅や家族）、部屋のタイプ、滞在期間、レビューの提出方法などを示します。
  - 残念ながら、これらのタグを使用するのは問題がある場合があります。以下のセクションでその有用性について説明します。

**レビューア関連の列**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - 推薦モデルでの要因になる可能性があります。例えば、数百件のレビューを投稿しているレビューアがネガティブなレビューを投稿する傾向があるかどうかを判断できる場合。ただし、特定のレビューのレビューアは一意のコードで識別されておらず、レビューのセットにリンクすることはできません。100件以上のレビューを投稿しているレビューアは30人いますが、推薦モデルにどのように役立つかは不明です。
- `Reviewer_Nationality`
  - 一部の人々は、特定の国籍がポジティブまたはネガティブなレビューを投稿する傾向があると考えるかもしれません。しかし、こうしたモデルにそのような経験則を組み込む際には注意が必要です。これらは国籍（時には人種）に基づくステレオタイプであり、各レビューアは個々の経験に基づいてレビューを書いています。それが以前のホテル滞在、移動距離、個人的な気質など多くの要因を通じてフィルタリングされている可能性があります。レビューのスコアが国籍によるものだと考えるのは正当化が難しいです。

##### 例

| 平均スコア | レビュー総数 | レビューアスコア | ネガティブ<br />レビュー                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | ポジティブレビュー                 | タグ                                                                                      |
| ---------- | ------------ | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8        | 1945         | 2.5              | 現在この場所はホテルではなく建設現場です。長旅の後に休んでいる間、また部屋で仕事をしている間、早朝から一日中許容できない建設騒音に悩まされました。隣の部屋ではジャックハンマーを使った作業が一日中行われていました。部屋の変更を求めましたが、静かな部屋はありませんでした。さらに悪いことに、料金を過剰請求されました。早朝のフライトのために夕方にチェックアウトし、適切な請求書を受け取りましたが、翌日ホテルは予約価格を超える金額を無断で請求しました。ひどい場所です。ここを予約して自分を罰しないでください。 | 何もありません。ひどい場所です。避けてください。 | 出張                                カップル スタンダードダブルルーム 2泊滞在 |

このように、このゲストはホテルでの滞在に満足していませんでした。このホテルは7.8の良い平均スコアと1945件のレビューを持っていますが、このレビューアは2.5を付け、115語で滞在のネガティブな点を述べています。ポジティブレビューの列に何も書かなかった場合、ポジティブな点がなかったと推測できますが、実際には警告の7語を書いています。単語の数だけを数えると、レビューアの意図が歪んでしまう可能性があります。不思議なことに、2.5というスコアは混乱を招きます。滞在がそれほど悪かったのなら、なぜ全く点数を付けないのではなく2.5を付けたのでしょうか？データセットを詳しく調査すると、最低スコアは2.5であり、0ではないことがわかります。最高スコアは10です。

##### タグ

前述のように、最初は`Tags`を使用してデータを分類するアイデアは理にかなっているように思えます。しかし、これらのタグは標準化されていないため、あるホテルでは*シングルルーム*、*ツインルーム*、*ダブルルーム*というオプションがあり、別のホテルでは*デラックスシングルルーム*、*クラシッククイーンルーム*、*エグゼクティブキングルーム*というオプションがあります。これらは同じものかもしれませんが、バリエーションが多すぎて選択肢は以下のようになります：

1. すべての用語を単一の標準に変更しようとする。ただし、各ケースで変換方法が明確ではないため非常に困難です（例：*クラシックシングルルーム*は*シングルルーム*に対応しますが、*中庭または市街地ビュー付きスーペリアクイーンルーム*は対応が非常に難しい）。

1. NLPアプローチを取り、各ホテルに適用される*ソロ*、*ビジネストラベラー*、*若い子供連れの家族*などの特定の用語の頻度を測定し、それを推薦に組み込む。

タグは通常（ただし常にではありません）、*旅行の種類*、*ゲストの種類*、*部屋の種類*、*滞在日数*、*レビューが提出されたデバイスの種類*に対応する5〜6個のカンマ区切りの値を含む単一のフィールドです。ただし、一部のレビューアが各フィールドを埋めない場合（空白のままにする場合）、値は常に同じ順序で並んでいるわけではありません。

例として、*グループの種類*を取り上げます。このフィールドには`Tags`列で1025のユニークな可能性がありますが、残念ながらそのうちの一部しかグループを指していません（他は部屋の種類などです）。*家族*に言及するものだけをフィルタリングすると、結果には多くの*ファミリールーム*タイプの結果が含まれます。*with*という用語を含めると、つまり*Family with*の値をカウントすると、結果は改善され、515,000件の結果のうち80,000件以上が「若い子供連れの家族」または「年長の子供連れの家族」というフレーズを含んでいます。

これにより、`Tags`列は完全に無用ではないことがわかりますが、有用にするには少し作業が必要です。

##### ホテルの平均スコア

データセットにはいくつかの奇妙な点や矛盾があり、それを解明することはできませんが、モデルを構築する際に注意するためにここで説明します。もし解明できたら、ディスカッションセクションで教えてください！

データセットには、平均スコアとレビュー数に関連する以下の列があります：

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

このデータセットで最もレビュー数が多いホテルは*Britannia International Hotel Canary Wharf*で、515,000件中4789件のレビューがあります。しかし、このホテルの`Total_Number_of_Reviews`値を見ると9086です。レビューの多くがスコアのみでレビューがないと推測するかもしれません。その場合、`Additional_Number_of_Scoring`列の値を加えるべきです。その値は2682で、4789に加えると7471になりますが、それでも`Total_Number_of_Reviews`の9086には1615足りません。

`Average_Score`列を取ると、それがデータセット内のレビューの平均であると推測するかもしれませんが、Kaggleの説明では「*過去1年間の最新コメントに基づいて計算されたホテルの平均スコア*」とされています。それはあまり有用ではないように思えますが、データセット内のレビューのスコアに基づいて独自の平均を計算することができます。同じホテルを例に取ると、平均ホテルスコアは7.1とされていますが、データセット内のレビューアスコアの計算平均は6.8です。これは近いですが、同じ値ではありません。`Additional_Number_of_Scoring`レビューで与えられたスコアが平均を7.1に引き上げたと推測するしかありません。しかし、その主張をテストまたは証明する方法がないため、`Average_Score`、`Additional_Number_of_Scoring`、`Total_Number_of_Reviews`を使用することは難しいです。

さらに複雑なのは、レビュー数が2番目に多いホテルの計算平均スコアが8.12で、データセットの`Average_Score`は8.1です。この正しいスコアは偶然なのか、それとも最初のホテルが例外なのか？

これらのホテルが外れ値である可能性があり、ほとんどの値が一致している（ただし一部は何らかの理由で一致していない）可能性を考慮して、次にデータセット内の値を探索し、値の正しい使用法（または使用しない方法）を決定する短いプログラムを書きます。
> 🚨 注意事項  
>  
> このデータセットを扱う際には、テキストを自分で読んだり分析したりすることなく、テキストから何かを計算するコードを書くことになります。これがNLPの本質であり、人間が直接関与せずに意味や感情を解釈することです。しかし、ネガティブなレビューを読む可能性もあります。読む必要はないので、ぜひ避けてください。中には「天気が良くなかった」など、ホテルや誰にもコントロールできないような、くだらない、または無関係なネガティブなホテルレビューもあります。しかし、レビューには暗い側面も存在します。時には、ネガティブなレビューが人種差別的、性差別的、または年齢差別的な内容を含むことがあります。これは残念なことですが、公共のウェブサイトから収集されたデータセットでは予想されることです。一部のレビュアーは、不快感を覚えたり、不安になったり、心を痛めたりするようなレビューを残すことがあります。コードに感情を測定させる方が、自分で読んで気分を害するよりも良いでしょう。とはいえ、そのようなレビューを書く人は少数派ですが、それでも存在していることは事実です。
## 演習 - データ探索
### データの読み込み

データを視覚的に確認するのはここまでにして、コードを書いて答えを導き出しましょう！このセクションでは pandas ライブラリを使用します。最初のタスクは、CSVデータを正しく読み込めることを確認することです。pandas ライブラリには高速なCSVローダーがあり、結果は前のレッスンで学んだようにデータフレームに格納されます。今回読み込むCSVには50万行以上のデータが含まれていますが、列は17個だけです。pandas はデータフレームを操作するための強力な機能を提供しており、各行に対して操作を実行することも可能です。

このレッスンでは、コードスニペットとその説明、そして結果についての議論が含まれます。コードは付属の _notebook.ipynb_ を使用してください。

まずは使用するデータファイルを読み込んでみましょう：

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

データが読み込まれたら、いくつかの操作を実行できます。このコードを次のセクションのプログラムの冒頭に置いてください。

## データの探索

今回のデータはすでに*クリーン*な状態です。つまり、英語以外の文字が含まれておらず、アルゴリズムが英語文字のみを期待している場合に問題を引き起こすことはありません。

✅ データを処理する前にフォーマットを整える必要がある場合もありますが、今回はその必要はありません。もし非英語文字を処理する必要がある場合、どのように対応しますか？

データが読み込まれたら、コードを使って探索できることを確認してください。`Negative_Review` と `Positive_Review` の列に注目したくなるかもしれません。これらの列には、NLPアルゴリズムで処理する自然言語テキストが含まれています。しかし、感情分析に飛び込む前に、以下のコードを使用して、データセット内の値がpandasで計算した値と一致するか確認してください。

## データフレーム操作

このレッスンの最初のタスクは、以下の主張が正しいかどうかを確認するために、データフレームを調査するコードを書くことです（データフレームを変更しないでください）。

> 多くのプログラミングタスクと同様に、これを完了する方法は複数ありますが、最も簡単で理解しやすい方法を選ぶのが良いアドバイスです。特に後でコードを見直す際に理解しやすい方法が望ましいです。データフレームには包括的なAPIがあり、効率的に目的を達成する方法がしばしば存在します。

以下の質問をコーディングタスクとして扱い、解答を試みてください。解答を見る前に自分で挑戦してください。

1. 読み込んだデータフレームの*形状*（行数と列数）を出力してください。
2. レビュアーの国籍の頻度を計算してください：
   1. `Reviewer_Nationality` 列にいくつの異なる値があるか、それらは何か？
   2. データセットで最も一般的なレビュアーの国籍は何か（国名とレビュー数を出力してください）？
   3. 次に多い上位10の国籍とその頻度を出力してください。
3. 上位10のレビュアー国籍ごとに最も頻繁にレビューされたホテルは何か？
4. データセット内のホテルごとのレビュー数（ホテルの頻度）を計算してください。
5. データセット内の各ホテルのレビュースコアの平均を計算し、新しい列 `Calc_Average_Score` をデータフレームに追加してください。この列には計算された平均値が含まれます。
6. 四捨五入して小数点第1位までの `Average_Score` と `Calc_Average_Score` が一致するホテルはありますか？
   1. Series（行）を引数として受け取り、値を比較し、一致しない場合にメッセージを出力するPython関数を書いてみてください。その後、`.apply()` メソッドを使用して各行を処理してください。
7. `Negative_Review` 列の値が "No Negative" の行数を計算して出力してください。
8. `Positive_Review` 列の値が "No Positive" の行数を計算して出力してください。
9. `Positive_Review` 列の値が "No Positive" **かつ** `Negative_Review` 列の値が "No Negative" の行数を計算して出力してください。

### コード解答

1. 読み込んだデータフレームの*形状*（行数と列数）を出力してください。

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. レビュアーの国籍の頻度を計算してください：

   1. `Reviewer_Nationality` 列にいくつの異なる値があるか、それらは何か？
   2. データセットで最も一般的なレビュアーの国籍は何か（国名とレビュー数を出力してください）？

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

   3. 次に多い上位10の国籍とその頻度を出力してください。

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

3. 上位10のレビュアー国籍ごとに最も頻繁にレビューされたホテルは何か？

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

4. データセット内のホテルごとのレビュー数（ホテルの頻度）を計算してください。

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
   
   データセット内で*カウントされた*結果が `Total_Number_of_Reviews` の値と一致しないことに気付くかもしれません。この値がホテルの総レビュー数を表しているのか、それともスクレイピングされなかったレビューがあるのか、または他の計算によるものなのかは不明です。この不明確さのため、`Total_Number_of_Reviews` はモデルで使用されません。

5. データセット内の各ホテルのレビュースコアの平均を計算し、新しい列 `Calc_Average_Score` をデータフレームに追加してください。列 `Hotel_Name`、`Average_Score`、`Calc_Average_Score` を出力してください。

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

   データセットの平均値と計算された平均値が異なる理由について疑問に思うかもしれません。一部の値が一致する一方で、他の値に差異がある理由は分かりません。この場合、レビューのスコアを使用して自分で平均を計算するのが安全です。ただし、差異は通常非常に小さいです。以下はデータセットの平均値と計算された平均値の差が最も大きいホテルです：

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

   スコアの差が1を超えるホテルが1つしかないため、差異を無視して計算された平均スコアを使用しても問題ないでしょう。

6. `Negative_Review` 列の値が "No Negative" の行数を計算して出力してください。

7. `Positive_Review` 列の値が "No Positive" の行数を計算して出力してください。

8. `Positive_Review` 列の値が "No Positive" **かつ** `Negative_Review` 列の値が "No Negative" の行数を計算して出力してください。

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

## 別の方法

Lambdaを使用せずにアイテムをカウントし、行数をカウントするためにsumを使用する別の方法：

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

   `Negative_Review` と `Positive_Review` の列に "No Negative" と "No Positive" の値が両方含まれる行が127行あることに気付いたかもしれません。つまり、レビュアーはホテルに数値スコアを付けましたが、肯定的または否定的なレビューを書くことを拒否しました。この行数は非常に少ない（515738行中127行、つまり0.02%）ため、モデルや結果に特定の方向への偏りを与えることはないでしょう。しかし、レビューがない行が含まれているデータセットを予期していなかったかもしれないので、このような行を発見するためにデータを探索する価値があります。

データセットを探索したので、次のレッスンではデータをフィルタリングし、感情分析を追加します。

---
## 🚀チャレンジ

このレッスンでは、前のレッスンで見たように、データとその特性を理解することがいかに重要かを示しています。特にテキストベースのデータは慎重に調査する必要があります。さまざまなテキスト中心のデータセットを掘り下げ、モデルにバイアスや偏った感情を導入する可能性のある領域を発見できるか試してみてください。

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

[NLPに関するこの学習パス](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott)を受講して、音声やテキスト中心のモデルを構築する際に試すべきツールを発見してください。

## 課題

[NLTK](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された原文が正式な情報源と見なされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用に起因する誤解や誤認について、当社は一切の責任を負いません。