<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a683e1fe430bb0d4a10b68f6ca15e0a6",
  "translation_date": "2025-09-03T22:37:31+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "ja"
}
-->
# Scikit-learnを使った回帰モデルの構築: データの準備と可視化

![データ可視化インフォグラフィック](../../../../translated_images/data-visualization.54e56dded7c1a804d00d027543f2881cb32da73aeadda2d4a4f10f3497526114.ja.png)

インフォグラフィック作成者: [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [講義前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [このレッスンはRでも利用可能です！](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## はじめに

Scikit-learnを使った機械学習モデル構築に必要なツールが整った今、データに対して質問を投げかける準備ができました。データを扱い、機械学習ソリューションを適用する際には、適切な質問をすることでデータセットの可能性を最大限に引き出すことが非常に重要です。

このレッスンでは以下を学びます:

- モデル構築のためのデータ準備方法
- Matplotlibを使ったデータ可視化方法

## データに対して適切な質問をする

解きたい質問によって使用する機械学習アルゴリズムの種類が決まります。そして、得られる回答の質はデータの性質に大きく依存します。

このレッスンで提供される[データ](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv)を見てみましょう。この.csvファイルはVS Codeで開くことができます。ざっと見ただけでも、空白や文字列と数値データの混在があることがわかります。また、「Package」という奇妙な列があり、そのデータは「sacks」や「bins」などの値が混在しています。実際、このデータは少し混乱しています。

[![初心者向けML - データセットの分析とクリーニング方法](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "初心者向けML - データセットの分析とクリーニング方法")

> 🎥 上の画像をクリックすると、このレッスンのデータ準備を進める短い動画が視聴できます。

実際には、完全に準備されたデータセットがそのまま機械学習モデルを作成するために提供されることは非常に稀です。このレッスンでは、標準的なPythonライブラリを使用して生データセットを準備する方法を学びます。また、データを可視化するためのさまざまな技術も学びます。

## ケーススタディ: 「かぼちゃ市場」

このフォルダには、ルート`data`フォルダ内に[US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv)という.csvファイルが含まれており、都市ごとに分類されたかぼちゃ市場に関する1757行のデータが含まれています。このデータは、米国農務省が配布する[Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)から抽出された生データです。

### データの準備

このデータはパブリックドメインに属しています。USDAのウェブサイトから都市ごとに分割された多くのファイルとしてダウンロードできます。あまりにも多くのファイルを避けるために、すべての都市データを1つのスプレッドシートに結合しました。つまり、すでにデータを少し「準備」しています。次に、このデータを詳しく見てみましょう。

### かぼちゃデータ - 初期の結論

このデータについて何に気づきますか？すでに文字列、数値、空白、奇妙な値が混在していることがわかっています。

回帰技術を使用してこのデータにどのような質問をすることができますか？例えば、「特定の月に販売されるかぼちゃの価格を予測する」という質問はどうでしょうか。このデータを再度見てみると、このタスクに必要なデータ構造を作成するためにいくつか変更を加える必要があることがわかります。

## 演習 - かぼちゃデータを分析する

[Python Data Analysis](https://pandas.pydata.org/)の略であるPandasを使用して、このかぼちゃデータを分析・準備しましょう。Pandasはデータの形状を整えるのに非常に便利なツールです。

### まず、欠損日付を確認する

まず、欠損日付を確認する手順を実行します:

1. 日付を月形式に変換します（これらは米国の日付形式なので、`MM/DD/YYYY`形式です）。
2. 月を新しい列に抽出します。

Visual Studio Codeで_notebook.ipynb_ファイルを開き、スプレッドシートを新しいPandasデータフレームにインポートします。

1. `head()`関数を使用して最初の5行を表示します。

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ 最後の5行を表示するにはどの関数を使用しますか？

1. 現在のデータフレームに欠損データがあるかどうかを確認します:

    ```python
    pumpkins.isnull().sum()
    ```

    欠損データがありますが、今回のタスクには影響がないかもしれません。

1. データフレームを扱いやすくするために、必要な列だけを選択します。`loc`関数を使用して、元のデータフレームから行（最初のパラメータとして渡す）と列（2番目のパラメータとして渡す）のグループを抽出します。以下の例では、`:`は「すべての行」を意味します。

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### 次に、かぼちゃの平均価格を決定する

特定の月におけるかぼちゃの平均価格を決定する方法を考えてみましょう。このタスクにはどの列を選択しますか？ヒント: 3つの列が必要です。

解決策: `Low Price`と`High Price`列の平均を取って新しい`Price`列を作成し、`Date`列を月だけを表示するように変換します。幸いにも、上記のチェックによると、日付や価格に欠損データはありません。

1. 平均を計算するには、以下のコードを追加します:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ `print(month)`を使用してデータを確認することができます。

2. 変換したデータを新しいPandasデータフレームにコピーします:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    データフレームを印刷すると、きれいで整ったデータセットが表示され、新しい回帰モデルを構築する準備が整います。

### しかし、奇妙な点があります

`Package`列を見ると、かぼちゃはさまざまな構成で販売されています。一部は「1 1/9 bushel」単位で販売され、一部は「1/2 bushel」単位で販売され、かぼちゃ1個単位やポンド単位、大きな箱単位で販売されるものもあります。

> かぼちゃは一貫して重さを測るのが非常に難しいようです

元のデータを掘り下げると、`Unit of Sale`が「EACH」または「PER BIN」の場合、`Package`タイプはインチ単位、ビン単位、または「各」単位であることがわかります。かぼちゃは一貫して重さを測るのが非常に難しいようです。そのため、`Package`列に「bushel」という文字列が含まれるかぼちゃだけを選択してフィルタリングします。

1. 初期の.csvインポートの下にフィルタを追加します:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    データを印刷すると、バスケット単位で販売される約415行のデータだけが取得されていることがわかります。

### しかし、もう1つやるべきことがあります

バスケットの量が行ごとに異なることに気づきましたか？価格をバスケット単位で標準化する必要があります。そのため、価格を標準化するための計算を行います。

1. new_pumpkinsデータフレームを作成するブロックの後に以下の行を追加します:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308)によると、バスケットの重さは生産物の種類によって異なります。これは体積測定です。「例えば、トマトのバスケットは56ポンドの重さがあるとされています... 葉物や緑の野菜はスペースを多く取り、重さが少ないため、ほうれん草のバスケットは20ポンドしかありません。」これは非常に複雑です！バスケットからポンドへの変換を行うのではなく、バスケット単位で価格を設定しましょう。しかし、かぼちゃのバスケットについてのこの研究は、データの性質を理解することがいかに重要であるかを示しています！

これで、バスケット測定に基づいて単位ごとの価格を分析することができます。データをもう一度印刷すると、標準化されたデータが表示されます。

✅ 半バスケット単位で販売されるかぼちゃが非常に高価であることに気づきましたか？その理由を考えることができますか？ヒント: 小さなかぼちゃは大きなかぼちゃよりもはるかに高価です。おそらく、大きな空洞のあるパイ用かぼちゃ1個が占めるスペースに比べて、小さなかぼちゃがバスケットにたくさん詰め込まれるためです。

## 可視化戦略

データサイエンティストの役割の一部は、扱っているデータの質と性質を示すことです。そのため、データのさまざまな側面を示す興味深い可視化、プロット、グラフ、チャートを作成することがよくあります。この方法で、関係性やギャップを視覚的に示すことができます。

[![初心者向けML - Matplotlibを使ったデータ可視化方法](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "初心者向けML - Matplotlibを使ったデータ可視化方法")

> 🎥 上の画像をクリックすると、このレッスンのデータ可視化を進める短い動画が視聴できます。

可視化は、データに最も適した機械学習技術を決定するのにも役立ちます。例えば、線に沿うように見える散布図は、そのデータが線形回帰演習に適していることを示します。

Jupyterノートブックでうまく動作するデータ可視化ライブラリの1つが[Matplotlib](https://matplotlib.org/)です（前のレッスンでも見ました）。

> データ可視化の経験をさらに積むには、[これらのチュートリアル](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott)をご覧ください。

## 演習 - Matplotlibで実験する

作成した新しいデータフレームを表示する基本的なプロットを作成してみましょう。基本的な折れ線グラフは何を示しますか？

1. ファイルの上部でPandasインポートの下にMatplotlibをインポートします:

    ```python
    import matplotlib.pyplot as plt
    ```

1. ノートブック全体を再実行して更新します。
1. ノートブックの下部にセルを追加してデータをボックスとしてプロットします:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![価格と月の関係を示す散布図](../../../../translated_images/scatterplot.b6868f44cbd2051c6680ccdbb1510697d06a3ff6cd4abda656f5009c0ed4e3fc.ja.png)

    このプロットは役に立ちますか？何か驚くことはありますか？

    特に役に立つものではありません。単にデータを特定の月における点の広がりとして表示するだけです。

### 役立つものにする

役立つチャートを表示するには、通常データを何らかの方法でグループ化する必要があります。y軸に月を表示し、データが分布を示すプロットを作成してみましょう。

1. グループ化された棒グラフを作成するセルを追加します:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![価格と月の関係を示す棒グラフ](../../../../translated_images/barchart.a833ea9194346d769c77a3a870f7d8aee51574cd1138ca902e5500830a41cbce.ja.png)

    これはより役立つデータ可視化です！かぼちゃの最高価格は9月と10月に発生しているようです。これはあなたの予想に合っていますか？その理由は何ですか？

---

## 🚀チャレンジ

Matplotlibが提供するさまざまな種類の可視化を探求してみましょう。回帰問題に最も適した種類はどれですか？

## [講義後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## 復習と自己学習

データを可視化するさまざまな方法を調べてみましょう。利用可能なライブラリのリストを作成し、2D可視化と3D可視化など、特定のタスクに最適なものを記録してください。何を発見しますか？

## 課題

[可視化の探求](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。元の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当方は責任を負いません。