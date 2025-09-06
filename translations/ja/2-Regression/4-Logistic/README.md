<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-06T09:26:24+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "ja"
}
-->
# カテゴリー予測のためのロジスティック回帰

![ロジスティック回帰と線形回帰のインフォグラフィック](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

> ### [このレッスンはRでも利用可能です！](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## はじめに

回帰に関する最後のレッスンでは、基本的な「クラシック」な機械学習技術の一つであるロジスティック回帰について学びます。この技術を使用して、パターンを発見し、二値カテゴリーを予測することができます。このキャンディはチョコレートか否か？この病気は伝染性か否か？この顧客はこの商品を選ぶか否か？

このレッスンで学ぶ内容：

- データ可視化のための新しいライブラリ
- ロジスティック回帰の技術

✅ この[学習モジュール](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott)で、このタイプの回帰についての理解を深めましょう。

## 前提条件

かぼちゃデータを扱ったことで、このデータには扱える二値カテゴリーがあることがわかりました。それは `Color` です。

いくつかの変数を基にして、_特定のかぼちゃの色がオレンジ 🎃 か白 👻 かを予測する_ ロジスティック回帰モデルを構築してみましょう。

> なぜ回帰に関するレッスンで二値分類について話しているのでしょうか？それは言語的な便宜上の理由だけであり、ロジスティック回帰は[実際には分類方法](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)であり、線形ベースのものです。次のレッスンでは、データを分類する他の方法について学びます。

## 問題を定義する

今回の目的では、これを二値として表現します：'白' または '白ではない'。データセットには '縞模様' というカテゴリーもありますが、インスタンスが少ないため使用しません。データセットから欠損値を削除すると、このカテゴリーは消えてしまいます。

> 🎃 面白い事実として、白いかぼちゃを時々「ゴーストかぼちゃ」と呼びます。彫刻するのは難しいため、オレンジのかぼちゃほど人気はありませんが、見た目はとてもクールです！したがって、質問を次のように再構成することもできます：'ゴースト' か 'ゴーストではない'。👻

## ロジスティック回帰について

ロジスティック回帰は、以前学んだ線形回帰とはいくつかの重要な点で異なります。

[![初心者向け機械学習 - ロジスティック回帰の理解](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "初心者向け機械学習 - ロジスティック回帰の理解")

> 🎥 上の画像をクリックすると、ロジスティック回帰の概要を説明する短い動画が視聴できます。

### 二値分類

ロジスティック回帰は線形回帰と同じ機能を提供しません。前者は二値カテゴリー（「白か白ではない」）についての予測を提供しますが、後者は連続値を予測することができます。例えば、かぼちゃの産地と収穫時期を基にして、_価格がどれだけ上昇するか_ を予測することができます。

![かぼちゃ分類モデル](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> インフォグラフィック作成者：[Dasani Madipalli](https://twitter.com/dasani_decoded)

### 他の分類

ロジスティック回帰には、他にも多項式や順序型などの種類があります：

- **多項式**: 複数のカテゴリーを持つ場合 - 「オレンジ、白、縞模様」。
- **順序型**: 順序付けられたカテゴリーを持つ場合。例えば、かぼちゃのサイズ（mini, sm, med, lg, xl, xxl）を論理的に順序付ける場合に役立ちます。

![多項式回帰 vs 順序型回帰](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### 変数は相関している必要はない

線形回帰がより相関のある変数でうまく機能することを覚えていますか？ロジスティック回帰はその逆で、変数が一致している必要はありません。このデータには相関が弱い変数があるため、適しています。

### 多くのクリーンなデータが必要

ロジスティック回帰は、データが多いほど正確な結果を提供します。私たちの小さなデータセットはこのタスクには最適ではないため、その点を考慮してください。

[![初心者向け機械学習 - ロジスティック回帰のためのデータ分析と準備](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "初心者向け機械学習 - ロジスティック回帰のためのデータ分析と準備")

> 🎥 上の画像をクリックすると、線形回帰のためのデータ準備についての短い動画が視聴できます。

✅ ロジスティック回帰に適したデータの種類について考えてみましょう。

## 演習 - データを整理する

まず、データを少し整理し、欠損値を削除していくつかの列を選択します：

1. 次のコードを追加してください：

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    新しいデータフレームを確認することもできます：

    ```python
    pumpkins.info
    ```

### 可視化 - カテゴリカルプロット

これまでに、かぼちゃデータを[スターターノートブック](../../../../2-Regression/4-Logistic/notebook.ipynb)に読み込み、いくつかの変数を含むデータセットを保持するように整理しました。ノートブックでデータフレームを可視化するために、以前使用したMatplotlibを基にした異なるライブラリ：[Seaborn](https://seaborn.pydata.org/index.html) を使用してみましょう。

Seabornはデータを可視化するための便利な方法を提供します。例えば、`Variety` と `Color` のデータ分布をカテゴリカルプロットで比較することができます。

1. `catplot` 関数を使用して、かぼちゃデータ `pumpkins` を指定し、各かぼちゃカテゴリー（オレンジまたは白）の色マッピングを指定してプロットを作成します：

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![データのグリッド表示](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    データを観察することで、`Color` データが `Variety` にどのように関連しているかを確認できます。

    ✅ このカテゴリカルプロットを基に、どのような興味深い探索が考えられますか？

### データ前処理：特徴量とラベルのエンコーディング

かぼちゃデータセットのすべての列には文字列値が含まれています。人間にとってカテゴリカルデータは直感的ですが、機械にとってはそうではありません。機械学習アルゴリズムは数値データでうまく機能します。そのため、エンコーディングはデータ前処理フェーズで非常に重要なステップです。これにより、カテゴリカルデータを数値データに変換し、情報を失うことなく処理できます。適切なエンコーディングは良いモデルの構築につながります。

特徴量エンコーディングには主に2つのタイプがあります：

1. 順序型エンコーダー：順序型変数に適しており、カテゴリカル変数のデータが論理的な順序に従う場合に使用します。例えば、データセットの `Item Size` 列です。このエンコーダーは各カテゴリーを数値で表し、その数値は列内のカテゴリーの順序を示します。

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. カテゴリカルエンコーダー：名義変数に適しており、カテゴリカル変数のデータが論理的な順序に従わない場合に使用します。例えば、データセットの `Item Size` 以外のすべての特徴量です。これはワンホットエンコーディングであり、各カテゴリーがバイナリ列で表されます。エンコードされた変数は、かぼちゃがその `Variety` に属している場合は1、そうでない場合は0になります。

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

次に、`ColumnTransformer` を使用して複数のエンコーダーを1つのステップにまとめ、適切な列に適用します。

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

一方、ラベルをエンコードするためには、scikit-learnの `LabelEncoder` クラスを使用します。このクラスはラベルを正規化し、0から n_classes-1（ここでは0と1）の値のみを含むようにします。

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

特徴量とラベルをエンコードしたら、それらを新しいデータフレーム `encoded_pumpkins` に統合できます。

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ `Item Size` 列に順序型エンコーダーを使用する利点は何ですか？

### 変数間の関係を分析する

データを前処理した後、特徴量とラベルの関係を分析して、モデルが特徴量を基にラベルをどれだけうまく予測できるかのアイデアをつかむことができます。
この種の分析を行う最良の方法はデータをプロットすることです。再びSeabornの `catplot` 関数を使用して、`Item Size`、`Variety`、`Color` の関係をカテゴリカルプロットで可視化します。データをより良くプロットするために、エンコードされた `Item Size` 列と未エンコードの `Variety` 列を使用します。

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```

![データのカテゴリカルプロット](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### スウォームプロットを使用する

`Color` は二値カテゴリー（白または白ではない）であるため、可視化には「[特化したアプローチ](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)」が必要です。このカテゴリーと他の変数との関係を可視化する他の方法もあります。

Seabornプロットを使用して変数を並べて可視化できます。

1. 値の分布を示す「スウォーム」プロットを試してみましょう：

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![データのスウォームプロット](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**注意**: 上記のコードは警告を生成する可能性があります。Seabornが大量のデータポイントをスウォームプロットに表現するのに失敗するためです。解決策として、マーカーのサイズを 'size' パラメータで減少させることができます。ただし、これによりプロットの読みやすさが影響を受ける可能性があります。

> **🧮 数学を見てみよう**
>
> ロジスティック回帰は「最大尤度」の概念に基づいており、[シグモイド関数](https://wikipedia.org/wiki/Sigmoid_function)を使用します。プロット上の「シグモイド関数」は「S」字型の形状をしています。この関数は値を取り、それを0から1の間のどこかにマッピングします。その曲線は「ロジスティック曲線」とも呼ばれます。その公式は次のようになります：
>
> ![ロジスティック関数](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> シグモイドの中点はxの0点に位置し、Lは曲線の最大値、kは曲線の急峻さを示します。この関数の結果が0.5以上の場合、そのラベルは二値選択の「1」として分類されます。それ以外の場合は「0」として分類されます。

## モデルを構築する

Scikit-learnでこの二値分類を見つけるモデルを構築するのは驚くほど簡単です。

[![初心者向け機械学習 - データ分類のためのロジスティック回帰](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "初心者向け機械学習 - データ分類のためのロジスティック回帰")

> 🎥 上の画像をクリックすると、線形回帰モデルの構築についての短い動画が視聴できます。

1. 分類モデルで使用したい変数を選択し、`train_test_split()` を呼び出してトレーニングセットとテストセットを分割します：

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. 次に、トレーニングデータを使用して `fit()` を呼び出し、モデルをトレーニングし、その結果を出力します：

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    モデルのスコアボードを確認してください。約1000行のデータしかないことを考えると、悪くありません：

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## 混同行列による理解の向上

上記の項目を印刷してスコアボードレポート[用語](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)を取得することができますが、[混同行列](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)を使用することでモデルをより簡単に理解できるかもしれません。

> 🎓 「[混同行列](https://wikipedia.org/wiki/Confusion_matrix)」（または「エラーマトリックス」）は、モデルの真の陽性と偽の陽性、真の陰性と偽の陰性を表す表であり、予測の精度を評価します。

1. 混同行列を使用するには、`confusion_matrix()` を呼び出します：

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    モデルの混同行列を確認してください：

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

Scikit-learnでは、混同行列の行（軸0）は実際のラベル、列（軸1）は予測されたラベルを表します。

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

ここで何が起こっているのでしょうか？モデルがかぼちゃを二値カテゴリー、つまり「白」と「白ではない」に分類するよう求められたとします。

- モデルがかぼちゃを「白ではない」と予測し、実際に「白ではない」カテゴリーに属している場合、それを真の陰性（TN）と呼びます。これは左上の数字で示されます。
- モデルがかぼちゃを「白」と予測し、実際には「白ではない」カテゴリーに属している場合、それを偽の陰性（FN）と呼びます。これは左下の数字で示されます。
- モデルがかぼちゃを「白ではない」と予測し、実際には「白」カテゴリーに属している場合、それを偽の陽性（FP）と呼びます。これは右上の数字で示されます。
- モデルがかぼちゃを「白」と予測し、実際に「白」カテゴリーに属している場合、それを真の陽性（TP）と呼びます。これは右下の数字で示されます。

予想される通り、真の陽性（TP）と真の陰性（TN）の数が多く、偽の陽性（FP）と偽の陰性（FN）の数が少ない方が、モデルの性能が良いことを意味します。
混同行列は、精度と再現率とどのように関連しているのでしょうか？上記で印刷された分類レポートでは、精度（0.85）と再現率（0.67）が示されていました。

精度 = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

再現率 = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ Q: 混同行列によると、モデルの性能はどうでしたか？  
A: 悪くはありません。真のネガティブが多くありますが、いくつかの偽陰性も存在します。

混同行列のTP/TNとFP/FNのマッピングを使って、以前見た用語を再確認しましょう：

🎓 精度: TP/(TP + FP)  
取得されたインスタンスの中で関連性のあるインスタンスの割合（例: ラベルが正しく付けられたもの）

🎓 再現率: TP/(TP + FN)  
関連性のあるインスタンスのうち、取得されたインスタンスの割合（正しくラベル付けされているかどうかは問わない）

🎓 f1スコア: (2 * 精度 * 再現率)/(精度 + 再現率)  
精度と再現率の加重平均。最高値は1、最低値は0。

🎓 サポート:  
取得された各ラベルの出現回数

🎓 正確度: (TP + TN)/(TP + TN + FP + FN)  
サンプルに対して正確に予測されたラベルの割合。

🎓 マクロ平均:  
各ラベルの重み付けされていない平均値を計算し、ラベルの不均衡を考慮しない。

🎓 重み付き平均:  
各ラベルの平均値を計算し、サポート（各ラベルの真のインスタンス数）によって重み付けしてラベルの不均衡を考慮する。

✅ 偽陰性の数を減らしたい場合、どの指標を注視すべきか考えられますか？

## このモデルのROC曲線を視覚化する

[![初心者向け機械学習 - ROC曲線を使ったロジスティック回帰の性能分析](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "初心者向け機械学習 - ROC曲線を使ったロジスティック回帰の性能分析")

> 🎥 上の画像をクリックしてROC曲線の概要を短い動画で確認してください

もう一つの視覚化を行い、いわゆる「ROC」曲線を見てみましょう：

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

Matplotlibを使用して、モデルの[受信者動作特性（ROC）](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc)をプロットします。ROC曲線は、分類器の出力を真陽性と偽陽性の観点から見るためによく使用されます。「ROC曲線は通常、Y軸に真陽性率、X軸に偽陽性率を特徴とします。」したがって、曲線の急峻さと中間線と曲線の間の空間が重要です。曲線が急速に上昇し、線を越えることが望ましいです。今回の場合、最初に偽陽性があり、その後線が適切に上昇して越えています：

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

最後に、Scikit-learnの[`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score)を使用して、実際の「曲線下面積（AUC）」を計算します：

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```  
結果は `0.9749908725812341` です。AUCは0から1の範囲であり、予測が100%正確なモデルはAUCが1になります。この場合、モデルは「かなり良い」と言えます。

今後の分類に関するレッスンでは、モデルのスコアを改善する方法を学びます。しかし、今のところおめでとうございます！これで回帰に関するレッスンを完了しました！

---
## 🚀チャレンジ

ロジスティック回帰にはまだまだ学ぶべきことがあります！しかし、最良の学習方法は実験です。このタイプの分析に適したデータセットを見つけて、モデルを構築してみましょう。何を学びますか？ヒント: [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets)で興味深いデータセットを探してみてください。

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

[スタンフォードのこの論文](https://web.stanford.edu/~jurafsky/slp3/5.pdf)の最初の数ページを読んで、ロジスティック回帰の実用的な使用例について学びましょう。これまで学んだ回帰タスクの中で、どのタスクがどちらの回帰タイプに適しているかを考えてみてください。どちらが最適でしょうか？

## 課題

[この回帰を再試行する](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。元の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当方は一切の責任を負いません。