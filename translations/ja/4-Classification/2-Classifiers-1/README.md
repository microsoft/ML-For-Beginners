<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9579f42e3ff5114c58379cc9e186a828",
  "translation_date": "2025-09-03T23:50:16+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "ja"
}
-->
# 料理分類器 1

このレッスンでは、前回のレッスンで保存した、料理に関するバランスの取れたクリーンなデータセットを使用します。

このデータセットを使って、さまざまな分類器を用いて_材料のグループに基づいて特定の国の料理を予測_します。その過程で、分類タスクにアルゴリズムを活用する方法についてさらに学びます。

## [講義前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/21/)
# 準備

[レッスン1](../1-Introduction/README.md)を完了している場合、これら4つのレッスンのために、ルートの`/data`フォルダに_cleaned_cuisines.csv_ファイルが存在することを確認してください。

## 演習 - 国の料理を予測する

1. このレッスンの_notebook.ipynb_フォルダで作業し、そのファイルとPandasライブラリをインポートします:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    データは以下のように見えます:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 次に、いくつかのライブラリをインポートします:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X座標とy座標を2つのデータフレームに分割してトレーニング用に準備します。`cuisine`をラベルのデータフレームとして使用します:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    以下のように表示されます:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0`列と`cuisine`列を`drop()`を使って削除します。残りのデータをトレーニング可能な特徴として保存します:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    特徴は以下のように見えます:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

これでモデルのトレーニングを開始する準備が整いました！

## 分類器の選択

データがクリーンでトレーニングの準備が整ったら、どのアルゴリズムを使用するかを決定する必要があります。

Scikit-learnでは分類が教師あり学習のカテゴリに分類されており、その中で多くの分類方法が提供されています。[その種類](https://scikit-learn.org/stable/supervised_learning.html)は最初は非常に多くて混乱するかもしれません。以下の方法はすべて分類技術を含んでいます:

- 線形モデル
- サポートベクターマシン
- 確率的勾配降下法
- 最近傍法
- ガウス過程
- 決定木
- アンサンブル法（投票分類器）
- マルチクラスおよびマルチ出力アルゴリズム（マルチクラス分類、マルチラベル分類、マルチクラス・マルチ出力分類）

> [ニューラルネットワークを使用してデータを分類する](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)こともできますが、このレッスンの範囲外です。

### どの分類器を選ぶべきか？

では、どの分類器を選ぶべきでしょうか？多くの場合、いくつか試して良い結果を探すのがテストの方法です。Scikit-learnは、作成されたデータセットでKNeighbors、SVC（2種類）、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB、QuadraticDiscriminationAnalysisを比較し、結果を視覚化する[比較](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)を提供しています:

![分類器の比較](../../../../translated_images/comparison.edfab56193a85e7fdecbeaa1b1f8c99e94adbf7178bed0de902090cf93d6734f.ja.png)
> Scikit-learnのドキュメントで生成されたプロット

> AutoMLはこれらの比較をクラウドで実行し、データに最適なアルゴリズムを選択できるため、この問題を簡単に解決します。[こちら](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)で試してみてください。

### より良いアプローチ

しかし、無作為に推測するよりも良い方法は、このダウンロード可能な[MLチートシート](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott)のアイデアに従うことです。ここでは、マルチクラス問題に対していくつかの選択肢があることがわかります:

![マルチクラス問題のチートシート](../../../../translated_images/cheatsheet.07a475ea444d22234cb8907a3826df5bdd1953efec94bd18e4496f36ff60624a.ja.png)
> Microsoftのアルゴリズムチートシートの一部、マルチクラス分類オプションを詳細に説明

✅ このチートシートをダウンロードして印刷し、壁に貼りましょう！

### 推論

制約を考慮して、さまざまなアプローチを推論してみましょう:

- **ニューラルネットワークは重すぎる**。クリーンだが最小限のデータセットを使用し、ノートブックを介してローカルでトレーニングを実行するため、ニューラルネットワークはこのタスクには重すぎます。
- **2クラス分類器は使用しない**。2クラス分類器は使用しないため、one-vs-allは除外されます。
- **決定木またはロジスティック回帰が適している可能性がある**。決定木やマルチクラスデータに対するロジスティック回帰が適しているかもしれません。
- **マルチクラスブースト決定木は異なる問題を解決する**。マルチクラスブースト決定木は非パラメトリックタスク、例えばランキングを構築するタスクに最適であり、私たちには役立ちません。

### Scikit-learnの使用

データを分析するためにScikit-learnを使用します。ただし、Scikit-learnでロジスティック回帰を使用する方法は多数あります。[渡すべきパラメータ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)を確認してください。

基本的に、Scikit-learnにロジスティック回帰を実行させる際に指定する必要がある重要なパラメータは、`multi_class`と`solver`の2つです。`multi_class`の値は特定の動作を適用します。solverの値は使用するアルゴリズムを指定します。すべてのsolverがすべての`multi_class`値と組み合わせられるわけではありません。

ドキュメントによると、マルチクラスの場合、トレーニングアルゴリズムは以下のように動作します:

- **one-vs-rest (OvR)方式を使用**する場合、`multi_class`オプションが`ovr`に設定されている場合
- **クロスエントロピー損失を使用**する場合、`multi_class`オプションが`multinomial`に設定されている場合（現在、`multinomial`オプションは‘lbfgs’, ‘sag’, ‘saga’, ‘newton-cg’ソルバーのみでサポートされています）

> 🎓 ここでの「方式」は、'ovr'（one-vs-rest）または'multinomial'のいずれかです。ロジスティック回帰は本来バイナリ分類をサポートするように設計されているため、これらの方式はマルチクラス分類タスクをより適切に処理できるようにします。[出典](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 「solver」は「最適化問題で使用するアルゴリズム」と定義されています。[出典](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)

Scikit-learnは、異なるデータ構造が引き起こす課題に対してsolverがどのように対応するかを説明するこの表を提供しています:

![ソルバー](../../../../translated_images/solvers.5fc648618529e627dfac29b917b3ccabda4b45ee8ed41b0acb1ce1441e8d1ef1.ja.png)

## 演習 - データを分割する

前回のレッスンでロジスティック回帰について学んだばかりなので、最初のトレーニング試行ではロジスティック回帰に焦点を当てることができます。
`train_test_split()`を呼び出してデータをトレーニングとテストグループに分割します:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 演習 - ロジスティック回帰を適用する

マルチクラスの場合、使用する_方式_と設定する_ソルバー_を選択する必要があります。LogisticRegressionを使用して、multi_classを`ovr`に設定し、**liblinear**ソルバーを使用してトレーニングします。

1. multi_classを`ovr`に設定し、solverを`liblinear`に設定したロジスティック回帰を作成します:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ デフォルトとしてよく設定される`lbfgs`のような別のソルバーを試してみてください
> 注意が必要な場合は、データを平坦化するためにPandasの[`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html)関数を使用してください。
精度は**80%以上**で良好です！

1. このモデルを試すには、データの1行（#50）をテストしてみてください：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    結果が出力されます：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 別の行番号を試して結果を確認してください。

1. さらに掘り下げて、この予測の精度を確認してみましょう：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    結果が出力されます - インド料理が最も可能性が高いと予測されています：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ なぜモデルがこれをインド料理だと確信しているのか説明できますか？

1. 回帰のレッスンで行ったように、分類レポートを出力して詳細を確認してください：

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀チャレンジ

このレッスンでは、クリーンアップしたデータを使用して、材料の組み合わせから国の料理を予測する機械学習モデルを構築しました。Scikit-learnが提供する分類オプションをじっくり読んでみてください。さらに「solver」の概念を掘り下げて、裏で何が行われているのかを理解してください。

## [講義後のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/22/)

## レビューと自己学習

ロジスティック回帰の数学的背景について、[このレッスン](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)を掘り下げて学んでください。
## 課題 

[solverについて学ぶ](assignment.md)

---

**免責事項**:  
この文書はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された文書を正式な情報源としてお考えください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当社は責任を負いません。