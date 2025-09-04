<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2f88fbc741d792890ff2f1430fe0dae0",
  "translation_date": "2025-09-03T22:17:06+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "ja"
}
-->
# Scikit-learnを使った回帰モデルの構築: 4つの方法で回帰を学ぶ

![線形回帰と多項式回帰のインフォグラフィック](../../../../translated_images/linear-polynomial.5523c7cb6576ccab0fecbd0e3505986eb2d191d9378e785f82befcf3a578a6e7.ja.png)
> インフォグラフィック作成者: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [講義前のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/13/)

> ### [このレッスンはRでも利用可能です！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### はじめに

これまでに、回帰とは何かを学び、このレッスン全体で使用するカボチャの価格データセットを用いてサンプルデータを調査しました。また、Matplotlibを使ってデータを可視化しました。

これからは、機械学習における回帰についてさらに深く掘り下げていきます。可視化はデータを理解する助けになりますが、機械学習の真の力は「モデルのトレーニング」にあります。モデルは過去のデータを基にトレーニングされ、データの依存関係を自動的に捉え、新しいデータに対して予測を行うことができます。この新しいデータはモデルがこれまで見たことのないものです。

このレッスンでは、回帰の2つのタイプ、つまり「基本的な線形回帰」と「多項式回帰」について学びます。これらのモデルを使って、異なる入力データに基づいてカボチャの価格を予測することができます。

[![初心者向け機械学習 - 線形回帰の理解](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "初心者向け機械学習 - 線形回帰の理解")

> 🎥 上の画像をクリックすると、線形回帰の概要を説明する短い動画が再生されます。

> このカリキュラム全体を通じて、数学の知識がほとんどないことを前提とし、他分野から来た学生にも理解しやすいように努めています。注釈、🧮 数学の説明、図解、その他の学習ツールを活用して理解を深めてください。

### 前提条件

これまでに、調査対象のカボチャデータの構造について理解しているはずです。このレッスンの_notebook.ipynb_ファイルには、事前にロードされ、クリーニングされたデータが含まれています。このファイルでは、カボチャの価格がブッシェル単位で新しいデータフレームに表示されています。Visual Studio Codeのカーネルでこれらのノートブックを実行できることを確認してください。

### 準備

データをロードする目的を思い出してください。

- カボチャを購入するのに最適な時期はいつか？
- ミニチュアカボチャのケースの価格はどれくらいか？
- 半ブッシェルバスケットで購入するべきか、それとも1 1/9ブッシェルボックスで購入するべきか？

このデータをさらに掘り下げてみましょう。

前のレッスンでは、Pandasデータフレームを作成し、元のデータセットの一部を標準化してブッシェル単位の価格を取得しました。しかし、その結果、約400のデータポイントしか得られず、秋の月に限定されました。

このレッスンの付属ノートブックに事前にロードされたデータを確認してください。このデータは事前にロードされ、初期の散布図が月ごとのデータを示すようにプロットされています。データをさらにクリーニングすることで、データの性質についてもう少し詳細を得られるかもしれません。

## 線形回帰線

レッスン1で学んだように、線形回帰の目的は次のような線をプロットすることです：

- **変数間の関係を示す**。変数間の関係を明らかにする
- **予測を行う**。新しいデータポイントがその線に対してどこに位置するかを正確に予測する

**最小二乗回帰**では、このタイプの線を描くことが一般的です。「最小二乗」とは、回帰線の周囲にあるすべてのデータポイントを二乗して合計することを意味します。この最終的な合計が可能な限り小さいことが理想的です。なぜなら、エラーの数を少なくしたいからです。

これは、すべてのデータポイントからの累積距離が最小になるような線をモデル化したいからです。また、方向ではなく大きさに関心があるため、項を二乗してから加算します。

> **🧮 数学を見せて** 
> 
> この線、つまり「最適な線」は[方程式](https://en.wikipedia.org/wiki/Simple_linear_regression)で表すことができます：
> 
> ```
> Y = a + bX
> ```
>
> `X`は「説明変数」、`Y`は「従属変数」です。線の傾きは`b`で、`a`はY切片を指します。これは`X = 0`のときの`Y`の値です。
>
>![傾きを計算する](../../../../translated_images/slope.f3c9d5910ddbfcf9096eb5564254ba22c9a32d7acd7694cab905d29ad8261db3.ja.png)
>
> まず、傾き`b`を計算します。インフォグラフィック作成者: [Jen Looper](https://twitter.com/jenlooper)
>
> 言い換えると、カボチャデータの元の質問「月ごとにブッシェル単位でカボチャの価格を予測する」に関連して、`X`は価格を指し、`Y`は販売月を指します。
>
>![方程式を完成させる](../../../../translated_images/calculation.a209813050a1ddb141cdc4bc56f3af31e67157ed499e16a2ecf9837542704c94.ja.png)
>
> `Y`の値を計算します。もし約4ドルを支払っているなら、それは4月に違いありません！インフォグラフィック作成者: [Jen Looper](https://twitter.com/jenlooper)
>
> この線を計算する数学は、線の傾きを示す必要があり、それはまた切片、つまり`X = 0`のときの`Y`の位置に依存します。
>
> これらの値の計算方法は[Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html)のウェブサイトで確認できます。また、[この最小二乗計算機](https://www.mathsisfun.com/data/least-squares-calculator.html)を訪れて、数値の値が線にどのように影響するかを確認してください。

## 相関

もう一つ理解すべき用語は、与えられたXとY変数間の**相関係数**です。散布図を使えば、この係数をすぐに視覚化できます。データポイントがきれいな線状に散らばっているプロットは高い相関を持ちますが、データポイントがXとYの間でどこにでも散らばっているプロットは低い相関を持ちます。

良い線形回帰モデルは、最小二乗回帰法を使用して回帰線を描いた場合に、相関係数が高い（1に近い）ものです。

✅ このレッスンに付属するノートブックを実行し、月と価格の散布図を確認してください。カボチャ販売の月と価格のデータは、散布図の視覚的解釈によると高い相関を持っているように見えますか？年の*日数*（つまり年初からの日数）を使用してより細かい測定を行った場合、結果は変わりますか？

以下のコードでは、データをクリーニングし、次のようなデータフレーム`new_pumpkins`を取得したと仮定します：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> データをクリーニングするコードは[`notebook.ipynb`](notebook.ipynb)にあります。前のレッスンと同じクリーニング手順を実行し、次の式を使用して`DayOfYear`列を計算しました：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

線形回帰の数学を理解したところで、カボチャの価格を予測するための回帰モデルを作成してみましょう。例えば、ホリデー用のカボチャ畑を運営する人が、畑用のカボチャパッケージの購入を最適化するためにこの情報を必要とするかもしれません。

## 相関を探る

[![初心者向け機械学習 - 相関を探る: 線形回帰の鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "初心者向け機械学習 - 相関を探る: 線形回帰の鍵")

> 🎥 上の画像をクリックすると、相関の概要を説明する短い動画が再生されます。

前のレッスンで、異なる月の平均価格が次のように見えることを確認しました：

<img alt="月ごとの平均価格" src="../2-Data/images/barchart.png" width="50%"/>

これにより、何らかの相関がある可能性が示唆されます。`Month`と`Price`、または`DayOfYear`と`Price`の関係を予測する線形回帰モデルをトレーニングしてみましょう。以下は後者の関係を示す散布図です：

<img alt="価格と年の日数の散布図" src="images/scatter-dayofyear.png" width="50%" /> 

`corr`関数を使って相関を確認してみましょう：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

相関はかなり小さいようです。`Month`では-0.15、`DayOfMonth`では-0.17ですが、別の重要な関係がある可能性があります。価格が異なるカボチャの品種に対応する異なるクラスターがあるように見えます。この仮説を確認するために、各カボチャカテゴリーを異なる色でプロットしてみましょう。`scatter`プロット関数に`ax`パラメータを渡すことで、すべてのポイントを同じグラフにプロットできます：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="価格と年の日数の散布図（色分け）" src="images/scatter-dayofyear-color.png" width="50%" /> 

調査結果は、販売日よりも品種が価格に大きな影響を与えることを示唆しています。これを棒グラフで確認できます：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="品種ごとの価格の棒グラフ" src="images/price-by-variety.png" width="50%" /> 

ここでは、特定のカボチャ品種「パイタイプ」に焦点を当て、日付が価格に与える影響を確認してみましょう：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="価格と年の日数の散布図（パイタイプ）" src="images/pie-pumpkins-scatter.png" width="50%" /> 

`corr`関数を使って`Price`と`DayOfYear`の相関を計算すると、`-0.27`程度の値が得られます。これは予測モデルをトレーニングする価値があることを意味します。

> 線形回帰モデルをトレーニングする前に、データがクリーンであることを確認することが重要です。線形回帰は欠損値に対してうまく機能しないため、空のセルをすべて削除するのが理にかなっています：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

別のアプローチとしては、空の値を対応する列の平均値で埋める方法があります。

## 単純線形回帰

[![初心者向け機械学習 - Scikit-learnを使った線形回帰と多項式回帰](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "初心者向け機械学習 - Scikit-learnを使った線形回帰と多項式回帰")

> 🎥 上の画像をクリックすると、線形回帰と多項式回帰の概要を説明する短い動画が再生されます。

線形回帰モデルをトレーニングするために、**Scikit-learn**ライブラリを使用します。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

まず、入力値（特徴量）と期待される出力（ラベル）を別々のnumpy配列に分けます：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 入力データに対して`reshape`を実行する必要があることに注意してください。これは、線形回帰パッケージが正しく理解するためです。線形回帰は入力として2次元配列を期待します。この配列の各行は入力特徴量のベクトルに対応します。今回の場合、入力が1つしかないため、形状がN×1の配列が必要です（Nはデータセットのサイズ）。

次に、データをトレーニングデータセットとテストデータセットに分割します。これにより、トレーニング後にモデルを検証できます：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後に、実際の線形回帰モデルのトレーニングはわずか2行のコードで完了します。`LinearRegression`オブジェクトを定義し、`fit`メソッドを使ってデータに適合させます：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`fit`後の`LinearRegression`オブジェクトには、回帰のすべての係数が含まれており、`.coef_`プロパティを使ってアクセスできます。今回の場合、係数は1つだけで、約`-0.017`になるはずです。これは、価格が時間とともに少しずつ下がることを示していますが、その変化は非常に小さく、1日あたり約2セント程度です。また、回帰線がY軸と交差する点を`lin_reg.intercept_`を使ってアクセスできます。今回の場合、約`21`になるはずで、これは年初の価格を示しています。

モデルの精度を確認するために、テストデータセットで価格を予測し、予測値が期待値にどれだけ近いかを測定します。これは平均二乗誤差（MSE）メトリクスを使って行います。MSEは、期待値と予測値のすべての差の二乗の平均です。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
私たちのエラーは約2つのポイントに集中しており、これは約17%です。あまり良くありません。モデルの品質を示すもう1つの指標は**決定係数**で、以下のように取得できます：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
値が0の場合、モデルは入力データを考慮せず、*最悪の線形予測器*として機能します。これは単に結果の平均値を示します。値が1の場合、すべての期待される出力を完全に予測できることを意味します。私たちの場合、決定係数は約0.06であり、非常に低い値です。

また、テストデータと回帰線をプロットして、私たちのケースで回帰がどのように機能するかをよりよく理解することができます：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="線形回帰" src="images/linear-results.png" width="50%" />

## 多項式回帰

線形回帰のもう一つのタイプは多項式回帰です。変数間に線形関係がある場合もありますが（例えば、カボチャの体積が大きいほど価格が高い）、これらの関係が平面や直線としてプロットできない場合もあります。

✅ [こちら](https://online.stat.psu.edu/stat501/lesson/9/9.8)に多項式回帰が適用できるデータの例があります。

日付と価格の関係をもう一度見てみましょう。この散布図は必ずしも直線で分析されるべきだと思いますか？価格は変動する可能性がありますよね？この場合、多項式回帰を試すことができます。

✅ 多項式は、1つ以上の変数と係数を含む数学的表現です。

多項式回帰は、非線形データにより適した曲線を作成します。私たちの場合、入力データに`DayOfYear`変数の二乗を含めることで、年内の特定の点で最小値を持つ放物線をデータに適合させることができます。

Scikit-learnには、データ処理の異なるステップを組み合わせるための便利な[パイプラインAPI](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)が含まれています。**パイプライン**は**推定器**のチェーンです。私たちの場合、まずモデルに多項式特徴を追加し、その後回帰をトレーニングするパイプラインを作成します：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)`を使用することで、入力データからすべての二次多項式を含めることができます。私たちの場合、これは単に`DayOfYear`<sup>2</sup>を意味しますが、2つの入力変数XとYがある場合、これによりX<sup>2</sup>、XY、Y<sup>2</sup>が追加されます。必要に応じて、より高次の多項式を使用することもできます。

パイプラインは元の`LinearRegression`オブジェクトと同じ方法で使用できます。つまり、パイプラインを`fit`し、その後`predict`を使用して予測結果を取得できます。以下はテストデータと近似曲線を示すグラフです：

<img alt="多項式回帰" src="images/poly-results.png" width="50%" />

多項式回帰を使用することで、MSEをわずかに低くし、決定係数をわずかに高くすることができますが、大きな改善はありません。他の特徴を考慮する必要があります！

> 最低のカボチャ価格がハロウィンの周辺で観察されることがわかります。これをどのように説明しますか？

🎃 おめでとうございます！パイカボチャの価格を予測するモデルを作成しました。同じ手順をすべてのカボチャの種類に対して繰り返すことができますが、それは面倒です。次に、モデルにカボチャの種類を考慮する方法を学びましょう！

## カテゴリカル特徴

理想的な世界では、同じモデルを使用して異なるカボチャの種類の価格を予測できるようにしたいです。しかし、`Variety`列は`Month`のような列とは異なり、非数値の値を含んでいます。このような列は**カテゴリカル**と呼ばれます。

[![初心者向けML - 線形回帰でカテゴリカル特徴を予測](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "初心者向けML - 線形回帰でカテゴリカル特徴を予測")

> 🎥 上の画像をクリックすると、カテゴリカル特徴の使用に関する短いビデオ概要が表示されます。

以下は、種類ごとの平均価格がどのように依存しているかを示しています：

<img alt="種類ごとの平均価格" src="images/price-by-variety.png" width="50%" />

種類を考慮するためには、まずそれを数値形式に変換する必要があります。これを**エンコード**と呼びます。いくつかの方法があります：

* 単純な**数値エンコード**は、異なる種類のテーブルを作成し、そのテーブル内のインデックスで種類名を置き換えます。これは線形回帰には最適ではありません。なぜなら、線形回帰はインデックスの実際の数値を結果に加え、ある係数で掛け算するからです。この場合、インデックス番号と価格の関係は明らかに非線形であり、特定の順序でインデックスを並べても同じです。
* **ワンホットエンコード**は、`Variety`列を4つの異なる列に置き換えます。それぞれの列は、対応する行が特定の種類である場合に`1`を含み、それ以外の場合は`0`を含みます。これにより、線形回帰ではカボチャの種類ごとに1つの係数が追加され、その種類の「開始価格」（または「追加価格」）を表します。

以下のコードは、種類をワンホットエンコードする方法を示しています：

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

ワンホットエンコードされた種類を入力として線形回帰をトレーニングするには、`X`と`y`データを正しく初期化するだけです：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

残りのコードは、上記で使用した線形回帰をトレーニングするコードと同じです。これを試してみると、平均二乗誤差はほぼ同じですが、決定係数が大幅に高くなり（約77%）、より正確な予測が可能になります。さらに正確な予測を得るためには、`Month`や`DayOfYear`などの数値特徴とともに、より多くのカテゴリカル特徴を考慮する必要があります。1つの大きな特徴配列を作成するには、`join`を使用できます：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

ここでは`City`や`Package`タイプも考慮しており、これによりMSEは2.84（10%）、決定係数は0.94になります！

## すべてをまとめる

最良のモデルを作成するには、上記の例からの結合データ（ワンホットエンコードされたカテゴリカル特徴＋数値特徴）を多項式回帰と組み合わせて使用します。以下は完全なコードです：

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

これにより、ほぼ97%の決定係数とMSE=2.23（約8%の予測誤差）を得ることができます。

| モデル | MSE | 決定係数 |
|-------|-----|---------------|
| `DayOfYear` 線形 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線形 | 5.24 (19.7%) | 0.77 |
| すべての特徴 線形 | 2.84 (10.5%) | 0.94 |
| すべての特徴 多項式 | 2.23 (8.25%) | 0.97 |

🏆 素晴らしい！1つのレッスンで4つの回帰モデルを作成し、モデルの品質を97%まで向上させました。回帰の最終セクションでは、カテゴリを決定するためのロジスティック回帰について学びます。

---
## 🚀チャレンジ

このノートブックでいくつかの異なる変数をテストし、相関がモデルの精度にどのように対応するかを確認してください。

## [講義後のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/14/)

## 復習と自己学習

このレッスンでは線形回帰について学びました。他にも重要な回帰の種類があります。ステップワイズ、リッジ、ラッソ、エラスティックネット技術について読んでみてください。さらに学ぶための良いコースは[スタンフォード統計学習コース](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)です。

## 課題

[モデルを構築する](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。元の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤った解釈について、当方は一切の責任を負いません。