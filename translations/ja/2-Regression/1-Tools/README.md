# Python と Scikit-learn で回帰モデルを始めよう

![回帰の概要をまとめたスケッチノート](../../../../translated_images/ja/ml-regression.4e4f70e3b3ed446e.webp)

> スケッチノート：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [講義前クイズ](https://ff-quizzes.netlify.app/en/ml/)

> ### [このレッスンは R でも利用可能です！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## はじめに

この4つのレッスンでは回帰モデルの構築方法を学びます。これらが何のためにあるのかもすぐに説明しますが、何かを始める前に、プロセスを開始するための適切なツールが揃っていることを確認してください！

このレッスンでは以下を学びます:

- ローカルの機械学習タスクに向けてパソコンを設定する方法。
- Jupyter ノートブックの扱い方。
- Scikit-learn の使い方とそのインストール方法。
- 実践的な演習を通じて線形回帰を理解する方法。

## インストールと設定

[![ML初心者向け - 機械学習モデル構築に必要なツールのセットアップ](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML初心者向け - 機械学習モデル構築に必要なツールのセットアップ")

> 🎥 上の画像をクリックすると、機械学習に向けたパソコンの設定を解説する短い動画を視聴できます。

1. **Python をインストールする**。パソコンに [Python](https://www.python.org/downloads/) がインストールされていることを確認してください。Python は多くのデータサイエンスや機械学習タスクに使われます。多くのコンピュータシステムにはPythonが標準搭載されています。セットアップを簡単にするための [Python コーディングパック](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) も利用可能です。

   ただし、Pythonは用途により使用するバージョンが異なる場合があるため、[仮想環境](https://docs.python.org/3/library/venv.html)を使って作業することが便利です。

2. **Visual Studio Code をインストールする**。パソコンに Visual Studio Code がインストールされていることを確認してください。基本的なインストール方法は [Visual Studio Code のインストール](https://code.visualstudio.com/) の手順に従ってください。このコースでは Visual Studio Code で Python を使うので、[Visual Studio Code の Python 開発環境設定](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)で設定方法をおさらいしておくと良いでしょう。

   > Python の理解を深めるために、[Learn モジュール](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) を試してみてください。
   >
   > [![Visual Studio Code での Python セットアップ](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Visual Studio Code での Python セットアップ")
   >
   > 🎥 上の画像をクリックすると、Visual Studio Code で Python を使う動画が再生されます。

3. **Scikit-learn をインストールする**。こちらの手順に従ってください [インストール手順](https://scikit-learn.org/stable/install.html)。Python 3 を使用する必要があるので、仮想環境の利用を推奨します。M1 Mac でのインストールにはページ内に特別な指示があります。

1. **Jupyter Notebook をインストールする**。 [Jupyter パッケージをインストールする](https://pypi.org/project/jupyter/) 必要があります。

## あなたの機械学習制作環境

コード開発と機械学習モデル作成には <strong>ノートブック</strong> を使います。この形式のファイルはデータサイエンティストによく使われ、拡張子が `.ipynb` で識別されます。

ノートブックは対話型の環境で、開発者はコードを書くだけでなく、その周りにメモやドキュメントも記述でき、実験的または研究志向のプロジェクトにとても役立ちます。

[![ML初心者向け - 回帰モデル構築開始のための Jupyter ノートブックのセットアップ](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML初心者向け - 回帰モデル構築開始のための Jupyter ノートブックのセットアップ")

> 🎥 上の画像をクリックすると、この演習を解説した短い動画を視聴できます。

### 演習 - ノートブックを使ってみる

このフォルダーには _notebook.ipynb_ ファイルがあります。

1. Visual Studio Code で _notebook.ipynb_ を開きます。

   Jupyter サーバーが Python 3+ で起動します。ノートブック内には `run` できるコードブロックがあります。再生ボタンのようなアイコンを選択してコードを実行できます。

1. `md` アイコンを選択し、マークダウンを少し追加して次のテキストを入れます **# Welcome to your notebook**。

   次に、Pythonコードを追加します。

1. コードブロックに **print('hello notebook')** と入力します。
1. 実行ボタンを選択してコードを動かします。

   表示される出力は次の通りです:

    ```output
    hello notebook
    ```

![ノートブックを開いた VS Code](../../../../translated_images/ja/notebook.4a3ee31f396b8832.webp)

コードとコメントを交互に挟んで、ノートブック自身をドキュメント化できます。

✅ ウェブ開発者の作業環境とデータサイエンティストのそれとがどのように違うか、少し考えてみてください。

## Scikit-learn のセットアップ完了

Python がローカル環境でセットアップでき、Jupyter ノートブックに慣れたら、Scikit-learn も同様に使いこなせるようにしましょう（読み方は `sci`、science のように発音します）。Scikit-learn は機械学習タスクをこなすための [幅広いAPI](https://scikit-learn.org/stable/modules/classes.html#api-ref)を提供しています。

公式の[ウェブサイト](https://scikit-learn.org/stable/getting_started.html)によると、「Scikit-learn は教師あり学習と教師なし学習をサポートするオープンソースの機械学習ライブラリです。モデル適合、データ前処理、モデル選択と評価、多くのユーティリティも提供しています」。

このコースでは Scikit-learn と他のツールを使って「従来型機械学習」と呼ばれるモデルを構築します。ニューラルネットワークや深層学習は別途用意する『AI 初心者向け』カリキュラムで学びます。

Scikit-learn はモデル作成と評価を簡単に行えます。主に数値データを扱い、学習用の多くの既成データセットが含まれています。学生が試行できる事前構築済みモデルも豊富です。まず既成データを読み込み、組み込み推定器を使って基本的なデータで最初の機械学習モデルを作りましょう。

## 演習 - Scikit-learn の最初のノートブック

> このチュートリアルは Scikit-learn のウェブサイトにある [線形回帰の例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) に着想を得ています。


[![ML初心者向け - Python で最初の線形回帰プロジェクト](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML初心者向け - Python で最初の線形回帰プロジェクト")

> 🎥 上の画像をクリックすると、この演習の解説動画を短く視聴できます。

関連する _notebook.ipynb_ ファイルのすべてのセルを ‘ゴミ箱’ アイコンで削除してクリーンにします。

このセクションでは、学習用に Scikit-learn に組み込まれている糖尿病に関する小規模なデータセットを扱います。糖尿病患者向けの治療法を試したいと想定してください。機械学習モデルは、変数の組み合わせからどの患者が治療により良く反応するかを判断する手助けをします。実際、単純な回帰モデルでも、変数に関する情報を視覚化することで理論的な臨床試験を組織するのに役立つかもしれません。

✅ 回帰方法には多くの種類があり、どれを選ぶかは求める答えによります。例えば、ある年齢の人の身長を予測したいなら、<strong>数値を求める</strong>ので線形回帰を使います。一方、ある料理がビーガンかどうかを判定したいなら、<strong>カテゴリー分け</strong>をするのでロジスティック回帰を使います。ロジスティック回帰は後で学びます。データに対してどんな質問をし、それにどの方法が適切か少し考えてみてください。

それでは、この課題を始めましょう。

### ライブラリのインポート

この課題でインポートするライブラリは以下です:

- **matplotlib**。有用な[グラフ作成ツール](https://matplotlib.org/)で、折れ線グラフを作成します。
- **numpy**。Pythonで数値データを扱うのに便利な [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) ライブラリです。
- **sklearn**。[Scikit-learn](https://scikit-learn.org/stable/user_guide.html) ライブラリです。

タスクを助けるためにいくつかのライブラリをインポートします。

1. 以下のコードを入力してインポートします:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上記では `matplotlib` 、 `numpy` をインポートし、 `sklearn` からは `datasets` 、 `linear_model` と `model_selection` をインポートしています。 `model_selection` はデータをトレーニングセットとテストセットに分けるのに使います。

### 糖尿病データセット

組み込みの [糖尿病データセット](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) には糖尿病に関する442サンプルのデータがあり、10の特徴変数が含まれます。その一部は以下の通りです:

- age: 年齢（歳）
- bmi: 体格指数（BMI）
- bp: 平均血圧
- s1 tc: T細胞（白血球の一種）

✅ このデータセットには糖尿病研究に重要な特徴変数として ‘性別’ の概念が含まれています。多くの医療データセットにおいてこの種の二分法的分類があります。こうした分類が特定の集団を治療対象から除外してしまうことについても考えてみてください。

では、X と y のデータを読み込みましょう。

> 🎓 これは教師あり学習なので、yという名前のターゲットが必要です。

新しいコードセルで、 `load_diabetes()` を呼び出して糖尿病データセットを読み込みます。引数 `return_X_y=True` は `X` がデータ行列、`y` が回帰目標であることを示します。

1. データ行列の形状と最初の要素を表示するためにプリント文を追加します:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    返されるのはタプルです。タプルの最初の2つの値をそれぞれ `X` と `y` に割り当てています。[タプルについてもっと学ぶ](https://wikipedia.org/wiki/Tuple)こともできます。

    このデータには442項目があり、それぞれ10要素の配列として構成されているのがわかります:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ データと回帰ターゲットの関係について少し考えてみてください。線形回帰は特徴X とターゲット変数yの関係を予測します。糖尿病データセットの[ターゲット](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)はドキュメントで何と示されていますか？このデータセットはそのターゲットを踏まえて何を示しているのでしょうか？

2. 次に、このデータセットの一部をプロット用に選びます。データセットの3列目を選択します。すべての行を対象に `:` を使い、3列目(インデックス2)を選びます。プロットには2次元配列が必要なので `reshape(n_rows, n_columns)` で形状を変換します。パラメータの一つが -1 の場合は対応する次元を自動計算します。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ いつでもプリントしてデータの形状を確認しましょう。

3. プロット用のデータが準備できたら、機械がデータの分割を論理的に判断できるか試しましょう。これにはデータ (X) とターゲット (y) の両方をテストセットとトレーニングセットに分割する必要があります。Scikit-learn は単純な手段を提供し、指定した地点でテストデータを分割できます。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. さあモデルの訓練準備完了です！線形回帰モデルを読み込み、トレーニングセットのX と y で `model.fit()` を使って訓練します:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` は TensorFlow など多くの機械学習ライブラリにある関数です

5. それからテストデータを使って予測を作成します。関数 `predict()` を使います。これでデータ群の間に線が引かれます。

    ```python
    y_pred = model.predict(X_test)
    ```

6. いよいよプロットでデータを表示しましょう。Matplotlib はこの作業にとても便利なツールです。全てのX と y のテストデータを散布図で描き、予測結果でモデルのデータ群の間に最適な線を引きます。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![糖尿病周辺の散布図](../../../../translated_images/ja/scatterplot.ad8b356bcbb33be6.webp)

   ✅ ここで何が起こっているか少し考えてみてください。多くの小さな点の間に直線が引かれていますが、これは何を意味していますか？この線を使って新しい未観測のデータ点が散布図の y 軸に対してどこに位置すべきか予測できることがわかりますか？このモデルの実用例を言葉にしてみましょう。

おめでとうございます、最初の線形回帰モデルを構築し、予測を作成し、プロットで表示できました！

---
## 🚀チャレンジ

このデータセットの別の変数をプロットしてみてください。ヒント：この行を編集します: `X = X[:,2]`。このデータセットのターゲットを踏まえて、糖尿病という病気の進行について何がわかりますか？
## [講義後クイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自主学習

このチュートリアルでは単回帰線形回帰を扱い、多変量回帰は扱いませんでした。これらの違いについて少し読んでみるか、[この動画](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef) を見てみてください。

回帰の概念についてさらに読み、この手法でどのような質問に答えられるか考えてみましょう。この[チュートリアル](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)を受けて理解を深めてください。

## 課題

[別のデータセット](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責事項**：
本書類は AI 翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を期していますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知おきください。原文の原語版が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。本翻訳の利用により生じたいかなる誤解や解釈違いについても、当方は責任を負いかねます。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->