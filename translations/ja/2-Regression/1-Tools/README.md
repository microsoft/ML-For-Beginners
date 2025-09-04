<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6b1cb0e46d4c5b747eff6e3607642760",
  "translation_date": "2025-09-03T22:32:34+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "ja"
}
-->
# PythonとScikit-learnを使った回帰モデルの入門

![回帰の概要をスケッチノートで表現](../../../../translated_images/ml-regression.4e4f70e3b3ed446e3ace348dec973e133fa5d3680fbc8412b61879507369b98d.ja.png)

> スケッチノート作成者: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [講義前のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [このレッスンはRでも利用可能です！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## はじめに

この4つのレッスンでは、回帰モデルの構築方法を学びます。これらが何のために使われるのかについては後ほど説明します。しかし、何かを始める前に、プロセスを開始するための適切なツールを準備してください！

このレッスンで学ぶ内容は以下の通りです：

- ローカルで機械学習タスクを実行するためのコンピュータの設定。
- Jupyterノートブックの使用方法。
- Scikit-learnのインストールと使用方法。
- 線形回帰を実際に体験する演習。

## インストールと設定

[![初心者向け機械学習 - 機械学習モデルを構築するためのツールを設定する](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "初心者向け機械学習 - 機械学習モデルを構築するためのツールを設定する")

> 🎥 上の画像をクリックすると、コンピュータをML用に設定する短い動画が再生されます。

1. **Pythonをインストール**。コンピュータに[Python](https://www.python.org/downloads/)がインストールされていることを確認してください。Pythonは多くのデータサイエンスや機械学習タスクで使用されます。ほとんどのコンピュータシステムにはすでにPythonがインストールされています。一部のユーザー向けにセットアップを簡単にするための便利な[Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)も利用可能です。

   ただし、Pythonの使用方法によっては、異なるバージョンが必要になる場合があります。このため、[仮想環境](https://docs.python.org/3/library/venv.html)を使用することが便利です。

2. **Visual Studio Codeをインストール**。コンピュータにVisual Studio Codeがインストールされていることを確認してください。基本的なインストールについては、[Visual Studio Codeのインストール手順](https://code.visualstudio.com/)に従ってください。このコースではVisual Studio CodeでPythonを使用するので、[Python開発のためのVisual Studio Codeの設定方法](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)を確認しておくと良いでしょう。

   > Pythonに慣れるために、この[Learnモジュールのコレクション](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)を試してみてください。
   >
   > [![Visual Studio CodeでPythonを設定する](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Visual Studio CodeでPythonを設定する")
   >
   > 🎥 上の画像をクリックすると、VS Code内でPythonを使用する方法の動画が再生されます。

3. **Scikit-learnをインストール**。[こちらの手順](https://scikit-learn.org/stable/install.html)に従ってインストールしてください。Python 3を使用する必要があるため、仮想環境を使用することをお勧めします。M1 Macにこのライブラリをインストールする場合、リンク先のページに特別な手順が記載されています。

4. **Jupyter Notebookをインストール**。[Jupyterパッケージ](https://pypi.org/project/jupyter/)をインストールしてください。

## ML作成環境

Pythonコードを開発し、機械学習モデルを作成するために**ノートブック**を使用します。このタイプのファイルはデータサイエンティストにとって一般的なツールであり、拡張子`.ipynb`で識別されます。

ノートブックは、コードを記述するだけでなく、コードに関するメモやドキュメントを追加することができるインタラクティブな環境です。これは実験的または研究指向のプロジェクトに非常に役立ちます。

[![初心者向け機械学習 - 回帰モデルを構築するためのJupyterノートブックの設定](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "初心者向け機械学習 - 回帰モデルを構築するためのJupyterノートブックの設定")

> 🎥 上の画像をクリックすると、この演習を進める短い動画が再生されます。

### 演習 - ノートブックを操作する

このフォルダには、_notebook.ipynb_というファイルがあります。

1. Visual Studio Codeで_notebook.ipynb_を開きます。

   JupyterサーバーがPython 3+で起動します。ノートブック内には`run`できるコードの部分があります。コードブロックを実行するには、再生ボタンのようなアイコンを選択します。

2. `md`アイコンを選択して、以下のテキストを追加します： **# Welcome to your notebook**。

   次に、Pythonコードを追加します。

3. コードブロックに**print('hello notebook')**と入力します。
4. 矢印を選択してコードを実行します。

   以下のように出力が表示されるはずです：

    ```output
    hello notebook
    ```

![VS Codeでノートブックを開いた状態](../../../../translated_images/notebook.4a3ee31f396b88325607afda33cadcc6368de98040ff33942424260aa84d75f2.ja.jpg)

コードとコメントを交互に記述することで、ノートブックを自己文書化することができます。

✅ データサイエンティストの作業環境がウェブ開発者の作業環境とどのように異なるか、少し考えてみてください。

## Scikit-learnの準備

Pythonがローカル環境で設定され、Jupyterノートブックに慣れたら、Scikit-learnにも慣れていきましょう（`sci`は`science`のように発音します）。Scikit-learnは、MLタスクを実行するための[広範なAPI](https://scikit-learn.org/stable/modules/classes.html#api-ref)を提供します。

[公式サイト](https://scikit-learn.org/stable/getting_started.html)によると、「Scikit-learnは、教師あり学習と教師なし学習をサポートするオープンソースの機械学習ライブラリです。また、モデルの適合、データの前処理、モデル選択と評価、その他多くのユーティリティを提供します。」

このコースでは、Scikit-learnやその他のツールを使用して、いわゆる「従来の機械学習」タスクを実行するためのモデルを構築します。ニューラルネットワークや深層学習は、今後の「AI for Beginners」カリキュラムで詳しく扱う予定です。

Scikit-learnを使用すると、モデルを簡単に構築し、それらを評価して使用することができます。主に数値データを使用することに焦点を当てており、学習ツールとして使用できるいくつかの既製のデータセットが含まれています。また、学生が試すための事前構築されたモデルも含まれています。まずは、事前パッケージ化されたデータをロードし、Scikit-learnの組み込み推定器を使用して最初のMLモデルを作成するプロセスを探ってみましょう。

## 演習 - 初めてのScikit-learnノートブック

> このチュートリアルは、Scikit-learnのウェブサイトにある[線形回帰の例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)に触発されています。

[![初心者向け機械学習 - Pythonで初めての線形回帰プロジェクト](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "初心者向け機械学習 - Pythonで初めての線形回帰プロジェクト")

> 🎥 上の画像をクリックすると、この演習を進める短い動画が再生されます。

このレッスンに関連する_notebook.ipynb_ファイル内のすべてのセルを、ゴミ箱アイコンを押してクリアしてください。

このセクションでは、Scikit-learnに組み込まれている学習用の小さなデータセットである糖尿病データセットを使用します。糖尿病患者の治療をテストしたいと仮定します。機械学習モデルは、変数の組み合わせに基づいて、どの患者が治療により良い反応を示すかを判断するのに役立つかもしれません。非常に基本的な回帰モデルでも、視覚化することで、理論的な臨床試験を整理するのに役立つ変数に関する情報を示すことができます。

✅ 回帰方法には多くの種類があり、どれを選ぶかは求めている答えによります。例えば、特定の年齢の人の予測される身長を知りたい場合は、**数値値**を求めているため線形回帰を使用します。一方、ある料理がビーガン料理と見なされるべきかどうかを知りたい場合は、**カテゴリの割り当て**を求めているためロジスティック回帰を使用します。後ほどロジスティック回帰について詳しく学びます。データに対してどのような質問ができるか、そしてどの方法が適切か少し考えてみてください。

では、このタスクを始めましょう。

### ライブラリのインポート

このタスクでは以下のライブラリをインポートします：

- **matplotlib**。便利な[グラフ作成ツール](https://matplotlib.org/)で、線グラフを作成するために使用します。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html)はPythonで数値データを扱うための便利なライブラリです。
- **sklearn**。これが[Scikit-learn](https://scikit-learn.org/stable/user_guide.html)ライブラリです。

タスクを助けるためにいくつかのライブラリをインポートします。

1. 以下のコードを入力してインポートを追加します：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上記では`matplotlib`、`numpy`をインポートし、`sklearn`から`datasets`、`linear_model`、`model_selection`をインポートしています。`model_selection`はデータをトレーニングセットとテストセットに分割するために使用されます。

### 糖尿病データセット

組み込みの[糖尿病データセット](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)には、糖尿病に関する442件のデータサンプルが含まれており、10個の特徴変数があります。その一部は以下の通りです：

- age: 年齢（年単位）
- bmi: ボディマス指数
- bp: 平均血圧
- s1 tc: T細胞（白血球の一種）

✅ このデータセットには、糖尿病に関する研究で重要な特徴変数として「性別」の概念が含まれています。多くの医療データセットには、このような二分分類が含まれています。このような分類が治療から特定の人口を除外する可能性について少し考えてみてください。

次に、Xとyデータをロードします。

> 🎓 これは教師あり学習であり、名前付きの「y」ターゲットが必要です。

新しいコードセルで、`load_diabetes()`を呼び出して糖尿病データセットをロードします。入力`return_X_y=True`は、`X`がデータマトリックス、`y`が回帰ターゲットになることを示します。

1. データマトリックスの形状とその最初の要素を表示するためにいくつかのprintコマンドを追加します：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    返されるのはタプルです。タプルの最初の2つの値をそれぞれ`X`と`y`に割り当てています。[タプルについて詳しく学ぶ](https://wikipedia.org/wiki/Tuple)。

    このデータには442件のアイテムがあり、10個の要素で構成された配列として形作られていることがわかります：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ データと回帰ターゲットの関係について少し考えてみてください。線形回帰は特徴Xとターゲット変数yの間の関係を予測します。このデータセットの[ターゲット](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)をドキュメントで見つけることができますか？このデータセットは何を示しているのでしょうか？

2. 次に、このデータセットの一部を選択してプロットします。データセットの3列目を選択するには、`:`演算子を使用してすべての行を選択し、インデックス（2）を使用して3列目を選択します。また、`reshape(n_rows, n_columns)`を使用してデータを2D配列に再構成することができます。パラメータの1つが-1の場合、対応する次元が自動的に計算されます。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ いつでもデータを印刷してその形状を確認できます。

3. データがプロットの準備が整ったら、このデータセットの数値間に論理的な分割を決定するために機械を使用できるか確認します。そのためには、データ（X）とターゲット（y）の両方をテストセットとトレーニングセットに分割する必要があります。Scikit-learnにはこれを簡単に行う方法があり、指定したポイントでテストデータを分割できます。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. これでモデルをトレーニングする準備が整いました！線形回帰モデルをロードし、`model.fit()`を使用してXとyのトレーニングセットでトレーニングします：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()`はTensorFlowなどの多くのMLライブラリで見られる関数です。

5. 次に、テストデータを使用して予測を作成します。これにより、データグループ間の線を描画することができます。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 次に、データをプロットで表示します。Matplotlibはこのタスクに非常に便利なツールです。すべてのXとyテストデータの散布図を作成し、予測を使用してデータグループ間の最適な位置に線を描画します。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![糖尿病に関するデータポイントを示す散布図](../../../../translated_images/scatterplot.ad8b356bcbb33be68d54050e09b9b7bfc03e94fde7371f2609ae43f4c563b2d7.ja.png)
✅ 少し考えてみましょう。たくさんの小さなデータ点を通る一直線がありますが、この線は一体何をしているのでしょうか？この線を使って、新しい未知のデータポイントがプロットのy軸に対してどこに位置するべきかを予測できる方法が見えてきませんか？このモデルの実用的な使い方を言葉で説明してみてください。

おめでとうございます！初めての線形回帰モデルを構築し、それを使って予測を行い、プロットに表示することができました！

---
## 🚀チャレンジ

このデータセットから別の変数をプロットしてみましょう。ヒント: この行を編集してください: `X = X[:,2]`。このデータセットのターゲットを考慮すると、糖尿病という病気の進行についてどのようなことが発見できるでしょうか？

## [講義後のクイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## 復習と自己学習

このチュートリアルでは、単回帰分析を扱いましたが、単変量回帰や重回帰分析ではありませんでした。これらの手法の違いについて少し調べてみるか、[この動画](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)を見てみてください。

回帰の概念についてさらに学び、この手法でどのような質問に答えられるのかを考えてみましょう。[このチュートリアル](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)を受講して理解を深めてください。

## 課題

[別のデータセット](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。元の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤った解釈について、当方は一切の責任を負いません。