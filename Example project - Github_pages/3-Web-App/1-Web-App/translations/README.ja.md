# 機械学習モデルを使うためのWebアプリを構築する

この講義では、この世界のものではないデータセットを使って機械学習モデルを学習させます。NUFORCのデータベースに登録されている「過去100年のUFO目撃情報」です。

あなたが学ぶ内容は以下の通りです。

- 学習したモデルを「塩漬け」にする方法
- モデルをFlaskアプリで使う方法

引き続きノートブックを使ってデータのクリーニングやモデルの学習を行いますが、さらに一歩進んでモデルを「野生で」、つまりWebアプリで使うのを検討することも可能です。

そのためには、Flaskを使ってWebアプリを構築する必要があります。

## [講義前の小テスト](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17?loc=ja)

## アプリの構築

機械学習モデルを使うためのWebアプリを構築する方法はいくつかあります。Webアーキテクチャはモデルの学習方法に影響を与える可能性があります。データサイエンスグループが学習したモデルをアプリで使用する、という業務があなたに任されている状況をイメージしてください。

### 検討事項

あなたがすべき質問はたくさんあります。

- **Webアプリですか？それともモバイルアプリですか？** モバイルアプリを構築している場合や、IoTの環境でモデルを使う必要がある場合は、[TensorFlow Lite](https://www.tensorflow.org/lite/) を使用して、AndroidまたはiOSアプリでモデルを使うことができます。
- **モデルはどこに保存しますか？** クラウドでしょうか？それともローカルでしょうか？
- **オフラインでのサポート。** アプリはオフラインで動作する必要がありますか？
- **モデルの学習にはどのような技術が使われていますか？** 選択された技術は使用しなければいけないツールに影響を与える可能性があります。
    - **Tensor flow を使っている。** 例えば TensorFlow を使ってモデルを学習している場合、 [TensorFlow.js](https://www.tensorflow.org/js/) を使って、Webアプリで使用できるように TensorFlow モデルを変換する機能をそのエコシステムは提供しています。
    - **PyTorchを使っている。** [PyTorch](https://pytorch.org/) などのライブラリを使用してモデルを構築している場合、[ONNX](https://onnx.ai/) (Open Neural Network Exchange) 形式で出力して、JavaScript のWebアプリで [Onnx Runtime](https://www.onnxruntime.ai/) を使用するという選択肢があります。この選択肢は、Scikit-learn で学習したモデルを使う今後の講義で調べます。
    - **Lobe.ai または Azure Custom Vision を使っている。** [Lobe.ai](https://lobe.ai/) や [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) のような機械学習SaaS (Software as a Service) システムを使用してモデルを学習している場合、この種のソフトウェアは多くのプラットフォーム向けにモデルを出力する方法を用意していて、これにはクラウド上のオンラインアプリケーションからリクエストされるような専用APIを構築することも含まれます。

また、ウェブブラウザ上でモデルを学習することができるFlaskのWebアプリを構築することもできます。JavaScript の場合でも TensorFlow.js を使うことで実現できます。

私たちの場合はPythonベースのノートブックを今まで使用してきたので、学習したモデルをそのようなノートブックからPythonで構築されたWebアプリで読める形式に出力するために必要な手順を探ってみましょう。

## ツール

ここでの作業には2つのツールが必要です。FlaskとPickleで、どちらもPython上で動作します。

✅ [Flask](https://palletsprojects.com/p/flask/) とは？制作者によって「マイクロフレームワーク」と定義されているFlaskは、Pythonを使ったWebフレームワークの基本機能と、Webページを構築するためのテンプレートエンジンを提供しています。Flaskでの構築を練習するために [この学習モジュール](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) を見てみてください。

✅ [Pickle](https://docs.python.org/3/library/pickle.html) とは？Pickle 🥒 は、Pythonのオブジェクト構造をシリアライズ・デシリアライズするPythonモジュールです。モデルを「塩漬け」にすると、Webで使用するためにその構造をシリアライズしたり平坦化したりします。pickleは本質的に安全ではないので、ファイルの 'un-pickle' を促された際は注意してください。塩漬けされたファイルの末尾は `.pkl` となります。

## 演習 - データをクリーニングする

この講義では、[NUFORC](https://nuforc.org) (The National UFO Reporting Center) が集めた8万件のUFO目撃情報のデータを使います。このデータには、UFOの目撃情報に関する興味深い記述があります。例えば以下のようなものです。

- **長い記述の例。** 「夜の草原を照らす光線から男が現れ、Texas Instruments の駐車場に向かって走った」
- **短い記述の例。** 「私たちを光が追いかけてきた」

[ufos.csv](../data/ufos.csv) のスプレッドシートには、目撃された場所の都市 (`city`)、州 (`state`)、国 (`country`)、物体の形状 (`shape`)、緯度 (`latitude`)、経度 (`longitude`) などの列が含まれています。

この講義に含んでいる空の [ノートブック](../notebook.ipynb) で、以下の手順に従ってください。

1. 前回の講義で行ったように `pandas`、`matplotlib`、`numpy` をインポートし、UFOのスプレッドシートをインポートしてください。サンプルのデータセットを見ることができます。

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. UFOのデータを新しいタイトルで小さいデータフレームに変換してください。また、`Country` 属性の一意な値を確認してください。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. ここで、null値をすべて削除し、1~60秒の目撃情報のみを読み込むことで処理すべきデータ量を減らすことができます。

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learn の `LabelEncoder` ライブラリをインポートして、国の文字列値を数値に変換してください。

   ✅ LabelEncoder はデータをアルファベット順にエンコードします。

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    データは以下のようになります。

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 演習 - モデルを構築する

これでデータを訓練グループとテストグループに分けてモデルを学習する準備ができました。

1. Xベクトルとして学習したい3つの特徴を選択し、Yベクトルには `Country` を指定します。`Seconds`、`Latitude`、`Longitude` を入力して国のIDを取得することにします。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ロジスティック回帰を使ってモデルを学習してください。

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

国 (`Country`) と緯度・経度 (`Latitude/Longitude`) は相関しているので当然ですが、精度は悪くないです。**（約95%）**

緯度 (`Latitude`) と経度 (`Longitude`) から国 (`Country`) を推測することができるので、作成したモデルは画期的なものではありませんが、クリーニングして出力した生のデータから学習を行い、このモデルをWebアプリで使用してみる良い練習にはなります。

## 演習 - モデルを「塩漬け」にする

さて、いよいよモデルを「塩漬け」にしてみましょう！これは数行のコードで実行できます。「塩漬け」にした後は、そのモデルを読み込んで、秒・緯度・経度を含むサンプルデータの配列でテストしてください。

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

モデルはイギリスの国番号である **「3」** を返します。すばらしい！👽

## 演習 - Flaskアプリを構築する

これでFlaskアプリを構築してモデルを呼び出すことができるようになり、これは同じような結果を返しますが、視覚的によりわかりやすい方法です。

1. まず、_ufo-model.pkl_ ファイルと _notebook.ipynb_ ファイルが存在する場所に **web-app** というフォルダを作成してください。

1. そのフォルダの中に、さらに3つのフォルダを作成してください。**css** というフォルダを含む **static** と、**templates** です。以下のようなファイルとディレクトリになっているはずです。

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 完成したアプリを見るには、solution フォルダを参照してください。

1. _web-app_ フォルダの中に作成する最初のファイルは **requirements.txt** です。JavaScript アプリにおける _package.json_ と同様に、このファイルはアプリに必要な依存関係をリストにしたものです。**requirements.txt** に以下の行を追加してください。

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 次に、_web-app_ に移動して、このファイルを実行します。

    ```bash
    cd web-app
    ```

1. _requirements.txt_ に記載されているライブラリをインストールするために、ターミナルで `pip install` と入力してください。

    ```bash
    pip install -r requirements.txt
    ```

1. アプリを完成させるために、さらに3つのファイルを作成する準備が整いました。

    1. ルートに **app.py** を作成してください。
    2. _templates_ ディレクトリに **index.html** を作成してください。
    3. _static/css_ ディレクトリに **styles.css** を作成してください。

1. 以下のスタイルで _styles.css_ ファイルを構築してください。

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. 次に _index.html_ を構築してください。

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    このファイルのテンプレートを見てみましょう。予測テキストのようにアプリから渡された変数を、 `{{}}` という「マスタッシュ」構文で囲んでいることに注目してください。また、`/predict` のパスに対して予測を送信するフォームもあります。

    ついに、モデルを使用して予測値を表示するpythonファイルを構築する準備ができました。

1. `app.py` に以下を追加してください。

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 ヒント: Flaskを使ったWebアプリを実行する際に [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) を加えると、サーバを再起動しなくてもアプリに加えた変更がすぐに反映されます。注意！本番アプリではこのモードを有効にしないでください。

`python app.py` もしくは `python3 app.py` を実行すると、Webサーバがローカルに起動し、UFOがどこで目撃されたのかという重要な質問に対する答えを、短いフォームに記入することで得られます。

その前に `app.py` を見てみましょう。

1. 最初に、依存関係が読み込まれてアプリが起動します。
1. 次に、モデルが読み込まれます。
1. 次に、ホームのパスで index.html がレンダリングされます。

`/predict` のパスにフォームを送信するといくつかのことが起こります。

1. フォームの変数が集められてnumpyの配列に変換されます。それらはモデルに送られ、予測が返されます。
2. 表示させたい国は、予測されたコードから読みやすい文字列に再レンダリングされて、index.html に送り返された後にテンプレートの中でレンダリングされます。

このように、Flaskとpickleされたモデルを使うのは比較的簡単です。一番難しいのは、予測を得るためにモデルに送らなければならないデータがどのような形をしているかを理解することです。それはモデルがどのように学習されたかによります。今回の場合は、予測を得るために入力すべきデータが3つあります。

プロの現場では、モデルを学習する人と、それをWebやモバイルアプリで使用する人との間に、良好なコミュニケーションが必要であることがわかります。今回はたった一人の人間であり、それはあなたです！

---

## 🚀 チャレンジ

ノートブックで作業してモデルをFlaskアプリにインポートする代わりに、Flaskアプリの中でモデルをトレーニングすることができます。おそらくデータをクリーニングした後になりますが、ノートブック内のPythonコードを変換して、アプリ内の `train` というパスでモデルを学習してみてください。この方法を採用することの長所と短所は何でしょうか？

## [講義後の小テスト](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18?loc=ja)

## 振り返りと自主学習

機械学習モデルを使用するWebアプリを構築する方法はたくさんあります。JavaScript やPythonを使って機械学習を活用するWebアプリを構築する方法を挙げてください。アーキテクチャに関する検討: モデルはアプリ内に置くべきでしょうか？それともクラウドに置くべきでしょうか？後者の場合、どのようにアクセスするでしょうか？機械学習を使ったWebソリューションのアーキテクチャモデルを描いてください。

## 課題

[違うモデルを試す](assignment.ja.md)
