<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T09:35:43+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ja"
}
-->
# Webアプリで機械学習モデルを活用する

このレッスンでは、機械学習モデルをトレーニングし、ユニークなデータセットを使用します。それは、_過去100年間のUFO目撃情報_ であり、NUFORCのデータベースから取得されたものです。

学ぶ内容:

- トレーニング済みモデルを「ピクル化」する方法
- Flaskアプリでそのモデルを使用する方法

ノートブックを使ってデータをクリーンアップし、モデルをトレーニングする作業を続けますが、さらに一歩進めて、モデルを「実際の環境」で使用する方法を探ります。つまり、Webアプリでの活用です。

これを実現するには、Flaskを使用してWebアプリを構築する必要があります。

## [講義前のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## アプリを構築する

機械学習モデルを利用するWebアプリを構築する方法は複数あります。Webアーキテクチャは、モデルのトレーニング方法に影響を与える可能性があります。例えば、データサイエンスチームがトレーニングしたモデルをアプリで使用するよう求められるビジネス環境を想像してみてください。

### 考慮事項

以下のような質問をする必要があります:

- **Webアプリかモバイルアプリか？** モバイルアプリを構築する場合やIoT環境でモデルを使用する必要がある場合、[TensorFlow Lite](https://www.tensorflow.org/lite/)を使用して、AndroidやiOSアプリでモデルを活用できます。
- **モデルはどこに配置されるのか？** クラウドかローカルか？
- **オフライン対応。** アプリはオフラインで動作する必要があるか？
- **モデルのトレーニングに使用された技術は何か？** 選択された技術は、使用するツールに影響を与える可能性があります。
    - **TensorFlowを使用する場合。** 例えば、TensorFlowを使用してモデルをトレーニングする場合、そのエコシステムは[TensorFlow.js](https://www.tensorflow.org/js/)を使用してWebアプリでモデルを活用するための変換機能を提供します。
    - **PyTorchを使用する場合。** [PyTorch](https://pytorch.org/)のようなライブラリを使用してモデルを構築する場合、[ONNX](https://onnx.ai/) (Open Neural Network Exchange)形式でエクスポートし、[Onnx Runtime](https://www.onnxruntime.ai/)を使用してJavaScript Webアプリで活用することができます。このオプションは、Scikit-learnでトレーニングされたモデルを使用する将来のレッスンで探ります。
    - **Lobe.aiやAzure Custom Visionを使用する場合。** [Lobe.ai](https://lobe.ai/)や[Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott)のようなML SaaS (Software as a Service)システムを使用してモデルをトレーニングする場合、この種のソフトウェアは、クラウドでオンラインアプリがクエリを実行するためのカスタムAPIを構築するなど、さまざまなプラットフォーム向けにモデルをエクスポートする方法を提供します。

また、ブラウザ内でモデルをトレーニングできるFlask Webアプリ全体を構築することも可能です。これもJavaScript環境でTensorFlow.jsを使用して実現できます。

私たちの目的では、Pythonベースのノートブックを使用してきたので、トレーニング済みモデルをノートブックからPythonで構築されたWebアプリで読み取れる形式にエクスポートする手順を探ります。

## ツール

このタスクには、FlaskとPickleという2つのツールが必要です。どちらもPython上で動作します。

✅ [Flask](https://palletsprojects.com/p/flask/)とは？ Flaskはその開発者によって「マイクロフレームワーク」と定義されており、Pythonを使用してWebページを構築するためのテンプレートエンジンを提供します。[この学習モジュール](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)を確認して、Flaskを使った構築を練習してください。

✅ [Pickle](https://docs.python.org/3/library/pickle.html)とは？ Pickle 🥒はPythonモジュールで、Pythonオブジェクト構造をシリアル化およびデシリアル化します。モデルを「ピクル化」する際には、その構造をシリアル化またはフラット化してWebで使用できるようにします。ただし、Pickleは本質的に安全ではないため、ファイルを「アンピクル化」するよう求められた場合は注意してください。ピクル化されたファイルの拡張子は`.pkl`です。

## 演習 - データをクリーンアップする

このレッスンでは、[NUFORC](https://nuforc.org) (The National UFO Reporting Center) によって収集された80,000件のUFO目撃情報を使用します。このデータには、以下のような興味深い目撃情報の説明が含まれています:

- **長い説明の例。** 「夜の草原に光のビームが照らされ、そこから男性が現れ、テキサスインスツルメンツの駐車場に向かって走る」
- **短い説明の例。** 「光が私たちを追いかけてきた」

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv)スプレッドシートには、目撃が発生した`city`、`state`、`country`、物体の`shape`、およびその`latitude`と`longitude`に関する列が含まれています。

このレッスンに含まれる空白の[ノートブック](../../../../3-Web-App/1-Web-App/notebook.ipynb)で以下を行います:

1. 前のレッスンで行ったように`pandas`、`matplotlib`、`numpy`をインポートし、UFOスプレッドシートをインポートします。サンプルデータセットを確認できます:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. UFOデータを新しいタイトルで小さなデータフレームに変換します。`Country`フィールドのユニークな値を確認してください。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 必要なデータ量を減らすために、null値を削除し、1〜60秒間の目撃情報のみをインポートします:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learnの`LabelEncoder`ライブラリをインポートして、国名のテキスト値を数値に変換します:

    ✅ LabelEncoderはデータをアルファベット順にエンコードします

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    データは以下のようになります:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 演習 - モデルを構築する

次に、データをトレーニング用とテスト用に分割してモデルをトレーニングする準備をします。

1. トレーニングする3つの特徴をXベクトルとして選択し、yベクトルは`Country`になります。`Seconds`、`Latitude`、`Longitude`を入力して国IDを返すようにします。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ロジスティック回帰を使用してモデルをトレーニングします:

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

精度は悪くありません **(約95%)**。`Country`と`Latitude/Longitude`が相関しているため、当然の結果です。

作成したモデルは非常に革新的なものではありませんが、クリーンアップした生データからトレーニングし、エクスポートしてWebアプリで使用する練習としては良い例です。

## 演習 - モデルを「ピクル化」する

次に、モデルを「ピクル化」します！数行のコードでこれを実現できます。ピクル化された後、ピクル化されたモデルをロードし、秒数、緯度、経度の値を含むサンプルデータ配列でテストします。

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

モデルは **「3」** を返します。これは英国の国コードです。驚きですね！👽

## 演習 - Flaskアプリを構築する

次に、Flaskアプリを構築してモデルを呼び出し、より視覚的に魅力的な方法で結果を返します。

1. _notebook.ipynb_ ファイルの隣に**web-app**というフォルダを作成します。その中に _ufo-model.pkl_ ファイルを配置します。

1. そのフォルダ内にさらに3つのフォルダを作成します: **static** (その中に**css**フォルダを作成)、および**templates**。以下のファイルとディレクトリが揃っているはずです:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 完成したアプリのビューについてはソリューションフォルダを参照してください

1. 最初に _web-app_ フォルダ内に**requirements.txt**ファイルを作成します。JavaScriptアプリの _package.json_ のように、このファイルはアプリに必要な依存関係をリストします。**requirements.txt**に以下を追加します:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 次に、このファイルを _web-app_ で実行します:

    ```bash
    cd web-app
    ```

1. ターミナルで`pip install`と入力し、_requirements.txt_ にリストされたライブラリをインストールします:

    ```bash
    pip install -r requirements.txt
    ```

1. 次に、アプリを完成させるためにさらに3つのファイルを作成します:

    1. ルートに**app.py**を作成します。
    2. _templates_ ディレクトリに**index.html**を作成します。
    3. _static/css_ ディレクトリに**styles.css**を作成します。

1. _styles.css_ ファイルにいくつかのスタイルを追加します:

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

1. 次に、_index.html_ ファイルを構築します:

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

    このファイルのテンプレート構文を確認してください。変数がアプリによって提供される`{{}}`の「マスタッシュ」構文に注目してください。また、`/predict`ルートに予測を投稿するフォームがあります。

    最後に、モデルの消費と予測の表示を駆動するPythonファイルを構築します:

1. `app.py`に以下を追加します:

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

    > 💡 ヒント: Flaskを使用してWebアプリを実行する際に[`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)を追加すると、アプリケーションに加えた変更が即座に反映され、サーバーを再起動する必要がなくなります。ただし、注意してください！本番アプリではこのモードを有効にしないでください。

`python app.py`または`python3 app.py`を実行すると、ローカルでWebサーバーが起動し、短いフォームに記入してUFO目撃情報に関する質問に答えを得ることができます！

その前に、`app.py`の各部分を確認してください:

1. まず、依存関係がロードされ、アプリが開始されます。
1. 次に、モデルがインポートされます。
1. その後、ホームルートでindex.htmlがレンダリングされます。

`/predict`ルートでは、フォームが投稿されると以下のことが行われます:

1. フォーム変数が収集され、numpy配列に変換されます。それがモデルに送られ、予測が返されます。
2. 表示したい国名が予測された国コードから読み取り可能なテキストとして再レンダリングされ、その値がindex.htmlに送られてテンプレートでレンダリングされます。

Flaskとピクル化されたモデルを使用してモデルを活用する方法は比較的簡単です。最も難しいのは、モデルに予測を得るために送信する必要があるデータの形状を理解することです。それはモデルがどのようにトレーニングされたかによります。このモデルでは、予測を得るために入力する必要があるデータポイントが3つあります。

プロフェッショナルな環境では、モデルをトレーニングする人々とWebやモバイルアプリでそれを活用する人々の間で良好なコミュニケーションが必要であることがわかります。私たちの場合、それはあなた一人です！

---

## 🚀 チャレンジ

ノートブックで作業し、モデルをFlaskアプリにインポートする代わりに、Flaskアプリ内でモデルをトレーニングすることができます！データをクリーンアップした後、ノートブック内のPythonコードをアプリ内でトレーニングするように変換してみてください。この方法を追求する利点と欠点は何でしょうか？

## [講義後のクイズ](https://ff-quizzes.netlify.app/en/ml/)

## 復習と自己学習

機械学習モデルを活用するWebアプリを構築する方法は多数あります。JavaScriptやPythonを使用してWebアプリを構築し、機械学習を活用する方法のリストを作成してください。アーキテクチャを考慮してください: モデルはアプリ内に保持すべきか、それともクラウドに配置すべきか？後者の場合、どのようにアクセスしますか？応用されたML Webソリューションのアーキテクチャモデルを描いてみてください。

## 課題

[別のモデルを試してみる](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な表現が含まれる可能性があります。元の言語で記載された原文が正式な情報源と見なされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤訳について、当社は一切の責任を負いません。
