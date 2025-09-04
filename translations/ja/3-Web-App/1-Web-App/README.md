<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2680c691fbdb6367f350761a275e2508",
  "translation_date": "2025-09-03T23:44:19+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ja"
}
-->
# Webアプリで機械学習モデルを使用する

このレッスンでは、過去1世紀にわたる_UFO目撃情報_というユニークなデータセットを使用して機械学習モデルをトレーニングします。このデータはNUFORCのデータベースから取得されています。

学ぶ内容:

- トレーニング済みモデルを「ピクル化」する方法
- Flaskアプリでそのモデルを使用する方法

ノートブックを使ってデータをクリーンアップし、モデルをトレーニングする作業を続けますが、さらに一歩進めて、モデルを「実際の環境」で使用する方法を探求します。つまり、Webアプリで使用する方法です。

これを実現するには、Flaskを使用してWebアプリを構築する必要があります。

## [事前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## アプリを構築する

機械学習モデルを利用するWebアプリを構築する方法はいくつかあります。Webアーキテクチャは、モデルのトレーニング方法に影響を与える可能性があります。例えば、データサイエンスチームがトレーニングしたモデルをアプリで使用するよう依頼された場合を想像してください。

### 考慮事項

以下のような質問を検討する必要があります:

- **Webアプリかモバイルアプリか?** モバイルアプリを構築する場合やIoT環境でモデルを使用する必要がある場合、[TensorFlow Lite](https://www.tensorflow.org/lite/)を使用してAndroidやiOSアプリでモデルを利用できます。
- **モデルはどこに配置されるのか?** クラウドかローカルか?
- **オフライン対応。** アプリはオフラインで動作する必要があるか?
- **モデルのトレーニングに使用された技術は何か?** 選択された技術は使用するツールに影響を与える可能性があります。
    - **TensorFlowを使用する場合。** 例えば、TensorFlowを使用してモデルをトレーニングする場合、そのエコシステムでは[TensorFlow.js](https://www.tensorflow.org/js/)を使用してWebアプリでモデルを利用できるように変換する機能を提供しています。
    - **PyTorchを使用する場合。** [PyTorch](https://pytorch.org/)のようなライブラリを使用してモデルを構築する場合、[ONNX](https://onnx.ai/) (Open Neural Network Exchange)形式でエクスポートし、[Onnx Runtime](https://www.onnxruntime.ai/)を使用してJavaScript Webアプリで利用することができます。このオプションは、Scikit-learnでトレーニングされたモデルを使用する将来のレッスンで探求します。
    - **Lobe.aiやAzure Custom Visionを使用する場合。** [Lobe.ai](https://lobe.ai/)や[Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott)のようなML SaaS (Software as a Service)システムを使用してモデルをトレーニングする場合、この種のソフトウェアは多くのプラットフォーム向けにモデルをエクスポートする方法を提供します。これには、オンラインアプリケーションでクラウドからクエリされるカスタムAPIを構築することも含まれます。

また、ブラウザ内でモデルをトレーニングできるFlask Webアプリ全体を構築する機会もあります。これもJavaScript環境でTensorFlow.jsを使用して実現可能です。

私たちの目的では、Pythonベースのノートブックを使用してきたので、トレーニング済みモデルをノートブックからPythonで構築されたWebアプリで読み取れる形式にエクスポートする手順を探ってみましょう。

## ツール

このタスクには、FlaskとPickleという2つのツールが必要です。どちらもPython上で動作します。

✅ [Flask](https://palletsprojects.com/p/flask/)とは? Flaskはその開発者によって「マイクロフレームワーク」と定義されており、Pythonを使用してWebページを構築するためのテンプレートエンジンを備えたWebフレームワークの基本機能を提供します。[この学習モジュール](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)を確認して、Flaskを使った構築を練習してください。

✅ [Pickle](https://docs.python.org/3/library/pickle.html)とは? Pickle 🥒はPythonモジュールで、Pythonオブジェクト構造をシリアル化およびデシリアル化します。モデルを「ピクル化」する際には、その構造をシリアル化またはフラット化してWebで使用できるようにします。ただし、Pickleは本質的に安全ではないため、ファイルを「アンピクル化」するよう求められた場合は注意してください。ピクル化されたファイルの拡張子は`.pkl`です。

## 演習 - データをクリーンアップする

このレッスンでは、[NUFORC](https://nuforc.org) (全米UFO報告センター)によって収集された80,000件のUFO目撃情報を使用します。このデータには、以下のような興味深い目撃情報の説明が含まれています:

- **長い説明の例。** 「夜の草原に光のビームが照らされ、そこから男性が現れ、テキサス・インスツルメンツの駐車場に向かって走っていく」
- **短い説明の例。** 「光が私たちを追いかけてきた」

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv)スプレッドシートには、目撃が発生した`city`、`state`、`country`、物体の`shape`、およびその`latitude`と`longitude`に関する列が含まれています。

このレッスンに含まれる空の[ノートブック](notebook.ipynb)で以下を行います:

1. 前のレッスンで行ったように`pandas`、`matplotlib`、`numpy`をインポートし、UFOのスプレッドシートをインポートします。サンプルデータセットを確認できます:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. UFOデータを新しいタイトル付きの小さなデータフレームに変換します。`Country`フィールドのユニークな値を確認してください。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 次に、必要なデータ量を減らすために、null値を削除し、1〜60秒間の目撃情報のみをインポートします:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learnの`LabelEncoder`ライブラリをインポートして、国のテキスト値を数値に変換します:

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

次に、データをトレーニンググループとテストグループに分割してモデルをトレーニングする準備をします。

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

作成したモデルは非常に革新的ではありませんが、クリーンアップした生データからトレーニングし、エクスポートしてWebアプリで使用する練習としては良い例です。

## 演習 - モデルを「ピクル化」する

次に、モデルをピクル化します! 数行のコードでこれを行うことができます。ピクル化された後、ピクル化されたモデルをロードし、秒数、緯度、経度の値を含むサンプルデータ配列でテストします。

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

モデルは**「3」**を返します。これは英国の国コードです。驚きですね! 👽

## 演習 - Flaskアプリを構築する

次に、Flaskアプリを構築してモデルを呼び出し、より視覚的に魅力的な方法で結果を返します。

1. _notebook.ipynb_ファイルの隣に**web-app**というフォルダを作成します。その中に_ufo-model.pkl_ファイルを配置します。

1. そのフォルダ内にさらに3つのフォルダを作成します: **static** (その中に**css**フォルダを作成)、および**templates**。以下のファイルとディレクトリが作成されます:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 完成したアプリのビューについてはソリューションフォルダを参照してください

1. _web-app_フォルダ内で最初に作成するファイルは**requirements.txt**です。JavaScriptアプリの_package.json_のように、このファイルはアプリに必要な依存関係をリストします。**requirements.txt**に以下の行を追加します:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 次に、このファイルを_web-app_で実行します:

    ```bash
    cd web-app
    ```

1. ターミナルで`pip install`を入力して、_requirements.txt_にリストされたライブラリをインストールします:

    ```bash
    pip install -r requirements.txt
    ```

1. 次に、アプリを完成させるためにさらに3つのファイルを作成します:

    1. ルートに**app.py**を作成します。
    2. _templates_ディレクトリに**index.html**を作成します。
    3. _static/css_ディレクトリに**styles.css**を作成します。

1. _styles.css_ファイルをいくつかのスタイルで構築します:

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

1. 次に、_index.html_ファイルを構築します:

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

    このファイルのテンプレート構文を確認してください。変数がアプリによって提供される`{{}}`の「マスタッシュ」構文に注意してください。また、`/predict`ルートに予測を投稿するフォームも含まれています。

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

    > 💡 ヒント: Flaskを使用してWebアプリを実行する際に[`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)を追加すると、アプリケーションに加えた変更が即座に反映され、サーバーを再起動する必要がなくなります。ただし、注意してください! 本番アプリではこのモードを有効にしないでください。

`python app.py`または`python3 app.py`を実行すると、ローカルでWebサーバーが起動し、短いフォームに入力してUFO目撃情報に関する疑問に答えることができます!

その前に、`app.py`の各部分を確認してください:

1. まず、依存関係がロードされ、アプリが開始されます。
1. 次に、モデルがインポートされます。
1. その後、ホームルートでindex.htmlがレンダリングされます。

`/predict`ルートでは、フォームが投稿されると以下のことが行われます:

1. フォーム変数が収集され、numpy配列に変換されます。それがモデルに送信され、予測が返されます。
2. 表示したい国名が予測された国コードから読み取り可能なテキストとして再レンダリングされ、その値がindex.htmlに送信され、テンプレートでレンダリングされます。

このようにして、Flaskとピクル化されたモデルを使用してモデルを利用するのは比較的簡単です。最も難しいのは、予測を得るためにモデルに送信する必要があるデータの形状を理解することです。それはモデルがどのようにトレーニングされたかによります。このモデルでは、予測を得るために入力する必要があるデータポイントが3つあります。

プロフェッショナルな環境では、モデルをトレーニングする人々とWebやモバイルアプリでそれを利用する人々の間で良好なコミュニケーションが必要であることがわかります。私たちの場合、それはあなた一人です!

---

## 🚀 チャレンジ

ノートブックで作業してモデルをFlaskアプリにインポートする代わりに、Flaskアプリ内でモデルをトレーニングすることもできます! データをクリーンアップした後、ノートブック内のPythonコードを変換して、`train`というルートでアプリ内でモデルをトレーニングしてみてください。この方法を追求する利点と欠点は何でしょうか?

## [事後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## 復習と自己学習

機械学習モデルを利用するWebアプリを構築する方法は多数あります。JavaScriptやPythonを使用してWebアプリを構築し、機械学習を活用する方法のリストを作成してください。アーキテクチャを考慮してください: モデルはアプリ内に保持すべきか、それともクラウドに配置すべきか? 後者の場合、どのようにアクセスしますか? 応用ML Webソリューションのアーキテクチャモデルを描いてみてください。

## 課題

[別のモデルを試してみる](assignment.md)

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があることをご承知ください。元の言語で記載された文書が正式な情報源とみなされるべきです。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当方は一切の責任を負いません。