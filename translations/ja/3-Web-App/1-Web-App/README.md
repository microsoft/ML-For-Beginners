# MLモデルを使用するWebアプリを構築する

このレッスンでは、NUFORCのデータベースから取得した「過去1世紀のUFO目撃情報」という、まさに非現実的なデータセットでMLモデルを訓練します。

あなたが学ぶこと:

- 訓練されたモデルを「ピクル」する方法
- Flaskアプリでそのモデルを使用する方法

データをクリーンアップし、モデルを訓練するためにノートブックを引き続き使用しますが、プロセスを一歩進めて、いわば「野生で」モデルを使用する方法を探求します：それはWebアプリの中でのことです。

これを実現するために、Flaskを使用してWebアプリを構築する必要があります。

## [講義前クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## アプリの構築

機械学習モデルを消費するためのWebアプリを構築する方法はいくつかあります。あなたのWebアーキテクチャは、モデルがどのように訓練されるかに影響を与えるかもしれません。データサイエンスグループが訓練したモデルをアプリで使用したいビジネスで働いていると想像してください。

### 考慮事項

考慮すべき多くの質問があります：

- **Webアプリですか、それともモバイルアプリですか？** モバイルアプリを構築している場合や、IoTのコンテキストでモデルを使用する必要がある場合は、[TensorFlow Lite](https://www.tensorflow.org/lite/)を使用して、AndroidまたはiOSアプリでモデルを使用できます。
- **モデルはどこに配置されますか？** クラウド上かローカルか？
- **オフラインサポート。** アプリはオフラインで動作する必要がありますか？
- **モデルを訓練するために使用された技術は何ですか？** 選択した技術は、使用するツールに影響を与えるかもしれません。
    - **TensorFlowを使用する。** 例えば、TensorFlowを使用してモデルを訓練している場合、そのエコシステムは[TensorFlow.js](https://www.tensorflow.org/js/)を使用してWebアプリで利用するためにTensorFlowモデルを変換する機能を提供します。
    - **PyTorchを使用する。** [PyTorch](https://pytorch.org/)のようなライブラリを使用してモデルを構築している場合、JavaScriptのWebアプリで使用できる[ONNX](https://onnx.ai/)（Open Neural Network Exchange）形式でエクスポートするオプションがあります。このオプションは、Scikit-learnで訓練されたモデルの将来のレッスンで探求されます。
    - **Lobe.aiまたはAzure Custom Visionを使用する。** [Lobe.ai](https://lobe.ai/)や[Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott)のようなML SaaS（Software as a Service）システムを使用してモデルを訓練している場合、この種のソフトウェアは、オンラインアプリケーションからクラウドでクエリされるカスタムAPIを構築するなど、多くのプラットフォーム向けにモデルをエクスポートする方法を提供します。

また、ブラウザでモデルを自ら訓練できる完全なFlask Webアプリを構築する機会もあります。これもJavaScriptコンテキストでTensorFlow.jsを使用して実現できます。

私たちの目的のために、Pythonベースのノートブックを使用しているので、そのようなノートブックから訓練されたモデルをPythonで構築されたWebアプリが読み取れる形式にエクスポートする手順を探ってみましょう。

## ツール

このタスクには2つのツールが必要です：FlaskとPickle、どちらもPythonで動作します。

✅ [Flask](https://palletsprojects.com/p/flask/)とは何ですか？ Flaskはその作成者によって「マイクロフレームワーク」と定義されており、Pythonを使用してWebページを構築するための基本機能を提供します。Flaskでの構築を練習するために、[このLearnモジュール](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)を見てみてください。

✅ [Pickle](https://docs.python.org/3/library/pickle.html)とは何ですか？ Pickle 🥒はPythonオブジェクト構造をシリアライズおよびデシリアライズするPythonモジュールです。モデルを「ピクル」すると、Webで使用するためにその構造をシリアライズまたはフラット化します。注意してください：pickleは本質的に安全ではないため、ファイルを「アンピクル」するように求められた場合は注意してください。ピクルされたファイルには接尾辞`.pkl`があります。

## 演習 - データをクリーンアップする

このレッスンでは、[NUFORC](https://nuforc.org)（全米UFO報告センター）が収集した80,000件のUFO目撃データを使用します。このデータには、UFO目撃の興味深い説明が含まれています。例えば：

- **長い例の説明。** 「光のビームから男が現れ、夜の草原に照らされ、テキサス・インスツルメンツの駐車場に向かって走る」。
- **短い例の説明。** 「光が私たちを追いかけてきた」。

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv)スプレッドシートには、目撃が発生した`city`、`state`、`country`に関する列、オブジェクトの`shape`、およびその`latitude`と`longitude`が含まれています。

このレッスンに含まれる空の[ノートブック](../../../../3-Web-App/1-Web-App/notebook.ipynb)で：

1. 前のレッスンで行ったように`pandas`、`matplotlib`、`numpy`をインポートし、ufosスプレッドシートをインポートします。サンプルデータセットを確認できます：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. ufosデータを新しいタイトルの小さなデータフレームに変換します。`Country`フィールドのユニークな値を確認します。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 次に、null値を削除し、1〜60秒の間の目撃情報のみをインポートすることで、処理するデータの量を減らすことができます：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Scikit-learnの`LabelEncoder`ライブラリをインポートして、国のテキスト値を数値に変換します：

    ✅ LabelEncoderはデータをアルファベット順にエンコードします

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    あなたのデータは次のようになります：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 演習 - モデルを構築する

データを訓練グループとテストグループに分割して、モデルを訓練する準備が整いました。

1. 訓練に使用する3つの特徴をXベクトルとして選択し、yベクトルは`Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude`で、国のIDを返します。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. ロジスティック回帰を使用してモデルを訓練します：

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

精度は悪くありません **（約95%）**、予想通り、`Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`ですが、クリーンアップし、エクスポートした生データから訓練し、このモデルをWebアプリで使用することを試みるのは良い練習です。

## 演習 - モデルを「ピクル」する

さて、モデルを_ピクル_する時が来ました！ いくつかの行のコードでそれを行うことができます。一度_ピクル_したら、ピクルされたモデルを読み込み、秒、緯度、経度の値を含むサンプルデータ配列に対してテストします。

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

モデルは**「3」**を返します。これは、イギリスの国コードです。すごい！ 👽

## 演習 - Flaskアプリを構築する

次に、モデルを呼び出して似たような結果を返すFlaskアプリを構築できますが、より視覚的に魅力的な方法で。

1. _ufo-model.pkl_ファイルが存在する_notebook.ipynb_ファイルの隣に**web-app**という名前のフォルダを作成します。

1. そのフォルダ内に、**static**というフォルダとその中に**css**フォルダ、さらに**templates**というフォルダを作成します。これで、次のファイルとディレクトリが揃っているはずです：

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 完成したアプリのビューについては、ソリューションフォルダを参照してください。

1. _web-app_フォルダで最初に作成するファイルは**requirements.txt**ファイルです。JavaScriptアプリの_package.json_のように、このファイルにはアプリに必要な依存関係がリストされています。**requirements.txt**に次の行を追加します：

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 次に、このファイルを実行するために_web-app_に移動します：

    ```bash
    cd web-app
    ```

1. ターミナルに`pip install`と入力し、_requirements.txt_にリストされたライブラリをインストールします：

    ```bash
    pip install -r requirements.txt
    ```

1. これで、アプリを完成させるためにさらに3つのファイルを作成する準備が整いました：

    1. ルートに**app.py**を作成します。
    2. _templates_ディレクトリに**index.html**を作成します。
    3. _static/css_ディレクトリに**styles.css**を作成します。

1. _styles.css_ファイルにいくつかのスタイルを追加します：

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

1. 次に、_index.html_ファイルを構築します：

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

    このファイルのテンプレートに目を通してください。アプリによって提供される変数の周りの「マスタッシュ」構文、例えば予測テキスト：`{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py`を追加します：

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

    > 💡 ヒント：[`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`を追加したときに、アプリのエラーを特定するのが簡単になります。この方法の利点と欠点は何ですか？

## [講義後クイズ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## まとめと自己学習

MLモデルを消費するためのWebアプリを構築する方法はいくつかあります。JavaScriptまたはPythonを使用して機械学習を活用するWebアプリを構築する方法のリストを作成してください。アーキテクチャを考慮してください：モデルはアプリ内に留まるべきか、それともクラウドに存在するべきか？ 後者の場合、どのようにアクセスしますか？ 実用的なML Webソリューションのアーキテクチャモデルを描いてみてください。

## 課題

[別のモデルを試す](assignment.md)

**免責事項**:  
この文書は、機械ベースのAI翻訳サービスを使用して翻訳されています。正確性を追求していますが、自動翻訳には誤りや不正確さが含まれる可能性があることをご理解ください。元の文書は、その原語での権威ある情報源と見なされるべきです。重要な情報については、専門の人間翻訳を推奨します。この翻訳の使用から生じる誤解や誤訳について、当社は一切の責任を負いません。