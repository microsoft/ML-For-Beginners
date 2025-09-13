<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T09:28:03+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "hk"
}
-->
# 建立一個使用機器學習模型的網頁應用程式

在這節課中，你將使用一個非常特別的數據集來訓練機器學習模型：_過去一個世紀的UFO目擊事件_，數據來源於NUFORC的資料庫。

你將學習：

- 如何將訓練好的模型進行“pickle”處理
- 如何在Flask應用程式中使用該模型

我們將繼續使用筆記本來清理數據並訓練模型，但你可以更進一步，探索如何在“真實世界”中使用模型：例如在網頁應用程式中。

為了做到這一點，你需要使用Flask來建立一個網頁應用程式。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 建立應用程式

有多種方法可以建立網頁應用程式來使用機器學習模型。你的網頁架構可能會影響模型的訓練方式。想像一下，你在一家公司工作，數據科學團隊已經訓練了一個模型，他們希望你在應用程式中使用該模型。

### 考慮事項

你需要問自己許多問題：

- **是網頁應用程式還是移動應用程式？** 如果你正在建立移動應用程式或需要在物聯網（IoT）環境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/) 並在Android或iOS應用程式中使用該模型。
- **模型將存放在哪裡？** 是在雲端還是本地？
- **離線支援。** 應用程式是否需要離線運作？
- **使用什麼技術訓練模型？** 選擇的技術可能會影響你需要使用的工具。
    - **使用TensorFlow。** 如果你使用TensorFlow訓練模型，該生態系統提供了將TensorFlow模型轉換為網頁應用程式使用的能力，例如使用 [TensorFlow.js](https://www.tensorflow.org/js/)。
    - **使用PyTorch。** 如果你使用像 [PyTorch](https://pytorch.org/) 這樣的庫來建立模型，你可以選擇將其導出為 [ONNX](https://onnx.ai/)（開放神經網絡交換格式），以便在使用 [Onnx Runtime](https://www.onnxruntime.ai/) 的JavaScript網頁應用程式中使用。這種選項將在未來的課程中探索，適用於使用Scikit-learn訓練的模型。
    - **使用Lobe.ai或Azure Custom Vision。** 如果你使用像 [Lobe.ai](https://lobe.ai/) 或 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 這樣的機器學習SaaS（軟件即服務）系統來訓練模型，這類軟件提供了多平台導出模型的方法，包括建立一個定制的API，供你的線上應用程式在雲端查詢。

你還可以建立一個完整的Flask網頁應用程式，該應用程式可以在網頁瀏覽器中自行訓練模型。這也可以在JavaScript環境中使用TensorFlow.js來完成。

對於我們的目的，由於我們一直在使用基於Python的筆記本，讓我們來探索如何將訓練好的模型從筆記本導出為Python建立的網頁應用程式可讀的格式。

## 工具

完成這項任務，你需要兩個工具：Flask和Pickle，它們都在Python上運行。

✅ [Flask](https://palletsprojects.com/p/flask/) 是什麼？由其創建者定義為“微框架”，Flask提供了使用Python和模板引擎建立網頁的基本功能。查看 [這個學習模組](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) 來練習使用Flask建立應用程式。

✅ [Pickle](https://docs.python.org/3/library/pickle.html) 是什麼？Pickle 🥒 是一個Python模組，用於序列化和反序列化Python物件結構。當你對模型進行“pickle”處理時，你會將其結構序列化或扁平化，以便在網頁上使用。注意：Pickle本身並不安全，因此如果被要求“un-pickle”文件時要小心。Pickle文件的後綴是 `.pkl`。

## 練習 - 清理數據

在這節課中，你將使用來自 [NUFORC](https://nuforc.org)（全國UFO報告中心）的80,000個UFO目擊事件數據。這些數據包含一些有趣的UFO目擊描述，例如：

- **長描述範例。** “一束光照在夜晚的草地上，一名男子從光束中走出，跑向德州儀器的停車場。”
- **短描述範例。** “燈光追著我們。”

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) 試算表包含有關目擊事件的 `city`、`state` 和 `country`，物體的 `shape` 以及其 `latitude` 和 `longitude` 的欄位。

在這節課提供的空白 [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) 中：

1. 像之前的課程一樣，導入 `pandas`、`matplotlib` 和 `numpy`，並導入UFO試算表。你可以查看一個數據集範例：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. 將UFO數據轉換為一個小型數據框，並重新命名欄位。檢查 `Country` 欄位中的唯一值。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 現在，你可以通過刪除任何空值並僅導入1-60秒之間的目擊事件來減少需要處理的數據量：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. 導入Scikit-learn的 `LabelEncoder` 庫，將國家的文字值轉換為數字：

    ✅ LabelEncoder 按字母順序編碼數據

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    你的數據應該看起來像這樣：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 練習 - 建立模型

現在你可以準備通過將數據分為訓練和測試組來訓練模型。

1. 選擇三個特徵作為你的X向量，y向量將是 `Country`。你希望能夠輸入 `Seconds`、`Latitude` 和 `Longitude`，並返回一個國家ID。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. 使用邏輯回歸訓練你的模型：

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

準確率不錯 **（約95%）**，這並不意外，因為 `Country` 和 `Latitude/Longitude` 是相關的。

你建立的模型並不算非常革命性，因為你應該能夠從 `Latitude` 和 `Longitude` 推斷出 `Country`，但這是一個很好的練習，嘗試從清理過的原始數據中訓練模型，導出模型，然後在網頁應用程式中使用它。

## 練習 - 將模型進行“pickle”處理

現在，是時候對你的模型進行 _pickle_ 處理了！你可以用幾行代碼完成這個操作。一旦完成 _pickle_ 處理，載入你的Pickle模型並用一個包含秒數、緯度和經度的樣本數據陣列進行測試，

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

模型返回 **'3'**，這是英國的國家代碼。太神奇了！👽

## 練習 - 建立Flask應用程式

現在你可以建立一個Flask應用程式來調用你的模型並以更具視覺吸引力的方式返回類似結果。

1. 首先在 _notebook.ipynb_ 文件旁邊建立一個名為 **web-app** 的資料夾，該資料夾中存放你的 _ufo-model.pkl_ 文件。

1. 在該資料夾中建立三個子資料夾：**static**，其中包含一個名為 **css** 的資料夾，以及 **templates**。你現在應該有以下文件和目錄：

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 參考解決方案資料夾以查看完成的應用程式

1. 在 _web-app_ 資料夾中建立第一個文件 **requirements.txt**。像JavaScript應用程式中的 _package.json_ 一樣，這個文件列出了應用程式所需的依賴項。在 **requirements.txt** 中添加以下行：

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 現在，通過導航到 _web-app_ 運行此文件：

    ```bash
    cd web-app
    ```

1. 在終端中輸入 `pip install`，以安裝 _requirements.txt_ 中列出的庫：

    ```bash
    pip install -r requirements.txt
    ```

1. 現在，你可以建立另外三個文件來完成應用程式：

    1. 在根目錄建立 **app.py**。
    2. 在 _templates_ 資料夾中建立 **index.html**。
    3. 在 _static/css_ 資料夾中建立 **styles.css**。

1. 在 _styles.css_ 文件中添加一些樣式：

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

1. 接下來，建立 _index.html_ 文件：

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

    查看此文件中的模板語法。注意變數周圍的“大括號”語法，例如預測文字：`{{}}`。還有一個表單，它將預測結果發送到 `/predict` 路由。

    最後，你準備好建立驅動模型消耗和預測顯示的Python文件：

1. 在 `app.py` 中添加：

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

    > 💡 提示：當你在使用Flask運行網頁應用程式時添加 [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)，你對應用程式所做的任何更改都會立即反映，而無需重新啟動伺服器。注意！不要在生產應用程式中啟用此模式。

如果你運行 `python app.py` 或 `python3 app.py` - 你的網頁伺服器將在本地啟動，你可以填寫一個簡短的表單來獲得關於UFO目擊地點的答案！

在此之前，查看 `app.py` 的各部分：

1. 首先，載入依賴項並啟動應用程式。
1. 然後，導入模型。
1. 接著，在主路由上渲染 index.html。

在 `/predict` 路由上，當表單提交時會發生以下幾件事：

1. 表單變數被收集並轉換為numpy陣列。然後它們被發送到模型並返回預測結果。
2. 我們希望顯示的國家代碼被重新渲染為可讀的文字，並將該值發送回 index.html，在模板中渲染。

以這種方式使用模型，結合Flask和Pickle模型，相對簡單。最困難的是理解必須發送到模型的數據形狀，以獲得預測結果。這完全取決於模型的訓練方式。這個模型需要輸入三個數據點才能獲得預測結果。

在專業環境中，你可以看到訓練模型的人和在網頁或移動應用程式中使用模型的人之間良好的溝通是多麼重要。在我們的情況下，只有一個人，那就是你！

---

## 🚀 挑戰

與其在筆記本中工作並將模型導入Flask應用程式，你可以直接在Flask應用程式中訓練模型！嘗試將筆記本中的Python代碼轉換為在應用程式內的 `train` 路由上訓練模型，可能是在清理數據之後。追求這種方法的優缺點是什麼？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

有許多方法可以建立網頁應用程式來使用機器學習模型。列出你可以使用JavaScript或Python建立網頁應用程式以利用機器學習的方法。考慮架構：模型應該保留在應用程式中還是存放在雲端？如果是後者，你將如何訪問它？繪製一個應用機器學習的網頁解決方案架構模型。

## 作業

[嘗試不同的模型](assignment.md)

---

**免責聲明**：  
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議使用專業的人工作業翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解讀概不負責。