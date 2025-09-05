<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T09:55:27+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "tw"
}
-->
# 建立一個使用機器學習模型的網頁應用程式

在這堂課中，你將使用一個來自外太空的數據集來訓練一個機器學習模型：_過去一個世紀的UFO目擊事件_，數據來源於NUFORC的資料庫。

你將學到：

- 如何將訓練好的模型進行“pickle”處理
- 如何在Flask應用程式中使用該模型

我們將繼續使用筆記本來清理數據並訓練模型，但你可以更進一步，探索如何在“真實世界”中使用模型，也就是在網頁應用程式中。

為此，你需要使用Flask來建立一個網頁應用程式。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

## 建立應用程式

有多種方式可以建立網頁應用程式來使用機器學習模型。你的網頁架構可能會影響模型的訓練方式。想像一下，你正在一家公司工作，數據科學團隊已經訓練了一個模型，他們希望你在應用程式中使用該模型。

### 考量因素

你需要問自己許多問題：

- **這是一個網頁應用程式還是行動應用程式？** 如果你正在建立一個行動應用程式或需要在物聯網環境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/)，並在Android或iOS應用程式中使用該模型。
- **模型將存放在哪裡？** 是在雲端還是本地？
- **是否需要離線支援？** 應用程式是否需要在離線狀態下運行？
- **訓練模型時使用了什麼技術？** 所選技術可能會影響你需要使用的工具。
    - **使用TensorFlow。** 如果你使用TensorFlow訓練模型，該生態系統提供了將TensorFlow模型轉換為可在網頁應用程式中使用的[TensorFlow.js](https://www.tensorflow.org/js/)的功能。
    - **使用PyTorch。** 如果你使用像[PyTorch](https://pytorch.org/)這樣的庫來建立模型，你可以選擇將其導出為[ONNX](https://onnx.ai/)（開放神經網絡交換格式），以便在可以使用[Onnx Runtime](https://www.onnxruntime.ai/)的JavaScript網頁應用程式中使用。這種選項將在未來的課程中探索，適用於Scikit-learn訓練的模型。
    - **使用Lobe.ai或Azure Custom Vision。** 如果你使用的是像[Lobe.ai](https://lobe.ai/)或[Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott)這樣的機器學習SaaS（軟體即服務）系統來訓練模型，這類軟體提供了多平台導出模型的方法，包括建立一個專屬的API，供你的線上應用程式在雲端查詢。

你還有機會建立一個完整的Flask網頁應用程式，該應用程式可以在網頁瀏覽器中自行訓練模型。這也可以使用JavaScript環境中的TensorFlow.js來完成。

對於我們的目的，因為我們一直在使用基於Python的筆記本，讓我們來探索將訓練好的模型從這樣的筆記本導出為Python構建的網頁應用程式可讀的格式所需的步驟。

## 工具

完成這項任務，你需要兩個工具：Flask和Pickle，這兩者都在Python上運行。

✅ [Flask](https://palletsprojects.com/p/flask/)是什麼？由其創建者定義為“微框架”，Flask提供了使用Python和模板引擎構建網頁的基本功能。查看[這個學習模組](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)來練習使用Flask構建應用程式。

✅ [Pickle](https://docs.python.org/3/library/pickle.html)是什麼？Pickle 🥒 是一個Python模組，用於序列化和反序列化Python對象結構。當你對模型進行“pickle”處理時，你會將其結構序列化或扁平化，以便在網頁上使用。注意：Pickle本身並不安全，因此如果被要求“un-pickle”一個文件時要小心。Pickle文件的後綴是`.pkl`。

## 練習 - 清理數據

在這堂課中，你將使用來自[NUFORC](https://nuforc.org)（全國UFO報告中心）的80,000條UFO目擊數據。這些數據包含一些有趣的UFO目擊描述，例如：

- **長描述範例。** “一名男子從夜晚照亮草地的光束中出現，並跑向德州儀器的停車場”。
- **短描述範例。** “燈光追逐著我們”。

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv)試算表包含有關目擊事件的`city`、`state`和`country`，物體的`shape`以及其`latitude`和`longitude`等欄位。

在這堂課提供的空白[筆記本](../../../../3-Web-App/1-Web-App/notebook.ipynb)中：

1. 像之前的課程一樣，匯入`pandas`、`matplotlib`和`numpy`，並匯入ufos試算表。你可以查看數據集的樣本：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. 將ufos數據轉換為一個小型的數據框，並重新命名欄位標題。檢查`Country`欄位中的唯一值。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 現在，你可以通過刪除任何空值並僅匯入1-60秒之間的目擊事件來減少需要處理的數據量：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. 匯入Scikit-learn的`LabelEncoder`庫，將國家的文字值轉換為數字：

    ✅ LabelEncoder會按字母順序編碼數據

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

現在，你可以準備通過將數據分為訓練組和測試組來訓練模型。

1. 選擇三個特徵作為你的X向量，y向量將是`Country`。你希望能輸入`Seconds`、`Latitude`和`Longitude`，並獲得一個國家代碼作為返回值。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. 使用邏輯回歸來訓練你的模型：

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

準確率還不錯 **（約95%）**，這並不令人意外，因為`Country`與`Latitude`和`Longitude`是相關的。

你創建的模型並不是非常革命性的，因為你應該能夠從`Latitude`和`Longitude`推斷出`Country`，但這是一個很好的練習，嘗試從你清理過的原始數據中訓練模型，導出模型，然後在網頁應用程式中使用它。

## 練習 - 將模型進行“pickle”處理

現在，是時候對你的模型進行_pickle_處理了！你可以用幾行代碼完成這個操作。一旦_pickle_處理完成，載入你的pickled模型，並用一個包含秒數、緯度和經度值的樣本數據陣列來測試它。

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

模型返回 **'3'**，這是英國的國家代碼。太酷了！👽

## 練習 - 建立Flask應用程式

現在，你可以建立一個Flask應用程式來調用你的模型，並以更具視覺吸引力的方式返回類似的結果。

1. 首先，在_notebook.ipynb_文件旁邊創建一個名為**web-app**的資料夾，該資料夾中存放你的_ufo-model.pkl_文件。

1. 在該資料夾中再創建三個資料夾：**static**（裡面有一個**css**資料夾）和**templates**。現在你應該有以下文件和目錄：

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 參考解決方案資料夾以查看完成的應用程式

1. 在_web-app_資料夾中創建第一個文件**requirements.txt**。像JavaScript應用程式中的_package.json_一樣，這個文件列出了應用程式所需的依賴項。在**requirements.txt**中添加以下內容：

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 現在，通過導航到_web-app_來運行此文件：

    ```bash
    cd web-app
    ```

1. 在終端中輸入`pip install`，以安裝_requirements.txt_中列出的庫：

    ```bash
    pip install -r requirements.txt
    ```

1. 現在，你已準備好創建另外三個文件來完成應用程式：

    1. 在根目錄中創建**app.py**。
    2. 在_templates_目錄中創建**index.html**。
    3. 在_static/css_目錄中創建**styles.css**。

1. 使用一些樣式來構建_styles.css_文件：

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

1. 接下來，構建_index.html_文件：

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

    查看此文件中的模板語法。注意變數周圍的“mustache”語法，例如：`{{}}`。還有一個表單，它將預測結果發送到`/predict`路由。

    最後，你已準備好構建驅動模型消費和顯示預測結果的Python文件：

1. 在`app.py`中添加：

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

    > 💡 提示：當你在使用Flask運行網頁應用程式時添加[`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)，任何對應用程式的更改都會立即反映，而無需重新啟動伺服器。注意！不要在生產應用程式中啟用此模式。

如果你運行`python app.py`或`python3 app.py`，你的本地網頁伺服器將啟動，你可以填寫一個簡短的表單，來回答你對UFO目擊地點的疑問！

在此之前，先看看`app.py`的各個部分：

1. 首先，載入依賴項並啟動應用程式。
1. 然後，匯入模型。
1. 接著，在主路由上渲染index.html。

在`/predict`路由上，當表單被提交時會發生以下幾件事：

1. 表單變數被收集並轉換為numpy陣列。然後將它們發送到模型，並返回一個預測結果。
2. 我們希望顯示的國家名稱從預測的國家代碼重新渲染為可讀的文本，並將該值發送回index.html以在模板中渲染。

使用Flask和pickled模型以這種方式使用模型相對簡單。最困難的部分是理解必須發送到模型的數據形狀，這取決於模型的訓練方式。這個模型需要輸入三個數據點才能獲得預測結果。

在專業環境中，你可以看到訓練模型的人員與在網頁或行動應用程式中使用模型的人員之間的良好溝通是多麼重要。在我們的案例中，這些工作都由你一人完成！

---

## 🚀 挑戰

與其在筆記本中工作並將模型匯入Flask應用程式，你可以直接在Flask應用程式中訓練模型！嘗試將筆記本中的Python代碼轉換為應用程式中的代碼，也許在清理數據後，從應用程式中的一個名為`train`的路由來訓練模型。嘗試這種方法的優缺點是什麼？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

有許多方式可以建立一個網頁應用程式來使用機器學習模型。列出你可以使用JavaScript或Python來建立網頁應用程式以利用機器學習的方法。考慮架構：模型應該保留在應用程式中還是存放在雲端？如果是後者，你將如何訪問它？畫出一個應用機器學習的網頁解決方案的架構模型。

## 作業

[嘗試一個不同的模型](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。