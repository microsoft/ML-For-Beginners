# 構建使用 ML 模型的 Web 應用程序

在本課中，你將在一個數據集上訓練一個 ML 模型，這個數據集來自世界各地：過去一個世紀的 UFO 目擊事件，來源於 [NUFORC 的數據庫](https://www.nuforc.org)。

你將學會：

- 如何「pickle」一個訓練有素的模型
- 如何在 Flask 應用程序中使用該模型

我們將繼續使用 notebook 來清理數據和訓練我們的模型，但你可以進一步探索在 web 應用程序中使用模型。

為此，你需要使用 Flask 構建一個 web 應用程序。

## [課前測](https://white-water-09ec41f0f.azurestaticapps.net/quiz/17/)

## 構建應用程序

有多種方法可以構建 Web 應用程序以使用機器學習模型。你的 web 架構可能會影響你的模型訓練方式。想象一下，你在一家企業工作，其中數據科學小組已經訓練了他們希望你在應用程序中使用的模型。

### 註意事項

你需要問很多問題：

- **它是 web 應用程序還是移動應用程序？** 如果你正在構建移動應用程序或需要在物聯網環境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/) 並在 Android 或 iOS 應用程序中使用該模型。
- **模型放在哪裏？** 在雲端還是本地？
- **離線支持**。該應用程序是否必須離線工作？
- **使用什麽技術來訓練模型？** 所選的技術可能會影響你需要使用的工具。
   - **使用 TensorFlow**。例如，如果你正在使用 TensorFlow 訓練模型，則該生態系統提供了使用 [TensorFlow.js](https://www.tensorflow.org/js/) 轉換 TensorFlow 模型以便在Web應用程序中使用的能力。
   - **使用 PyTorch**。如果你使用 [PyTorch](https://pytorch.org/) 等庫構建模型，則可以選擇將其導出到 [ONNX](https://onnx.ai/)（開放神經網絡交換）格式，用於可以使用 [Onnx Runtime](https://www.onnxruntime.ai/)的JavaScript Web 應用程序。此選項將在 Scikit-learn-trained 模型的未來課程中進行探討。
   - **使用 Lobe.ai 或 Azure 自定義視覺**。如果你使用 ML SaaS（軟件即服務）系統，例如 [Lobe.ai](https://lobe.ai/) 或 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 來訓練模型，這種類型的軟件提供了為許多平臺導出模型的方法，包括構建一個定製A PI，供在線應用程序在雲中查詢。

你還有機會構建一個完整的 Flask Web 應用程序，該應用程序能夠在 Web瀏覽器中訓練模型本身。這也可以在 JavaScript 上下文中使用 TensorFlow.js 來完成。

出於我們的目的，既然我們一直在使用基於 Python 的 notebook，那麽就讓我們探討一下將經過訓練的模型從 notebook 導出為 Python 構建的 web 應用程序可讀的格式所需要采取的步驟。

## 工具

對於此任務，你需要兩個工具：Flask 和 Pickle，它們都在 Python 上運行。

✅ 什麽是 [Flask](https://palletsprojects.com/p/flask/)？ Flask 被其創建者定義為「微框架」，它提供了使用 Python 和模板引擎構建網頁的 Web 框架的基本功能。看看[本學習單元](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)練習使用 Flask 構建應用程序。

✅ 什麽是 [Pickle](https://docs.python.org/3/library/pickle.html)？ Pickle🥒是一個 Python 模塊，用於序列化和反序列化 Python 對象結構。當你「pickle」一個模型時，你將其結構序列化或展平以在 Web 上使用。小心：pickle 本質上不是安全的，所以如果提示「un-pickle」文件，請小心。生產的文件具有後綴 `.pkl`。

## 練習 - 清理你的數據 

在本課中，你將使用由 [NUFORC](https://nuforc.org)（國家 UFO 報告中心）收集的 80,000 次 UFO 目擊數據。這些數據對 UFO 目擊事件有一些有趣的描述，例如：

- **詳細描述**。"一名男子從夜間照射在草地上的光束中出現，他朝德克薩斯儀器公司的停車場跑去"。
- **簡短描述**。 「燈光追著我們」。

[ufos.csv](./data/ufos.csv) 電子表格包括有關目擊事件發生的 `city`、`state` 和 `country`、對象的 `shape` 及其 `latitude` 和 `longitude` 的列。

在包含在本課中的空白 [notebook](notebook.ipynb) 中：

1. 像在之前的課程中一樣導入 `pandas`、`matplotlib` 和 `numpy`，然後導入 ufos 電子表格。你可以查看一個示例數據集：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('../data/ufos.csv')
    ufos.head()
    ```

2. 將 ufos 數據轉換為帶有新標題的小 dataframe。檢查 `country` 字段中的唯一值。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

3. 現在，你可以通過刪除任何空值並僅導入 1-60 秒之間的目擊數據來減少我們需要處理的數據量：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

4. 導入 Scikit-learn 的 `LabelEncoder` 庫，將國家的文本值轉換為數字：

   ✅ LabelEncoder 按字母順序編碼數據

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    你的數據應如下所示：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3	    53.200000	-2.916667
    3	20.0	4	    28.978333	-96.645833
    14	30.0	4	    35.823889	-80.253611
    23	60.0	4	    45.582778	-122.352222
    24	3.0	    3	    51.783333	-0.783333
    ```

## 練習 - 建立你的模型

現在，你可以通過將數據劃分為訓練和測試組來準備訓練模型。

1. 選擇要訓練的三個特征作為 X 向量，y 向量將是 `Country` 你希望能夠輸入 `Seconds`、`Latitude` 和 `Longitude` 並獲得要返回的國家/地區 ID。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

2. 使用邏輯回歸訓練模型：

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

準確率還不錯 **（大約 95%）**，不出所料，因為 `Country` 和 `Latitude/Longitude` 相關。

你創建的模型並不是非常具有革命性，因為你應該能夠從其 `Latitude` 和 `Longitude` 推斷出 `Country`，但是，嘗試從清理、導出的原始數據進行訓練，然後在 web 應用程序中使用此模型是一個很好的練習。

## 練習 - 「pickle」你的模型

現在，是時候 _pickle_ 你的模型了！你可以在幾行代碼中做到這一點。一旦它是 _pickled_，加載你的 pickled 模型並針對包含秒、緯度和經度值的示例數據數組對其進行測試，

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

該模型返回 **'3'**，這是英國的國家代碼。👽

## 練習 - 構建Flask應用程序 

現在你可以構建一個Flask應用程序來調用你的模型並返回類似的結果，但以一種更美觀的方式。

1. 首先在你的 _ufo-model.pkl_ 文件所在的 _notebook.ipynb_ 文件旁邊創建一個名為 **web-app** 的文件夾。

2. 在該文件夾中創建另外三個文件夾：**static**，其中有文件夾 **css** 和 **templates**。 你現在應該擁有以下文件和目錄

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ``` 

   ✅ 請參閱解決方案文件夾以查看已完成的應用程序

3. 在 _web-app_ 文件夾中創建的第一個文件是 **requirements.txt** 文件。與 JavaScript 應用程序中的 _package.json_ 一樣，此文件列出了應用程序所需的依賴項。在 **requirements.txt** 中添加以下幾行：  

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

4. 現在，進入 web-app 文件夾：

   ```bash
   cd web-app
   ```

5. 在你的終端中輸入 `pip install`，以安裝 _reuirements.txt_ 中列出的庫：

   ```bash
   pip install -r requirements.txt
   ```

6. 現在，你已準備好創建另外三個文件來完成應用程序：

    1. 在根目錄中創建 **app.py**。
    2. 在 _templates_ 目錄中創建**index.html**。
    3. 在 _static/css_ 目錄中創建**styles.css**。

7. 使用一些樣式構建 _styles.css_ 文件：

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

8. 接下來，構建 _index.html_ 文件：

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

    看看這個文件中的模板。請註意應用程序將提供的變量周圍的「mustache」語法，例如預測文本：`{{}}`。還有一個表單可以將預測發布到 `/predict` 路由。

    最後，你已準備好構建使用模型和顯示預測的 python 文件：

9. 在`app.py`中添加:

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

   > 💡 提示：當你在使用 Flask 運行 Web 應用程序時添加 [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)時你對應用程序所做的任何更改將立即反映，無需重新啟動服務器。註意！不要在生產應用程序中啟用此模式

如果你運行 `python app.py` 或 `python3 app.py` - 你的網絡服務器在本地啟動，你可以填寫一個簡短的表格來回答你關於在哪裏看到 UFO 的問題！

在此之前，先看一下 `app.py` 的實現：

1. 首先，加載依賴項並啟動應用程序。
2. 然後，導入模型。
3. 然後，在 home 路由上渲染 index.html。

在 `/predict` 路由上，當表單被發布時會發生幾件事情： 

1. 收集表單變量並轉換為 numpy 數組。然後將它們發送到模型並返回預測。
2. 我們希望顯示的國家/地區根據其預測的國家/地區代碼重新呈現為可讀文本，並將該值發送回 index.html 以在模板中呈現。

以這種方式使用模型，包括 Flask 和 pickled 模型，是相對簡單的。最困難的是要理解數據是什麽形狀的，這些數據必須發送到模型中才能得到預測。這完全取決於模型是如何訓練的。有三個數據要輸入，以便得到一個預測。

在一個專業的環境中，你可以看到訓練模型的人和在 Web 或移動應用程序中使用模型的人之間的良好溝通是多麽的必要。在我們的情況下，只有一個人，你！

---

## 🚀 挑戰

你可以在 Flask 應用程序中訓練模型，而不是在 notebook 上工作並將模型導入 Flask 應用程序！嘗試在 notebook 中轉換 Python 代碼，可能是在清除數據之後，從應用程序中的一個名為 `train` 的路徑訓練模型。采用這種方法的利弊是什麽？

## [課後測](https://white-water-09ec41f0f.azurestaticapps.net/quiz/18/)

## 復習與自學

有很多方法可以構建一個Web應用程序來使用ML模型。列出可以使用JavaScript或Python構建Web應用程序以利用機器學習的方法。考慮架構：模型應該留在應用程序中還是存在於雲中？如果是後者，你將如何訪問它？為應用的ML Web解決方案繪製架構模型。

## 任務

[嘗試不同的模型](./assignment.zh-tw.md)


