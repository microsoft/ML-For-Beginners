# 构建一个使用机器学习模型的Web应用

在本课中，你将使用一个非常特别的数据集训练一个机器学习模型：_过去一个世纪的UFO目击事件_，这些数据来源于NUFORC的数据库。

你将学习到：

- 如何“pickle”一个训练好的模型
- 如何在Flask应用中使用该模型

我们将继续使用notebook来清理数据和训练我们的模型，但你可以更进一步，探索在实际环境中使用模型：在一个web应用中。

要做到这一点，你需要使用Flask构建一个web应用。

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## 构建应用

有几种方法可以构建web应用来使用机器学习模型。你的web架构可能会影响你训练模型的方式。想象一下，你在一个企业中工作，数据科学团队已经训练了一个模型，他们希望你在应用中使用。

### 考虑因素

你需要问自己很多问题：

- **这是一个web应用还是一个移动应用？** 如果你正在构建一个移动应用或需要在物联网环境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/) 并在Android或iOS应用中使用该模型。
- **模型将驻留在哪里？** 在云端还是本地？
- **离线支持。** 应用是否需要离线工作？
- **使用什么技术训练模型？** 所选技术可能会影响你需要使用的工具。
    - **使用TensorFlow。** 如果你使用TensorFlow训练模型，例如，该生态系统提供了使用 [TensorFlow.js](https://www.tensorflow.org/js/) 将TensorFlow模型转换为web应用使用的能力。
    - **使用PyTorch。** 如果你使用诸如 [PyTorch](https://pytorch.org/) 之类的库构建模型，你可以选择将其导出为 [ONNX](https://onnx.ai/) (开放神经网络交换) 格式，用于可以使用 [Onnx Runtime](https://www.onnxruntime.ai/) 的JavaScript web应用。这种选择将在未来的课程中探索，用于一个Scikit-learn训练的模型。
    - **使用Lobe.ai或Azure Custom Vision。** 如果你使用诸如 [Lobe.ai](https://lobe.ai/) 或 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 之类的ML SaaS（软件即服务）系统训练模型，这类软件提供了为多种平台导出模型的方法，包括构建一个定制API，通过你的在线应用在云端查询。

你还有机会构建一个完整的Flask web应用，该应用可以在web浏览器中自行训练模型。这也可以在JavaScript环境中使用TensorFlow.js完成。

为了我们的目的，因为我们一直在使用基于Python的notebook，让我们来探索将训练好的模型从这样的notebook导出为一个Python构建的web应用可读的格式所需的步骤。

## 工具

完成这项任务，你需要两个工具：Flask和Pickle，它们都运行在Python上。

✅ 什么是 [Flask](https://palletsprojects.com/p/flask/)? 由其创建者定义为“微框架”，Flask提供了使用Python和一个模板引擎构建网页的基本web框架功能。看看 [这个学习模块](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) 来练习使用Flask构建。

✅ 什么是 [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 是一个Python模块，用于序列化和反序列化Python对象结构。当你“pickle”一个模型时，你将其结构序列化或扁平化，以便在web上使用。注意：pickle本质上是不安全的，因此如果被提示“un-pickle”一个文件时要小心。一个pickled文件的后缀是 `.pkl`。

## 练习 - 清理数据

在本课中，你将使用由 [NUFORC](https://nuforc.org)（国家UFO报告中心）收集的80,000个UFO目击事件数据。这些数据包含一些有趣的UFO目击描述，例如：

- **长描述示例。** “一个人从夜间照在草地上的光束中出现，并跑向德州仪器的停车场”。
- **短描述示例。** “灯光追逐我们”。

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) 电子表格包括关于`city`、`state`和`country`的列，记录了目击事件发生的地点、物体的`shape`、其`latitude`和`longitude`。

在本课包含的空白 [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) 中：

1. 导入`pandas`、`matplotlib`和`numpy`，并导入ufos电子表格。你可以查看一个样本数据集：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. 将ufos数据转换为一个具有新标题的小数据框。检查`Country`字段中的唯一值。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. 现在，你可以通过删除任何空值并仅导入1-60秒之间的目击事件来减少需要处理的数据量：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. 导入Scikit-learn的`LabelEncoder`库，将国家的文本值转换为数字：

    ✅ LabelEncoder按字母顺序编码数据

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    你的数据应该看起来像这样：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 练习 - 构建你的模型

现在你可以通过将数据分为训练组和测试组来准备训练模型。

1. 选择你要训练的三个特征作为X向量，y向量将是`Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude`，并获得一个国家ID以返回。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. 使用逻辑回归训练你的模型：

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

准确率不错 **（约95%）**，不出所料，因为`Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`，但这是一个从你清理、导出并在web应用中使用的原始数据中尝试训练的好练习。

## 练习 - “pickle”你的模型

现在，是时候_pickle你的模型了！你可以用几行代码完成。一旦_pickle完毕，加载你的pickled模型并用包含秒数、纬度和经度值的样本数据数组进行测试，

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

模型返回**'3'**，这是英国的国家代码。太神奇了！👽

## 练习 - 构建一个Flask应用

现在你可以构建一个Flask应用来调用你的模型并返回类似的结果，但以更具视觉吸引力的方式。

1. 先在_notebook.ipynb_文件所在的地方创建一个名为**web-app**的文件夹，其中包含你的_ufo-model.pkl_文件。

1. 在该文件夹中创建三个文件夹：**static**，其中包含一个**css**文件夹，以及**templates**。你现在应该有以下文件和目录：

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 参考解决方案文件夹以查看完成的应用

1. 在_web-app_文件夹中创建的第一个文件是**requirements.txt**文件。就像JavaScript应用中的_package.json_一样，该文件列出了应用所需的依赖项。在**requirements.txt**中添加以下几行：

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 现在，通过导航到_web-app_运行此文件：

    ```bash
    cd web-app
    ```

1. 在你的终端中键入`pip install`，以安装_requirements.txt_中列出的库：

    ```bash
    pip install -r requirements.txt
    ```

1. 现在，你准备创建另外三个文件以完成应用：

    1. 在根目录中创建**app.py**。
    2. 在_templates_目录中创建**index.html**。
    3. 在_static/css_目录中创建**styles.css**。

1. 使用一些样式构建_styles.css_文件：

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

1. 接下来，构建_index.html_文件：

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

    查看此文件中的模板。注意将由应用提供的变量周围的“胡须”语法，如预测文本：`{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py`中添加：

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

    > 💡 提示：当你添加[`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

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

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`时。追求这种方法的优缺点是什么？

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## 复习与自学

有很多方法可以构建一个使用机器学习模型的web应用。列出你可以使用JavaScript或Python构建一个利用机器学习的web应用的方法。考虑架构：模型应该留在应用中还是驻留在云端？如果是后者，你将如何访问它？画出一个应用机器学习web解决方案的架构模型。

## 作业

[尝试一个不同的模型](assignment.md)

**免责声明**：
本文件是使用基于机器的人工智能翻译服务翻译的。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。应将原文档的母语版本视为权威来源。对于关键信息，建议进行专业人工翻译。对于使用本翻译而引起的任何误解或误读，我们概不负责。