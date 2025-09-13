<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T09:06:07+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "zh"
}
-->
# 构建一个使用机器学习模型的网页应用

在本课中，你将使用一个非常特别的数据集来训练一个机器学习模型：_过去一个世纪的UFO目击事件_，数据来源于NUFORC的数据库。

你将学习：

- 如何对训练好的模型进行“pickle”处理
- 如何在Flask应用中使用该模型

我们将继续使用notebook来清理数据并训练模型，但你可以更进一步，尝试在“真实世界”中使用模型，也就是在一个网页应用中。

为此，你需要使用Flask构建一个网页应用。

## [课前小测验](https://ff-quizzes.netlify.app/en/ml/)

## 构建一个应用

有多种方法可以构建网页应用来使用机器学习模型。你的网页架构可能会影响模型的训练方式。想象一下，你正在一个企业中工作，数据科学团队已经训练了一个模型，他们希望你在应用中使用它。

### 需要考虑的问题

你需要问自己许多问题：

- **这是一个网页应用还是一个移动应用？** 如果你正在构建一个移动应用，或者需要在物联网环境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/) 并在Android或iOS应用中使用该模型。
- **模型将存储在哪里？** 是在云端还是本地？
- **是否需要离线支持？** 应用是否需要在离线状态下运行？
- **训练模型使用了什么技术？** 所选技术可能会影响你需要使用的工具。
    - **使用TensorFlow。** 如果你使用TensorFlow训练模型，该生态系统提供了将TensorFlow模型转换为网页应用中使用的能力，例如通过 [TensorFlow.js](https://www.tensorflow.org/js/)。
    - **使用PyTorch。** 如果你使用 [PyTorch](https://pytorch.org/) 等库构建模型，你可以选择将其导出为 [ONNX](https://onnx.ai/)（开放神经网络交换）格式，用于支持JavaScript网页应用的 [Onnx Runtime](https://www.onnxruntime.ai/)。在未来的课程中，我们将探索如何将Scikit-learn训练的模型导出为ONNX格式。
    - **使用Lobe.ai或Azure Custom Vision。** 如果你使用 [Lobe.ai](https://lobe.ai/) 或 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 等机器学习SaaS（软件即服务）系统来训练模型，这类软件提供了多平台导出模型的方法，包括构建一个定制的API，通过云端供在线应用查询。

你还可以选择构建一个完整的Flask网页应用，该应用能够在网页浏览器中自行训练模型。这也可以通过JavaScript环境中的TensorFlow.js实现。

对于我们的目的，由于我们一直在使用基于Python的notebook，让我们来探索将训练好的模型从notebook导出为Python构建的网页应用可读取的格式所需的步骤。

## 工具

完成此任务，你需要两个工具：Flask和Pickle，它们都运行在Python上。

✅ 什么是 [Flask](https://palletsprojects.com/p/flask/)？Flask被其创建者定义为一个“微框架”，它使用Python和模板引擎来构建网页，提供了网页框架的基本功能。可以参考 [这个学习模块](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) 来练习使用Flask构建应用。

✅ 什么是 [Pickle](https://docs.python.org/3/library/pickle.html)？Pickle 🥒 是一个Python模块，用于序列化和反序列化Python对象结构。当你对模型进行“pickle”处理时，你会将其结构序列化或扁平化，以便在网页上使用。需要注意的是：Pickle本身并不安全，因此在被提示“un-pickle”文件时要小心。Pickle文件的后缀为`.pkl`。

## 练习 - 清理数据

在本课中，你将使用来自 [NUFORC](https://nuforc.org)（国家UFO报告中心）的80,000条UFO目击数据。这些数据中包含一些有趣的UFO目击描述，例如：

- **长描述示例。** “一个人从夜晚草地上的一道光束中出现，跑向德州仪器的停车场。”
- **短描述示例。** “灯光追逐我们。”

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) 表格包含关于目击发生的 `city`（城市）、`state`（州）和 `country`（国家），物体的 `shape`（形状），以及其 `latitude`（纬度）和 `longitude`（经度）的列。

在本课提供的空白 [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) 中：

1. 像之前的课程一样，导入 `pandas`、`matplotlib` 和 `numpy`，并导入ufos表格。你可以查看数据集的样本：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. 将ufos数据转换为一个小型数据框，并重新命名列标题。检查 `Country` 字段中的唯一值。

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

1. 导入Scikit-learn的 `LabelEncoder` 库，将国家的文本值转换为数字：

    ✅ LabelEncoder 按字母顺序对数据进行编码

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    你的数据应如下所示：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## 练习 - 构建模型

现在你可以准备通过将数据分为训练组和测试组来训练模型。

1. 选择三个特征作为你的X向量，y向量将是 `Country`。你希望能够输入 `Seconds`、`Latitude` 和 `Longitude`，并返回一个国家ID。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. 使用逻辑回归训练模型：

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

模型的准确率还不错 **（大约95%）**，这并不奇怪，因为 `Country` 和 `Latitude/Longitude` 是相关的。

你创建的模型并不是非常具有革命性，因为你应该能够从 `Latitude` 和 `Longitude` 推断出 `Country`，但这是一个很好的练习，可以尝试从清理过的原始数据中训练模型，导出模型，然后在网页应用中使用它。

## 练习 - 对模型进行“pickle”处理

现在，是时候对你的模型进行“pickle”处理了！你可以用几行代码完成这一步。一旦完成“pickle”处理，加载你的Pickle模型，并用一个包含秒数、纬度和经度值的样本数据数组进行测试，

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

模型返回了 **'3'**，这是英国的国家代码。太神奇了！👽

## 练习 - 构建一个Flask应用

现在你可以构建一个Flask应用来调用你的模型，并以更直观的方式返回类似的结果。

1. 首先，在 _notebook.ipynb_ 文件旁边创建一个名为 **web-app** 的文件夹，其中存放你的 _ufo-model.pkl_ 文件。

1. 在该文件夹中再创建三个文件夹：**static**（其中包含一个名为 **css** 的文件夹）和 **templates**。你现在应该有以下文件和目录：

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ 参考解决方案文件夹以查看完成的应用

1. 在 _web-app_ 文件夹中创建第一个文件 **requirements.txt**。像JavaScript应用中的 _package.json_ 一样，此文件列出了应用所需的依赖项。在 **requirements.txt** 中添加以下内容：

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. 现在，通过导航到 _web-app_ 运行此文件：

    ```bash
    cd web-app
    ```

1. 在终端中输入 `pip install`，以安装 _requirements.txt_ 中列出的库：

    ```bash
    pip install -r requirements.txt
    ```

1. 现在，你可以创建另外三个文件来完成应用：

    1. 在根目录创建 **app.py**。
    2. 在 _templates_ 目录中创建 **index.html**。
    3. 在 _static/css_ 目录中创建 **styles.css**。

1. 在 _styles.css_ 文件中添加一些样式：

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

1. 接下来，构建 _index.html_ 文件：

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

    查看此文件中的模板语法。注意变量周围的“大括号”语法，例如预测文本：`{{}}`。还有一个表单会将预测结果发布到 `/predict` 路由。

    最后，你已经准备好构建驱动模型使用和预测显示的Python文件：

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

    > 💡 提示：当你在使用Flask运行网页应用时添加 [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)，任何对应用的更改都会立即反映出来，而无需重启服务器。但要小心！不要在生产应用中启用此模式。

如果你运行 `python app.py` 或 `python3 app.py`，你的网页服务器会在本地启动，你可以填写一个简短的表单，获取关于UFO目击地点的答案！

在此之前，先看看 `app.py` 的各个部分：

1. 首先，加载依赖项并启动应用。
1. 然后，导入模型。
1. 接着，在主页路由上渲染 index.html。

在 `/predict` 路由上，当表单被提交时，会发生以下几件事：

1. 表单变量被收集并转换为一个numpy数组。然后将其发送到模型，并返回一个预测结果。
2. 我们希望显示的国家代码被重新渲染为可读的文本值，并将该值发送回 index.html，在模板中渲染。

通过Flask和Pickle模型以这种方式使用模型是相对简单的。最难的部分是理解必须发送到模型的数据形状，以获得预测结果。这完全取决于模型的训练方式。这个模型需要输入三个数据点才能获得预测。

在专业环境中，你可以看到训练模型的团队和在网页或移动应用中使用模型的团队之间良好沟通的重要性。在我们的案例中，只有一个人，那就是你！

---

## 🚀 挑战

与其在notebook中工作并将模型导入Flask应用，你可以直接在Flask应用中训练模型！尝试将notebook中的Python代码转换为在应用中的 `train` 路由上训练模型。尝试这种方法的优缺点是什么？

## [课后小测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

构建一个使用机器学习模型的网页应用有很多方法。列出你可以使用JavaScript或Python构建网页应用以利用机器学习的方法。考虑架构：模型应该保留在应用中还是存储在云端？如果是后者，你将如何访问它？绘制一个应用机器学习网页解决方案的架构模型。

## 作业

[尝试一个不同的模型](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。对于因使用本翻译而引起的任何误解或误读，我们概不负责。