# 构建使用 ML 模型的 Web 应用程序

在本课中，你将在一个数据集上训练一个 ML 模型，这个数据集来自世界各地：过去一个世纪的 UFO 目击事件，来源于 [NUFORC 的数据库](https://www.nuforc.org)。

你将学会：

- 如何“pickle”一个训练有素的模型
- 如何在 Flask 应用程序中使用该模型

我们将继续使用 notebook 来清理数据和训练我们的模型，但你可以进一步探索在 web 应用程序中使用模型。

为此，你需要使用 Flask 构建一个 web 应用程序。

## [课前测](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## 构建应用程序

有多种方法可以构建 Web 应用程序以使用机器学习模型。你的 web 架构可能会影响你的模型训练方式。想象一下，你在一家企业工作，其中数据科学小组已经训练了他们希望你在应用程序中使用的模型。

### 注意事项

你需要问很多问题：

- **它是 web 应用程序还是移动应用程序？** 如果你正在构建移动应用程序或需要在物联网环境中使用模型，你可以使用 [TensorFlow Lite](https://www.tensorflow.org/lite/) 并在 Android 或 iOS 应用程序中使用该模型。
- **模型放在哪里？** 在云端还是本地？
- **离线支持**。该应用程序是否必须离线工作？
- **使用什么技术来训练模型？** 所选的技术可能会影响你需要使用的工具。
   - **使用 TensorFlow**。例如，如果你正在使用 TensorFlow 训练模型，则该生态系统提供了使用 [TensorFlow.js](https://www.tensorflow.org/js/) 转换 TensorFlow 模型以便在Web应用程序中使用的能力。
   - **使用 PyTorch**。如果你使用 [PyTorch](https://pytorch.org/) 等库构建模型，则可以选择将其导出到 [ONNX](https://onnx.ai/)（开放神经网络交换）格式，用于可以使用 [Onnx Runtime](https://www.onnxruntime.ai/)的JavaScript Web 应用程序。此选项将在 Scikit-learn-trained 模型的未来课程中进行探讨。
   - **使用 Lobe.ai 或 Azure 自定义视觉**。如果你使用 ML SaaS（软件即服务）系统，例如 [Lobe.ai](https://lobe.ai/) 或 [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) 来训练模型，这种类型的软件提供了为许多平台导出模型的方法，包括构建一个定制A PI，供在线应用程序在云中查询。

你还有机会构建一个完整的 Flask Web 应用程序，该应用程序能够在 Web浏览器中训练模型本身。这也可以在 JavaScript 上下文中使用 TensorFlow.js 来完成。

出于我们的目的，既然我们一直在使用基于 Python 的 notebook，那么就让我们探讨一下将经过训练的模型从 notebook 导出为 Python 构建的 web 应用程序可读的格式所需要采取的步骤。

## 工具

对于此任务，你需要两个工具：Flask 和 Pickle，它们都在 Python 上运行。

✅ 什么是 [Flask](https://palletsprojects.com/p/flask/)？ Flask 被其创建者定义为“微框架”，它提供了使用 Python 和模板引擎构建网页的 Web 框架的基本功能。看看[本学习单元](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott)练习使用 Flask 构建应用程序。

✅ 什么是 [Pickle](https://docs.python.org/3/library/pickle.html)？ Pickle🥒是一个 Python 模块，用于序列化和反序列化 Python 对象结构。当你“pickle”一个模型时，你将其结构序列化或展平以在 Web 上使用。小心：pickle 本质上不是安全的，所以如果提示“un-pickle”文件，请小心。生产的文件具有后缀 `.pkl`。

## 练习 - 清理你的数据 

在本课中，你将使用由 [NUFORC](https://nuforc.org)（国家 UFO 报告中心）收集的 80,000 次 UFO 目击数据。这些数据对 UFO 目击事件有一些有趣的描述，例如：

- **详细描述**。"一名男子从夜间照射在草地上的光束中出现，他朝德克萨斯仪器公司的停车场跑去"。
- **简短描述**。 “灯光追着我们”。

[ufos.csv](./data/ufos.csv) 电子表格包括有关目击事件发生的 `city`、`state` 和 `country`、对象的 `shape` 及其 `latitude` 和 `longitude` 的列。

在包含在本课中的空白 [notebook](notebook.ipynb) 中：

1. 像在之前的课程中一样导入 `pandas`、`matplotlib` 和 `numpy`，然后导入 ufos 电子表格。你可以查看一个示例数据集：

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('../data/ufos.csv')
    ufos.head()
    ```

2. 将 ufos 数据转换为带有新标题的小 dataframe。检查 `country` 字段中的唯一值。

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

3. 现在，你可以通过删除任何空值并仅导入 1-60 秒之间的目击数据来减少我们需要处理的数据量：

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

4. 导入 Scikit-learn 的 `LabelEncoder` 库，将国家的文本值转换为数字：

   ✅ LabelEncoder 按字母顺序编码数据

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    你的数据应如下所示：

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3	    53.200000	-2.916667
    3	20.0	4	    28.978333	-96.645833
    14	30.0	4	    35.823889	-80.253611
    23	60.0	4	    45.582778	-122.352222
    24	3.0	    3	    51.783333	-0.783333
    ```

## 练习 - 建立你的模型

现在，你可以通过将数据划分为训练和测试组来准备训练模型。

1. 选择要训练的三个特征作为 X 向量，y 向量将是 `Country` 你希望能够输入 `Seconds`、`Latitude` 和 `Longitude` 并获得要返回的国家/地区 ID。

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

2. 使用逻辑回归训练模型：

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

准确率还不错 **（大约 95%）**，不出所料，因为 `Country` 和 `Latitude/Longitude` 相关。

你创建的模型并不是非常具有革命性，因为你应该能够从其 `Latitude` 和 `Longitude` 推断出 `Country`，但是，尝试从清理、导出的原始数据进行训练，然后在 web 应用程序中使用此模型是一个很好的练习。

## 练习 - “pickle”你的模型

现在，是时候 _pickle_ 你的模型了！你可以在几行代码中做到这一点。一旦它是 _pickled_，加载你的 pickled 模型并针对包含秒、纬度和经度值的示例数据数组对其进行测试，

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

该模型返回 **'3'**，这是英国的国家代码。👽

## 练习 - 构建Flask应用程序 

现在你可以构建一个Flask应用程序来调用你的模型并返回类似的结果，但以一种更美观的方式。

1. 首先在你的 _ufo-model.pkl_ 文件所在的 _notebook.ipynb_ 文件旁边创建一个名为 **web-app** 的文件夹。

2. 在该文件夹中创建另外三个文件夹：**static**，其中有文件夹 **css** 和 **templates**。 你现在应该拥有以下文件和目录

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ``` 

   ✅ 请参阅解决方案文件夹以查看已完成的应用程序

3. 在 _web-app_ 文件夹中创建的第一个文件是 **requirements.txt** 文件。与 JavaScript 应用程序中的 _package.json_ 一样，此文件列出了应用程序所需的依赖项。在 **requirements.txt** 中添加以下几行：  

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

4. 现在，进入 web-app 文件夹：

   ```bash
   cd web-app
   ```

5. 在你的终端中输入 `pip install`，以安装 _reuirements.txt_ 中列出的库：

   ```bash
   pip install -r requirements.txt
   ```

6. 现在，你已准备好创建另外三个文件来完成应用程序：

    1. 在根目录中创建 **app.py**。
    2. 在 _templates_ 目录中创建**index.html**。
    3. 在 _static/css_ 目录中创建**styles.css**。

7. 使用一些样式构建 _styles.css_ 文件：

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

8. 接下来，构建 _index.html_ 文件：

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

    看看这个文件中的模板。请注意应用程序将提供的变量周围的“mustache”语法，例如预测文本：`{{}}`。还有一个表单可以将预测发布到 `/predict` 路由。

    最后，你已准备好构建使用模型和显示预测的 python 文件：

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

   > 💡 提示：当你在使用 Flask 运行 Web 应用程序时添加 [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode)时你对应用程序所做的任何更改将立即反映，无需重新启动服务器。注意！不要在生产应用程序中启用此模式

如果你运行 `python app.py` 或 `python3 app.py` - 你的网络服务器在本地启动，你可以填写一个简短的表格来回答你关于在哪里看到 UFO 的问题！

在此之前，先看一下 `app.py` 的实现：

1. 首先，加载依赖项并启动应用程序。
2. 然后，导入模型。
3. 然后，在 home 路由上渲染 index.html。

在 `/predict` 路由上，当表单被发布时会发生几件事情： 

1. 收集表单变量并转换为 numpy 数组。然后将它们发送到模型并返回预测。
2. 我们希望显示的国家/地区根据其预测的国家/地区代码重新呈现为可读文本，并将该值发送回 index.html 以在模板中呈现。

以这种方式使用模型，包括 Flask 和 pickled 模型，是相对简单的。最困难的是要理解数据是什么形状的，这些数据必须发送到模型中才能得到预测。这完全取决于模型是如何训练的。有三个数据要输入，以便得到一个预测。

在一个专业的环境中，你可以看到训练模型的人和在 Web 或移动应用程序中使用模型的人之间的良好沟通是多么的必要。在我们的情况下，只有一个人，你！

---

## 🚀 挑战

你可以在 Flask 应用程序中训练模型，而不是在 notebook 上工作并将模型导入 Flask 应用程序！尝试在 notebook 中转换 Python 代码，可能是在清除数据之后，从应用程序中的一个名为 `train` 的路径训练模型。采用这种方法的利弊是什么？

## [课后测](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## 复习与自学

有很多方法可以构建一个Web应用程序来使用ML模型。列出可以使用JavaScript或Python构建Web应用程序以利用机器学习的方法。考虑架构：模型应该留在应用程序中还是存在于云中？如果是后者，你将如何访问它？为应用的ML Web解决方案绘制架构模型。

## 任务

[尝试不同的模型](./assignment.zh-cn.md)


