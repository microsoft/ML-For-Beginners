<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-09-03T18:07:17+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "zh"
}
-->
# 构建一个美食推荐网页应用

在本课中，您将使用之前课程中学到的一些技术以及贯穿整个系列的美食数据集，构建一个分类模型。此外，您还将构建一个小型网页应用来使用保存的模型，并利用 Onnx 的网页运行时。

机器学习最实用的用途之一是构建推荐系统，今天您可以迈出这一方向的第一步！

[![展示这个网页应用](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "应用机器学习")

> 🎥 点击上方图片观看视频：Jen Looper 使用分类美食数据构建网页应用

## [课前测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

在本课中，您将学习：

- 如何构建模型并将其保存为 Onnx 模型
- 如何使用 Netron 检查模型
- 如何在网页应用中使用您的模型进行推理

## 构建您的模型

构建应用型机器学习系统是将这些技术应用于业务系统的重要部分。通过使用 Onnx，您可以在网页应用中使用模型（因此在需要时也可以离线使用）。

在[之前的课程](../../3-Web-App/1-Web-App/README.md)中，您构建了一个关于 UFO 目击事件的回归模型，将其“封装”并在 Flask 应用中使用。虽然这种架构非常有用，但它是一个完整的 Python 应用，而您的需求可能包括使用 JavaScript 应用。

在本课中，您可以构建一个基于 JavaScript 的基本推理系统。不过，首先需要训练一个模型并将其转换为 Onnx 格式。

## 练习 - 训练分类模型

首先，使用我们之前清理过的美食数据集训练一个分类模型。

1. 首先导入有用的库：

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    您需要 '[skl2onnx](https://onnx.ai/sklearn-onnx/)' 来帮助将 Scikit-learn 模型转换为 Onnx 格式。

1. 然后，像之前课程中一样使用 `read_csv()` 读取 CSV 文件处理数据：

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. 删除前两列不必要的数据，并将剩余数据保存为 'X'：

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. 将标签保存为 'y'：

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### 开始训练流程

我们将使用具有良好准确性的 'SVC' 库。

1. 从 Scikit-learn 导入相关库：

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. 分离训练集和测试集：

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. 像之前课程中一样构建一个 SVC 分类模型：

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. 现在，测试您的模型，调用 `predict()`：

    ```python
    y_pred = model.predict(X_test)
    ```

1. 打印分类报告以检查模型质量：

    ```python
    print(classification_report(y_test,y_pred))
    ```

    如我们之前所见，准确性很好：

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### 将模型转换为 Onnx

确保使用正确的张量数量进行转换。此数据集列出了 380 种成分，因此您需要在 `FloatTensorType` 中标注该数量：

1. 使用张量数量 380 进行转换。

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. 创建 onx 文件并保存为 **model.onnx**：

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > 注意，您可以在转换脚本中传入[选项](https://onnx.ai/sklearn-onnx/parameterized.html)。在本例中，我们传入了 'nocl' 为 True 和 'zipmap' 为 False。由于这是一个分类模型，您可以选择移除 ZipMap，它会生成一个字典列表（不必要）。`nocl` 指的是模型中是否包含类别信息。通过将 `nocl` 设置为 'True' 来减小模型的大小。

运行整个笔记本后，现在会构建一个 Onnx 模型并将其保存到此文件夹中。

## 查看您的模型

Onnx 模型在 Visual Studio Code 中不太直观，但有一个非常好的免费软件，许多研究人员使用它来可视化模型，以确保模型构建正确。下载 [Netron](https://github.com/lutzroeder/Netron) 并打开您的 model.onnx 文件。您可以看到您的简单模型被可视化，列出了其 380 个输入和分类器：

![Netron 可视化](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.zh.png)

Netron 是一个查看模型的有用工具。

现在您可以在网页应用中使用这个简洁的模型了。让我们构建一个应用，当您查看冰箱并试图确定可以用剩余食材组合制作哪种美食时，这个应用会派上用场。

## 构建推荐网页应用

您可以直接在网页应用中使用您的模型。这种架构还允许您在本地运行它，甚至在需要时离线运行。首先，在存储 `model.onnx` 文件的同一文件夹中创建一个 `index.html` 文件。

1. 在这个文件 _index.html_ 中，添加以下标记：

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. 现在，在 `body` 标签内，添加一些标记以显示一些食材的复选框列表：

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    注意，每个复选框都被赋予了一个值。这反映了根据数据集找到食材的索引。例如，苹果在这个按字母顺序排列的列表中占据第五列，因此其值为 '4'，因为我们从 0 开始计数。您可以查阅 [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) 来发现某个食材的索引。

    在 index.html 文件中继续工作，在最后一个关闭的 `</div>` 后添加一个脚本块，其中调用了模型。

1. 首先，导入 [Onnx Runtime](https://www.onnxruntime.ai/)：

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime 用于支持在广泛的硬件平台上运行您的 Onnx 模型，包括优化和使用的 API。

1. 一旦 Runtime 就位，您可以调用它：

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

在这段代码中，发生了以下几件事：

1. 您创建了一个包含 380 个可能值（1 或 0）的数组，用于设置并发送到模型进行推理，具体取决于某个食材复选框是否被选中。
2. 您创建了一个复选框数组以及一种在应用启动时通过 `init` 函数确定它们是否被选中的方法。当复选框被选中时，`ingredients` 数组会被修改以反映所选食材。
3. 您创建了一个 `testCheckboxes` 函数，用于检查是否有复选框被选中。
4. 当按钮被按下时，您使用 `startInference` 函数，如果有复选框被选中，则开始推理。
5. 推理流程包括：
   1. 设置模型的异步加载
   2. 创建一个发送到模型的张量结构
   3. 创建反映您在训练模型时创建的 `float_input` 输入的 'feeds'（您可以使用 Netron 验证该名称）
   4. 将这些 'feeds' 发送到模型并等待响应

## 测试您的应用

在存放 index.html 文件的文件夹中打开 Visual Studio Code 的终端会话。确保您已全局安装 [http-server](https://www.npmjs.com/package/http-server)，然后在提示符下输入 `http-server`。一个本地主机应该打开，您可以查看您的网页应用。检查根据各种食材推荐的美食：

![食材网页应用](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.zh.png)

恭喜，您已经创建了一个带有几个字段的“推荐”网页应用。花点时间完善这个系统吧！

## 🚀挑战

您的网页应用非常简约，因此请继续使用 [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) 数据中的食材及其索引来完善它。哪些风味组合可以制作出某个国家的特色菜？

## [课后测验](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## 复习与自学

虽然本课只是简单介绍了为食材创建推荐系统的实用性，但这一领域的机器学习应用有非常丰富的示例。阅读更多关于这些系统如何构建的内容：

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## 作业 

[构建一个新的推荐系统](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们不对因使用此翻译而产生的任何误解或误读承担责任。