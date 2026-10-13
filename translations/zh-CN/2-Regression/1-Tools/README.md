# 使用 Python 和 Scikit-learn 入门回归模型

![回归的草图总结](../../../../translated_images/zh-CN/ml-regression.4e4f70e3b3ed446e.webp)

> 草图作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课也提供 R 语言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介绍

在这四节课中，您将学会如何构建回归模型。稍后我们将讨论它们的用途。但在开始之前，请确保您已经准备好适当的工具！

在本课中，您将学习如何：

- 为本地机器学习任务配置计算机。
- 使用 Jupyter 笔记本。
- 使用 Scikit-learn，包括安装过程。
- 通过实践练习探索线性回归。

## 安装与配置

[![面向初学者的机器学习 - 准备好你的工具以构建机器学习模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "面向初学者的机器学习 - 准备好你的工具以构建机器学习模型")

> 🎥 点击上方图片观看短视频，讲解如何配置你的计算机以进行机器学习。

1. **安装 Python**。确保你的计算机上安装了[Python](https://www.python.org/downloads/)。Python 是许多数据科学和机器学习任务的基础。大多数计算机系统已经预装了 Python。还有实用的[Python 编码包](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)，能简化某些用户的安装流程。

   但是，Python 的某些用法需要不同的版本，因此使用[虚拟环境](https://docs.python.org/3/library/venv.html)会很有帮助。

2. **安装 Visual Studio Code**。确保你计算机上安装了 Visual Studio Code。按照[安装 Visual Studio Code](https://code.visualstudio.com/)的说明完成基本安装。本课程将使用 Visual Studio Code 运行 Python，因此你可能想了解如何[配置 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)以进行 Python 开发。

   > 通过本集合中的[学习模块](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)熟悉 Python。
   >
   > [![在 Visual Studio Code 中设置 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "在 Visual Studio Code 中设置 Python")
   >
   > 🎥 点击上方图片观看视频：如何在 VS Code 中使用 Python。

3. **安装 Scikit-learn**，请按照[这些说明](https://scikit-learn.org/stable/install.html)操作。需要确保使用 Python 3，建议在虚拟环境中安装。如果你在 M1 Mac 上安装，有特殊说明请参见上述链接页面。

1. **安装 Jupyter Notebook**。你需要[安装 Jupyter 包](https://pypi.org/project/jupyter/)。

## 你的机器学习创作环境

你将使用 <strong>笔记本</strong> 来编写 Python 代码和创建机器学习模型。这类文件是数据科学家的常用工具，文件后缀是 `.ipynb`。

笔记本是交互式环境，允许开发者编写代码的同时添加注释和文档，对于实验或研究项目非常有用。

[![面向初学者的机器学习 - 设置 Jupyter 笔记本开始构建回归模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "面向初学者的机器学习 - 设置 Jupyter 笔记本开始构建回归模型")

> 🎥 点击上方图片观看本练习的短视频。

### 练习 - 使用笔记本

在此文件夹中，你会找到文件 _notebook.ipynb_。

1. 在 Visual Studio Code 中打开 _notebook.ipynb_。

   一个 Jupyter 服务器会启动，运行 Python 3+。你会看到笔记本中可以 `run` 的代码块。点击看起来像播放按钮的图标即可运行代码块。

1. 选择 `md` 图标，添加一些 markdown 内容，写入以下文本 **# 欢迎使用你的笔记本**。

   接着，添加一些 Python 代码。

1. 在代码块中输入 **print('hello notebook')**。
1. 点击箭头运行代码。

   你应该会看到输出：

    ```output
    hello notebook
    ```

![VS Code 中打开的笔记本](../../../../translated_images/zh-CN/notebook.4a3ee31f396b8832.webp)

你可以在代码中夹杂注释，自我文档化你的笔记本。

✅ 想一想，网页开发者和数据科学家的工作环境有多大不同。

## 使用 Scikit-learn 起步

现在你已经在本地环境配置好 Python，并熟悉了 Jupyter 笔记本，我们来熟悉一下 Scikit-learn（发音为“sci”，如英文单词“science”）。Scikit-learn 提供了一个[丰富的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)，帮助你完成机器学习任务。

根据他们的[官网](https://scikit-learn.org/stable/getting_started.html)，“Scikit-learn 是一个开源的机器学习库，支持监督学习和无监督学习。还提供了模型拟合、数据预处理、模型选择和评估以及多种实用工具。”

在本课程中，你将使用 Scikit-learn 和其他工具构建机器学习模型，执行我们称为“传统机器学习”的任务。我们故意避开了神经网络和深度学习，因为它们将在即将推出的“面向初学者的 AI”课程中详细讲解。

Scikit-learn 让构建模型和评估模型变得简单。它主要针对数值数据，包含若干现成数据集用于学习，也包括预制模型供学生尝试。让我们先探索加载预装数据及使用内置估计器构建基础机器学习模型的过程。

## 练习 - 你的第一个 Scikit-learn 笔记本

> 本教程灵感来源于 Scikit-learn 网站上的[线性回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![面向初学者的机器学习 - 你的第一个 Python 线性回归项目](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "面向初学者的机器学习 - 你的第一个 Python 线性回归项目")

> 🎥 点击上方图片观看本练习的短视频。

在本课对应的 _notebook.ipynb_ 文件中，通过点击“垃圾桶”图标清空所有单元格。

在本节中，你将使用一个关于糖尿病的内置小型数据集，它为学习而设计。假设你想测试糖尿病患者的某种治疗方案。机器学习模型可以帮助你根据变量组合预测哪些患者能更好地响应治疗。即使是非常基础的回归模型，经过可视化，也可能揭示有助于设计理论临床试验的变量信息。

✅ 回归方法有多种，选用哪一种取决于你想回答的问题。如果你想预测某年龄段的人的可能身高，你会用线性回归，因为你在寻找一个<strong>数值</strong>。如果你想判断某种菜肴是否为素食，你在做<strong>类别归属</strong>的判断，会用逻辑回归。稍后你会学到更多逻辑回归知识。想想你可以向数据询问哪些问题，以及哪种方法更合适。

让我们开始这个任务吧。

### 导入库

本任务我们将导入一些库：

- **matplotlib**。这是一个实用的[绘图库](https://matplotlib.org/)，我们用它来绘制折线图。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是处理 Python 数值数据的实用库。
- **sklearn**。这是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 库。

导入一些库以辅助完成任务。

1. 输入以下代码导入：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上面代码导入了 `matplotlib`、`numpy`，以及从 `sklearn` 导入了 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用于将数据分割成训练集和测试集。

### 糖尿病数据集

内置的[糖尿病数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)包含 442 条糖尿病相关样本数据，有 10 个特征变量，部分包括：

- age：年龄（岁）
- bmi：身体质指数
- bp：平均血压
- s1 tc：T细胞（一种白细胞）

✅ 该数据集中的“性别”作为研究糖尿病的重要特征变量。许多医疗数据集都包含这种二分类特征。思考下这样的分类方式如何可能将某些人口群体排除在治疗之外。

现在，加载 X 和 y 数据。

> 🎓 记住，这是监督学习，我们需要一个名为 y 的目标变量。

在新代码单元中，通过调用 `load_diabetes()` 加载糖尿病数据集。参数 `return_X_y=True` 表示 `X` 是数据矩阵，`y` 是回归目标。

1. 添加一些打印命令，显示数据矩阵的形状及其第一个元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你获得的响应是一个元组。你将元组的前两个值分别赋给 `X` 和 `y`。了解更多[元组](https://wikipedia.org/wiki/Tuple)。

    可见数据有 442 条，形状是包含 10 个元素的数组：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 思考一下数据与回归目标之间的关系。线性回归预测特征 X 与目标变量 y 之间的关系。你能在文档中找到糖尿病数据集的[目标](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)吗？考虑该目标，数据集展示了什么？

2. 接下来，选取数据集的一个部分绘图，选择第 3 列数据。你可以使用冒号操作符选取所有行，然后用索引（2）选取第 3 列。你也可以使用 `reshape(n_rows, n_columns)` 重塑为二维数组，以满足绘图需求。如果某个维度设为 -1，则自动计算该维度大小。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 随时打印数据检查其形状。

3. 现在数据准备好绘图了，可以看看机器是否能帮忙找到这些数据的合理分割点。为此，你需要将数据（X）和目标（y）都拆分成训练集和测试集。Scikit-learn 提供了简单方法，可在指定位置进行数据拆分。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 现在你可以训练模型了！加载线性回归模型，用你的 X 和 y 训练集调用 `model.fit()`：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是许多机器学习库（如 TensorFlow）常见的函数。

5. 接着，用测试数据调用 `predict()` 生成预测值。利用此结果在模型数据组之间绘制判别线。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 现在展示数据图。Matplotlib 是极有用的工具。创建散点图展示所有 X 和 y 测试数据，用预测值画线，展示模型数据分布的分界。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![展示糖尿病数据点的散点图](../../../../translated_images/zh-CN/scatterplot.ad8b356bcbb33be6.webp)


   ✅ 想一想这里到底发生了什么。一条直线穿过了许多小数据点，但它到底在做什么？你能看出怎样利用这条直线来预测一个未见新数据点应该如何相对于图的 y 轴进行定位吗？试着用语言表达一下这个模型的实际用途。

恭喜你，构建了你的第一个线性回归模型，用它做出了预测，并在图中展示了出来！

---
## 🚀挑战

绘制这个数据集中不同的变量。提示：编辑这一行：`X = X[:,2]`。结合这个数据集的目标，你能发现糖尿病作为一种疾病在发展的过程中有哪些变化吗？
## [课后小测](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

本教程中，你使用的是简单线性回归，而非单变量或多元线性回归。了解一下这些方法之间的差异，或者看看[这个视频](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)。

多读一些关于回归概念的内容，思考这种技术可以回答哪些类型的问题。做做这个[教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)来加深理解。

## 作业

[一个不同的数据集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->