<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T08:58:09+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "zh"
}
-->
# 使用 Python 和 Scikit-learn 构建回归模型

![回归模型的简要概述](../../../../sketchnotes/ml-regression.png)

> 由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 绘制的手绘笔记

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课程也提供 R 版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 简介

在这四节课中，您将学习如何构建回归模型。我们很快会讨论这些模型的用途。但在开始之前，请确保您已准备好正确的工具来进行学习！

在本课中，您将学习：

- 配置您的计算机以进行本地机器学习任务。
- 使用 Jupyter 笔记本。
- 安装并使用 Scikit-learn。
- 通过动手练习探索线性回归。

## 安装和配置

[![机器学习入门 - 配置工具以构建机器学习模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "机器学习入门 - 配置工具以构建机器学习模型")

> 🎥 点击上方图片观看短视频，了解如何配置您的计算机以进行机器学习。

1. **安装 Python**。确保您的计算机上已安装 [Python](https://www.python.org/downloads/)。您将使用 Python 来完成许多数据科学和机器学习任务。大多数计算机系统已经预装了 Python。此外，还有一些有用的 [Python 编码包](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)，可以简化某些用户的设置过程。

   不过，某些 Python 的使用场景可能需要不同版本的软件。因此，建议您使用 [虚拟环境](https://docs.python.org/3/library/venv.html)。

2. **安装 Visual Studio Code**。确保您的计算机上已安装 Visual Studio Code。按照这些说明完成 [Visual Studio Code 的安装](https://code.visualstudio.com/)。在本课程中，您将使用 Python 在 Visual Studio Code 中进行开发，因此您可能需要了解如何 [配置 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) 以进行 Python 开发。

   > 通过学习这组 [模块](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)，熟悉 Python。
   >
   > [![使用 Visual Studio Code 设置 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "使用 Visual Studio Code 设置 Python")
   >
   > 🎥 点击上方图片观看视频：在 VS Code 中使用 Python。

3. **安装 Scikit-learn**，按照 [这些说明](https://scikit-learn.org/stable/install.html) 进行安装。由于需要确保使用 Python 3，建议您使用虚拟环境。如果您在 M1 Mac 上安装此库，请参考上述页面中的特殊说明。

4. **安装 Jupyter Notebook**。您需要 [安装 Jupyter 包](https://pypi.org/project/jupyter/)。

## 您的机器学习开发环境

您将使用 **笔记本** 来开发 Python 代码并创建机器学习模型。这种文件类型是数据科学家常用的工具，其文件后缀为 `.ipynb`。

笔记本是一种交互式环境，允许开发者编写代码并添加注释和文档，非常适合实验或研究项目。

[![机器学习入门 - 设置 Jupyter 笔记本以开始构建回归模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "机器学习入门 - 设置 Jupyter 笔记本以开始构建回归模型")

> 🎥 点击上方图片观看短视频，了解如何完成此练习。

### 练习 - 使用笔记本

在此文件夹中，您会找到文件 _notebook.ipynb_。

1. 在 Visual Studio Code 中打开 _notebook.ipynb_。

   一个 Jupyter 服务器将启动，并使用 Python 3+。您会发现笔记本中可以运行的代码块。您可以通过选择播放按钮图标运行代码块。

2. 选择 `md` 图标并添加一些 markdown，输入以下文本 **# 欢迎来到您的笔记本**。

   接下来，添加一些 Python 代码。

3. 在代码块中输入 **print('hello notebook')**。
4. 选择箭头运行代码。

   您应该会看到打印的结果：

    ```output
    hello notebook
    ```

![在 VS Code 中打开的笔记本](../../../../2-Regression/1-Tools/images/notebook.jpg)

您可以在代码中插入注释，以便自我记录笔记本内容。

✅ 思考一下，网页开发者的工作环境与数据科学家的工作环境有何不同。

## 使用 Scikit-learn 入门

现在，Python 已在您的本地环境中设置完毕，并且您已经熟悉了 Jupyter 笔记本，接下来让我们熟悉一下 Scikit-learn（发音为 `sci`，像 `science`）。Scikit-learn 提供了一个 [广泛的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)，帮助您完成机器学习任务。

根据其 [官网](https://scikit-learn.org/stable/getting_started.html) 的介绍，“Scikit-learn 是一个开源机器学习库，支持监督学习和无监督学习。它还提供了各种工具，用于模型拟合、数据预处理、模型选择和评估，以及许多其他实用功能。”

在本课程中，您将使用 Scikit-learn 和其他工具构建机器学习模型，以完成我们称为“传统机器学习”的任务。我们特意避开了神经网络和深度学习，因为这些内容将在即将推出的“AI 入门”课程中详细介绍。

Scikit-learn 使构建模型并评估其使用变得简单。它主要专注于使用数值数据，并包含几个现成的数据集供学习使用。它还包括一些预构建的模型供学生尝试。让我们探索加载预打包数据并使用内置估算器构建第一个机器学习模型的过程。

## 练习 - 您的第一个 Scikit-learn 笔记本

> 本教程的灵感来源于 Scikit-learn 网站上的 [线性回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。

[![机器学习入门 - 您的第一个 Python 线性回归项目](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "机器学习入门 - 您的第一个 Python 线性回归项目")

> 🎥 点击上方图片观看短视频，了解如何完成此练习。

在与本课相关的 _notebook.ipynb_ 文件中，按下“垃圾桶”图标清空所有单元格。

在本节中，您将使用 Scikit-learn 中内置的一个关于糖尿病的小型数据集进行学习。假设您想测试一种针对糖尿病患者的治疗方法。机器学习模型可能会帮助您根据变量的组合确定哪些患者对治疗的反应更好。即使是一个非常基础的回归模型，当可视化时，也可能显示有关变量的信息，帮助您组织理论临床试验。

✅ 回归方法有很多种，选择哪一种取决于您想要回答的问题。如果您想预测某个年龄段的人的可能身高，您可以使用线性回归，因为您在寻找一个 **数值**。如果您想确定某种菜肴是否应该被归类为素食，您在寻找一个 **类别分配**，因此您可以使用逻辑回归。稍后您将学习更多关于逻辑回归的内容。思考一下，您可以向数据提出哪些问题，以及哪种方法更适合回答这些问题。

让我们开始这个任务。

### 导入库

在此任务中，我们将导入一些库：

- **matplotlib**。这是一个有用的 [绘图工具](https://matplotlib.org/)，我们将用它来创建折线图。
- **numpy**。 [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是一个处理 Python 数值数据的有用库。
- **sklearn**。这是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 库。

导入一些库以帮助完成任务。

1. 通过输入以下代码添加导入：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上述代码导入了 `matplotlib` 和 `numpy`，并从 `sklearn` 中导入了 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用于将数据分割为训练集和测试集。

### 糖尿病数据集

内置的 [糖尿病数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 包括 442 个关于糖尿病的数据样本，包含 10 个特征变量，其中一些包括：

- age：年龄（以年为单位）
- bmi：身体质量指数
- bp：平均血压
- s1 tc：T 细胞（白细胞的一种）

✅ 此数据集包含“性别”这一特征变量，这在糖尿病研究中很重要。许多医学数据集都包含这种二元分类。思考一下，这种分类可能会如何将某些群体排除在治疗之外。

现在，加载 X 和 y 数据。

> 🎓 请记住，这是监督学习，我们需要一个名为“y”的目标变量。

在新的代码单元中，通过调用 `load_diabetes()` 加载糖尿病数据集。输入参数 `return_X_y=True` 表示 `X` 将是数据矩阵，而 `y` 将是回归目标。

1. 添加一些打印命令以显示数据矩阵的形状及其第一个元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    您得到的响应是一个元组。您将元组的前两个值分别赋给 `X` 和 `y`。了解更多 [关于元组](https://wikipedia.org/wiki/Tuple)。

    您可以看到这些数据有 442 个项目，每个项目是包含 10 个元素的数组：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 思考一下数据与回归目标之间的关系。线性回归预测特征 X 和目标变量 y 之间的关系。您能在文档中找到糖尿病数据集的 [目标](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 吗？这个数据集展示了什么？

2. 接下来，通过选择数据集的第 3 列来绘制部分数据。您可以使用 `:` 操作符选择所有行，然后使用索引（2）选择第 3 列。您还可以使用 `reshape(n_rows, n_columns)` 将数据重塑为二维数组（绘图所需）。如果其中一个参数为 -1，则对应的维度会自动计算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 随时打印数据以检查其形状。

3. 现在您已经准备好绘制数据，可以看看机器是否能帮助确定数据集中的逻辑分割。为此，您需要将数据（X）和目标（y）分割为测试集和训练集。Scikit-learn 提供了一种简单的方法，您可以在给定点分割测试数据。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 现在您可以训练模型了！加载线性回归模型，并使用 `model.fit()` 用 X 和 y 训练集训练模型：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是一个您会在许多机器学习库（如 TensorFlow）中看到的函数。

5. 然后，使用测试数据创建预测，使用 `predict()` 函数。这将用于绘制数据组之间的分割线。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 现在是时候用图表展示数据了。Matplotlib 是一个非常有用的工具。创建一个所有 X 和 y 测试数据的散点图，并使用预测结果在数据组之间绘制一条最合适的线。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![显示糖尿病数据点的散点图](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ 想一想这里发生了什么。一条直线穿过许多小数据点，但它究竟在做什么？你能看出如何利用这条线来预测一个新的、未见过的数据点在图表的 y 轴上的位置吗？试着用语言描述这个模型的实际用途。

恭喜你！你已经构建了第一个线性回归模型，用它进行了预测，并将结果显示在图表中！

---
## 🚀挑战

绘制该数据集中不同变量的图表。提示：编辑这行代码：`X = X[:,2]`。根据该数据集的目标，你能发现关于糖尿病作为一种疾病的进展的什么信息？

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

在本教程中，你使用了简单线性回归，而不是单变量或多变量线性回归。阅读一些关于这些方法之间差异的内容，或者观看[这个视频](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)。

阅读更多关于回归的概念，并思考这种技术可以回答哪些类型的问题。通过[这个教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)来加深你的理解。

## 作业

[一个不同的数据集](assignment.md)

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应以原始语言的文档作为权威来源。对于关键信息，建议使用专业人工翻译。因使用本翻译而导致的任何误解或误读，我们概不负责。