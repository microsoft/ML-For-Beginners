# 使用 Python 和 Scikit-learn 入门回归模型

![回归摘要手绘笔记](../../../../translated_images/zh-CN/ml-regression.4e4f70e3b3ed446e.webp)

> 手绘笔记作者 [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课程也有 R 语言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介绍

在这四节课中，你将学习如何构建回归模型。我们稍后会讨论这些模型的用途。但在你做任何事情之前，务必确保你拥有启动该过程所需的正确工具！

在本节课中，你将学习如何：

- 配置你的计算机以用于本地机器学习任务。
- 使用 Jupyter 笔记本。
- 使用 Scikit-learn，包括安装。
- 通过动手练习探索线性回归。

## 安装与配置

[![面向初学者的机器学习 - 准备好搭建机器学习模型的工具](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "面向初学者的机器学习 - 准备好搭建机器学习模型的工具")

> 🎥 点击上方图片观看一个简短视频，演示如何配置你的计算机以支持机器学习。

1. **安装 Python**。确保你的计算机已安装 [Python](https://www.python.org/downloads/)。Python 是许多数据科学和机器学习任务中使用的工具。大多数计算机系统已经预装了 Python。也有一些实用的 [Python 编码包](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)，可以简化部分用户的安装过程。

   但 Python 的某些用法要求使用特定版本的软件，另一部分又需要使用不同版本。因此，使用 [虚拟环境](https://docs.python.org/3/library/venv.html) 是非常有用的。

2. **安装 Visual Studio Code**。请确保你已经在计算机上安装了 Visual Studio Code。按照这些指南进行 [Visual Studio Code 安装](https://code.visualstudio.com/)。在本课程中你将使用 Visual Studio Code 运行 Python，因此建议你熟悉如何为 Python 开发 [配置 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)。

   > 通过完成这个集合的 [学习模块](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)增强你对 Python 的熟悉度
   >
   > [![在 Visual Studio Code 中配置 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "在 Visual Studio Code 中配置 Python")
   >
   > 🎥 点击上方图片观看一个关于如何在 VS Code 中使用 Python 的视频。

3. **安装 Scikit-learn**，请参照 [此处指南](https://scikit-learn.org/stable/install.html)。由于你需要确保使用 Python 3，建议使用虚拟环境安装此库。注意，如果你使用的是 M1 Mac，上述链接页面提供了特别的安装说明。

1. **安装 Jupyter Notebook**。你需要 [安装 Jupyter 包](https://pypi.org/project/jupyter/)。

## 你的机器学习开发环境

你将使用 <strong>笔记本</strong> 来开发 Python 代码及创建机器学习模型。这类文件是数据科学家常用的工具，通常以 `.ipynb` 作为后缀。

笔记本是一种交互式环境，允许开发者一边编写代码，一边添加注释和文档，非常适合实验性或研究导向的项目。

[![面向初学者的机器学习 - 配置 Jupyter 笔记本开始构建回归模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "面向初学者的机器学习 - 配置 Jupyter 笔记本开始构建回归模型")

> 🎥 点击上方图片观看简短视频，演示该练习过程。

### 练习 - 使用笔记本

在此文件夹中，你会找到文件 _notebook.ipynb_。

1. 在 Visual Studio Code 中打开 _notebook.ipynb_。

   一个 Jupyter 服务器将启动，使用 Python 3+。你会看到笔记本中有可执行的代码块，你可以点击类似播放按钮的图标运行它们。

1. 选择 `md` 图标，添加一段 markdown，内容是 **# 欢迎使用你的笔记本**。

   接下来，添加一些 Python 代码。

1. 在代码块中输入 **print('hello notebook')**。
1. 点击箭头运行代码。

   你应该能看到打印语句：

    ```output
    hello notebook
    ```

![打开笔记本的 VS Code 界面](../../../../translated_images/zh-CN/notebook.4a3ee31f396b8832.webp)

你可以在代码间穿插注释，为笔记本自我记录说明。

✅ 思考一下， Web 开发者的工作环境与数据科学家的工作环境究竟有何不同。

## 使用 Scikit-learn 快速上手

现在本地环境中已设置好 Python，你也熟悉了 Jupyter 笔记本，我们来一起熟悉 Scikit-learn（发音为“sci”如同“science”）。Scikit-learn 提供了一个 [丰富的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref) ，能帮助你完成机器学习任务。

根据其 [官网](https://scikit-learn.org/stable/getting_started.html)，"Scikit-learn 是一个开源机器学习库，支持监督学习和无监督学习。它还提供了多种模型拟合、数据预处理、模型选择与评估等工具，以及许多实用功能。"

在本课程中，你将使用 Scikit-learn 及其它工具构建机器学习模型，执行我们称为“传统机器学习”的任务。我们有意避开神经网络和深度学习，这些会在即将上线的“初学者人工智能”课程中详细介绍。

Scikit-learn 使构建模型和评估变得简单。它主要关注数值型数据，并内置了几个可用于学习的现成数据集。它还包含预制模型，供学生试用。让我们探索加载预包装数据、使用内置估计器创建第一个基于基础数据的机器学习模型的流程。

## 练习 - 你的第一个 Scikit-learn 笔记本

> 本教程灵感来自 Scikit-learn 网站上的 [线性回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![面向初学者的机器学习 - 你的第一个 Python 线性回归项目](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "面向初学者的机器学习 - 你的第一个 Python 线性回归项目")

> 🎥 点击上方图片观看简短视频，演示该练习过程。

在本节课关联的 _notebook.ipynb_ 文件中，通过点击“垃圾桶”图标清空所有单元格内容。

这部分中，你将使用 Scikit-learn 内置的一个关于糖尿病的小型数据集用于学习。假设你想测试一种糖尿病患者治疗方法。机器学习模型可以通过变量组合帮助你判定哪些患者可能对治疗反应较好。甚至是一个非常基础的回归模型，经过可视化后，就可能提供关于变量的信息，帮助你组织理论上的临床试验。

✅ 回归方法有很多种，选用哪一种取决于你想要的答案。如果你想预测某人特定年龄的预期身高，应使用线性回归，因为你想预测的是一个 <strong>数值</strong>。如果你想判断某种菜肴是否属于素食之一类，你是在寻找 <strong>类别分类</strong>，这时应使用逻辑回归。稍后你会学习逻辑回归。花点时间思考你可以提出哪些数据问题，这些方法里哪种更合适。

那么，现在开始这个任务吧。

### 导入库

本任务中我们将导入一些库：

- **matplotlib**。它是一个实用的 [绘图工具](https://matplotlib.org/)，我们将用它创建折线图。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是处理数值数据的有用 Python 库。
- **sklearn**。即 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 库。

导入一些库以帮助完成任务。

1. 输入以下代码进行导入：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上面导入了 `matplotlib`、`numpy`，并从 `sklearn` 中导入了 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用于将数据拆分为训练集和测试集。

### 糖尿病数据集

内置的 [糖尿病数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)包含442个有关糖尿病的样本，共有10个特征变量，其中有：

- age：年龄（岁）
- bmi：身体质量指数（体重指数）
- bp：平均血压
- s1 tc：T细胞（一种白细胞）

✅ 该数据集包括'性别'这一对糖尿病研究非常重要的特征变量。许多医学数据集都包含这种二元分类。思考一下，这类分类可能会如何将某些群体排除在治疗之外。

现在，加载 X 和 y 数据。

> 🎓 请记住，这是监督学习，需要一个名为 'y' 的目标变量。

在一个新的代码单元里，通过调用 `load_diabetes()` 加载糖尿病数据集。`return_X_y=True` 表示 `X` 是数据矩阵，`y` 是回归目标。

1. 添加一些打印语句，显示数据矩阵的形状和第一个元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你得到的是一个元组。你将元组的前两个值分别赋给 `X` 和 `y`。更多详情请参见 [关于元组](https://wikipedia.org/wiki/Tuple)。

    你可以看到该数据包含442条记录，每条记录有10个元素：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 思考数据与回归目标之间的关系。线性回归预测特征X与目标变量y之间的关系。你能在文档中找到该糖尿病数据集的 [目标变量](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)吗？考虑目标变量，这个数据集展示了什么内容？

2. 接下来，选取该数据集的一部分进行绘图，即选取第3列。你可以使用 `:` 选择所有行，再用索引 (2) 选取第3列。还可以用 `reshape(n_rows, n_columns)` 将数据重塑成二维数组——绘图所需的格式。参数中如果有 -1，表示该维度会自动计算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 任何时候都可以打印数据检查其形状。

3. 准备好要绘制的数据后，可以看看机器是否能帮忙找到该数据集中数字的合理划分。为此，需要把数据（X）和目标（y）拆分为训练集和测试集。Scikit-learn 中有简单方法，可以在指定位置拆分测试数据。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 现在开始训练模型！加载线性回归模型，通过 `model.fit()` 用你的 X 和 y 训练集训练它：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是许多机器学习库（如 TensorFlow）都能看到的函数。

5. 接着，使用测试数据通过 `predict()` 函数创建预测。这将用于绘制数据群组之间的线条。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 现在是展示数据绘图的时候了。Matplotlib 在这方面非常有用。创建一个散点图，显示所有 X 和 y 测试数据，再用预测结果绘制一条线，划分模型中的数据群。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![显示糖尿病数据点的散点图](../../../../translated_images/zh-CN/scatterplot.ad8b356bcbb33be6.webp)

   ✅ 思考这里发生了什么。一条直线穿过许多小数据点，这条线具体在做什么？你能看出如何用它预测一个新的、未见过的数据点应在图的 y 轴上的哪一位置吗？尝试用文字描述该模型的实际用途。

恭喜，你已构建了你的第一个线性回归模型，基于它做出了预测，并在图中展示了结果！

---
## 🚀挑战

从数据集中绘制不同的变量。提示：编辑这行代码： `X = X[:,2]`。考虑该数据集的目标变量，你能发现糖尿病作为一种疾病的进展规律吗？
## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

在本教程中，你使用了简单线性回归，而非单变量或多元线性回归。可稍作了解这几种方法的区别，或者观看 [此视频](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

阅读更多关于回归概念的内容，并思考通过此技术可以回答哪些类型的问题。参加此[教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)以加深理解。

## 作业

[一个不同的数据集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->