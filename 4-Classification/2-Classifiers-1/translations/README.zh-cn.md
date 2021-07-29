# 菜品分类器1

在本节中，将使用你在上一个课程中所保存的全部经过均衡和清洗的菜品数据。

You will use this dataset with a variety of classifiers to _predict a given national cuisine based on a group of ingredients_. While doing so, you'll learn more about some of the ways that algorithms can be leveraged for classification tasks.
你将使用这份数据集，并通过多种分类器 _在给出了各种配料后预测这是那一个国家的菜品_。在此过程中，你将学到更多能够用来调节分类任务算法的方法。   

## [课前测试](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/21/)
# 准备工作

假设你已经完成了[课程1](../1-Introduction/README.md), 确保在根目录的`/data`文件夹中有 _cleaned_cuisines.csv_ 文件来进行接下来的四节课程。

## 练习 - 预测某国的菜品

1. 在本节课的 _notebook.ipynb_ 文件中，导入Pandas的同时载入相应的数据文件：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../../data/cleaned_cuisine.csv")
    cuisines_df.head()
    ```

    数据如下所示:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. 现在，再多导入一些库：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 接下来需要将数据分训练模型所需的X和y两个dataframe。首先可将`cuisine`列的数据单独保存为标签（label）的dataframe。

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    输出看上去会是这样:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. 调用`drop()`函数将 `Unnamed: 0`和 `cuisine`列删除，并将余下的数据作为可以用于训练的特证（feature）:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    你的特证（feature）数据看上去将会是这样:

    | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |     |
    | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: | --- |
    |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |    0 |     ... |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0   |
    |      1 |        1 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |    0 |     ... |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0   |
    |      2 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |    0 |     ... |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0   |
    |      3 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |    0 |     ... |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0   |
    |      4 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |    0 |     ... |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        1 | 0   |

现在，你已经准备好可以开始训练你的模型了！

## 选则你的分类器

你的数据已经清洗干净并已经准备好可以进行训练了，现在需要决定你想使用的算法来完成这项任务。

Scikit_learn将分类任务归在了监督学习目录中，在这个目录中你将会找到很多方法来进行分类。乍一看上去，有点[琳琅满目](https://scikit-learn.org/stable/supervised_learning.html)。下面这些方法都包含了分类技术：

- 线性模型（Linear Models）
- 支持向量机（Support Vector Machines）
- 随机梯度下降（Stochastic Gradient Descent）
- 最近邻（Nearest Neighbors）
- 高斯过程（Gaussian Processes）
- 决策树（Decision Trees）
- 集成方法（投票分类器）（Ensemble methods（voting classifier）） 
- 多类别多输出算法（多类别多标签分类，多类别多输出分类）（Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)）

> 你也可以使用[神经网络来分类数据](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), 但这对于本课程来说有点超纲了。

### 如何选择分类器?

那么，你应该选择哪一个分类器呢？一般来说，可以选择多个方法并对比他们运行后的结果。Scikit-learn提供了各种算法（包括KNeighbors、 SVC two ways、 GaussianProcessClassifier、 DecisionTreeClassifier、 RandomForestClassifier、 MLPClassifier、 AdaBoostClassifier、 GaussianNB以及QuadraticDiscrinationAnalysis）的[比较](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，并且对比较的结果进行了可视化的展示：

![各分类器比较](../images/comparison.png)
> 图表来源于Scikit-learn的官方文档

> AutoML通过在云端运行这些比较非常完美地解决的这个问题，使得你能够根据你的数据选择最佳的算法。试试[这里](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-15963-cxa)。

### 一种更好的方法

不过，比起无脑地猜测，根据这份可以下载的[机器学习作弊表]中的方法是一个更好的选择。在表中我们可以发现对于这个多类型的分类任务，可以有一些选择：

A better way than wildly guessing, however, is to follow the ideas on this downloadable [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-15963-cxa). Here, we discover that, for our multiclass problem, we have some choices:

![多类型问题作弊表](../images/cheatsheet.png)
> 微软算法作弊表中关于多类型分类任务可选算法的部分

✅ 下载这份作弊表，打印出来，挂在你的墙上吧！

### 推导过程

Let's see if we can reason our way through different approaches given the constraints we have:让我们看看根据我们所有的限制条件推导下各中方法的可行性：

- **神经网络（Neural Network）太过复杂了**。我们的数据很清晰但数据量比较小，此外我们是通过notebook在本地进行训练，神经网络对于这个任务来说过于复杂了。
- **二分类法(two-class classifier)不可行**。我们不能使用二分类法,所以这就排除了一对多（one-vs-all）算法。 
- **决策树以及逻辑回归可行**. 决策树也许有用，或者也可以使用逻辑回归来处理多类型数据。
- **多类型增强决策数可以解决不同的问题**. 多类型增强决策树最适合非参数化的任务，即任务目标是建立一个排序，这对我们的任务并没有作用。

### 使用Scikit-learn 

我们将使用Scikit-learn来分析我们的数据。然而，在Scikit-learn中有很多种方法来使用逻辑回归。可以看一看逻辑回归算法可以[传递的参数](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

当我们需要Scikit-learn进行逻辑回归运算时，`multi_class` 以及 `solver`是最重要的两个参数，我们需要特别说明一下哎。 `multi_class` 的值决定了特定的行为。`solver`的值是我们需要使用的算法。并不是所有的solvers都可以匹配`multi_class`的值的。

According to the docs, in the multiclass case, the training algorithm根据文档，在多类型问题种，训练的算法:

- **使用“一对其余”(OvR)策略（scheme）**, 如果`multi_class`被设置为`ovr`
- **使用交叉熵损失（cross entropy loss）**, 如果`multi_class`被设置为`multinomial` (目前，`multinomial`只支持‘lbfgs’, ‘sag’, ‘saga’以及‘newton-cg’等 solver)。

> 🎓 其中“scheme”可以是“ovr(one-vs-rest)”也可以是“multinomial”。 因为逻辑回归事实上是设计用于支持二分类任务的，这些scheme将使其可以更好的支持多类型分类任务。[来源](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 “solver”被定义为是"用于解决优化问题的算法"。[来源](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn提供了以下这个表格来解释solver是如何应对的不同的数据结构所带来的不同的挑战的:

![solvers](../images/solvers.png)

## 练习 - 分割数据

因为你刚刚在上一节课中学习了逻辑回归，因此我们可以聚焦于此，来演练一下如何进行第一个模型的训练。通过调用`train_test_split()`可以把你的数据分割成训练集和测试集：


```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 练习 - 应用逻辑回归

因为我们正在进行多类型分类的案例，你需要决定选用什么  _scheme_ 以及使用什么 _solver_ 。使用带有multiclass设置的LogisticRegression，并将solver设置为**liblinear**来进行模型训练。

1. 创建逻辑回归，并将multi_class设置为`ovr`，同时将solver设置为 `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ 也可以试试其他solver比如`lbfgs`, 它通常可以作为默认的设置

    > 注意, 使用Pandas的[`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) 函数可以在需要的时候将你的数据进行降维

    准确率高达了**80%**!

1. 你也可以通过查看一行数据（比如第50行）来观察到模型运行的情况:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    运行后的输出如下:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 试试不同的行号来检查一下结果吧

1. 让我们再深入研究一下，你可以检查一下这回预测的准确率:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    运行后的输出如下———这是一道印度菜的可能性最大，是最合理的猜测:

    |          |        0 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | -------: | -------: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    |   indian | 0.715851 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    |  chinese | 0.229475 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | japanese | 0.029763 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    |   korean | 0.017277 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    |     thai | 0.007634 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |

    ✅ 你能解释下为什么模型会如此确定这是一道印度菜么？

1. 就和你在回归的课程中所做的一样，通过输出分类的报告，我们可以得到更多的细节：

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    | precision    | recall | f1-score | support |      |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | ------------ | ------ | -------- | ------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | chinese      | 0.73   | 0.71     | 0.72    | 229  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | indian       | 0.91   | 0.93     | 0.92    | 254  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | japanese     | 0.70   | 0.75     | 0.72    | 220  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | korean       | 0.86   | 0.76     | 0.81    | 242  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | thai         | 0.79   | 0.85     | 0.82    | 254  |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | accuracy     | 0.80   | 1199     |         |      |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | macro avg    | 0.80   | 0.80     | 0.80    | 1199 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    | weighted avg | 0.80   | 0.80     | 0.80    | 1199 |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |

## 挑战

在本课程中，你使用了清洗后的数据建立了一个机器学习的模型，能够根据一系列的配料来预测菜品来自于哪个国家。请再花点时间阅读一下Scikit-learn所提供的可以用来分类数据的其他选择。同时也可以深入研究一下“solver”的概念并尝试一下理解其背后的原理。

## [课后小测](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/22/)
## 回顾与自学

[这个课程](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)将对逻辑回归背后的数学原理进行更加深入的讲解

## 作业 

[学习solver](assignment.md)
