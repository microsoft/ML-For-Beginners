# 逻辑回归用于预测类别

![逻辑回归与线性回归信息图](../../../../translated_images/zh-CN/linear-vs-logistic.ba180bf95e7ee667.webp)

## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课也提供 R 语言版本！](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## 简介

在本回归课的最后一个单元中，我们将学习逻辑回归（Logistic Regression），这是基础且经典的机器学习技术之一。您可以使用此技术发现模式以预测二元类别。这个糖果是巧克力吗？这种疾病会传染吗？这个客户会选择这个产品吗？

本课您将学到：

- 一个用于数据可视化的新库
- 逻辑回归的相关技术

✅ 在此 [学习模块](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott) 中加深您对这类回归工作的理解

## 先决条件

在使用过南瓜数据后，我们已经足够熟悉数据，以发现其中有一个二元类别可以使用：`Color`（颜色）。

让我们建立一个逻辑回归模型来预测，基于某些变量，_一个给定南瓜可能是什么颜色_（橘色🎃或白色👻）。

> 为什么在关于回归的一组课程里讨论二元分类？只是语言上的方便，逻辑回归其实是[一种分类方法](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)，只是基于线性模型。下一组课程将介绍其他分类数据的方法。

## 确定问题

本次我们将把它表示为二元：“白色”或“非白色”。数据里还有“条纹”类别，但数量很少，我们不会使用它。且删除缺失值后它会消失。

> 🎃 有趣的事实：我们有时称白南瓜为“鬼南瓜”。它们不易雕刻，所以没有橘色南瓜那么受欢迎，但外形很酷！因此，我们的提问也可以改成“鬼”或“非鬼”。👻

## 关于逻辑回归

逻辑回归与之前讲过的线性回归在几个重要方面不同。

[![初学者机器学习 - 理解用于分类的逻辑回归](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "初学者机器学习 - 理解用于分类的逻辑回归")

> 🎥 点击上图观看逻辑回归简短视频概述。

### 二元分类

逻辑回归不具备线性回归的所有特性。前者针对二元类别（“白色或非白色”）进行预测，而后者能够预测连续数值，例如给出南瓜产地和收获时间，预测_价格会上涨多少_。

![南瓜分类模型](../../../../translated_images/zh-CN/pumpkin-classifier.562771f104ad5436.webp)
> 信息图由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 制作

### 其他分类

逻辑回归还有其他类型，包含多项式回归和有序回归：

- <strong>多项式回归</strong>，存在多个类别——“橙色、白色和条纹”。
- <strong>有序回归</strong>，类别有顺序，适用于如我们按尺寸（迷你、小、中、大、加大、加加大）排序的南瓜。

![多项式与有序回归](../../../../translated_images/zh-CN/multinomial-vs-ordinal.36701b4850e37d86.webp)

### 变量不必相关

记得线性回归变量越相关表现越好吗？逻辑回归正好相反——变量不必对齐。这适合本数据，相关性较弱。

### 你需要大量干净数据

逻辑回归使用越多数据结果越准确；我们的数据集较小，不是最优，需注意。

[![初学者机器学习 - 逻辑回归数据分析与准备](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "初学者机器学习 - 逻辑回归数据分析与准备")

> 🎥 点击上图观看线性回归数据准备简短视频概述

✅ 思考哪些数据适合用逻辑回归

## 练习 - 整理数据

先稍作清理，删除缺失值，选取部分列：

1. 添加以下代码：

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    你随时可以查看你的新数据框：

    ```python
    pumpkins.info
    ```

### 可视化 - 类别图

现在你已经再次加载了 [starter notebook](./notebook.ipynb) 的南瓜数据，并清理了数据以保留一些变量，包括`Color`。让我们用另一个基于 Matplotlib 的库 [Seaborn](https://seaborn.pydata.org/index.html) 在笔记本中可视化数据框。

Seaborn 提供了非常棒的数据可视化方式。例如，使用类别绘图比较每个`Variety`和`Color`的分布。

1. 使用`catplot`函数创建该图，使用我们的南瓜数据`pumpkins`，并为每个南瓜类别（橘色或白色）指定颜色映射：

    ```python
    import seaborn as sns
    
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }

    sns.catplot(
    data=pumpkins, y="Variety", hue="Color", kind="count",
    palette=palette, 
    )
    ```

    ![可视化数据网格](../../../../translated_images/zh-CN/pumpkins_catplot_1.c55c409b71fea2ec.webp)

    观察该数据，你可以看到颜色数据与品种的关联。

    ✅ 根据此类别绘图，你能想象出哪些有趣的探索？

### 数据预处理：特征和标签编码
我们的南瓜数据中所有列都是字符串。人能直观理解分类数据，但机器理解较难。机器学习算法对数字处理更有效。编码是在数据预处理阶段非常重要步骤，它帮我们将分类数据转换成数字，且不丢失信息。良好编码有助构建优良模型。

特征编码主要有两类编码器：

1. 有序编码器：适用于有顺序的分类变量，如我们数据集的`Item Size`列。它创建映射，将每个类别用数字表示，即该类别在列中的顺序。

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. 类别编码器：适用于无序分类变量，如我们数据集中除`Item Size`外的特征。它是独热编码，即每个类别用一个二元列表示：如果某南瓜属于该品种，编码变量为1，否则为0。

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```
然后，用`ColumnTransformer`将多个编码器合并为一步并应用于相应列。

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```
另一方面，对标签编码，我们用 Scikit-learn 的`LabelEncoder`类工具，把标签规范化为只包含0和1之间值（这里是0和1）。

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```
编码特征和标签完成后，可合并成新数据框`encoded_pumpkins`。

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```
✅ 使用有序编码器对`Item Size`列有什么优势？

### 分析变量之间的关系

预处理完数据后，我们可以分析特征和标签之间的关系，初步判断模型预测标签的能力。
最佳做法是绘图。我们将再次用 Seaborn 的 `catplot` 函数，在类别图中展示 `Item Size`、`Variety` 和 `Color` 间的关系。为了更好地绘图，我们使用编码后的 `Item Size` 列和未编码的 `Variety` 列。

```python
    palette = {
    'ORANGE': 'orange',
    'WHITE': 'wheat',
    }
    pumpkins['Item Size'] = encoded_pumpkins['ord__Item Size']

    g = sns.catplot(
        data=pumpkins,
        x="Item Size", y="Color", row='Variety',
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        height=1.8, aspect=4, palette=palette,
    )
    g.set(xlabel="Item Size", ylabel="").set(xlim=(0,6))
    g.set_titles(row_template="{row_name}")
```
![可视化数据的类别图](../../../../translated_images/zh-CN/pumpkins_catplot_2.87a354447880b388.webp)

### 使用 swarm plot

因为颜色是二元类别（白或非白），它需要“[特殊方法](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar) 可视化”。还有其他方式展示该分类与变量的关系。

你可以用 Seaborn 的图表并排可视化变量。

1. 试试“swarm”图，显示数值分布：

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![数据分布的群聚图](../../../../translated_images/zh-CN/swarm_2.efeacfca536c2b57.webp)

<strong>注意</strong>：上述代码可能产生警告，因为 seaborn 无法很好地将大量数据点放入 swarm 图。解决方案是用“size”参数减小标记尺寸，但这会影响图的可读性。


> **🧮 给我数学原理**
>
> 逻辑回归基于“最大似然”概念，使用[Sigmoid函数](https://wikipedia.org/wiki/Sigmoid_function)。Sigmoid函数的图形像S形。它把一个值映射到0到1之间。其曲线又叫“逻辑曲线”。公式如下：
>
> ![逻辑函数](../../../../translated_images/zh-CN/sigmoid.8b7ba9d095c789cf.webp)
>
> 其中 Sigmoid 中点在 x=0 处，L 是曲线最大值，k 是曲线陡峭程度。若函数值大于0.5，则该二元分类标签为概率的“1”；否则归为“0”。

## 构建模型

在 Scikit-learn 中构建此类二元分类模型异常简便。

[![初学者机器学习 - 逻辑回归分类](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "初学者机器学习 - 逻辑回归分类")

> 🎥 点击上图观看构建线性回归模型简短视频

1. 选择要用在分类模型中的变量，并使用`train_test_split()`拆分训练集和测试集：

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. 通过调用`fit()`训练模型，然后打印结果：

    ```python
    from sklearn.metrics import f1_score, classification_report 
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('F1-score: ', f1_score(y_test, predictions))
    ```

    查看模型得分情况，考虑到只有约1000行数据，这表现不错：

    ```output
                       precision    recall  f1-score   support
    
                    0       0.94      0.98      0.96       166
                    1       0.85      0.67      0.75        33
    
        accuracy                                0.92       199
        macro avg           0.89      0.82      0.85       199
        weighted avg        0.92      0.92      0.92       199
    
        Predicted labels:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0
        0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 1 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 1 1 0
        0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 1 0 0 0 0 0 0 0 0 1 1]
        F1-score:  0.7457627118644068
    ```

## 通过混淆矩阵更好理解

你可以打印出[报告](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)来获得分数，但用[混淆矩阵](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)更直观理解模型表现。

> 🎓 “[混淆矩阵](https://wikipedia.org/wiki/Confusion_matrix)”（或“错误矩阵”）是表格，表达模型真阳性与假阳性、真阴性与假阴性的统计，评估预测准确度。

1. 使用混淆矩阵，调用 `confusion_matrix()`：

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    查看模型的混淆矩阵：

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

在 Scikit-learn 中，混淆矩阵的行（轴0）是实际标签，列（轴1）是预测标签。

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

这里发生了什么？假设模型要在两个类别中分类南瓜，“白色”和“非白色”。

- 如果模型预测为非白色，且实际上确实是非白色，我们称其为真阴性，即左上数字。
- 如果模型预测为白色，但实际上是非白色，我们称其为假阳性，即左下数字。
- 如果模型预测为非白色，但实际上是白色，我们称其为假阴性，即右上数字。
- 如果模型预测为白色，且实际上是白色，我们称其为真阳性，即右下数字。


如你所料，拥有较多的真正例和真负例以及较少的假正例和假负例是更理想的，这意味着模型表现更好。

混淆矩阵是如何与精确率和召回率相关联的？请记住，上面打印的分类报告显示了精确率（0.85）和召回率（0.67）。

精确率 = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

召回率 = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ 问题：根据混淆矩阵，模型表现如何？答：还不错；真负例数量较多，但也存在一些假负例。 

让我们回顾一下之前看到的术语，借助混淆矩阵中 TP/TN 和 FP/FN 的映射关系：

🎓 精确率：TP/(TP + FP) 在检索的实例中相关实例的比例（例如哪些标签被准确标注）

🎓 召回率：TP/(TP + FN) 检索到的相关实例的比例，无论是否标注正确

🎓 f1-score: (2 * 精确率 * 召回率)/(精确率 + 召回率) 精确率和召回率的加权平均，最佳值为1，最差为0

🎓 支持度：检索到的每个标签的出现次数

🎓 准确率：(TP + TN)/(TP + TN + FP + FN) 对样本的标签预测准确的百分比。

🎓 宏平均：为每个标签计算未加权的平均指标，不考虑标签不平衡。

🎓 加权平均：为每个标签计算的平均指标，通过其支持度（每个标签的真实实例数量）加权，考虑标签不平衡。

✅ 如果你想让模型减少假负例的数量，你会关注哪个指标？

## 可视化该模型的 ROC 曲线

[![ML 入门 - 使用 ROC 曲线分析逻辑回归性能](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "ML 入门 - 使用 ROC 曲线分析逻辑回归性能")

> 🎥 点击上方图片观看关于 ROC 曲线的简短视频介绍

我们再做一个可视化，来看看所谓的“ROC”曲线：

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

fig = plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

使用 Matplotlib 绘制模型的[接收者操作特征](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc)或ROC曲线。ROC曲线常用于观察分类器输出的真阳性与假阳性的关系。“ROC 曲线通常在 Y 轴表示真正例率，在 X 轴表示假正例率。”因此，曲线的陡峭程度以及中间线与曲线之间的空间很重要：你想要一条快速上升并越过这条线的曲线。在我们的例子中，起始处存在假正例，然后曲线正常上升并且越过了那条线：

![ROC](../../../../translated_images/zh-CN/ROC_2.777f20cdfc4988ca.webp)

最后，使用 Scikit-learn 的 [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score)计算实际的“曲线下面积”（AUC）：

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```
 结果为 `0.9749908725812341`。鉴于 AUC 范围是 0 到 1，分数越大越好，因为预测完全正确的模型的 AUC 是 1；在这种情况下，该模型 _相当不错_。

在未来的分类课程中，你将学习如何迭代以提高模型分数。但现在，恭喜你！你已经完成了这些回归课程！

---
## 🚀挑战

逻辑回归还有很多内容值得深入！但最好的学习方式是实验。找到适合这类分析的数据集并用它建立一个模型。你学到了什么？提示：尝试 [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) 寻找有趣的数据集。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

阅读 [斯坦福大学的这篇论文](https://web.stanford.edu/~jurafsky/slp3/5.pdf) 的前几页，了解逻辑回归的一些实用应用。思考我们到目前为止学习的各种回归任务中，哪些任务更适合哪种回归类型。哪种方法效果最好？

## 作业 

[重试这个回归](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：
本文件由 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻译完成。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。原始语言版文件应视为权威来源。对于重要信息，建议使用专业人工翻译。我们对因使用本翻译而产生的任何误解或误释不承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->