# 使用 Scikit-learn 构建回归模型：四种回归方法

## 初学者提示

线性回归用于预测<strong>数值型</strong>（例如房价、温度或销售额）。
它通过找到最能代表输入特征与输出关系的直线来实现。

本课将重点理解概念，然后再探索更高级的回归技术。
![线性回归与多项式回归信息图](../../../../translated_images/zh-CN/linear-polynomial.5523c7cb6576ccab.webp)
> 信息图由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 制作
## [课前测验](https://ff-quizzes.netlify.app/en/ml/)

> ### [本课内容也提供 R 版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介绍

迄今为止，你已经通过使用本课程将会继续使用的南瓜定价数据集了解了回归是什么。你也用 Matplotlib 对数据进行了可视化。

现在你准备深入了解机器学习中的回归。可视化有助于理解数据，但机器学习的真正能力在于<strong>训练模型</strong>。模型通过历史数据训练，自动捕捉数据之间的依赖关系，并允许你预测模型之前未见过的新数据结果。

本课你将学习两种回归类型：<strong>基本线性回归</strong> 和 <strong>多项式回归</strong>，以及这些技术背后的一些数学原理。通过这些模型，我们可以根据不同输入数据预测南瓜价格。

[![面向初学者的机器学习——理解线性回归](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "面向初学者的机器学习——理解线性回归")

> 🎥 点击上图观看线性回归简短视频介绍。

> 在整个课程中，我们假设你有最少的数学知识背景，力图为来自其他领域的学生提供可访问的内容，因此请关注笔记、🧮 提示、图示及其他学习工具，以帮助理解。

### 先决条件

你现在应该熟悉我们正在研究的南瓜数据的结构。你可以在本课的 _notebook.ipynb_ 文件中找到已预加载和预处理的数据。在该文件中，南瓜价格以每蒲式耳价格显示在一个新的数据框中。确保你能在 Visual Studio Code 的内核中运行这些笔记本。

### 准备工作

提醒一下，我们加载这些数据是为了提问：

- 什么时候买南瓜最合适？
- 一箱迷你南瓜的价格大概是多少？
- 该以半蒲式耳篮买，还是1 1/9蒲式耳箱买？
让我们继续深入挖掘这些数据。

在上一课中，你创建了 Pandas 数据框，并用部分原始数据填充，标准化为每蒲式耳价格。这样做后，你只能获得大约 400 条数据点，且仅限秋季月份。

看看我们在本课随附笔记本中预加载的数据。数据已经加载完毕，并绘制了显示月份数据的初始散点图。也许通过进一步清洗，我们能更详细了解数据的本质。

## 线性回归线

如你在第一课中所学，线性回归的目标是绘制一条直线来：

- <strong>显示变量关系</strong>。展示变量间的关系
- <strong>进行预测</strong>。准确预测新数据点在直线上的位置

通常采用<strong>最小二乘回归</strong>法绘制此类直线。“最小二乘”指的是尽量减少模型总误差的过程。对于每个数据点，我们测量实际点与回归线之间的垂直距离（称为残差）。

对这些距离平方的原因有两个：

1. <strong>忽略方向只看大小</strong>：我们希望将误差 -5 和 +5 等同看待，平方可将所有值变为正数。

2. <strong>惩罚离群值</strong>：平方更看重大的误差，使回归线尽量贴近远离的点。

然后我们将所有平方的距离相加。我们的目标是找到使该总和最小的特定直线（即“最小二乘”）。

> **🧮 数学详解**
>
> 这条称为<strong>最佳拟合线</strong>的直线可用[一个方程表示](https://en.wikipedia.org/wiki/Simple_linear_regression)： 
>
> ```
> Y = a + bX
> ```
>
> `X` 是“解释变量”，“Y”是“因变量”。直线的斜率是 `b`，`a` 是 y 轴截距，即当 `X=0` 时 `Y` 的值。
>
>![计算斜率](../../../../translated_images/zh-CN/slope.f3c9d5910ddbfcf9.webp)
>
> 首先，计算斜率 `b`。信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作
>
> 换句话说，针对我们原始的南瓜数据问题：“按月份预测南瓜每蒲式耳价格”，`X` 是价格，`Y` 是销售月份。
>
>![完成方程](../../../../translated_images/zh-CN/calculation.a209813050a1ddb1.webp)
>
> 计算 `Y` 的值。如果你支付大约4美元，那一定是四月！信息图由 [Jen Looper](https://twitter.com/jenlooper) 制作
>
> 计算此直线的数学方法必须展示斜率，同时也会以截距为依据，即当 `X=0` 时 `Y` 的位置。
>
> 你可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 网站上观察计算步骤。也可访问[最小二乘计算器](https://www.mathsisfun.com/data/least-squares-calculator.html)，观看数字如何影响直线。

## 相关性

另一个需要了解的概念是给定 `X` 和 `Y` 变量之间的<strong>相关系数</strong>。通过散点图可以快速直观地看出这个系数。散点沿直线排列的图显示高相关性，点散布无序则显示相关性低。

一个好的线性回归模型，应该在最小二乘法下，拥有较高（接近 1 而非 0）的相关系数。

✅ 运行本课附带的笔记本，查看“月份 vs 价格”的散点图。在直观解释该散点图后，南瓜销售相关的“月份 vs 价格”数据看起来相关性高还是低？如果改用更细粒度指标，比如“年份中的天数”（即年初以来的天数），相关性会改变吗？

以下代码假设我们已清理数据，并获得名为 `new_pumpkins` 的数据框，类似于：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理数据的代码见 [`notebook.ipynb`](notebook.ipynb)，我们采用了与上一课相同的清理步骤，并通过以下表达式计算了 `DayOfYear` 列：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

既然你了解了线性回归背后的数学，让我们创建一个回归模型，看看是否能预测哪种南瓜包装价格最好。假如有人为节日南瓜园买南瓜，这些信息有助于优化购买。

## 寻找相关性

[![面向初学者的机器学习 - 寻找相关性：线性回归的关键](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "面向初学者的机器学习 - 寻找相关性：线性回归的关键")

> 🎥 点击上图观看相关性简短视频介绍。

上一课可见，不同月份的平均价格如下图所示：

<img alt="按月份的平均价格" src="../../../../translated_images/zh-CN/barchart.a833ea9194346d76.webp" width="50%"/>

这暗示两者之间应有相关性，我们可训练线性回归模型来预测 `Month` 和 `Price`，或 `DayOfYear` 和 `Price` 之间的关系。下面是后者的散点图：

<img alt="价格与年中天数的散点图" src="../../../../translated_images/zh-CN/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

使用 `corr` 函数检测是否相关：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

相关性看起来较小，月份相关系数约为 -0.15，年中天数相关系数约为 -0.17，但或许存在其他重要关系。似乎不同价格簇对应不同南瓜品种。为验证这个假设，我们给各品种使用不同颜色绘制散点图。通过向 `scatter` 函数传递 `ax` 参数，可以将所有点绘制于同一图表：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="价格与年中天数散点图（彩色区分）" src="../../../../translated_images/zh-CN/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

调研表明，品种对价格影响更大于销售日期。下图通过柱状图展示：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="按品种划分的价格柱状图" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

暂时只关注“派类型”南瓜，看看日期对价格的影响：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="价格与年中天数的散点图" src="../../../../translated_images/zh-CN/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

如果用 `corr` 函数计算 `Price` 与 `DayOfYear` 的相关系数，会得到大约 `-0.27`，说明训练预测模型是合理的。

> 训练线性回归模型前，确保数据干净很重要。线性回归对空值敏感，因此最好去除所有空单元格：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一种方法是用对应列的均值填充空值。

## 简单线性回归

[![面向初学者的机器学习 - 使用 Scikit-learn 的线性和多项式回归](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "面向初学者的机器学习 - 使用 Scikit-learn 的线性和多项式回归")

> 🎥 点击上图观看线性与多项式回归简短视频介绍。

我们将用 **Scikit-learn** 库训练线性回归模型。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

首先将输入特征和预期输出（标签）分成不同的 numpy 数组：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意，为了让线性回归库正确识别输入，我们做了 `reshape` 操作。线性回归期望输入为二维数组，每行是特征向量。本例只用一个输入特征，因此需要形状为 N × 1 的数组，N 为数据集大小。

接着将数据拆分为训练集和测试集，方便训练后验证模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最后，训练线性回归模型只需两行代码。定义 `LinearRegression` 对象，用 `fit` 方法拟合数据：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 对象在 `fit` 后包含了回归的所有系数，可以通过 `.coef_` 属性访问。在我们的例子中，只有一个系数，应该在 `-0.017` 附近。这意味着价格随着时间似乎略有下降，但幅度不大，大约每天下降两分钱。我们还可以通过 `lin_reg.intercept_` 访问回归与 Y 轴的交点——在我们的例子中大约是 `21`，表示年初的价格。

为了查看模型的准确性，我们可以在测试数据集上预测价格，然后衡量预测值与预期值的接近程度。这可以使用均方根误差（RMSE）指标完成，即期望值与预测值全部平方差的均值开根号。

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

我们的误差约为 2 点，即约 17%。不是特别好。另一个模型质量指标是<strong>决定系数</strong>，可以这样获得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

如果值为 0，说明模型不考虑输入数据，表现为<em>最差的线性预测器</em>，即结果的均值。值为 1 则意味着我们能完美预测所有期望输出。在我们的例子中，系数约为 0.06，较低。

我们也可以绘制测试数据和回归线，以更直观地了解回归效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-CN/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多项式回归

另一种线性回归是多项式回归。尽管有时候变量间存在线性关系——比如体积越大的南瓜价格越高——但有时这些关系无法用平面或直线表达。

✅ 这里有一些[更多示例](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples)展示了适合使用多项式回归的数据

再看看日期和价格的关系。这个散点图一定要用直线分析吗？价格难道不会波动吗？这时，就可以尝试多项式回归。

✅ 多项式是由一个或多个变量和系数组成的数学表达式

多项式回归通过曲线更好地拟合非线性数据。在我们的案例中，如果将平方的 `DayOfYear` 变量加入输入数据中，应能以抛物线拟合数据，并在一年中某点处有一个极小值。

Scikit-learn 包含了有用的[流水线 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)用于组合数据处理的多个步骤。<strong>流水线</strong>是一系列<strong>估计器</strong>的链。在我们的例子中，我们将创建一个先添加多项式特征、然后进行回归训练的流水线：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 意味着我们会包含输入数据的所有二阶多项式。在我们的例子中仅有 `DayOfYear`<sup>2</sup>，但若输入有两个变量 X 和 Y，则会加上 X<sup>2</sup>、XY 和 Y<sup>2</sup>。如果需要，也可以使用更高阶的多项式。

流水线的用法与原始 `LinearRegression` 对象相同，即可以对流水线 `fit`，然后用 `predict` 得到预测结果：

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

为了绘制平滑的拟合曲线，我们用 `np.linspace` 创建均匀的输入值范围，而不是在无序的测试数据上直接绘图（那样会出现锯齿形曲线）：

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

下面是测试数据和拟合曲线的图表：

<img alt="Polynomial regression" src="../../../../translated_images/zh-CN/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多项式回归，我们能稍微降低 RMSE 并提高决定系数，但提升不大。我们需要考虑更多特征！

> 你可以看到南瓜价格的最低点大约出现在万圣节左右。你能解释这现象吗？ 

🎃 恭喜，你刚刚建立了一个可以预测派南瓜价格的模型。你也可以用同样方法处理所有南瓜类型，但那样比较繁琐。接下来我们学习如何将南瓜品种纳入模型！

## 分类特征

理想情况下，我们希望用同一个模型预测不同南瓜品种的价格。然而，`Variety` 列与 `Month` 等列不同，因为它包含非数字值。这类列称为<strong>分类变量</strong>。

[![针对初学者的机器学习——使用线性回归进行分类特征预测](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "针对初学者的机器学习——使用线性回归进行分类特征预测")

> 🎥 点击上方图片观看关于使用分类特征的简短视频介绍。

这里显示了平均价格如何随品种变化：

<img alt="Average price by variety" src="../../../../translated_images/zh-CN/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考虑品种，首先需要将其转换为数值形式，也就是<strong>编码</strong>。有几种常用方法：

* 简单的<strong>数字编码</strong>会建立一个不同品种的表，然后用表中的索引替代品种名称。对于线性回归来说，这不是最佳方法，因为线性回归会将索引的数值直接加权，然而索引和价格之间显然是非线性的，即便我们保证索引按特定方式排序。
* <strong>独热编码</strong>会用四个不同的列替代 `Variety` 列，每个品种一个列。相应行是该品种则该列为 `1`，否则为 `0`。这样线性回归就有了四个系数，对应四个南瓜品种，分别代表该品种的“起始价”（或更准确地说是“附加价”）。

下面代码展示了如何对品种进行独热编码：

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

为了用独热编码后的品种训练线性回归，只需正确初始化 `X` 和 `y`：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其余代码与之前训练线性回归相同。尝试后你会发现均方误差差不多，但决定系数提高到了约 77%。要获得更准确的预测，可以同时考虑更多分类特征和数值特征，比如 `Month` 或 `DayOfYear`。要合并成一个完整特征数组，可使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

这里我们还考虑了 `City` 和 `Package` 类型，得到 RMSE 2.84（10.5%），决定系数 0.94！

## 综合应用

为了获得最佳模型，我们可以将上述示例中的组合（独热编码分类 + 数值特征）数据与多项式回归结合。下面是完整代码，方便参考：

```python
# 设置训练数据
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 制作训练-测试集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 设置并训练管道
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 预测测试数据结果
pred = pipeline.predict(X_test)

# 计算均方根误差和决定系数
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

这将给出近 97% 的最高决定系数和 RMSE=2.23（约 8% 预测误差）。

| 模型 | RMSE | 决定系数 |
|-------|-----|---------------|
| `DayOfYear` 线性 | 2.77（17.2%） | 0.07 |
| `DayOfYear` 多项式 | 2.73（17.0%） | 0.08 |
| `Variety` 线性 | 5.24（19.7%） | 0.77 |
| 所有特征 线性 | 2.84（10.5%） | 0.94 |
| 所有特征 多项式 | 2.23（8.25%） | 0.97 |

🏆 太棒了！你在本节课里创建了四个回归模型，并将模型质量提高到 97%。在回归的最后一部分，你将学习如何用逻辑回归做分类判定。

---
## 🚀挑战

在此笔记本中测试多个不同变量，观察相关性与模型准确度的关系。

## [课后测验](https://ff-quizzes.netlify.app/en/ml/)

## 复习与自学

本节课我们学习了线性回归。还有其他重要的回归类型。请阅读关于逐步回归、岭回归（Ridge）、套索回归（Lasso）和弹性网（Elasticnet）技术。推荐学习的好课程是[斯坦福统计学习课程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 练习

[构建模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免责声明**：  
本文档使用 AI 翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。虽然我们力求准确，但请注意自动翻译可能存在错误或不准确之处。原始文档的原语言版本应被视为权威来源。对于重要信息，建议采用专业人工翻译。我们不对因使用本翻译而产生的任何误解或误释承担责任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->