# 分类的介绍

在这四节课中，您将探索经典机器学习的一个基本重点——_分类_ 。我们将利用各种分类算法对泛亚的佳肴数据集进行演练。希望你如饥似渴了！

![就放一点儿！](../images/pinch.png)

> 在这些课程中享受泛亚美食吧！图片由 [Jen Looper](https://twitter.com/jenlooper) 提供

分类是[监督学习](https://wikipedia.org/wiki/Supervised_learning)的一种形式，与回归技术有很多共同之处。如果说机器学习就是通过使用数据集来预测事物的值或名称，那么分类通常就是其中两类：_二元分类_ 和 _多元分类_ 。

[![分类的介绍](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "分类的介绍")

> 🎥 点击上面的图片来观看：MIT的 John Guttag 向您介绍分类

请记住：

- **线性回归** 帮助您预测变量之间的关系，并准确预测新数据点与该线的关系。举个例子，您可以预测 *南瓜在九月与十二月的价格* 。
- **逻辑回归** 帮助您发现“二元归类”：在这个价位上，*这个南瓜是橙色还是非橙色* ？

分类利用各种算法提供了确定数据点的标签或类别的另外一种方法。让我们使用这些美食数据，看看能否通过观察菜谱，来确定其美食的来源。

## [课前测验](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/19/)

### 简介

分类是机器学习研究人员和数据科学家的基本活动之一。从基本的二元分类（“这是否是垃圾邮件？”），到使用计算机视觉的复杂图像识别。能够将数据分类并提出问题总是很有帮助的。

用更严谨的方式说明这一过程——您的分类方法创建了一个预测模型，使您能够将输入变量之间的关系映射到输出变量。

![binary vs. multiclass classification](../images/binary-multiclass.png)

> 适合使用分类算法处理的二元与多元问题。由 [Jen Looper](https://twitter.com/jenlooper) 绘制的图示

在开始着手清理数据、可视化以及为机器学习任务准备数据之前，让我们先了解一下机器学习之中可用于对数据进行分类的各种方式。

依据[维基百科的统计](https://wikipedia.org/wiki/Statistical_classification)，经典的机器学习分类法利用一些特征，如`吸烟`、`体重`与`年龄`等，来确定患某些疾病的可能性。跟您之前做过的的回归练习类似，在监督学习中，机器学习算法利用数据集的类别（或“特征”），来进行归类和预测，并得出一个分类或结果。

✅ 让我们花点时间来思考关于美食的数据集。多元模型能够回答什么？二元模型能够回答什么？如果您想确定给定的菜肴是否会使用胡芦巴怎么办？如果您想知道拥有一个装满八角、朝鲜蓟、花椰菜和芥末的食材包，您是否可以制作出典型的印度菜？

[![Crazy mystery baskets](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Crazy mystery baskets")

> 🎥 单击上面的图片观看视频。“疯厨胡炖（Chopped）”节目会拿出一个“神秘篮子”，其中装有一些随机的食材，厨师必须用这些食材来制作菜肴。在这种情况下，机器学习当然会很有帮助！
## 你好呀 '分类器'

关于这个美食数据集的问题实际上是一个**多元分类**问题——如果我们有一些选定国家的美食，知道了它们的配料，推测出他们是哪个国家的美食。

Scikit-learn 提供了适用于不同问题类型的几种用于对数据进行分类的算法模型。在接下来的两课中，您将了解其中的几种算法。

## 练习 - 清理和平衡您的数据 

The first task at hand, before starting this project, is to clean and **balance** your data to get better results. Start with the blank _notebook.ipynb_ file in the root of this folder.

The first thing to install is [imblearn](https://imbalanced-learn.org/stable/). This is a Scikit-learn package that will allow you to better balance the data (you will learn more about this task in a minute).

1. To install `imblearn`, run `pip install`, like so:

    ```python
    pip install imblearn
    ```

1. Import the packages you need to import your data and visualize it, also import `SMOTE` from `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Now you are set up to read import the data next.

1. The next task will be to import the data:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Using `read_csv()` will read the content of the csv file _cusines.csv_ and place it in the variable `df`.

1. Check the data's shape:

    ```python
    df.head()
    ```

   The first five rows look like this:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Get info about this data by calling `info()`:

    ```python
    df.info()
    ```

    Your out resembles:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Exercise - learning about cuisines

Now the work starts to become more interesting. Let's discover the distribution of data, per cuisine 

1. Plot the data as bars by calling `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![cuisine data distribution](images/cuisine-dist.png)

    There are a finite number of cuisines, but the distribution of data is uneven. You can fix that! Before doing so, explore a little more. 

1. Find out how much data is available per cuisine and print it out:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    the output looks like so:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Discovering ingredients

Now you can dig deeper into the data and learn what are the typical ingredients per cuisine. You should clean out recurrent data that creates confusion between cuisines, so let's learn about this problem.

1. Create a function `create_ingredient()` in Python to create an ingredient dataframe. This function will start by dropping an unhelpful column and sort through ingredients by their count:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False
        inplace=False)
        return ingredient_df
    ```

   Now you can use that function to get an idea of top ten most popular ingredients by cuisine.

1. Call `create_ingredient()` and plot it calling `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](images/thai.png)

1. Do the same for the japanese data:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](images/japanese.png)

1. Now for the chinese ingredients:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](images/chinese.png)

1. Plot the indian ingredients:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](images/indian.png)

1. Finally, plot the korean ingredients:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](images/korean.png)

1. Now, drop the most common ingredients that create confusion between distinct cuisines, by calling `drop()`: 

   Everyone loves rice, garlic and ginger!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Balance the dataset

Now that you have cleaned the data, use [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - to balance it.

1. Call `fit_resample()`, this strategy generates new samples by interpolation.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    By balancing your data, you'll have better results when classifying it. Think about a binary classification. If most of your data is one class, a ML model is going to predict that class more frequently, just because there is more data for it. Balancing the data takes any skewed data and helps remove this imbalance. 

1. Now you can check the numbers of labels per ingredient:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Your output looks like so:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    The data is nice and clean, balanced, and very delicious! 

1. The last step is to save your balanced data, including labels and features, into a new dataframe that can be exported into a file:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. You can take one more look at the data using `transformed_df.head()` and `transformed_df.info()`. Save a copy of this data for use in future lessons:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisine.csv")
    ```

    This fresh CSV can now be found in the root data folder.

---

## 🚀Challenge

This curriculum contains several interesting datasets. Dig through the `data` folders and see if any contain datasets that would be appropriate for binary or multi-class classification? What questions would you ask of this dataset?

## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/20/)

## Review & Self Study

Explore SMOTE's API. What use cases is it best used for? What problems does it solve?

## Assignment 

[Explore classification methods](assignment.md)
