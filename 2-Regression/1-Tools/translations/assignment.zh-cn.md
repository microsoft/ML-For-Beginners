# 用 Scikit-learn 实现一次回归算法

## 说明

先看看 Scikit-learn 中的 [Linnerud 数据集](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud)
这个数据集中有多个[目标变量（target）](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset)，其中包含了三种运动（训练数据）和三个生理指标（目标变量）组成，这些数据都是从一个健身俱乐部中的 20 名中年男子收集到的。

之后用自己的方式，创建一个可以描述腰围和完成仰卧起坐个数关系的回归模型。用同样的方式对这个数据集中的其它数据也建立一下模型探究一下其中的关系。

## 评判标准

| 标准                       | 优秀                           | 中规中矩                      | 仍需努力          |
| ------------------------------ | ----------------------------------- | ----------------------------- | -------------------------- |
| 需要提交一段能描述数据集中关系的文字 | 很好的描述了数据集中的关系 | 只能描述少部分的关系 | 啥都没有提交 |
