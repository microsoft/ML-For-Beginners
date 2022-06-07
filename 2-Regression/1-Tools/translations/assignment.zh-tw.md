# 用 Scikit-learn 實現一次回歸算法

## 說明

先看看 Scikit-learn 中的 [Linnerud 數據集](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud)
這個數據集中有多個[目標變量（target）](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset)，其中包含了三種運動（訓練數據）和三個生理指標（目標變量）組成，這些數據都是從一個健身俱樂部中的 20 名中年男子收集到的。

之後用自己的方式，創建一個可以描述腰圍和完成仰臥起坐個數關系的回歸模型。用同樣的方式對這個數據集中的其它數據也建立一下模型探究一下其中的關系。

## 評判標準

| 標準                       | 優秀                           | 中規中矩                      | 仍需努力          |
| ------------------------------ | ----------------------------------- | ----------------------------- | -------------------------- |
| 需要提交一段能描述數據集中關系的文字 | 很好的描述了數據集中的關系 | 只能描述少部分的關系 | 啥都沒有提交 |
