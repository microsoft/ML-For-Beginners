<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T09:16:12+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "mo"
}
-->
# 美食分類器 1

在這節課中，您將使用上一節課保存的數據集，這是一個關於美食的平衡且乾淨的數據集。

您將使用這個數據集和多種分類器來_根據一組食材預測特定的國家美食_。在此過程中，您將深入了解算法如何用於分類任務。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)
# 準備工作

假設您已完成[第一課](../1-Introduction/README.md)，請確保在根目錄的 `/data` 文件夾中存在一個 _cleaned_cuisines.csv_ 文件，供這四節課使用。

## 練習 - 預測國家美食

1. 在本課的 _notebook.ipynb_ 文件夾中，導入該文件以及 Pandas 庫：

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    數據看起來如下：

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. 現在，導入更多的庫：

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. 將 X 和 y 坐標分成兩個數據框進行訓練。`cuisine` 可以作為標籤數據框：

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    它看起來如下：

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. 使用 `drop()` 刪除 `Unnamed: 0` 列和 `cuisine` 列。將剩餘的數據保存為可訓練的特徵：

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    您的特徵看起來如下：

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

現在您可以開始訓練您的模型了！

## 選擇分類器

現在您的數據已清理並準備好訓練，您需要決定使用哪種算法來完成任務。

Scikit-learn 將分類歸類為監督學習，在這個類別中，您會發現許多分類方法。[種類繁多](https://scikit-learn.org/stable/supervised_learning.html)，乍看之下可能令人眼花繚亂。以下方法都包含分類技術：

- 線性模型
- 支持向量機
- 隨機梯度下降
- 最近鄰
- 高斯過程
- 決策樹
- 集成方法（投票分類器）
- 多類和多輸出算法（多類多標籤分類，多類多輸出分類）

> 您也可以使用[神經網絡進行分類](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)，但這超出了本課的範圍。

### 選擇哪個分類器？

那麼，應該選擇哪個分類器呢？通常，通過嘗試多種分類器並尋找良好的結果是一種測試方法。Scikit-learn 提供了一個[並排比較](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，在創建的數據集上比較 KNeighbors、SVC 兩種方式、GaussianProcessClassifier、DecisionTreeClassifier、RandomForestClassifier、MLPClassifier、AdaBoostClassifier、GaussianNB 和 QuadraticDiscrinationAnalysis，並以可視化方式展示結果：

![分類器比較](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> 圖片來自 Scikit-learn 的文檔

> AutoML 可以輕鬆解決這個問題，通過在雲端運行這些比較，幫助您選擇最適合您數據的算法。試試看[這裡](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### 更好的方法

比盲目猜測更好的方法是遵循這份可下載的[機器學習速查表](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott)。在這裡，我們發現針對我們的多類問題，我們有一些選擇：

![多類問題速查表](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> 微軟算法速查表的一部分，詳細介紹了多類分類選項

✅ 下載這份速查表，打印出來，掛在牆上！

### 推理

讓我們看看是否可以根據我們的限制推理出不同的方法：

- **神經網絡太重**。考慮到我們的數據集雖然乾淨但規模較小，以及我們通過筆記本本地運行訓練的事實，神經網絡對於這項任務來說太過繁重。
- **不使用二類分類器**。我們不使用二類分類器，因此排除了一對多（one-vs-all）。
- **決策樹或邏輯回歸可能有效**。決策樹可能有效，或者多類數據的邏輯回歸也可能有效。
- **多類增強決策樹解決不同問題**。多類增強決策樹最適合非參數任務，例如設計排名的任務，因此對我們來說並不適用。

### 使用 Scikit-learn 

我們將使用 Scikit-learn 來分析數據。然而，在 Scikit-learn 中有許多方法可以使用邏輯回歸。查看[可傳遞的參數](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

基本上有兩個重要的參數 - `multi_class` 和 `solver` - 我們需要指定，當我們要求 Scikit-learn 執行邏輯回歸時。`multi_class` 值應用某種行為。solver 的值則是使用的算法。並非所有 solver 都可以與所有 `multi_class` 值配對。

根據文檔，在多類情況下，訓練算法：

- **使用一對多（OvR）方案**，如果 `multi_class` 選項設置為 `ovr`
- **使用交叉熵損失**，如果 `multi_class` 選項設置為 `multinomial`。（目前 `multinomial` 選項僅支持 ‘lbfgs’、‘sag’、‘saga’ 和 ‘newton-cg’ solver。）

> 🎓 這裡的“方案”可以是 'ovr'（一對多）或 'multinomial'。由於邏輯回歸主要設計用於支持二類分類，這些方案使其能更好地處理多類分類任務。[來源](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 solver 定義為“用於優化問題的算法”。[來源](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)。

Scikit-learn 提供了這張表格來解釋 solver 如何處理不同數據結構帶來的挑戰：

![solver](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## 練習 - 分割數據

我們可以專注於邏輯回歸作為我們的第一次訓練嘗試，因為您在上一節課中剛剛學習了它。
通過調用 `train_test_split()` 將數據分為訓練和測試組：

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 練習 - 應用邏輯回歸

由於您正在使用多類情況，您需要選擇使用哪種_方案_以及設置哪種_solver_。使用 LogisticRegression 並設置 multi_class 為 `ovr` 和 solver 為 `liblinear` 進行訓練。

1. 創建一個邏輯回歸，將 multi_class 設置為 `ovr`，solver 設置為 `liblinear`：

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ 嘗試使用其他 solver，例如 `lbfgs`，它通常設置為默認值
> 注意，當需要將資料展平時，可以使用 Pandas 的 [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) 函數。
準確率超過 **80%**！

1. 您可以透過測試第 50 行數據來查看此模型的運作情況：

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    結果如下：

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ 嘗試不同的行數並檢查結果

1. 更深入地探討，您可以檢查此預測的準確性：

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    結果如下 - 印度料理是模型的最佳猜測，且概率相當高：

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ 您能解釋為什麼模型非常確定這是印度料理嗎？

1. 透過列印分類報告獲取更多細節，就像您在回歸課程中所做的一樣：

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀挑戰

在本課程中，您使用清理過的數據建立了一個機器學習模型，該模型可以根據一系列食材預測國家料理。花些時間閱讀 Scikit-learn 提供的多種分類數據選項。深入了解 "solver" 的概念，以理解其背後的運作原理。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

深入了解邏輯回歸背後的數學原理：[這篇課程](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## 作業 

[研究 solvers](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對於因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。