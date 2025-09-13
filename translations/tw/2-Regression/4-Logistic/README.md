<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-05T09:43:09+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "tw"
}
-->
# 使用邏輯回歸預測分類

![邏輯回歸與線性回歸資訊圖](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 語言版本！](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## 簡介

在這堂關於回歸的最後一課中，我們將探討邏輯回歸，這是機器學習中一種經典的基本技術。你可以使用這種技術來發現模式並預測二元分類。例如：這顆糖果是巧克力嗎？這種疾病是否具有傳染性？這位顧客會選擇這個產品嗎？

在本課中，你將學到：

- 一個新的數據視覺化庫
- 邏輯回歸的技術

✅ 在這個 [Learn 模組](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott) 中深入了解這種類型回歸的應用。

## 前置條件

在之前的課程中，我們已經熟悉了南瓜數據，並意識到其中有一個可以使用的二元分類：`Color`。

現在，我們將建立一個邏輯回歸模型，根據一些變量來預測南瓜的顏色是什麼（橙色 🎃 還是白色 👻）。

> 為什麼我們在回歸的課程中討論二元分類？這只是語言上的方便，因為邏輯回歸[實際上是一種分類方法](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)，儘管它是基於線性的。在下一組課程中，你將學到其他分類數據的方法。

## 定義問題

對於我們的目的，我們將問題表述為二元分類：「白色」或「非白色」。數據集中還有一個「條紋」分類，但由於樣本數量很少，我們將不使用它。無論如何，當我們從數據集中移除空值時，它也會消失。

> 🎃 有趣的是，我們有時會稱白色南瓜為「幽靈」南瓜。它們不太容易雕刻，因此不像橙色南瓜那麼受歡迎，但它們看起來很酷！所以我們也可以將問題重新表述為：「幽靈」或「非幽靈」。👻

## 關於邏輯回歸

邏輯回歸與之前學到的線性回歸在幾個重要方面有所不同。

[![機器學習初學者 - 理解邏輯回歸用於分類](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "機器學習初學者 - 理解邏輯回歸用於分類")

> 🎥 點擊上方圖片觀看邏輯回歸的簡短視頻概述。

### 二元分類

邏輯回歸無法提供與線性回歸相同的功能。前者提供的是對二元分類（例如「白色或非白色」）的預測，而後者則能預測連續值，例如根據南瓜的產地和收穫時間，_價格將上漲多少_。

![南瓜分類模型](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> 資訊圖由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供

### 其他分類

邏輯回歸還有其他類型，包括多項式和序數分類：

- **多項式分類**：涉及多個分類，例如「橙色、白色和條紋」。
- **序數分類**：涉及有序的分類，適用於需要邏輯排序的結果，例如按大小排序的南瓜（迷你、小、中、大、特大、超大）。

![多項式與序數回歸](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### 變量不需要相關性

還記得線性回歸在變量相關性更高時效果更好嗎？邏輯回歸則相反——變量不需要相關性。這對於我們的數據很有用，因為它的相關性較弱。

### 需要大量乾淨的數據

如果使用更多數據，邏輯回歸的結果會更準確；我們的小型數據集並不是這項任務的最佳選擇，因此請記住這一點。

[![機器學習初學者 - 邏輯回歸的數據分析與準備](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "機器學習初學者 - 邏輯回歸的數據分析與準備")

✅ 思考哪些類型的數據適合邏輯回歸。

## 練習 - 整理數據

首先，稍微清理一下數據，刪除空值並選擇一些列：

1. 添加以下代碼：

    ```python
  
    columns_to_select = ['City Name','Package','Variety', 'Origin','Item Size', 'Color']
    pumpkins = full_pumpkins.loc[:, columns_to_select]

    pumpkins.dropna(inplace=True)
    ```

    你可以隨時查看新的數據框：

    ```python
    pumpkins.info
    ```

### 視覺化 - 類別圖

到目前為止，你已經再次使用南瓜數據加載了[起始筆記本](../../../../2-Regression/4-Logistic/notebook.ipynb)，並對其進行了清理，以保留包含一些變量（包括 `Color`）的數據集。讓我們使用一個不同的庫 [Seaborn](https://seaborn.pydata.org/index.html) 在筆記本中視覺化數據框。Seaborn 是基於我們之前使用的 Matplotlib 構建的。

Seaborn 提供了一些很棒的方式來視覺化數據。例如，你可以在類別圖中比較每個 `Variety` 和 `Color` 的數據分佈。

1. 使用 `catplot` 函數創建這樣的圖，使用我們的南瓜數據 `pumpkins`，並為每個南瓜分類（橙色或白色）指定顏色映射：

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

    ![一個數據視覺化網格](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    通過觀察數據，你可以看到 `Color` 數據與 `Variety` 的關係。

    ✅ 根據這個類別圖，你能想到哪些有趣的探索？

### 數據預處理：特徵和標籤編碼

我們的南瓜數據集的所有列都包含字符串值。對於人類來說，處理類別數據是直觀的，但對於機器來說並非如此。機器學習算法更適合處理數字數據。因此，編碼是數據預處理階段中非常重要的一步，因為它使我們能夠將類別數據轉換為數字數據，而不丟失任何信息。良好的編碼有助於構建良好的模型。

對於特徵編碼，主要有兩種類型的編碼器：

1. 序數編碼器：適用於序數變量，即數據具有邏輯順序的類別變量，例如數據集中的 `Item Size` 列。它創建一個映射，使每個類別由一個數字表示，該數字是該列中類別的順序。

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. 類別編碼器：適用於名義變量，即數據沒有邏輯順序的類別變量，例如數據集中除 `Item Size` 以外的所有特徵。這是一種獨熱編碼，意味著每個類別由一個二進制列表示：如果南瓜屬於該 `Variety`，則編碼變量等於 1，否則為 0。

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

然後，使用 `ColumnTransformer` 將多個編碼器合併為一個步驟，並將它們應用於適當的列。

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

另一方面，為了編碼標籤，我們使用 scikit-learn 的 `LabelEncoder` 類，這是一個實用類，用於將標籤標準化，使其僅包含 0 到 n_classes-1（此處為 0 和 1）之間的值。

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

一旦我們編碼了特徵和標籤，就可以將它們合併到一個新的數據框 `encoded_pumpkins` 中。

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ 為什麼對於 `Item Size` 列使用序數編碼器有優勢？

### 分析變量之間的關係

現在我們已經對數據進行了預處理，可以分析特徵與標籤之間的關係，以了解模型在給定特徵的情況下預測標籤的能力。

分析這類關係的最佳方式是繪製數據。我們將再次使用 Seaborn 的 `catplot` 函數，來視覺化 `Item Size`、`Variety` 和 `Color` 之間的關係。為了更好地繪製數據，我們將使用編碼後的 `Item Size` 列和未編碼的 `Variety` 列。

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

![一個類別圖的數據視覺化](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### 使用 swarm plot

由於 `Color` 是一個二元分類（白色或非白色），它需要「[專門的方法](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)來進行視覺化」。還有其他方法可以視覺化此分類與其他變量的關係。

你可以使用 Seaborn 圖表並排視覺化變量。

1. 嘗試使用「swarm」圖來顯示值的分佈：

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![一個 swarm 圖的數據視覺化](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**注意**：上述代碼可能會產生警告，因為 Seaborn 無法在 swarm 圖中表示如此多的數據點。一種可能的解決方案是通過使用 `size` 參數減小標記的大小。然而，請注意這會影響圖表的可讀性。

> **🧮 數學解釋**
>
> 邏輯回歸依賴於「最大似然」的概念，使用[Sigmoid 函數](https://wikipedia.org/wiki/Sigmoid_function)。Sigmoid 函數在圖表上看起來像一個「S」形。它接受一個值並將其映射到 0 和 1 之間。其曲線也被稱為「邏輯曲線」。其公式如下：
>
> ![邏輯函數](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> 其中，Sigmoid 的中點位於 x 的 0 點，L 是曲線的最大值，k 是曲線的陡峭程度。如果函數的結果大於 0.5，則該標籤將被歸為二元選擇中的「1」類別。如果不是，則歸為「0」。

## 構建模型

在 Scikit-learn 中構建一個二元分類模型出奇地簡單。

[![機器學習初學者 - 使用邏輯回歸進行數據分類](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "機器學習初學者 - 使用邏輯回歸進行數據分類")

> 🎥 點擊上方圖片觀看構建邏輯回歸模型的簡短視頻概述。

1. 選擇你想在分類模型中使用的變量，並通過調用 `train_test_split()` 分割訓練集和測試集：

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. 現在你可以通過調用 `fit()` 使用訓練數據來訓練模型，並打印出結果：

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

    查看模型的評分報告。考慮到你只有大約 1000 行數據，這個結果還不錯：

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

## 通過混淆矩陣更好地理解模型

雖然你可以通過打印上述項目獲得評分報告[術語](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)，但使用[混淆矩陣](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)可能更容易理解模型的表現。

> 🎓 一個「[混淆矩陣](https://wikipedia.org/wiki/Confusion_matrix)」（或「錯誤矩陣」）是一個表格，用於表示模型的真陽性、假陽性、真陰性和假陰性，從而評估預測的準確性。

1. 要使用混淆矩陣，調用 `confusion_matrix()`：

    ```python
    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_test, predictions)
    ```

    查看模型的混淆矩陣：

    ```output
    array([[162,   4],
           [ 11,  22]])
    ```

在 Scikit-learn 中，混淆矩陣的行（軸 0）是實際標籤，列（軸 1）是預測標籤。

|       |   0   |   1   |
| :---: | :---: | :---: |
|   0   |  TN   |  FP   |
|   1   |  FN   |  TP   |

這裡發生了什麼？假設我們的模型被要求在兩個二元分類之間對南瓜進行分類：「白色」和「非白色」。

- 如果模型預測南瓜為非白色，且實際上屬於「非白色」分類，我們稱之為真陰性，顯示在左上角。
- 如果模型預測南瓜為白色，且實際上屬於「非白色」分類，我們稱之為假陰性，顯示在左下角。
- 如果模型預測南瓜為非白色，且實際上屬於「白色」分類，我們稱之為假陽性，顯示在右上角。
- 如果模型預測南瓜為白色，且實際上屬於「白色」分類，我們稱之為真陽性，顯示在右下角。

正如你可能猜到的，真陽性和真陰性的數量越多，假陽性和假陰性的數量越少，模型的表現就越好。
混淆矩陣如何與精確率（Precision）和召回率（Recall）相關？請記住，上述的分類報告顯示精確率為 0.85，召回率為 0.67。

精確率 = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

召回率 = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ 問：根據混淆矩陣，模型表現如何？  
答：還不錯，有相當多的真負例（True Negatives），但也有一些假負例（False Negatives）。

讓我們藉助混淆矩陣中 TP/TN 和 FP/FN 的映射，重新回顧之前提到的術語：

🎓 精確率（Precision）：TP/(TP + FP)  
檢索到的實例中，相關實例的比例（例如，哪些標籤被正確標記）。

🎓 召回率（Recall）：TP/(TP + FN)  
相關實例中被檢索到的比例，無論是否正確標記。

🎓 F1 分數（f1-score）：(2 * 精確率 * 召回率)/(精確率 + 召回率)  
精確率和召回率的加權平均值，最佳為 1，最差為 0。

🎓 支持度（Support）：檢索到的每個標籤的出現次數。

🎓 準確率（Accuracy）：(TP + TN)/(TP + TN + FP + FN)  
樣本中標籤被正確預測的百分比。

🎓 宏平均（Macro Avg）：  
對每個標籤的度量進行無權重平均計算，不考慮標籤的不平衡。

🎓 加權平均（Weighted Avg）：  
對每個標籤的度量進行平均計算，考慮標籤的不平衡，並根據支持度（每個標籤的真實實例數）進行加權。

✅ 你能想到如果想讓模型減少假負例（False Negatives）的數量，應該關注哪個指標嗎？

## 視覺化此模型的 ROC 曲線

[![機器學習初學者 - 使用 ROC 曲線分析邏輯回歸性能](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "機器學習初學者 - 使用 ROC 曲線分析邏輯回歸性能")

> 🎥 點擊上方圖片觀看關於 ROC 曲線的簡短影片

讓我們再做一次視覺化，看看所謂的 "ROC" 曲線：

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

使用 Matplotlib 繪製模型的 [接收者操作特徵曲線（Receiving Operating Characteristic, ROC）](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc)。ROC 曲線通常用於查看分類器在真陽性與假陽性方面的輸出表現。"ROC 曲線通常以真陽性率作為 Y 軸，假陽性率作為 X 軸。" 因此，曲線的陡峭程度以及曲線與中線之間的空間很重要：你希望曲線快速向上並超過中線。在我們的例子中，起初有一些假陽性，然後曲線正確地向上並超過中線：

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

最後，使用 Scikit-learn 的 [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) 計算實際的 "曲線下面積"（Area Under the Curve, AUC）：

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```  
結果是 `0.9749908725812341`。由於 AUC 的範圍是 0 到 1，你希望分數越高越好，因為一個 100% 正確預測的模型將有 AUC 為 1；在這個例子中，模型表現 _相當不錯_。

在未來的分類課程中，你將學習如何迭代以改進模型的分數。但目前為止，恭喜你！你已完成這些回歸課程！

---

## 🚀 挑戰

關於邏輯回歸還有很多值得探索的內容！但學習的最佳方式是實驗。找一個適合這類分析的數據集，並用它構建一個模型。你學到了什麼？提示：試試 [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) 上有趣的數據集。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

閱讀 [這篇來自 Stanford 的論文](https://web.stanford.edu/~jurafsky/slp3/5.pdf) 的前幾頁，了解邏輯回歸的一些實際應用。思考哪些任務更適合我們到目前為止學習的回歸類型。哪種方法效果最好？

## 作業

[重試這個回歸](assignment.md)

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。