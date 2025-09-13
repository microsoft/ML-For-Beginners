<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "abf86d845c84330bce205a46b382ec88",
  "translation_date": "2025-09-06T09:07:21+00:00",
  "source_file": "2-Regression/4-Logistic/README.md",
  "language_code": "mo"
}
-->
# 使用邏輯迴歸預測分類

![邏輯迴歸與線性迴歸資訊圖表](../../../../2-Regression/4-Logistic/images/linear-vs-logistic.png)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 版本！](../../../../2-Regression/4-Logistic/solution/R/lesson_4.html)

## 簡介

在這堂關於迴歸的最後一課中，我們將探討邏輯迴歸，這是基本的 _經典_ 機器學習技術之一。你可以使用這種技術來發現模式並預測二元分類。例如：這顆糖果是巧克力嗎？這種疾病是否具有傳染性？這位顧客是否會選擇這個產品？

在本課程中，你將學到：

- 一個新的資料視覺化庫
- 邏輯迴歸的技術

✅ 在這個 [學習模組](https://docs.microsoft.com/learn/modules/train-evaluate-classification-models?WT.mc_id=academic-77952-leestott) 中深入了解如何使用這種類型的迴歸。

## 前置條件

在使用南瓜數據後，我們已經足夠熟悉它，並意識到有一個二元分類可以使用：`Color`。

讓我們建立一個邏輯迴歸模型，根據一些變數來預測 _某個南瓜可能的顏色_（橙色 🎃 或白色 👻）。

> 為什麼我們在迴歸課程中討論二元分類？僅僅是語言上的方便，因為邏輯迴歸實際上是 [一種分類方法](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)，儘管它是基於線性的。在下一組課程中了解其他分類數據的方法。

## 定義問題

對於我們的目的，我們將其表述為二元分類：'白色' 或 '非白色'。我們的數據集中還有一個 '條紋' 類別，但它的樣本數量很少，因此我們不會使用它。事實上，當我們從數據集中移除空值時，它就消失了。

> 🎃 有趣的事實：我們有時會稱白色南瓜為 '幽靈南瓜'。它們不容易雕刻，因此不像橙色南瓜那麼受歡迎，但它們看起來很酷！所以我們也可以將問題重新表述為：'幽靈' 或 '非幽靈'。👻

## 關於邏輯迴歸

邏輯迴歸與之前學到的線性迴歸在幾個重要方面有所不同。

[![初學者的機器學習 - 理解邏輯迴歸](https://img.youtube.com/vi/KpeCT6nEpBY/0.jpg)](https://youtu.be/KpeCT6nEpBY "初學者的機器學習 - 理解邏輯迴歸")

> 🎥 點擊上方圖片觀看邏輯迴歸的簡短視頻概述。

### 二元分類

邏輯迴歸不提供與線性迴歸相同的功能。前者提供關於二元分類（"白色或非白色"）的預測，而後者能夠預測連續值，例如根據南瓜的產地和收穫時間，_價格將上漲多少_。

![南瓜分類模型](../../../../2-Regression/4-Logistic/images/pumpkin-classifier.png)
> 資訊圖表由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供

### 其他分類

邏輯迴歸還有其他類型，包括多項式和序列：

- **多項式**：涉及多個類別，例如 "橙色、白色和條紋"。
- **序列**：涉及有序的類別，適合我們希望按邏輯順序排列結果，例如南瓜按有限數量的大小排序（迷你、小、中、大、特大、超大）。

![多項式與序列迴歸](../../../../2-Regression/4-Logistic/images/multinomial-vs-ordinal.png)

### 變數不需要相關

還記得線性迴歸在變數相關性更高時效果更好嗎？邏輯迴歸則相反——變數不需要相關性。這適用於我們的數據，因為它的相關性較弱。

### 需要大量乾淨的數據

邏輯迴歸在使用更多數據時會提供更準確的結果；我們的小型數據集並不適合這項任務，因此請記住這一點。

[![初學者的機器學習 - 邏輯迴歸的數據分析與準備](https://img.youtube.com/vi/B2X4H9vcXTs/0.jpg)](https://youtu.be/B2X4H9vcXTs "初學者的機器學習 - 邏輯迴歸的數據分析與準備")

✅ 思考哪些類型的數據適合邏輯迴歸。

## 練習 - 整理數據

首先，稍微清理一下數據，刪除空值並選擇部分列：

1. 添加以下程式碼：

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

到目前為止，你已經在 [起始筆記本](../../../../2-Regression/4-Logistic/notebook.ipynb) 中載入了南瓜數據，並清理了它以保留包含一些變數（包括 `Color`）的數據集。讓我們使用不同的庫 [Seaborn](https://seaborn.pydata.org/index.html) 在筆記本中視覺化數據框。Seaborn 是基於我們之前使用的 Matplotlib。

Seaborn 提供了一些很棒的方式來視覺化數據。例如，你可以在類別圖中比較每個 `Variety` 和 `Color` 的數據分佈。

1. 使用 `catplot` 函數創建這樣的圖，使用南瓜數據 `pumpkins`，並為每個南瓜類別（橙色或白色）指定顏色映射：

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

    ![一組視覺化數據的網格](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_1.png)

    通過觀察數據，你可以看到 `Color` 數據與 `Variety` 的關係。

    ✅ 根據這個類別圖，你能想到哪些有趣的探索？

### 數據預處理：特徵和標籤編碼

我們的南瓜數據集包含所有列的字串值。對人類來說，處理類別數據是直觀的，但對機器來說並非如此。機器學習算法更適合處理數字。因此，編碼是數據預處理階段非常重要的一步，因為它使我們能夠將類別數據轉換為數字數據，而不丟失任何信息。良好的編碼有助於建立良好的模型。

對於特徵編碼，有兩種主要的編碼器：

1. 序列編碼器：適合序列變數，即類別變數，其數據遵循邏輯順序，例如數據集中的 `Item Size` 列。它創建一個映射，使每個類別由一個數字表示，該數字是列中類別的順序。

    ```python
    from sklearn.preprocessing import OrdinalEncoder

    item_size_categories = [['sml', 'med', 'med-lge', 'lge', 'xlge', 'jbo', 'exjbo']]
    ordinal_features = ['Item Size']
    ordinal_encoder = OrdinalEncoder(categories=item_size_categories)
    ```

2. 類別編碼器：適合名義變數，即類別變數，其數據不遵循邏輯順序，例如數據集中除 `Item Size` 以外的所有特徵。它是一種獨熱編碼，意味著每個類別由一個二元列表示：如果南瓜屬於該 `Variety`，則編碼變數等於 1，否則為 0。

    ```python
    from sklearn.preprocessing import OneHotEncoder

    categorical_features = ['City Name', 'Package', 'Variety', 'Origin']
    categorical_encoder = OneHotEncoder(sparse_output=False)
    ```

接著，使用 `ColumnTransformer` 將多個編碼器合併為一個步驟，並應用於適當的列。

```python
    from sklearn.compose import ColumnTransformer
    
    ct = ColumnTransformer(transformers=[
        ('ord', ordinal_encoder, ordinal_features),
        ('cat', categorical_encoder, categorical_features)
        ])
    
    ct.set_output(transform='pandas')
    encoded_features = ct.fit_transform(pumpkins)
```

另一方面，對於標籤編碼，我們使用 scikit-learn 的 `LabelEncoder` 類，這是一個工具類，用於將標籤標準化，使其僅包含 0 到 n_classes-1（此處為 0 和 1）之間的值。

```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    encoded_label = label_encoder.fit_transform(pumpkins['Color'])
```

一旦我們編碼了特徵和標籤，就可以將它們合併到新的數據框 `encoded_pumpkins` 中。

```python
    encoded_pumpkins = encoded_features.assign(Color=encoded_label)
```

✅ 使用序列編碼器處理 `Item Size` 列有哪些優勢？

### 分析變數之間的關係

現在我們已經對數據進行了預處理，可以分析特徵和標籤之間的關係，以了解模型在給定特徵的情況下預測標籤的能力。

進行這類分析的最佳方式是繪製數據。我們將再次使用 Seaborn 的 `catplot` 函數，視覺化 `Item Size`、`Variety` 和 `Color` 之間的關係。為了更好地繪製數據，我們將使用編碼後的 `Item Size` 列和未編碼的 `Variety` 列。

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

![視覺化數據的類別圖](../../../../2-Regression/4-Logistic/images/pumpkins_catplot_2.png)

### 使用 swarm plot

由於 `Color` 是二元分類（白色或非白色），它需要一種 [專門的視覺化方法](https://seaborn.pydata.org/tutorial/categorical.html?highlight=bar)。還有其他方法可以視覺化此分類與其他變數的關係。

你可以使用 Seaborn 圖表並排視覺化變數。

1. 嘗試使用 'swarm' 圖來顯示值的分佈：

    ```python
    palette = {
    0: 'orange',
    1: 'wheat'
    }
    sns.swarmplot(x="Color", y="ord__Item Size", data=encoded_pumpkins, palette=palette)
    ```

    ![視覺化數據的 swarm 圖](../../../../2-Regression/4-Logistic/images/swarm_2.png)

**注意**：上述程式碼可能會產生警告，因為 Seaborn 無法在 swarm 圖中表示如此多的數據點。一種可能的解決方案是使用 'size' 參數減小標記的大小，但請注意這會影響圖表的可讀性。

> **🧮 數學解析**
>
> 邏輯迴歸依賴於 '最大似然' 的概念，使用 [S 型函數](https://wikipedia.org/wiki/Sigmoid_function)。在圖表上，S 型函數看起來像一個 'S' 形。它將一個值映射到 0 和 1 之間的某個位置。其曲線也被稱為 '邏輯曲線'。公式如下：
>
> ![邏輯函數](../../../../2-Regression/4-Logistic/images/sigmoid.png)
>
> 其中 S 型函數的中點位於 x 的 0 點，L 是曲線的最大值，k 是曲線的陡度。如果函數的結果大於 0.5，該標籤將被分配為二元選擇中的 '1' 類別；否則，將被分配為 '0' 類別。

## 建立模型

在 Scikit-learn 中建立二元分類模型的過程相當簡單。

[![初學者的機器學習 - 使用邏輯迴歸進行數據分類](https://img.youtube.com/vi/MmZS2otPrQ8/0.jpg)](https://youtu.be/MmZS2otPrQ8 "初學者的機器學習 - 使用邏輯迴歸進行數據分類")

> 🎥 點擊上方圖片觀看建立線性迴歸模型的簡短視頻概述。

1. 選擇你想在分類模型中使用的變數，並使用 `train_test_split()` 分割訓練集和測試集：

    ```python
    from sklearn.model_selection import train_test_split
    
    X = encoded_pumpkins[encoded_pumpkins.columns.difference(['Color'])]
    y = encoded_pumpkins['Color']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    ```

2. 現在你可以通過使用訓練數據調用 `fit()` 來訓練模型，並打印出結果：

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

    查看模型的得分板。考慮到你只有大約 1000 行數據，結果還不錯：

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

## 使用混淆矩陣更好地理解模型

雖然你可以通過打印上述項目獲得得分板報告 [術語](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=classification_report#sklearn.metrics.classification_report)，但使用 [混淆矩陣](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) 可能更容易理解模型的表現。

> 🎓 '[混淆矩陣](https://wikipedia.org/wiki/Confusion_matrix)'（或 '錯誤矩陣'）是一個表格，用於表達模型的真實與錯誤的正例和負例，從而評估預測的準確性。

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

這裡發生了什麼？假設模型被要求在兩個二元分類中對南瓜進行分類，分類為 '白色' 和 '非白色'。

- 如果模型預測南瓜為非白色，且實際屬於 '非白色' 類別，我們稱之為真負例，顯示在左上角。
- 如果模型預測南瓜為白色，且實際屬於 '非白色' 類別，我們稱之為假負例，顯示在左下角。
- 如果模型預測南瓜為非白色，且實際屬於 '白色' 類別，我們稱之為假正例，顯示在右上角。
- 如果模型預測南瓜為白色，且實際屬於 '白色' 類別，我們稱之為真正例，顯示在右下角。

如你所料，真正例和真負例的數量越多，假正例和假負例的數量越少，模型的表現就越好。
混淆矩陣如何與精確度和召回率相關？記住，上面列出的分類報告顯示精確度 (0.85) 和召回率 (0.67)。

精確度 = tp / (tp + fp) = 22 / (22 + 4) = 0.8461538461538461

召回率 = tp / (tp + fn) = 22 / (22 + 11) = 0.6666666666666666

✅ 問：根據混淆矩陣，模型表現如何？  
答：還不錯；有相當多的真負例，但也有一些假負例。

讓我們再次回顧之前提到的術語，並結合混淆矩陣中 TP/TN 和 FP/FN 的映射來理解：

🎓 精確度：TP/(TP + FP)  
檢索到的實例中，相關實例的比例（例如，哪些標籤被正確標記）。

🎓 召回率：TP/(TP + FN)  
被檢索到的相關實例比例，無論是否被正確標記。

🎓 f1-score：(2 * 精確度 * 召回率)/(精確度 + 召回率)  
精確度和召回率的加權平均值，最佳值為 1，最差值為 0。

🎓 支援度：檢索到的每個標籤的出現次數。

🎓 準確率：(TP + TN)/(TP + TN + FP + FN)  
樣本中標籤被正確預測的百分比。

🎓 宏平均：每個標籤的未加權平均值計算，不考慮標籤的不平衡。

🎓 加權平均：每個標籤的平均值計算，根據支援度（每個標籤的真實實例數量）進行加權。

✅ 你能想到如果想減少假負例，應該關注哪個指標嗎？

## 可視化此模型的 ROC 曲線

[![初學者機器學習 - 使用 ROC 曲線分析邏輯回歸性能](https://img.youtube.com/vi/GApO575jTA0/0.jpg)](https://youtu.be/GApO575jTA0 "初學者機器學習 - 使用 ROC 曲線分析邏輯回歸性能")

> 🎥 點擊上方圖片觀看 ROC 曲線的簡短視頻概述

讓我們進行另一個可視化，來查看所謂的 "ROC" 曲線：

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

使用 Matplotlib 繪製模型的 [接收者操作特徵曲線](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html?highlight=roc) 或 ROC。ROC 曲線通常用於查看分類器的輸出，從真陽性和假陽性的角度分析。"ROC 曲線通常在 Y 軸上顯示真陽性率，在 X 軸上顯示假陽性率。" 因此，曲線的陡峭程度以及曲線與中線之間的空間很重要：你希望曲線快速向上並超過中線。在我們的例子中，起初有一些假陽性，然後曲線正確地向上並超過中線：

![ROC](../../../../2-Regression/4-Logistic/images/ROC_2.png)

最後，使用 Scikit-learn 的 [`roc_auc_score` API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html?highlight=roc_auc#sklearn.metrics.roc_auc_score) 計算實際的 "曲線下面積" (AUC)：

```python
auc = roc_auc_score(y_test,y_scores[:,1])
print(auc)
```  
結果是 `0.9749908725812341`。由於 AUC 的範圍是 0 到 1，你希望分數越高越好，因為一個完全正確預測的模型的 AUC 為 1；在這個例子中，模型表現 _相當不錯_。

在未來的分類課程中，你將學習如何迭代以改進模型的分數。但目前，恭喜你！你已完成這些回歸課程！

---

## 🚀挑戰

關於邏輯回歸還有很多值得探索的內容！但最好的學習方式是進行實驗。找到一個適合此類分析的數據集並用它建立模型。你學到了什麼？提示：試試 [Kaggle](https://www.kaggle.com/search?q=logistic+regression+datasets) 上有趣的數據集。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

閱讀 [這篇來自斯坦福的論文](https://web.stanford.edu/~jurafsky/slp3/5.pdf) 的前幾頁，了解邏輯回歸的一些實際用途。思考哪些任務更適合我們到目前為止所學的不同回歸類型。哪種方法效果最好？

## 作業

[重試此回歸](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵信息，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或誤釋不承擔責任。