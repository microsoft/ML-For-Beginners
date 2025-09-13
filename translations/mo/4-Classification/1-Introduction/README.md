<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-06T09:17:41+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "mo"
}
-->
# 分類簡介

在這四節課中，您將探索經典機器學習的一個核心主題——_分類_。我們將使用一個關於亞洲和印度美食的數據集，逐步了解各種分類算法的應用。希望您準備好大快朵頤了！

![只需一點點！](../../../../4-Classification/1-Introduction/images/pinch.png)

> 在這些課程中慶祝泛亞洲美食！圖片由 [Jen Looper](https://twitter.com/jenlooper) 提供

分類是一種[監督式學習](https://wikipedia.org/wiki/Supervised_learning)，與回歸技術有許多相似之處。如果機器學習的核心是通過使用數據集來預測事物的值或名稱，那麼分類通常分為兩類：_二元分類_和_多類分類_。

[![分類簡介](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "分類簡介")

> 🎥 點擊上方圖片觀看影片：麻省理工學院的 John Guttag 介紹分類

請記住：

- **線性回歸** 幫助您預測變量之間的關係，並準確預測新數據點在該線性關係中的位置。例如，您可以預測_南瓜在九月和十二月的價格_。
- **邏輯回歸** 幫助您發現「二元類別」：在這個價格範圍內，_這個南瓜是橙色還是非橙色_？

分類使用各種算法來確定數據點的標籤或類別。讓我們使用這些美食數據，看看是否可以通過觀察一組食材來確定其美食的來源。

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程提供 R 版本！](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### 簡介

分類是機器學習研究者和數據科學家的一項基本活動。從基本的二元值分類（「這封電子郵件是垃圾郵件還是非垃圾郵件？」），到使用計算機視覺進行複雜的圖像分類和分割，能夠將數據分成類別並提出問題始終是非常有用的。

用更科學的方式來描述這個過程，您的分類方法會創建一個預測模型，使您能夠將輸入變量與輸出變量之間的關係進行映射。

![二元分類與多類分類](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> 二元分類與多類分類問題，供分類算法處理。信息圖由 [Jen Looper](https://twitter.com/jenlooper) 提供

在開始清理數據、可視化數據並為機器學習任務做好準備之前，讓我們先了解一下機器學習分類數據的各種方式。

分類源於[統計學](https://wikipedia.org/wiki/Statistical_classification)，使用經典機器學習技術，通過特徵（例如 `smoker`、`weight` 和 `age`）來確定_患某種疾病的可能性_。作為一種類似於您之前進行的回歸練習的監督式學習技術，您的數據是有標籤的，機器學習算法使用這些標籤來分類和預測數據集的類別（或「特徵」），並將它們分配到某個組或結果中。

✅ 花點時間想像一個關於美食的數據集。一個多類模型能回答什麼問題？一個二元模型能回答什麼問題？如果您想確定某種美食是否可能使用葫蘆巴葉呢？如果您想知道，假如收到一袋包含八角、洋薊、花椰菜和辣根的雜貨，您是否能做出一道典型的印度菜？

[![瘋狂的神秘食材籃](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "瘋狂的神秘食材籃")

> 🎥 點擊上方圖片觀看影片。節目《Chopped》的整個主題是「神秘食材籃」，廚師必須用隨機選擇的食材做出一道菜。機器學習模型肯定能幫上忙！

## 你好，分類器

我們想要問這個美食數據集的問題實際上是一個**多類問題**，因為我們有多個潛在的國家美食可以選擇。給定一批食材，這些食材會屬於哪一類？

Scikit-learn 提供了多種不同的算法來分類數據，具體取決於您想要解決的問題類型。在接下來的兩節課中，您將學習其中幾種算法。

## 練習 - 清理並平衡數據

在開始這個項目之前，第一項任務是清理並**平衡**您的數據，以獲得更好的結果。從此文件夾根目錄中的空白 _notebook.ipynb_ 文件開始。

首先需要安裝 [imblearn](https://imbalanced-learn.org/stable/)。這是一個 Scikit-learn 套件，可以幫助您更好地平衡數據（您將在稍後了解更多相關任務）。

1. 安裝 `imblearn`，運行 `pip install`，如下所示：

    ```python
    pip install imblearn
    ```

1. 導入所需的包以導入數據並進行可視化，還需從 `imblearn` 中導入 `SMOTE`。

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    現在您已準備好接下來導入數據。

1. 下一步是導入數據：

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   使用 `read_csv()` 會讀取 csv 文件 _cusines.csv_ 的內容並將其放入變量 `df` 中。

1. 檢查數據的形狀：

    ```python
    df.head()
    ```

   前五行看起來像這樣：

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. 通過調用 `info()` 獲取有關此數據的信息：

    ```python
    df.info()
    ```

    您的輸出類似於：

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## 練習 - 了解美食

現在工作開始變得更有趣了。讓我們探索每種美食的數據分佈。

1. 通過調用 `barh()` 將數據繪製為條形圖：

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![美食數據分佈](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    美食的數量是有限的，但數據分佈是不均勻的。您可以修正這一點！在修正之前，先多探索一下。

1. 找出每種美食的數據量並打印出來：

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

    輸出如下所示：

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## 探索食材

現在您可以更深入地挖掘數據，了解每種美食的典型食材。您應該清理掉那些在美食之間造成混淆的重複數據，因此讓我們了解這個問題。

1. 在 Python 中創建一個名為 `create_ingredient()` 的函數，用於創建食材數據框。此函數將首先刪除一個無用的列，並根據食材的數量進行排序：

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   現在您可以使用該函數了解每種美食中最受歡迎的前十種食材。

1. 調用 `create_ingredient()` 並通過調用 `barh()` 繪製圖表：

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![泰國](../../../../4-Classification/1-Introduction/images/thai.png)

1. 對日本數據執行相同操作：

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![日本](../../../../4-Classification/1-Introduction/images/japanese.png)

1. 現在是中國食材：

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![中國](../../../../4-Classification/1-Introduction/images/chinese.png)

1. 繪製印度食材：

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![印度](../../../../4-Classification/1-Introduction/images/indian.png)

1. 最後，繪製韓國食材：

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![韓國](../../../../4-Classification/1-Introduction/images/korean.png)

1. 現在，通過調用 `drop()` 刪除那些在不同美食之間造成混淆的最常見食材：

   每個人都喜歡米飯、大蒜和薑！

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## 平衡數據集

現在您已清理數據，使用 [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html)——「合成少數類別過採樣技術」——來平衡數據。

1. 調用 `fit_resample()`，此策略通過插值生成新樣本。

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    通過平衡數據，您在分類時會獲得更好的結果。想想二元分類。如果您的大部分數據屬於一個類別，機器學習模型會更頻繁地預測該類別，僅僅因為該類別的數據更多。平衡數據可以消除這種不平衡。

1. 現在您可以檢查每種食材的標籤數量：

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    您的輸出如下所示：

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

    數據現在乾淨、平衡，而且非常美味！

1. 最後一步是將平衡後的數據，包括標籤和特徵，保存到一個新的數據框中，並導出到文件中：

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. 您可以使用 `transformed_df.head()` 和 `transformed_df.info()` 再次查看數據。保存一份此數據以供未來課程使用：

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    此新鮮的 CSV 現在可以在根目錄的數據文件夾中找到。

---

## 🚀挑戰

此課程包含多個有趣的數據集。翻閱 `data` 文件夾，看看是否有適合二元或多類分類的數據集？您會向這些數據集提出什麼問題？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

探索 SMOTE 的 API。它最適合用於哪些用例？它解決了哪些問題？

## 作業 

[探索分類方法](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵信息，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋不承擔責任。