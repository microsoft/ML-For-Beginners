<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T09:45:20+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "tw"
}
-->
# 使用 Scikit-learn 建立回歸模型：準備與視覺化數據

![數據視覺化資訊圖表](../../../../2-Regression/2-Data/images/data-visualization.png)

資訊圖表由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 語言版本！](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## 簡介

現在您已經準備好使用 Scikit-learn 開始建立機器學習模型，接下來可以開始對數據提出問題。在處理數據並應用機器學習解決方案時，了解如何提出正確的問題以充分發揮數據集的潛力是非常重要的。

在本課程中，您將學到：

- 如何為模型建立準備數據。
- 如何使用 Matplotlib 進行數據視覺化。

## 對數據提出正確的問題

您需要回答的問題將決定您使用哪種類型的機器學習算法。而您獲得答案的質量將在很大程度上取決於數據的性質。

看看為本課程提供的[數據](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv)。您可以在 VS Code 中打開這個 .csv 文件。快速瀏覽會發現其中有空白值，還有字符串和數字數據的混合。此外，還有一個奇怪的名為 "Package" 的列，其中的數據混合了 "sacks"、"bins" 和其他值。事實上，這些數據有點混亂。

[![機器學習初學者 - 如何分析和清理數據集](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "機器學習初學者 - 如何分析和清理數據集")

> 🎥 點擊上方圖片觀看一段短視頻，了解如何準備本課程的數據。

事實上，很少能直接獲得一個完全準備好用於建立機器學習模型的數據集。在本課程中，您將學習如何使用標準 Python 庫準備原始數據集，並學習各種數據視覺化技術。

## 案例研究：'南瓜市場'

在此文件夾中，您會在根目錄的 `data` 文件夾中找到一個名為 [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) 的 .csv 文件，其中包含 1757 行有關南瓜市場的數據，按城市分組。這是從美國農業部發布的[特殊作物終端市場標準報告](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice)中提取的原始數據。

### 準備數據

這些數據屬於公共領域。它可以從 USDA 網站下載，並以每個城市為單位分成多個文件。為了避免過多的文件，我們已將所有城市數據合併到一個電子表格中，因此我們已經對數據進行了一些_準備_。接下來，讓我們仔細看看這些數據。

### 南瓜數據 - 初步結論

您對這些數據有什麼觀察？您可能已經注意到其中混合了字符串、數字、空白值和一些需要理解的奇怪值。

使用回歸技術，您可以對這些數據提出什麼問題？例如，"預測某個月份出售的南瓜價格"。再次查看數據，您會發現需要進行一些更改，以創建適合此任務的數據結構。

## 練習 - 分析南瓜數據

讓我們使用 [Pandas](https://pandas.pydata.org/)（名稱代表 `Python Data Analysis`），這是一個非常有用的數據處理工具，來分析和準備這些南瓜數據。

### 首先，檢查是否有缺失日期

您需要首先檢查是否有缺失的日期：

1. 將日期轉換為月份格式（這些是美國日期格式，為 `MM/DD/YYYY`）。
2. 提取月份到一個新列中。

在 Visual Studio Code 中打開 _notebook.ipynb_ 文件，並將電子表格導入到一個新的 Pandas dataframe 中。

1. 使用 `head()` 函數查看前五行。

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ 您會使用什麼函數來查看最後五行？

1. 檢查當前 dataframe 中是否有缺失數據：

    ```python
    pumpkins.isnull().sum()
    ```

    有缺失數據，但可能對當前任務無關緊要。

1. 為了使 dataframe 更易於處理，使用 `loc` 函數選擇您需要的列。該函數從原始 dataframe 中提取一組行（作為第一個參數傳遞）和列（作為第二個參數傳遞）。以下示例中的表達式 `:` 表示 "所有行"。

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### 其次，確定南瓜的平均價格

思考如何確定某個月份南瓜的平均價格。您會選擇哪些列來完成此任務？提示：您需要 3 列。

解決方案：取 `Low Price` 和 `High Price` 列的平均值來填充新的 Price 列，並將 Date 列轉換為僅顯示月份。幸運的是，根據上面的檢查，日期和價格的數據沒有缺失。

1. 要計算平均值，添加以下代碼：

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ 您可以使用 `print(month)` 打印任何您想檢查的數據。

2. 現在，將轉換後的數據複製到一個新的 Pandas dataframe 中：

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    打印出 dataframe，您會看到一個乾淨整潔的數據集，您可以基於此建立新的回歸模型。

### 但是等等！這裡有些奇怪的地方

如果您查看 `Package` 列，會發現南瓜以多種不同的方式出售。有些以 "1 1/9 bushel" 為單位，有些以 "1/2 bushel" 為單位，有些按個數出售，有些按重量出售，還有一些以不同寬度的大箱出售。

> 南瓜似乎很難以一致的方式稱重

深入研究原始數據，您會發現任何 `Unit of Sale` 等於 "EACH" 或 "PER BIN" 的項目，其 `Package` 類型也包括每英寸、每箱或 "each"。南瓜似乎很難以一致的方式稱重，因此我們只選擇 `Package` 列中包含 "bushel" 字符串的南瓜。

1. 在文件頂部的 .csv 導入下方添加一個篩選器：

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    如果您現在打印數據，您會看到只剩下大約 415 行包含以 bushel 為單位的南瓜數據。

### 但是等等！還有一件事要做

您是否注意到每行的 bushel 數量不同？您需要對價格進行標準化，以顯示每 bushel 的價格，因此需要進行一些數學運算來標準化它。

1. 在創建 new_pumpkins dataframe 的代碼塊後添加以下行：

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ 根據 [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308)，bushel 的重量取決於農產品的類型，因為它是一種體積測量單位。例如，一 bushel 的番茄應該重 56 磅……葉類和綠色蔬菜佔用更多空間但重量較輕，因此一 bushel 的菠菜只有 20 磅。這一切都相當複雜！我們不需要進行 bushel 到磅的轉換，而是按 bushel 定價。不過，這些關於南瓜 bushel 的研究表明，了解數據的性質是多麼重要！

現在，您可以根據 bushel 測量單位分析每單位的定價。如果您再打印一次數據，您會看到它已經標準化。

✅ 您是否注意到按半 bushel 出售的南瓜非常昂貴？您能找出原因嗎？提示：小南瓜比大南瓜貴得多，可能是因為每 bushel 中小南瓜的數量更多，而大南瓜的空心部分佔用了更多空間。

## 視覺化策略

數據科學家的部分工作是展示他們正在處理的數據的質量和性質。為此，他們通常會創建有趣的視覺化圖表，例如散點圖、柱狀圖和折線圖，展示數據的不同方面。通過這種方式，他們能夠直觀地顯示數據中難以發現的關係和差距。

[![機器學習初學者 - 如何使用 Matplotlib 視覺化數據](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "機器學習初學者 - 如何使用 Matplotlib 視覺化數據")

> 🎥 點擊上方圖片觀看一段短視頻，了解如何視覺化本課程的數據。

視覺化還可以幫助確定最適合數據的機器學習技術。例如，看起來像一條線的散點圖表明該數據非常適合線性回歸練習。

在 Jupyter notebook 中，一個非常好用的數據視覺化庫是 [Matplotlib](https://matplotlib.org/)（您在上一課中也見過它）。

> 在[這些教程](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott)中獲得更多數據視覺化的經驗。

## 練習 - 嘗試使用 Matplotlib

嘗試創建一些基本圖表來顯示您剛剛創建的新 dataframe。基本的折線圖會顯示什麼？

1. 在文件頂部的 Pandas 導入下方導入 Matplotlib：

    ```python
    import matplotlib.pyplot as plt
    ```

1. 重新運行整個 notebook 以刷新。
1. 在 notebook 底部添加一個單元格，將數據繪製為箱形圖：

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![顯示價格與月份關係的散點圖](../../../../2-Regression/2-Data/images/scatterplot.png)

    這是一個有用的圖表嗎？它有什麼讓您感到驚訝的地方嗎？

    它並不是特別有用，因為它只是在給定月份中顯示數據的分佈。

### 讓它更有用

為了讓圖表顯示有用的數據，通常需要以某種方式對數據進行分組。我們來嘗試創建一個圖表，其中 y 軸顯示月份，數據顯示數據的分佈。

1. 添加一個單元格來創建分組柱狀圖：

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![顯示價格與月份關係的柱狀圖](../../../../2-Regression/2-Data/images/barchart.png)

    這是一個更有用的數據視覺化！它似乎表明南瓜的最高價格出現在九月和十月。這符合您的預期嗎？為什麼或為什麼不符合？

---

## 🚀挑戰

探索 Matplotlib 提供的不同類型的視覺化。哪些類型最適合回歸問題？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

了解數據視覺化的多種方式。列出可用的各種庫，並記下哪些庫最適合特定類型的任務，例如 2D 視覺化與 3D 視覺化。您發現了什麼？

## 作業

[探索視覺化](assignment.md)

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或誤釋不承擔責任。