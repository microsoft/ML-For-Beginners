<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T09:17:57+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "hk"
}
-->
# 使用 Python 和 Scikit-learn 建立回歸模型

![回歸模型的手繪筆記摘要](../../../../sketchnotes/ml-regression.png)

> 手繪筆記由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 提供

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程亦提供 R 語言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 簡介

在這四節課中，你將學習如何建立回歸模型。我們稍後會討論這些模型的用途。但在開始之前，請確保你已經準備好合適的工具來進行這個過程！

在本課程中，你將學到如何：

- 配置你的電腦以進行本地機器學習任務。
- 使用 Jupyter notebooks。
- 安裝並使用 Scikit-learn。
- 通過實際操作探索線性回歸。

## 安裝與配置

[![機器學習初學者 - 配置工具以建立機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "機器學習初學者 - 配置工具以建立機器學習模型")

> 🎥 點擊上方圖片觀看一段短片，了解如何配置你的電腦以進行機器學習。

1. **安裝 Python**。確保你的電腦已安裝 [Python](https://www.python.org/downloads/)。Python 是進行數據科學和機器學習任務的重要工具。大多數電腦系統已經預裝了 Python。你也可以使用一些方便的 [Python 編碼包](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)，以簡化安裝過程。

   不過，某些 Python 的使用場景可能需要特定版本的軟件，因此建議使用 [虛擬環境](https://docs.python.org/3/library/venv.html) 來管理不同的版本。

2. **安裝 Visual Studio Code**。確保你的電腦已安裝 Visual Studio Code。按照這些指引完成 [Visual Studio Code 的基本安裝](https://code.visualstudio.com/)。在本課程中，你將在 Visual Studio Code 中使用 Python，因此建議熟悉如何為 Python 開發 [配置 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)。

   > 透過這個 [學習模組集合](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) 來熟悉 Python。
   >
   > [![使用 Visual Studio Code 配置 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "使用 Visual Studio Code 配置 Python")
   >
   > 🎥 點擊上方圖片觀看影片：在 VS Code 中使用 Python。

3. **安裝 Scikit-learn**，按照 [這些指引](https://scikit-learn.org/stable/install.html) 進行安裝。由於需要使用 Python 3，建議使用虛擬環境。如果你使用的是 M1 Mac，請參考上述頁面中的特別指引。

4. **安裝 Jupyter Notebook**。你需要 [安裝 Jupyter 套件](https://pypi.org/project/jupyter/)。

## 你的機器學習開發環境

你將使用 **notebooks** 來開發 Python 程式碼並建立機器學習模型。這種文件格式是數據科學家常用的工具，其文件副檔名為 `.ipynb`。

Notebooks 是一種互動式環境，允許開發者撰寫程式碼並添加註解或文件，這對於實驗性或研究導向的項目非常有幫助。

[![機器學習初學者 - 配置 Jupyter Notebooks 開始建立回歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "機器學習初學者 - 配置 Jupyter Notebooks 開始建立回歸模型")

> 🎥 點擊上方圖片觀看一段短片，了解如何進行此操作。

### 練習 - 使用 Notebook

在此資料夾中，你會找到檔案 _notebook.ipynb_。

1. 在 Visual Studio Code 中打開 _notebook.ipynb_。

   Jupyter 伺服器將啟動，並使用 Python 3+。你會看到 Notebook 中可以執行的程式碼區塊。點擊類似播放按鈕的圖標即可執行程式碼區塊。

2. 選擇 `md` 圖標，添加一些 Markdown，並輸入以下文字：**# Welcome to your notebook**。

   接著，添加一些 Python 程式碼。

3. 在程式碼區塊中輸入 **print('hello notebook')**。
4. 點擊箭頭執行程式碼。

   你應該會看到以下輸出：

    ```output
    hello notebook
    ```

![VS Code 中打開的 Notebook](../../../../2-Regression/1-Tools/images/notebook.jpg)

你可以在程式碼中穿插註解，為 Notebook 添加自我文件化的功能。

✅ 想一想，網頁開發者的工作環境與數據科學家的工作環境有何不同。

## 開始使用 Scikit-learn

現在你的本地環境已經配置好 Python，並且你對 Jupyter Notebooks 感到熟悉，接下來讓我們熟悉一下 Scikit-learn（發音為 `sci`，如 `science`）。Scikit-learn 提供了一個[廣泛的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)，幫助你執行機器學習任務。

根據其[官方網站](https://scikit-learn.org/stable/getting_started.html)的描述，"Scikit-learn 是一個開源的機器學習庫，支持監督式和非監督式學習。它還提供了多種工具，用於模型擬合、數據預處理、模型選擇與評估，以及其他多種實用功能。"

在本課程中，你將使用 Scikit-learn 和其他工具來建立機器學習模型，執行我們所稱的「傳統機器學習」任務。我們刻意避開了神經網絡和深度學習，因為這些內容將在我們即將推出的「AI 初學者」課程中詳細介紹。

Scikit-learn 使得建立模型並評估其使用變得簡單。它主要專注於處理數值數據，並包含多個現成的數據集作為學習工具。它還提供了預建模型供學生嘗試。接下來，我們將探索如何加載預打包數據並使用內建的估算器來建立第一個機器學習模型。

## 練習 - 你的第一個 Scikit-learn Notebook

> 本教程靈感來自 Scikit-learn 網站上的 [線性回歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。

[![機器學習初學者 - 在 Python 中的第一個線性回歸項目](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "機器學習初學者 - 在 Python 中的第一個線性回歸項目")

> 🎥 點擊上方圖片觀看一段短片，了解如何進行此練習。

在與本課程相關的 _notebook.ipynb_ 文件中，清空所有程式碼區塊，點擊「垃圾桶」圖標。

在本節中，你將使用 Scikit-learn 中內建的一個小型糖尿病數據集進行操作。假設你想測試一種針對糖尿病患者的治療方法。機器學習模型可能幫助你根據變量的組合，確定哪些患者對治療的反應更好。即使是一個非常基本的回歸模型，當可視化時，也可能顯示出有助於組織理論臨床試驗的變量信息。

✅ 回歸方法有很多種類型，選擇哪一種取決於你想要解答的問題。如果你想預測某個年齡段的人的可能身高，你會使用線性回歸，因為你尋求的是一個**數值結果**。如果你想判斷某種菜餚是否應被歸類為素食，你尋求的是一個**類別分配**，因此你會使用邏輯回歸。稍後你會學到更多關於邏輯回歸的內容。想一想，你可以向數據提出哪些問題，並考慮哪種方法更合適。

讓我們開始這個任務。

### 導入庫

在這個任務中，我們將導入一些庫：

- **matplotlib**。這是一個有用的[繪圖工具](https://matplotlib.org/)，我們將用它來創建折線圖。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是一個處理 Python 數值數據的有用庫。
- **sklearn**。這是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 庫。

導入一些幫助完成任務的庫。

1. 輸入以下程式碼以添加導入：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上述程式碼中，你導入了 `matplotlib` 和 `numpy`，並從 `sklearn` 中導入了 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用於將數據分割為訓練集和測試集。

### 糖尿病數據集

內建的 [糖尿病數據集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 包含 442 條糖尿病相關的數據樣本，具有 10 個特徵變量，其中一些包括：

- age：年齡（以年為單位）
- bmi：身體質量指數
- bp：平均血壓
- s1 tc：T 細胞（白血球的一種類型）

✅ 此數據集包含「性別」作為研究糖尿病的重要特徵變量。許多醫學數據集都包含這種類型的二元分類。想一想，這樣的分類可能如何將某些群體排除在治療之外。

現在，載入 X 和 y 數據。

> 🎓 記住，這是監督式學習，我們需要一個名為 'y' 的目標變量。

在新的程式碼區塊中，通過調用 `load_diabetes()` 載入糖尿病數據集。輸入參數 `return_X_y=True` 表示 `X` 將是數據矩陣，而 `y` 將是回歸目標。

1. 添加一些打印命令以顯示數據矩陣的形狀及其第一個元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你得到的回應是一個元組。你將元組的前兩個值分別賦值給 `X` 和 `y`。了解更多關於[元組](https://wikipedia.org/wiki/Tuple)。

    你可以看到這些數據包含 442 條記錄，每條記錄由 10 個元素組成的數組構成：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 想一想數據與回歸目標之間的關係。線性回歸預測特徵 X 和目標變量 y 之間的關係。你能在文檔中找到糖尿病數據集的[目標](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)嗎？這個數據集展示了什麼？

2. 接下來，選擇數據集的一部分進行繪圖，選擇數據集的第 3 列。你可以使用 `:` 運算符選擇所有行，然後使用索引（2）選擇第 3 列。你還可以使用 `reshape(n_rows, n_columns)` 將數據重塑為 2D 數組（繪圖所需）。如果其中一個參數為 -1，則對應的維度會自動計算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 隨時打印數據以檢查其形狀。

3. 現在你已準備好繪製數據，可以看看機器是否能幫助確定數據集中數字之間的邏輯分界。為此，你需要將數據（X）和目標（y）分割為測試集和訓練集。Scikit-learn 提供了一種簡單的方法來完成這個操作；你可以在給定點分割測試數據。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 現在你可以訓練模型了！加載線性回歸模型，並使用 `model.fit()` 訓練 X 和 y 訓練集：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是你在許多機器學習庫（如 TensorFlow）中會看到的函數。

5. 然後，使用測試數據創建預測，通過函數 `predict()` 完成。這將用於繪製數據組之間的分界線。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在是時候在圖中顯示數據了。Matplotlib 是完成此任務的非常有用的工具。創建所有 X 和 y 測試數據的散點圖，並使用預測結果繪製一條線，該線位於模型數據組之間的最合適位置。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![顯示糖尿病數據點的散點圖](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ 想一想這裡發生了什麼。一條直線穿過了許多小數據點，但它究竟在做什麼？你能否看出應該如何利用這條直線來預測一個新的、未見過的數據點應該在圖表的 y 軸上對應的位置？試著用語言描述這個模型的實際用途。

恭喜你！你已經建立了你的第一個線性回歸模型，使用它進行了預測，並將結果顯示在圖表中！

---
## 🚀挑戰

繪製這個數據集中的另一個變量。提示：編輯這一行：`X = X[:,2]`。根據這個數據集的目標，你能發現糖尿病作為一種疾病的進展有什麼特點？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

在本教程中，你使用了簡單線性回歸，而不是單變量或多變量線性回歸。閱讀一些關於這些方法差異的資料，或者觀看[這段影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)。

深入了解回歸的概念，並思考這種技術可以回答哪些類型的問題。參加這個[教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)來加深你的理解。

## 作業

[另一個數據集](assignment.md)

---

**免責聲明**：  
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或誤釋不承擔責任。