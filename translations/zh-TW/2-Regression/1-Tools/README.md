# 使用 Python 和 Scikit-learn 開始建立回歸模型

![回歸模型的筆記總覽](../../../../translated_images/zh-TW/ml-regression.4e4f70e3b3ed446e.webp)

> 筆記作者 [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程亦提供 R 語言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介紹

在這四課中，你將學會如何建立回歸模型。我們稍後會談談它們的用途。但在操作之前，請確保你已準備好所有正確的工具來開始此過程！

在本課程中，你將學會如何：

- 為本機的機器學習任務設定電腦環境。
- 使用 Jupyter 筆記本。
- 使用 Scikit-learn，包含安裝步驟。
- 透過實作練習了解線性回歸。

## 安裝與設定

[![初學者機器學習 - 設定工具準備建構機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "初學者機器學習 - 設定工具準備建構機器學習模型")

> 🎥 點擊上方圖片觀看快速教學影片，教你如何設定電腦環境進行機器學習。

1. **安裝 Python**。請確保你的電腦已有安裝[Python](https://www.python.org/downloads/)。你將使用 Python 來進行許多資料科學和機器學習任務。大多數電腦系統已內建 Python。也有實用的[Python 程式碼套件](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)可用於簡化設定步驟。

   不過有些 Python 程式碼需要某一版本，有些則需要其他版本，因此使用[虛擬環境](https://docs.python.org/3/library/venv.html)會很有幫助。

2. **安裝 Visual Studio Code**。確保你已在電腦上安裝了 Visual Studio Code。請依據下列指引[安裝 Visual Studio Code](https://code.visualstudio.com/)。在本課程中你將使用 Visual Studio Code 來開發 Python 程式，因此建議你熟悉如何為 Python 開發[設定 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)。

   > 透過一系列[學習模組](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)熟悉 Python。
   >
   > [![在 Visual Studio Code 中設定 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "在 Visual Studio Code 中設定 Python")
   >
   > 🎥 點擊上方圖片觀看如何在 VS Code 中使用 Python 的教學影片。

3. **安裝 Scikit-learn**，請依據[此處說明](https://scikit-learn.org/stable/install.html)操作。由於需要確保使用 Python 3，建議使用虛擬環境。如你是在 M1 Mac 上安裝該庫，請留意該頁面上的特別指示。

4. **安裝 Jupyter Notebook**。你需要[安裝 Jupyter 套件](https://pypi.org/project/jupyter/)。

## 你的機器學習撰寫環境

你將使用 **筆記本（notebook）** 來撰寫 Python 程式碼並建立機器學習模型。這類檔案是資料科學家常用的工具，檔案副檔名為 `.ipynb`。

筆記本是一個互動式環境，允許開發者同時撰寫程式碼、加入註記與文件，對於實驗性或研究導向的專案非常有幫助。

[![初學者機器學習 - 設定 Jupyter 筆記本開始建構回歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "初學者機器學習 - 設定 Jupyter 筆記本開始建構回歸模型")

> 🎥 點擊上方圖片觀看本練習快速教學影片。

### 練習 - 使用筆記本

在此資料夾中，你會找到 _notebook.ipynb_ 檔案。

1. 在 Visual Studio Code 中打開 _notebook.ipynb_。

   Jupyter 伺服器會啟動，使用 Python 3+。你會看到筆記本中有可執行的程式碼區塊。點擊類似播放按鈕的圖示，即可執行該區塊。

1. 選取 `md` 圖示並加入 markdown，並輸入以下文字 **# Welcome to your notebook**。

   接著，加入一些 Python 程式碼。

1. 在程式碼區塊中輸入 **print('hello notebook')**。
1. 點擊箭頭執行程式碼。

   你應該會看到輸出：

    ```output
    hello notebook
    ```

![VS Code 開啟筆記本](../../../../translated_images/zh-TW/notebook.4a3ee31f396b8832.webp)

你可以在程式碼中加入註解，以自我紀錄筆記本內容。

✅ 思考一下網頁開發者的工作環境和資料科學家的工作環境有何不同。

## 啟用 Scikit-learn

既然已在本機設置好 Python，且熟悉 Jupyter 筆記本，現在讓我們熟悉 Scikit-learn（發音為 sci，類似 science）。Scikit-learn 提供了[豐富的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)幫助你執行機器學習任務。

根據官方[網站](https://scikit-learn.org/stable/getting_started.html)說明：「Scikit-learn 是一個支援監督式與非監督式學習的開源機器學習函式庫。它還提供了模型擬合、資料前處理、模型選擇與評估以及其他多種工具。」

在本課程中，將會使用 Scikit-learn 與其他工具來建立我們所謂的「傳統機器學習」模型。我們刻意避免神經網路與深度學習，因為這些內容我們會在即將推出的「AI 初學者」課程中深入講解。

Scikit-learn 使建立模型與評估變得簡單。它主要針對數值資料，包含多個內建數據集作為學習工具，也提供預建模型供學生嘗試。現在讓我們先探索如何載入預先封裝的資料，並使用內建估計器來建立 Scikit-learn 的第一個機器學習模型。

## 練習 - 你的第一個 Scikit-learn 筆記本

> 本教學靈感來源於 Scikit-learn 官網的[線性回歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![初學者機器學習 - 你的第一個 Python 線性回歸專案](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "初學者機器學習 - 你的第一個 Python 線性回歸專案")

> 🎥 點擊上方圖片觀看本練習的快速教學影片。

在本課程配套的 _notebook.ipynb_ 檔案中，按下「垃圾桶」圖示清空所有儲存格。

本節將使用一個內建於 Scikit-learn 的小型糖尿病資料集作為學習素材。假設你想測試一種糖尿病病患的治療方式。機器學習模型或許能協助你根據變數組合，判斷哪些病患對治療反應較佳。即使是非常基礎的回歸模型，透過視覺化，也能顯示出哪些變數有助於設計理論性臨床試驗。

✅ 有很多種類型的回歸方法，選擇哪種取決於你想得到的答案。若你想預測某年齡對應可能身高，會使用線性回歸，因為你希望得到的是<strong>數值</strong>。若你想判別某種料理是否為純素食，則是在尋求<strong>分類</strong>，會使用邏輯斯回歸。邏輯斯回歸稍後會介紹。想一想你能問數據哪些問題，以及這些方法中哪一種更合適。

那麼我們開始吧。

### 載入函式庫

這個任務會用到以下函式庫：

- **matplotlib**。這是一個實用的[繪圖工具](https://matplotlib.org/)，我們將用來創建折線圖。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是 Python 中處理數值資料的好幫手。
- **sklearn**。這是[Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 函式庫。

載入函式庫以協助完成任務。

1. 輸入以下程式碼來加入函式庫：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上述程式碼匯入了 `matplotlib`、`numpy`，並從 `sklearn` 中匯入 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用於將資料分為訓練和測試集。

### 糖尿病資料集

內建的[糖尿病資料集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)包含 442 筆與糖尿病相關的樣本資料，帶有 10 個特徵變數，部分包括：

- age：年齡（歲）
- bmi：身體質量指數（Body Mass Index）
- bp：平均血壓
- s1 tc：T 細胞（一種白血球）

✅ 此資料集將「性別」列為重要的特徵變數之一。許多醫學資料集都包含這種二元分類。請思考此類分類如何可能使某些族群在治療上被排除。

接下來，載入 X 與 y 資料。

> 🎓 記住，這是監督式學習，我們需要命名的目標變數 'y'。

在新程式碼區塊中，透過呼叫 `load_diabetes()` 載入糖尿病資料集。參數 `return_X_y=True` 表示 `X` 是資料矩陣，`y` 是回歸目標。

1. 加入列印指令，顯示資料矩陣的形狀及其第一筆元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你回傳得到的是一個元組。你把這元組前兩個元素分別指派給 `X` 和 `y`。瞭解更多[元組](https://wikipedia.org/wiki/Tuple)。

    你會看到此資料中有 442 筆資料，每筆有 10 個元素的陣列：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 思考一下資料與回歸目標之間的關係。線性回歸旨在預測特徵 X 與目標變數 y 間的關係。你能在文件中找到糖尿病資料集的[目標](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)嗎？此資料集的目標展示了什麼？

2. 接著，選擇資料集中的一部分繪圖，即取資料集的第 3 欄。你可以用 `:` 選擇所有列，然後以索引（2）選擇第 3 欄。你也可以用 `reshape(n_rows, n_columns)` 將資料重塑成 2D 陣列以便繪圖。如果其中一個參數是 -1，該維度會自動計算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 隨時印出資料以確認其形狀。

3. 現在你有資料準備繪圖，可以試著讓機器判斷資料中的合理分割點。為此，你需將資料（X）與目標（y）拆分為測試與訓練套件。Scikit-learn 提供了簡便的方法來分割測試資料。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 接著準備訓練模型！載入線性回歸模型，並用你的 X、y 訓練集使用 `model.fit()` 進行訓練：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是你會在很多 ML 函式庫中看到的函數，例如 TensorFlow。

5. 接著，用測試資料建立預測，使用 `predict()` 函數。此預測會用於在資料群之間畫出分隔線。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在該用圖形展示資料了。Matplotlib 是完成此任務很有用的工具。使用散佈圖表現所有 X 與 y 測試資料，並利用預測來畫出模型間分群最合適的分割線。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![一個展示糖尿病資料散佈點的圖形](../../../../translated_images/zh-TW/scatterplot.ad8b356bcbb33be6.webp)


   ✅ 想一想這裡發生了什麼。一條直線穿過了許多小點的資料，但它到底在做什麼？你能看出應該如何使用這條線來預測一個新的、未見過的資料點在圖表的 y 軸上的相對位置嗎？試著用文字描述這個模型的實際用途。

恭喜，你建立了你的第一個線性迴歸模型，使用它做出預測，並將結果顯示在圖表上！

---
## 🚀挑戰

繪製該資料集中不同的變數。提示：編輯這一行：`X = X[:,2]`。考慮這個資料集的目標，你能發現糖尿病作為一種疾病的進展狀況嗎？
## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

在本教學中，你操作的是簡單線性迴歸，而不是單變數或多變數線性迴歸。閱讀一些關於這些方法差異的資料，或者看看這則 [影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

多閱讀關於迴歸概念的內容，並思考這種技術可以回答哪些問題。完成這個 [教學](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) 來加深你的理解。

## 作業

[另一個資料集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->