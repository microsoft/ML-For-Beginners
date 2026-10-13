# 使用 Python 和 Scikit-learn 開始迴歸模型

![迴歸總結手繪筆記](../../../../translated_images/zh-MO/ml-regression.4e4f70e3b3ed446e.webp)

> 手繪筆記由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 製作

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [此課程亦有 R 版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介紹

在這四課中，你將學習如何建立迴歸模型。我們稍後將討論它們的用途。但在開始操作之前，請先確認你已準備好正確的工具！

本課程你將學習如何：

- 設置電腦以進行本地機器學習任務。
- 使用 Jupyter 筆記本。
- 使用 Scikit-learn，包括安裝。
- 透過實作探索線性迴歸。

## 安裝與配置

[![機器學習初學者 - 安裝您的工具準備建立機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "機器學習初學者 - 安裝您的工具準備建立機器學習模型")

> 🎥 點擊上圖觀看教學影片，示範如何設定電腦以執行機器學習。

1. **安裝 Python**。確保您的電腦已安裝 [Python](https://www.python.org/downloads/)。Python 是許多資料科學及機器學習工作所需的語言。大多數電腦系統已預裝 Python。也有實用的 [Python 編碼套件](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott)，可幫助部分用戶簡化設定流程。

   然而，不同的 Python 用途可能需要不同版本，因此使用 [虛擬環境](https://docs.python.org/3/library/venv.html) 是很實用的做法。

2. **安裝 Visual Studio Code**。確保您的電腦已安裝 Visual Studio Code。請按照這些說明 [安裝 Visual Studio Code](https://code.visualstudio.com/) 以完成基礎安裝。您將在本課程中使用 Visual Studio Code 編寫 Python，建議了解如何 [設定 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) 作為 Python 開發環境。

   > 透過這系列 [學習模組](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) 熟悉 Python。
   >
   > [![視覺工作室碼中設定 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "視覺工作室碼中設定 Python")
   >
   > 🎥 點擊上圖觀看教學影片：在 VS Code 使用 Python。

3. **安裝 Scikit-learn**，請遵循 [這些指示](https://scikit-learn.org/stable/install.html)。因需使用 Python 3，建議使用虛擬環境。若您在 M1 Mac 上安裝，本頁面提供了特別指引。

1. **安裝 Jupyter Notebook**。您需要 [安裝 Jupyter 套件](https://pypi.org/project/jupyter/)。

## 你的機器學習編輯環境

你將使用 **notebooks** 來開發 Python 程式碼並建立機器學習模型。這類檔案是資料科學家常用工具，副檔名為 `.ipynb`。

筆記本是一個互動式環境，允許開發人員同時撰寫程式碼、加入筆記及撰寫程式說明，這對實驗性或研究導向專案非常有用。

[![機器學習初學者 - 設定 Jupyter 筆記本開始建立迴歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "機器學習初學者 - 設定 Jupyter 筆記本開始建立迴歸模型")

> 🎥 點擊上圖觀看示範本練習的短片。

### 練習 - 操作一個 notebook

在此資料夾中，你會找到檔案 _notebook.ipynb_。

1. 在 Visual Studio Code 中開啟 _notebook.ipynb_。

   會啟動一個 Python 3+ 的 Jupyter 伺服器。你會看到可執行的程式碼區域。可按一個看似播放鍵的圖示，執行該程式碼區塊。

1. 選擇 `md` 圖示，加入一些 markdown 語法和以下文字 **# Welcome to your notebook**。

   接著，加入一些 Python 程式碼。

1. 在程式碼區塊鍵入 **print('hello notebook')**。
1. 按箭頭執行程式碼。

   你應該會看到輸出如下：

    ```output
    hello notebook
    ```

![VS Code 開啟 notebook 的畫面](../../../../translated_images/zh-MO/notebook.4a3ee31f396b8832.webp)

你可以穿插程式碼與註解，自我記錄 notebook 內容。

✅ 想想看，網頁開發者的工作環境與資料科學家的環境有何不同。

## 使用 Scikit-learn 上手

現在你的本地環境已設定好 Python，且熟悉 Jupyter 筆記本，接下來讓我們熟悉 Scikit-learn（發音為「sci」，類似「science」）。Scikit-learn 提供了[完整的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)協助你執行機器學習任務。

根據其[官方網站](https://scikit-learn.org/stable/getting_started.html)說明：「Scikit-learn 是一個支持監督式與非監督式學習的開源機器學習函式庫。它還提供各種模型擬合、資料前處理、模型選擇與評估等工具，及其他許多實用功能。」

在本課程，你將使用 Scikit-learn 與其他工具建立機器學習模型，以完成我們所稱的“傳統機器學習”任務。我們故意避免使用神經網絡和深度學習，這部分會在即將推出的「AI 初學者」課程中更完整說明。

Scikit-learn 讓建立模型及評估變得簡單。它主要使用數值資料，並包含多組預先準備好的學習用資料集，也有學生可試用的預建模型。讓我們先探索如何載入預設資料，並用內建的估算器建立首個 Scikit-learn 機器學習模型。

## 練習 - 你的第一個 Scikit-learn 筆記本

> 本教學靈感來自 Scikit-learn 網站上的[線性迴歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![機器學習初學者 - 你的第一個 Python 線性迴歸專案](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "機器學習初學者 - 你的第一個 Python 線性迴歸專案")

> 🎥 點擊上圖觀看示範本練習的短片。

在本課程附帶的 _notebook.ipynb_ 檔案中，請按「垃圾桶」圖示清空所有儲存格。

本節中，你將使用 Scikit-learn 內建的一組關於糖尿病的迷你資料集學習。假設你想測試一種糖尿病患者的治療方法。機器學習模型有助判斷哪些病患會較適合此治療方案，基於各變數組合。即使是基本迴歸模型，經視覺化後，也能揭示幫助理論性臨床試驗設計的變數資訊。

✅ 有許多迴歸方法，選擇哪種取決於你想得到什麼答案。若想預測特定年齡的人大約身高，則用線性迴歸，因為你想要預測的是<strong>數值數據</strong>。若要判斷某類料理是不是純素，則是<strong>分類任務</strong>，會用邏輯迴歸。後續你會進一步學習邏輯迴歸。思考一下你能向數據提出哪些問題，以及用哪種方法較適合。

現在開始動手做吧。

### 匯入函式庫

本任務我們將匯入一些函式庫：

- **matplotlib**。這是個實用的[繪圖工具](https://matplotlib.org/)，用來產生折線圖。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是 Python 中處理數值資料的重要函式庫。
- **sklearn**。即是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 函式庫。

匯入函式庫以協助你的任務。

1. 輸入以下程式碼匯入：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   你匯入了 `matplotlib`、`numpy`，以及從 `sklearn` 匯入 `datasets`、`linear_model`、`model_selection`。其中 `model_selection` 用來將資料拆分為訓練集和測試集。

### 糖尿病資料集

內建的 [糖尿病資料集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)含有 442 個糖尿病相關樣本資料，特徵共 10 項，其中若干如下：

- age：年齡（歲）
- bmi：身體質量指數
- bp：平均血壓
- s1 tc：T 細胞（一種白血球）

✅ 此資料集中包含 'sex' 作為對糖尿病研究重要的特徵變數。許多醫療資料集都包含這類二元分類。思考一下這種分類是否可能導致部分族群被排除在治療之外。

接下來，載入 X 與 y 資料。

> 🎓 記住，這是監督式學習，因此需要命名為 'y' 的目標變量。

在新的程式碼儲存格中，呼叫 `load_diabetes()` 載入資料集。輸入參數 `return_X_y=True` 表示 `X` 會是資料矩陣，`y` 則是迴歸目標。

1. 加入印出指令，顯示資料矩陣形狀與第一筆資料：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你得到的回傳結果是個元組。你將元組的前兩個值分別指定給 `X` 與 `y`。可深入了解[元組](https://wikipedia.org/wiki/Tuple)。

    可以看到此資料集有 442 筆資料，每筆資料有 10 個元素的陣列：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 想一想資料與迴歸目標的關係。線性迴歸預測特徵 X 與目標變量 y 的連結。你能否在文件中找到該糖尿病資料集的[目標](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)？給定該目標，此資料集試圖說明什麼？

2. 接著，挑選資料集其中一部分來繪圖，選取第 3 欄資料。使用 `:` 選取所有列，再用索引 (2) 選取第 3 欄。可用 `reshape(n_rows, n_columns)` 將資料改為二維陣列，是繪圖的需求。若其中一參數是 -1，則自動計算相應維度大小。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 隨時印出資料確認其形狀。

3. 現在你有資料可繪圖，嘗試讓機器判定該資料中數值的合理分界。這需將資料 (X) 與目標 (y) 依比例拆成測試與訓練集。Scikit-learn 有簡單方法可拆分資料。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 準備好後開始訓練模型！載入線性迴歸模型，並用 X 與 y 的訓練集以 `model.fit()` 訓練：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是許多機器學習函式庫（如 TensorFlow）通用的函式。

5. 接著，用測試資料使用 `predict()` 產生預測。它用來繪製模型數據分組間的線條。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在該將資料以圖表呈現。Matplotlib 是此任務中很實用的工具。繪製所有 X 與 y 測試資料的散點圖，再用預測值畫出最適合的分界線。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![顯示糖尿病資料點的散點圖](../../../../translated_images/zh-MO/scatterplot.ad8b356bcbb33be6.webp)


   ✅ 想一想這裡發生了什麼。一直線穿過許多小點數據，但它到底在做什麼？你能否看出應該如何利用這條線來預測一個未見過的新數據點應該在圖中 y 軸的哪個位置？試著用文字表達這個模型的實際用途。

恭喜，你建立了第一個線性回歸模型，使用它創建了預測，並且將其顯示在圖中！

---
## 🚀挑戰

繪製來自此數據集的不同變量的圖表。提示：編輯這一行：`X = X[:,2]`。根據此數據集的目標，你能發現糖尿病作為一種疾病的進展有什麼特點？
## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

在本教程中，你使用的是簡單線性回歸，而非單變量或多元線性回歸。了解這些方法之間的差異，或觀看[這個影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

閱讀更多關於回歸概念的內容，思考這個技術能回答哪些問題。使用這個[教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)加深你的理解。

## 作業

[另一個數據集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->