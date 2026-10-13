# 使用 Python 和 Scikit-learn 開始迴歸模型之旅

![迴歸總結手繪筆記](../../../../translated_images/zh-HK/ml-regression.4e4f70e3b3ed446e.webp)

> 手繪筆記由 [Tomomi Imura](https://www.twitter.com/girlie_mac) 製作

## [課前小測](https://ff-quizzes.netlify.app/en/ml/)

> ### [這堂課也有 R 語言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介紹

在這四堂課中，你將學習如何建立迴歸模型。我們很快會討論這些模型的用途。但在做任何事之前，先確保你已經準備好合適的工具來開展流程！

在這堂課，你將學習如何：

- 配置你的電腦進行本地機器學習任務。
- 使用 Jupyter 筆記本。
- 使用 Scikit-learn，包括安裝步驟。
- 通過實作練習探索線性迴歸。

## 安裝與設定

[![為初學者準備的 ML - 配置你的工具開始建立機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "為初學者準備的 ML - 配置你的工具開始建立機器學習模型")

> 🎥 點擊上方圖片觀看短片，學習如何配置電腦以進行 ML。

1. **安裝 Python**。確保你的電腦已安裝好 [Python](https://www.python.org/downloads/)。Python 是許多數據科學及機器學習任務的重要工具。大多數系統已內建 Python。也有實用的 [Python 編碼套件](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) 可供部分用戶簡化設定。

   然而部分 Python 的應用需要特定版本，因此使用 [虛擬環境](https://docs.python.org/3/library/venv.html) 是非常有用的。

2. **安裝 Visual Studio Code**。確保你的電腦已安裝 Visual Studio Code。遵循此說明[安裝 Visual Studio Code](https://code.visualstudio.com/)完成基礎安裝。因為你會在這課程中使用 Python，所以建議熟悉如何[為 Python 開發設定 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott)。

   > 透過一系列 [Learn 模組](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)熟悉 Python。
   >
   > [![使用 Visual Studio Code 設定 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "使用 Visual Studio Code 設定 Python")
   >
   > 🎥 點擊上方圖片觀看如何在 VS Code 裡使用 Python 的教學影片。

3. **安裝 Scikit-learn**，請參考[這些指示](https://scikit-learn.org/stable/install.html)。因必須使用 Python 3，建議在虛擬環境中安裝。如果你在 M1 Mac 上安裝，請注意頁面上有特別說明。

1. **安裝 Jupyter Notebook**。你需要 [安裝 Jupyter 軟件包](https://pypi.org/project/jupyter/)。

## 你的機器學習開發環境

你將使用 <strong>筆記本</strong> 來編寫 Python 程式碼和建立機器學習模型。這類文件是數據科學家常用工具，可以透過副檔名 `.ipynb` 識別。

筆記本是互動環境，開發者可同時撰寫程式碼、添加註解與文件，非常適合實驗或研究型項目。

[![為初學者準備的 ML - 設置 Jupyter 筆記本開始建立迴歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "為初學者準備的 ML - 設置 Jupyter 筆記本開始建立迴歸模型")

> 🎥 點擊上方圖片觀看此練習的教學短片。

### 練習 - 使用筆記本

在此資料夾中，你會找到 _notebook.ipynb_ 檔案。

1. 在 Visual Studio Code 中打開 _notebook.ipynb_。

   Jupyter 服務會以 Python 3+ 启动。你會看到筆記本中可 `執行` 的程式碼區塊。點擊類似播放按鈕的圖示即可執行區塊。

1. 選擇 `md` 圖標並添加一些 markdown，寫入以下文字 **# 歡迎使用你的筆記本**。

   接著，加入一些 Python 程式碼。

1. 在程式碼區塊中輸入 **print('hello notebook')**。
1. 點擊箭頭執行程式碼。

   你應該會看到列印的訊息：

    ```output
    hello notebook
    ```

![VS Code 中開啟的筆記本](../../../../translated_images/zh-HK/notebook.4a3ee31f396b8832.webp)

你可以在程式碼中夾帶註解，讓筆記本自我文件化。

✅ 花點時間想想，網頁開發者的工作環境與數據科學家的工作環境有多不同。

## 使用 Scikit-learn 快速起步

現在你的本地環境已安裝好 Python，且熟悉 Jupyter 筆記本，接下來讓我們來熟悉 Scikit-learn（發音為「sci」就像「science」中的發音）。Scikit-learn 提供了[豐富的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref)，幫助你執行機器學習任務。

根據他們的[網站](https://scikit-learn.org/stable/getting_started.html)描述：「Scikit-learn 是一個開源機器學習函式庫，支援監督式及非監督式學習。它同時提供模型擬合、資料預處理、模型選擇與評估以及許多其他實用工具。」

在本課程中，你將使用 Scikit-learn 和其他工具，建立所謂「傳統機器學習」的模型。我們有意避開神經網絡與深度學習，這些會在即將推出的「初學者 AI」課程中深入介紹。

Scikit-learn 讓建立與評估模型變得簡單方便。它主要聚焦於使用數值資料，並內建多個現成數據集作學習範例，也包含已建好的模型供學生嘗試。讓我們先探索如何載入內建數據，使用 Scikit-learn 的預建估計器建立第一個機器學習模型，並分析簡單的數據。

## 練習 - 你的第一個 Scikit-learn 筆記本

> 本教程靈感來自 Scikit-learn 網站上的[線性迴歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![為初學者準備的 ML - 你的第一個 Python 線性迴歸專案](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "為初學者準備的 ML - 你的第一個 Python 線性迴歸專案")

> 🎥 點擊上方圖片觀看此練習的教學短片。

在本課程附帶的 _notebook.ipynb_ 檔案中，按下「垃圾桶」圖示清除所有儲存格。

這部分你將使用 Scikit-learn 內建一個關於糖尿病的小型數據集。假設你想測試某項糖尿病病患的治療方法。機器學習模型可幫助你依據變數組合判斷哪些病患的反應會較佳。即使是非常基礎的迴歸模型，當視覺化後，也可揭露變數間的信息，幫你規劃理論臨床試驗。

✅ 有許多類型的迴歸方法，不同方法適合解決不同問題。若你想預測特定年齡的身高，會用線性迴歸，因為你尋求的是<strong>數值結果</strong>；若你想判定一道菜是否屬於素食料理，則是<strong>分類問題</strong>，會用邏輯迴歸。稍後你將學習邏輯迴歸。花點時間思考可向資料提出的問題，以及哪種方法更合適。

現在開始著手這項任務。

### 引入程式庫

這項任務會用到以下一些程式庫：

- **matplotlib**。這是有用的[繪圖工具](https://matplotlib.org/)，我們將用它來畫折線圖。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是 Python 中處理數值資料的重要函式庫。
- **sklearn**。這是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)。

引入所需的程式庫協助完成任務。

1. 輸入以下代碼以加入引用：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   你以上引用了 `matplotlib`、`numpy`，還從 `sklearn` 引入了 `datasets`、`linear_model` 和 `model_selection`。`model_selection` 用於將資料分成訓練組與測試組。

### 糖尿病數據集

內建的 [糖尿病數據集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)包含 442 筆糖尿病相關資料，含 10 項特徵變數，其中包括：

- 年齡：years
- 體重指數 (bmi)
- 血壓平均值 (bp)
- s1 tc：T 細胞（一種白血球）

✅ 此資料集將「性別」視為一項重要特徵變數，醫學資料中常包含此類二元分類。思考一下這種分類方式可能導致部分族群因特徵被排除於療法之外。

現在載入資料矩陣 X 和目標 y。

> 🎓 請記得，這是監督式學習，需要一個命名為 y 的目標變量。

在新程式碼區塊中，透過呼叫 `load_diabetes()` 載入糖尿病資料集。參數 `return_X_y=True` 表示回傳的 X 是資料矩陣，y 是迴歸目標。

1. 加入 print 指令以顯示資料矩陣形狀與第一筆資料：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    回傳結果是一個元組。你是將元組的前兩個值分別賦值給 `X` 和 `y`。進一步了解 [元組](https://wikipedia.org/wiki/Tuple)。

    你可以看到資料有 442 筆，且每筆資料包含 10 個元素：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 想想資料與迴歸目標之間的關係。線性迴歸預測特徵 X 與目標變數 y 的關係。你能在說明文檔找到糖尿病資料集的[目標](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)是什麼嗎？這資料集想展示什麼？

2. 接著選取此資料集中的一部份畫圖，方法是選取第 3 欄數據。你可以使用 `:` 選擇所有列，再用索引 (2) 選擇第 3 欄。為符合繪圖需求，可以用 `reshape(n_rows, n_columns)` 將資料重新塑形。當一個參數為 -1 時，該維度大小會自動計算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 需要時隨時列印資料檢查其形狀。

3. 現在你有了可以繪圖的資料，接著可讓機器判斷資料分組的合理界線。為此，需要將資料 (X) 與目標 (y) 分割成測試集與訓練集。Scikit-learn 提供簡單方法指定分割點。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 現在準備訓練模型！載入線性迴歸模型並用 X 與 y 訓練資料集執行 `model.fit()` 進行訓練：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是許多 ML 函式庫（例如 TensorFlow）常見的訓練函式

5. 使用測試資料透過 `predict()` 函式創建預測，繪圖時將用這條預測線分隔資料群組

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在是時候用圖形顯示資料。Matplotlib 是非常有用的工具。繪製所有 X 與 y 測試資料的散點圖，並利用預測結果畫出最合適的分界線，顯示模型資料分組。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![展示糖尿病資料點的散點圖](../../../../translated_images/zh-HK/scatterplot.ad8b356bcbb33be6.webp)


   ✅ 想一想這裡發生了什麼。一條直線穿過許多小點資料，但它到底在做什麼？你能否看到如何利用這條線去預測一個新的、未見過的資料點應該在圖上的 y 軸上的位置？試著用文字描述這個模型的實際用途。

恭喜你，建立了你的第一個線性回歸模型，使用它做出一個預測，並將結果顯示在圖表中！

---
## 🚀挑戰

繪製這個資料集中的另一個變數。提示：編輯這行：`X = X[:,2]`。根據這個資料集的目標，你能發現糖尿病作為一種疾病的發展過程嗎？
## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

在本教學中，你使用的是簡單線性回歸，而非單變量或多重線性回歸。閱讀一些關於這些方法差異的資料，或觀看[這個影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

多閱讀關於回歸這個概念的內容，並思考這種技術能回答哪些類型的問題。參加這個[教學](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott)來深化你的理解。

## 作業

[另一個資料集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->