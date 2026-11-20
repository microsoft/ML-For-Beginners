# 使用 Python 與 Scikit-learn 建立迴歸模型入門

![迴歸模型摘要筆記](../../../../translated_images/zh-TW/ml-regression.4e4f70e3b3ed446e.webp)

> 筆記作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程亦提供 R 版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介紹

這四堂課將帶你學習如何建立迴歸模型，稍後會討論其用途。但在開始之前，請先確認你已準備好正確的工具！

本課程你將學習：

- 配置你的電腦以執行本地機器學習任務。
- 使用 Jupyter 筆記本。
- 使用 Scikit-learn，包括安裝步驟。
- 藉由實作了解線性迴歸。

## 安裝與設定

[![機器學習新手——準備好環境以建立機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "機器學習新手——準備好環境以建立機器學習模型")

> 🎥 點擊上方圖片觀看設定電腦環境的短片。

1. **安裝 Python**。請確定你的電腦已安裝 [Python](https://www.python.org/downloads/)。你將在許多資料科學與機器學習任務中使用 Python。大多電腦系統內建 Python。也有實用的 [Python 編碼套件](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) 可協助部分用戶快速安裝。

   不同用途的 Python 可能需要不同版本，因此使用 [虛擬環境](https://docs.python.org/3/library/venv.html) 非常有幫助。

2. **安裝 Visual Studio Code**。請確定你的電腦已安裝 Visual Studio Code。依照這裡的指示 [安裝 Visual Studio Code](https://code.visualstudio.com/) 作基本安裝。課程中將於 Visual Studio Code 內使用 Python，建議熟悉如何 [設定 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) 以便 Python 開發。

   > 透過這系列的 [學習模組](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) 熟悉 Python。
   >
   > [![使用 Visual Studio Code 安裝 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "使用 Visual Studio Code 安裝 Python")
   >
   > 🎥 點擊上方圖片觀看在 VS Code 內使用 Python 的影片。

3. **安裝 Scikit-learn**，請依照 [此處指示](https://scikit-learn.org/stable/install.html) 操作。因需使用 Python 3，建議使用虛擬環境。如於 M1 Mac 安裝，頁面中有特殊指引。

1. **安裝 Jupyter Notebook**。你需要安裝 [Jupyter 套件](https://pypi.org/project/jupyter/)。

## 你的機器學習開發環境

你將使用 **筆記本（notebooks）** 來撰寫 Python 程式碼並建立機器學習模型。這種檔案常見於資料科學家工具，副檔名為 `.ipynb`。

筆記本提供互動環境，允許開發者同時撰寫程式碼、加筆記與文件，對實驗或研究專案非常有用。

[![機器學習新手——設定 Jupyter 筆記本以開始建立迴歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "機器學習新手——設定 Jupyter 筆記本以開始建立迴歸模型")

> 🎥 點擊上方圖片觀看完成此練習的短片。

### 練習 - 使用筆記本

在此資料夾中，你會找到 _notebook.ipynb_ 檔案。

1. 在 Visual Studio Code 中開啟 _notebook.ipynb_。

   Jupyter 伺服器隨即啟動並運行 Python 3+。你會看到筆記本中可 `run` 的程式碼區塊。按右側播放按鈕執行該段程式碼。

1. 選擇 `md` 圖示並加入一些 markdown，輸入文字 **# Welcome to your notebook**。

   接著，加入一些 Python 程式碼。

1. 在程式碼區塊中輸入 **print('hello notebook')**。
1. 按下播放箭頭執行程式。

   你會看到輸出的結果：

    ```output
    hello notebook
    ```

![VS Code 開啟筆記本畫面](../../../../translated_images/zh-TW/notebook.4a3ee31f396b8832.webp)

你可在程式碼中插入註解，自我記錄筆記本內容。

✅ 思考一下，網頁開發者的工作環境與資料科學家的工作環境有何不同？

## 使用 Scikit-learn 順利啟動

現在你的本地環境已設置好 Python，並熟悉 Jupyter 筆記本，接著讓我們熟悉 Scikit-learn（讀作 「sci」，如同 science）。Scikit-learn 提供 [完整的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref) 協助你執行機器學習任務。

根據他們的 [網站](https://scikit-learn.org/stable/getting_started.html) 描述，「Scikit-learn 是一個支援監督式與非監督式學習的開源機器學習函式庫，提供模型擬合、資料前處理、模型選擇與評估等多種工具及許多實用功能。」

在本課程中，你將利用 Scikit-learn 與其它工具來建立我們所稱的「傳統機器學習」模型。我們刻意避免使用神經網路與深度學習，這面向將在未來的「AI 初學者」課程中深入介紹。

Scikit-learn 讓建立與評估模型變得簡單。它主要針對數值資料，並包含多組內建數據供學習使用，也提供預建模型供學生嘗試。讓我們一起探索如何載入預打包資料並使用內建估算器，藉此用 Scikit-learn 建立你的第一個機器學習模型並使用基本資料。

## 練習 - 你的第一個 Scikit-learn 筆記本

> 本教學靈感來自 Scikit-learn 網站上的 [線性迴歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![機器學習新手——你的第一個 Python 線性迴歸專案](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "機器學習新手——你的第一個 Python 線性迴歸專案")

> 🎥 點擊上方圖片觀看完成此練習的短片。

在本課程所附的 _notebook.ipynb_ 檔案中，按下「垃圾桶」圖示清空所有儲存格。

在此階段，你會使用 Scikit-learn 內建的小型糖尿病數據集作為學習範例。假設你想測試一項糖尿病治療方案。機器學習模型或可幫助判斷有哪些病患基於各項變數組合，反應會較佳。即使是非常基礎的迴歸模型，視覺化後亦能顯示某些變數訊息，有助你規劃理論性臨床試驗。

✅ 迴歸方法種類繁多，選擇哪種取決於你想要的答案。例如，若你想依年齡預測某人可能的身高，因為你需要的是 <strong>數值</strong>，就會用線性迴歸。若你想判別某料理類型是否為純素，則屬於 <strong>類別分配</strong>，會用邏輯迴歸。後面你會學到邏輯迴歸。想一想可向資料提出的問題及較適合的迴歸類型。

讓我們開始這個任務吧。

### 載入函式庫

本任務中我們將載入以下函式庫：

- **matplotlib**，一個實用的 [繪圖工具](https://matplotlib.org/)，用來產生線圖。
- **numpy**，[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是處理 Python 中數值資料的實用函式庫。
- **sklearn**，也就是 [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 函式庫。

載入函式庫協助完成你的任務。

1. 輸入以下程式碼新增 import：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上述程式碼載入了 `matplotlib`、`numpy`，並從 `sklearn` 載入 `datasets`、`linear_model` 及 `model_selection`。其中，`model_selection` 用來將資料切分成訓練集與測試集。

### 糖尿病數據集

內建的 [糖尿病數據集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)含有 442 筆有關糖尿病的資料樣本，共有 10 個特徵變數，其中包括：

- age：年齡（歲）
- bmi：身體質量指數
- bp：平均血壓
- s1 tc：T 細胞（一種白血球）

✅ 此數據集包含『性別』特徵，此為糖尿病研究的重要分類。許多醫療數據集都含此類二元分類。思考一下這類分類可能導致哪些族群被排除於治療之外。

現在，載入 X 與 y 的資料。

> 🎓 記住，這是監督式學習，我們需要一個命名為 'y' 的目標變數。

在新程式碼儲存格中，呼叫 `load_diabetes()` 載入糖尿病數據集。輸入參數 `return_X_y=True` 表示 `X` 會是一個資料矩陣，`y` 則是迴歸目標。

1. 新增 print 命令輸出資料矩陣形狀與首項：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你取得的回傳值為一個元組，程式是將元組中的前兩個值分別指派給 `X` 和 `y`。更多資料可參考 [關於元組](https://wikipedia.org/wiki/Tuple)。

    你會看到資料包含 442 筆、每筆有 10 個元素的陣列：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 思考資料與迴歸目標間的關聯。線性迴歸預測 X 特徵與目標變數 y 間的關係。可在文件中找到糖尿病數據集的 [目標](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)？目標為何？數據集展示了什麼？

2. 接著，從數據集中擇一部分作繪圖，選出第 3 欄。透過 `:` 選出所有列，使用索引 (2) 取得第三欄資料。你也可以用 `reshape(n_rows, n_columns)` 將資料重塑為二維陣列，繪圖時需要。當參數設為 -1，維度將自動計算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 隨時印出資料檢查其形狀。

3. 資料已準備好繪圖後，可以讓機器判斷資料間合理的區分點。為此，將資料（X）和目標（y）同時分為訓練集與測試集。Scikit-learn 提供簡單方法，能在指定位置分割測試資料。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 現在準備訓練模型！載入線性迴歸模型，使用 `model.fit()` 用 X 與 y 的訓練資料訓練模型：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是許多機器學習函式庫，如 TensorFlow 中常見的函數。

5. 接著，使用測試資料呼叫 `predict()` 函數預測，將用以繪製資料群的分界線。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在該在圖中呈現資料了。Matplotlib 是個非常實用的工具。繪製所有 X 與 y 測試資料的散佈圖，並利用預測結果繪製一條分界線，劃分模型資料群。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![以糖尿病資料點繪製的散佈圖](../../../../translated_images/zh-TW/scatterplot.ad8b356bcbb33be6.webp)

   ✅ 思考此圖所呈現的意義。一直線穿越許多小點，但它究竟在做什麼？你能理解這條線的用途，能用來預測新資料點在 y 軸上的對應位置？試著用自己的話描述此模型的實際用途。

恭喜，你成功建立你的第一個線性迴歸模型，並用它做出預測，最後用圖形顯示結果！

---
## 🚀 挑戰

從此數據集中繪製不同的變數。提示：修改此行 `X = X[:,2]`。根據該資料集的目標，你能發現糖尿病作為疾病的進展情況嗎？
## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

教學中你使用了簡單線性迴歸，而非單變量或多元線性迴歸。可閱讀它們間差異，或觀賞 [此影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

進一步了解迴歸的概念，並思考這種技術能回答什麼樣的問題。請參加這個 [教學](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) 以加深你的理解。

## 作業

[不同的資料集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
此文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們努力追求準確性，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於關鍵資訊，建議採用專業人工翻譯。我們不對因使用此翻譯所產生的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->