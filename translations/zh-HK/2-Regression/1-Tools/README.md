# 使用 Python 和 Scikit-learn 入門迴歸模型

![迴歸模型概要的手繪筆記](../../../../translated_images/zh-HK/ml-regression.4e4f70e3b3ed446e.webp)

> 手繪筆記作者 [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前問答](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程亦提供 R 語言版本！](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## 介紹

在這四課中，你將學習如何建立迴歸模型。我們稍後會討論迴歸模型的用途。但在開始之前，請確保你已準備好適合的工具來開始這個流程！

在本課程中，你將學會：

- 配置你的電腦以便進行本地機器學習任務。
- 使用 Jupyter 筆記本。
- 使用 Scikit-learn，包括安裝。
- 透過動手練習探索線性迴歸。

## 安裝與設定

[![初學者機器學習 - 準備好安裝你的工具來建立機器學習模型](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "初學者機器學習 - 準備好安裝你的工具來建立機器學習模型")

> 🎥 點擊上圖觀看設定電腦進行機器學習的短片教學。

1. **安裝 Python**。確保你的電腦已安裝 [Python](https://www.python.org/downloads/)。Python 將用於許多資料科學及機器學習任務。大多數電腦系統預設已安裝 Python。對某些使用者來說，也有實用的 [Python 編程套件](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) 供配置時使用。

   然而，部分 Python 用途需要不同版本的軟體，因此使用 [虛擬環境](https://docs.python.org/3/library/venv.html) 會很方便。

2. **安裝 Visual Studio Code**。請確保你電腦已安裝 Visual Studio Code。請依照指示[安裝 Visual Studio Code](https://code.visualstudio.com/) 完成基本安裝。這堂課你將於 Visual Studio Code 中使用 Python，建議先熟悉如何[配置 Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) 以便 Python 開發。

   > 建議透過一系列 [學習單元](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) 來熟悉 Python。
   >
   > [![在 Visual Studio Code 設定 Python](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "在 Visual Studio Code 設定 Python")
   >
   > 🎥 點擊上圖觀看在 VS Code 內使用 Python 的示範影片。

3. **安裝 Scikit-learn**，請依照 [這些指示](https://scikit-learn.org/stable/install.html) 進行。確保你使用的是 Python 3，建議在虛擬環境中安裝。若你在 M1 Mac 上安裝此套件，請注意上述連結頁面內的特別指示。

1. **安裝 Jupyter Notebook**。你需要 [安裝 Jupyter 套件](https://pypi.org/project/jupyter/)。

## 你的機器學習開發環境

你將使用 <strong>筆記本</strong> 來撰寫 Python 代碼並建立機器學習模型。此類型檔案為資料科學家常用工具，副檔名為 `.ipynb`。

筆記本是一個互動式環境，讓開發者可以同時撰寫程式碼、加入筆記與文件說明，對於實驗或研究導向的專案非常有幫助。

[![初學者機器學習 - 設定 Jupyter 筆記本以開始建立迴歸模型](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "初學者機器學習 - 設定 Jupyter 筆記本以開始建立迴歸模型")

> 🎥 點擊上圖觀看這個練習的影片示範。

### 練習 — 使用筆記本

在此資料夾中，你會找到檔案 _notebook.ipynb_。

1. 在 Visual Studio Code 中打開 _notebook.ipynb_。

   Jupyter 伺服器會啟動並使用 Python 3+。你會在筆記本中看到可以 `運行` 的區塊，即程式碼片段。你可以點選類似播放鍵的圖示來運行該程式碼區塊。

1. 選取 `md` 圖示並加入一點 Markdown 格式，輸入以下文字：**# 歡迎使用你的筆記本**。

   接著，加入一些 Python 代碼。

1. 在程式碼區塊輸入 **print('hello notebook')**。
1. 點選執行箭頭跑這段程式碼。

   你應該會看到印出的訊息：

    ```output
    hello notebook
    ```

![VS Code 中打開的筆記本](../../../../translated_images/zh-HK/notebook.4a3ee31f396b8832.webp)

你可以在程式碼中穿插注釋，以自我文件化筆記本內容。

✅ 花點時間思考，網頁開發者的工作環境與資料科學家的工作環境有什麼不同。

## 開始使用 Scikit-learn

當你已經在本地環境中設好了 Python，且熟悉 Jupyter 筆記本後，我們接著來熟悉 Scikit-learn（讀作 sci，類似 science）。Scikit-learn 提供了 [廣泛的 API](https://scikit-learn.org/stable/modules/classes.html#api-ref) 幫助你完成機器學習任務。

根據其 [官方網站](https://scikit-learn.org/stable/getting_started.html) 記載，「Scikit-learn 是一個支援監督與非監督學習的開源機器學習庫。它也提供多種模型擬合、資料前處理、模型選擇與評估工具，以及其他許多實用功能。」

在本課程中，你將使用 Scikit-learn 與其他工具來建立機器學習模型，執行我們所謂的「傳統機器學習」任務。我們有意避開神經網絡與深度學習，因這部分將由即將推出的「初學者 AI 課程」涵蓋。

Scikit-learn 讓模型建立與評估變得簡單，主要聚焦於數值型資料，也內建多個適合學習使用的資料集，並提供多種預建模型供學生嘗試。現在讓我們探索如何載入出廠包裝資料，並使用內建估算器建立你的第一個 Scikit-learn ML 模型。

## 練習 — 你的第一個 Scikit-learn 筆記本

> 本教學靈感來自 Scikit-learn 官方網站上的 [線性迴歸範例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)。


[![初學者機器學習 - 你的第一個 Python 線性迴歸專案](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "初學者機器學習 - 你的第一個 Python 線性迴歸專案")

> 🎥 點擊上圖觀看此練習的短片示範。

在本課程相關的 _notebook.ipynb_ 檔案中，按下「垃圾桶」圖示清除所有儲存格內容。

本節將使用 Scikit-learn 內建的一個關於糖尿病的小資料集，作為學習用途。假設你想測試一種糖尿病患者的治療方法，機器學習模型可能幫助你判斷哪類病患會對此治療反應較佳，基於多種變數組合。即使是非常基礎的迴歸模型，在視覺化時也可能顯示出可用於設計理論臨床試驗的變數相關資訊。

✅ 迴歸方法有很多種，你選用何種取決於你想解答的問題。如果你想預測特定年齡者的身高，會用線性迴歸，因為你追求一個<strong>數值</strong>。若你想判斷某種料理是否為純素，則屬分類問題，會用邏輯迴歸。稍後你會學到邏輯迴歸。請思考你可以對資料提出什麼問題，並判斷哪些方法更適合。

現在開始這個任務吧。

### 載入函式庫

這個任務我們將導入下列函式庫：

- **matplotlib**。它是一個實用的[繪圖工具](https://matplotlib.org/)，我們將用它來繪製折線圖。
- **numpy**。[numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) 是 Python 中處理數值資料的有用函式庫。
- **sklearn**。即是[Scikit-learn](https://scikit-learn.org/stable/user_guide.html) 函式庫。

匯入部分函式庫幫助你完成任務。

1. 在程式碼區塊輸入以下匯入指令：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   上面你匯入了 `matplotlib` 和 `numpy`，並從 `sklearn` 匯入 `datasets`、`linear_model` 及 `model_selection`。`model_selection` 用於將資料切分為訓練及測試集。

### 糖尿病資料集

內建的 [糖尿病資料集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 包含 442 筆糖尿病相關樣本資料，含 10 項特徵變數，其中包括：

- 年齡：以年為單位
- BMI：身體質量指數
- 血壓：平均血壓
- s1 tc：T 細胞（一種白血球）

✅ 此資料集將「性別」納為糖尿病研究的重要特徵變數。許多醫學資料集包含此二元分類。請思考這類分類方式可能使某些族群遭排除於治療之外。

現在載入 X 與 y 資料。

> 🎓 請記得，這是監督式學習，我們需要一個命名為 y 的目標變數。

在新程式碼儲存格中，呼叫 `load_diabetes()` 載入糖尿病資料集。引數 `return_X_y=True` 表示 `X` 是資料矩陣，而 `y` 為迴歸目標。

1. 新增一些 print 指令以顯示資料矩陣形狀及其第一個元素：

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    你會收到一個 Tuple 作為回傳值。你做的動作是將元組的兩個第一個值依序指定給 `X` 和 `y`。想了解更多可查看 [關於元組](https://wikipedia.org/wiki/Tuple)。

    你可以看到資料含有 442 筆樣本，每筆由 10 個元素的陣列構成：

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ 想一想資料與迴歸目標的關係。線性迴歸預測特徵 X 與目標變數 y 之間的關係。你能在文件中找到糖尿病資料集的[目標值](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)是什麼嗎？由此資料集及目標值你能了解什麼？

2. 接著，選出資料集中的一部分繪圖，選取第 3 欄資料。你可用 `:` 選取所有列，再用索引（2）選擇第 3 欄。你也可以將資料重塑為 2D 陣列，以符合繪圖需求，使用 `reshape(n_rows, n_columns)`。其中若參數為 -1，則對應維度會自動計算。

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ 隨時可印出資料確認其形狀。

3. 現在資料已備妥可繪圖，你可檢查機器能否找出資料的合理分界。為此，你必須將資料（X）和目標（y）切分為測試與訓練集。Scikit-learn 提供簡便方法，你可指定分割點拆分資料。

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. 現在準備訓練模型了！載入線性迴歸模型，並用你的 X 和 y 訓練資料集用 `model.fit()` 進行訓練：

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` 是許多機器學習庫（如 TensorFlow）常見的函式。

5. 接著，使用測試資料用 `predict()` 函式進行預測。此預測可用來畫出分界線。

    ```python
    y_pred = model.predict(X_test)
    ```

6. 現在是時候透過繪圖展現資料了。Matplotlib 是非常實用的工具。繪製測試資料 X 與 y 的散點圖，並利用預測結果在資料群組中畫出一條最適合的線。

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![糖尿病資料點散佈圖](../../../../translated_images/zh-HK/scatterplot.ad8b356bcbb33be6.webp)

   ✅ 思考一下這裡發生了什麼。一條直線穿過許多小點，但它究竟在做什麼？你是否能理解如何利用這條線預測一個新、未看過的資料點應該如何相對於 y 軸排列？試著將此模型的實用意義用語言描述出來。

恭喜你，完成了你的第一個線性迴歸模型，做出預測，並以圖形展示出來！

---
## 🚀挑戰

從此資料集中繪製另一個變數。提示：修改這行：`X = X[:,2]`。根據此資料集的目標，你能從中發現糖尿病作為疾病的進展狀況嗎？
## [課後問答](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

在本教程中，你涉獵了簡單線性迴歸，而非單變量或多變量線性迴歸。可閱讀關於這些方法差異的資料，或參考 [此影片](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)。

閱讀有關迴歸概念的更多內容，並思考這種技術可以回答哪些類型的問題。參加此 [教程](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) 以加深你的理解。

## 作業

[另一個數據集](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->