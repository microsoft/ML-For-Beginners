<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-06T09:05:38+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "mo"
}
-->
# 使用 Scikit-learn 建立回歸模型：四種回歸方法

![線性回歸與多項式回歸資訊圖表](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> 資訊圖表由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 製作
## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 簡介

到目前為止，您已經透過南瓜價格數據集的樣本數據了解了什麼是回歸分析，並使用 Matplotlib 進行了可視化。

現在，您準備深入了解機器學習中的回歸分析。雖然可視化可以幫助您理解數據，但機器學習的真正力量來自於 _訓練模型_。模型基於歷史數據進行訓練，能夠自動捕捉數據之間的依賴關係，並幫助您預測模型未曾見過的新數據的結果。

在本課程中，您將學習更多關於兩種回歸方法：_基本線性回歸_ 和 _多項式回歸_，以及這些技術背後的一些數學原理。這些模型將幫助我們根據不同的輸入數據預測南瓜的價格。

[![初學者的機器學習 - 理解線性回歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "初學者的機器學習 - 理解線性回歸")

> 🎥 點擊上方圖片觀看線性回歸的簡短視頻概述。

> 在整個課程中，我們假設學生的數學知識有限，並努力使內容對來自其他領域的學生更易理解，因此請留意筆記、🧮 數學提示、圖表以及其他學習工具以幫助理解。

### 先決條件

到目前為止，您應該已熟悉我們正在分析的南瓜數據的結構。您可以在本課程的 _notebook.ipynb_ 文件中找到預加載和預清理的數據。在該文件中，南瓜的價格以每蒲式耳的形式顯示在新的數據框中。請確保您可以在 Visual Studio Code 的內核中運行這些筆記本。

### 準備工作

提醒一下，您正在加載這些數據以便提出問題。

- 什麼時候是購買南瓜的最佳時機？
- 我可以預期一箱迷你南瓜的價格是多少？
- 我應該選擇半蒲式耳籃子還是 1 1/9 蒲式耳箱來購買？

讓我們繼續深入挖掘這些數據。

在上一課中，您創建了一個 Pandas 數據框，並用原始數據集的一部分填充它，將價格標準化為每蒲式耳。然而，通過這樣做，您只能收集到大約 400 個數據點，而且僅限於秋季月份。

查看本課程附帶的筆記本中預加載的數據。數據已預加載，並繪製了初始散點圖以顯示月份數據。也許通過進一步清理數據，我們可以更詳細地了解數據的性質。

## 線性回歸線

正如您在第一課中所學，線性回歸的目標是能夠繪製一條線以：

- **顯示變數關係**。展示變數之間的關係
- **進行預測**。準確預測新數據點在該線上的位置

通常使用 **最小平方回歸** 來繪製這種類型的線。'最小平方' 的意思是回歸線周圍的所有數據點的距離平方後相加。理想情況下，最終的總和應該盡可能小，因為我們希望誤差數量低，也就是 `最小平方`。

我們這樣做是因為我們希望建模一條距離所有數據點累積距離最小的線。我們在相加之前對項進行平方，因為我們關注的是其大小而不是方向。

> **🧮 數學展示**
> 
> 這條線，稱為 _最佳擬合線_，可以用 [一個方程](https://en.wikipedia.org/wiki/Simple_linear_regression) 表示：
> 
> ```
> Y = a + bX
> ```
>
> `X` 是 '解釋變數'，`Y` 是 '依賴變數'。線的斜率是 `b`，而 `a` 是 y 截距，指的是當 `X = 0` 時 `Y` 的值。
>
>![計算斜率](../../../../2-Regression/3-Linear/images/slope.png)
>
> 首先，計算斜率 `b`。資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 換句話說，參考我們南瓜數據的原始問題："根據月份預測每蒲式耳南瓜的價格"，`X` 代表價格，`Y` 代表銷售月份。
>
>![完成方程](../../../../2-Regression/3-Linear/images/calculation.png)
>
> 計算 Y 的值。如果您支付大約 $4，那一定是四月！資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作
>
> 計算這條線的數學必須展示線的斜率，這也取決於截距，即當 `X = 0` 時 `Y` 的位置。
>
> 您可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站上觀察這些值的計算方法。也可以訪問 [最小平方計算器](https://www.mathsisfun.com/data/least-squares-calculator.html) 來查看數字值如何影響線。

## 相關性

另一個需要理解的術語是 **相關係數**，即給定 X 和 Y 變數之間的相關性。使用散點圖，您可以快速可視化該係數。數據點整齊排列成一條線的圖表具有高相關性，而數據點在 X 和 Y 之間隨意分佈的圖表則具有低相關性。

一個好的線性回歸模型應該是使用最小平方回歸方法和回歸線，並且具有高（接近 1 而非 0）的相關係數。

✅ 運行本課程附帶的筆記本，查看月份與價格的散點圖。根據您的視覺解讀，南瓜銷售的月份與價格之間的數據是否具有高或低相關性？如果您使用更精細的測量方式（例如 *一年中的天數*，即自年初以來的天數），結果是否會有所改變？

在下面的代碼中，我們假設已清理數據，並獲得了一個名為 `new_pumpkins` 的數據框，類似於以下內容：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理數據的代碼可在 [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb) 中找到。我們執行了與上一課相同的清理步驟，並使用以下表達式計算了 `DayOfYear` 列：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

現在您已了解線性回歸背後的數學原理，讓我們建立一個回歸模型，看看是否可以預測哪種南瓜包裝的價格最划算。想要為節日南瓜園購買南瓜的人可能需要這些信息來優化南瓜包裝的購買。

## 尋找相關性

[![初學者的機器學習 - 尋找相關性：線性回歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "初學者的機器學習 - 尋找相關性：線性回歸的關鍵")

> 🎥 點擊上方圖片觀看相關性的簡短視頻概述。

從上一課中，您可能已經看到不同月份的平均價格如下所示：

<img alt="按月份的平均價格" src="../2-Data/images/barchart.png" width="50%"/>

這表明應該存在某種相關性，我們可以嘗試訓練線性回歸模型來預測 `Month` 與 `Price` 或 `DayOfYear` 與 `Price` 之間的關係。以下是顯示後者關係的散點圖：

<img alt="價格與一年中的天數的散點圖" src="images/scatter-dayofyear.png" width="50%" /> 

讓我們使用 `corr` 函數檢查是否存在相關性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看起來相關性很小，`Month` 為 -0.15，`DayOfYear` 為 -0.17，但可能存在另一個重要的關係。看起來不同的南瓜品種對價格的影響更大。為了確認這一假設，讓我們用不同的顏色繪製每個南瓜品種。通過向 `scatter` 繪圖函數傳遞 `ax` 參數，我們可以將所有點繪製在同一圖上：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格與一年中的天數的散點圖" src="images/scatter-dayofyear-color.png" width="50%" /> 

我們的調查表明，品種對整體價格的影響比實際銷售日期更大。我們可以通過柱狀圖看到這一點：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="價格與品種的柱狀圖" src="images/price-by-variety.png" width="50%" /> 

讓我們暫時只關注一種南瓜品種——'pie type'，看看日期對價格的影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格與一年中的天數的散點圖" src="images/pie-pumpkins-scatter.png" width="50%" /> 

如果我們現在使用 `corr` 函數計算 `Price` 與 `DayOfYear` 之間的相關性，我們會得到類似 `-0.27` 的結果——這意味著訓練一個預測模型是有意義的。

> 在訓練線性回歸模型之前，確保數據清理是很重要的。線性回歸對缺失值的處理效果不佳，因此清理掉所有空單元格是有意義的：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種方法是用相應列的平均值填充這些空值。

## 簡單線性回歸

[![初學者的機器學習 - 使用 Scikit-learn 進行線性和多項式回歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "初學者的機器學習 - 使用 Scikit-learn 進行線性和多項式回歸")

> 🎥 點擊上方圖片觀看線性和多項式回歸的簡短視頻概述。

為了訓練我們的線性回歸模型，我們將使用 **Scikit-learn** 庫。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

首先，我們將輸入值（特徵）和預期輸出（標籤）分離到不同的 numpy 陣列中：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 請注意，我們需要對輸入數據進行 `reshape`，以便線性回歸包正確理解它。線性回歸需要一個 2D 陣列作為輸入，其中陣列的每一行對應於輸入特徵的向量。在我們的情況下，由於我們只有一個輸入——我們需要一個形狀為 N×1 的陣列，其中 N 是數據集的大小。

接著，我們需要將數據分為訓練集和測試集，以便在訓練後驗證模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練實際的線性回歸模型只需要兩行代碼。我們定義 `LinearRegression` 對象，並使用 `fit` 方法將其擬合到數據中：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 對象在 `fit` 後包含所有回歸係數，可以通過 `.coef_` 屬性訪問。在我們的情況下，只有一個係數，應該大約是 `-0.017`。這意味著價格似乎隨時間略有下降，但幅度不大，每天大約下降 2 美分。我們還可以通過 `lin_reg.intercept_` 訪問回歸線與 Y 軸的交點——在我們的情況下，大約是 `21`，表示年初的價格。

為了查看模型的準確性，我們可以在測試數據集上預測價格，然後測量預測值與預期值的接近程度。這可以通過均方誤差（MSE）指標來完成，MSE 是所有預期值與預測值之間平方差的平均值。

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
我們的錯誤似乎集中在兩個點上，大約是 17%。表現不太理想。另一個衡量模型品質的指標是 **決定係數**，可以通過以下方式獲得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
如果值為 0，表示模型未考慮輸入數據，並且表現為*最差的線性預測器*，即僅僅是結果的平均值。值為 1 則表示我們可以完美地預測所有期望的輸出。在我們的情況下，決定係數約為 0.06，這相當低。

我們還可以繪製測試數據與回歸線的圖表，以更好地了解回歸在我們的案例中的表現：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="線性回歸" src="images/linear-results.png" width="50%" />

## 多項式回歸

另一種線性回歸是多項式回歸。有時候，變量之間存在線性關係，例如南瓜的體積越大，價格越高；但有時候這些關係無法用平面或直線來表示。

✅ 這裡有一些[更多例子](https://online.stat.psu.edu/stat501/lesson/9/9.8)，展示了可以使用多項式回歸的數據。

再看看日期與價格之間的關係。這個散點圖看起來是否一定要用直線來分析？價格難道不會波動嗎？在這種情況下，你可以嘗試使用多項式回歸。

✅ 多項式是可能包含一個或多個變量和係數的數學表達式。

多項式回歸會創建一條曲線，以更好地擬合非線性數據。在我們的案例中，如果我們在輸入數據中加入平方的 `DayOfYear` 變量，我們應該能用一條拋物線來擬合數據，該拋物線在一年中的某個點會有一個最低值。

Scikit-learn 提供了一個方便的 [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)，可以將不同的數據處理步驟結合在一起。**Pipeline** 是一個由**估算器**組成的鏈。在我們的案例中，我們將創建一個 pipeline，首先向模型添加多項式特徵，然後訓練回歸：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

使用 `PolynomialFeatures(2)` 表示我們將包含所有二次多項式特徵。在我們的案例中，這僅意味著 `DayOfYear`<sup>2</sup>，但如果有兩個輸入變量 X 和 Y，這將添加 X<sup>2</sup>、XY 和 Y<sup>2</sup>。如果需要，我們也可以使用更高次的多項式。

Pipeline 的使用方式與原始的 `LinearRegression` 對象相同，例如我們可以 `fit` pipeline，然後使用 `predict` 獲得預測結果。以下是顯示測試數據和近似曲線的圖表：

<img alt="多項式回歸" src="images/poly-results.png" width="50%" />

使用多項式回歸，我們可以獲得稍低的 MSE 和稍高的決定係數，但提升並不顯著。我們需要考慮其他特徵！

> 你可以看到南瓜價格的最低點大約出現在萬聖節附近。你能解釋這個現象嗎？

🎃 恭喜！你剛剛創建了一個可以幫助預測南瓜派價格的模型。你可能可以對所有南瓜類型重複相同的過程，但這樣會很繁瑣。接下來，我們將學習如何在模型中考慮南瓜品種！

## 類別特徵

在理想情況下，我們希望能夠使用同一個模型來預測不同南瓜品種的價格。然而，`Variety` 列與 `Month` 等列有所不同，因為它包含非數值型的值。這類列被稱為**類別型**。

[![初學者的機器學習 - 使用線性回歸進行類別特徵預測](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "初學者的機器學習 - 使用線性回歸進行類別特徵預測")

> 🎥 點擊上方圖片觀看有關使用類別特徵的簡短視頻概述。

以下是品種與平均價格的關係：

<img alt="按品種劃分的平均價格" src="images/price-by-variety.png" width="50%" />

為了考慮品種，我們首先需要將其轉換為數值形式，或者**編碼**。有幾種方法可以做到：

* 簡單的**數值編碼**會建立一個不同品種的表，然後用表中的索引替換品種名稱。這對線性回歸來說不是最好的方法，因為線性回歸會將索引的實際數值加到結果中，並乘以某個係數。在我們的案例中，索引號與價格之間的關係顯然是非線性的，即使我們確保索引按某種特定方式排序。
* **獨熱編碼**會用 4 個不同的列替換 `Variety` 列，每個列對應一個品種。如果某行屬於某品種，該列的值為 `1`，否則為 `0`。這意味著線性回歸中會有四個係數，每個南瓜品種都有一個，負責該品種的“起始價格”（或“附加價格”）。

以下代碼展示了如何對品種進行獨熱編碼：

```python
pd.get_dummies(new_pumpkins['Variety'])
```  

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE  
----|-----------|-----------|--------------------------|----------  
70 | 0 | 0 | 0 | 1  
71 | 0 | 0 | 0 | 1  
... | ... | ... | ... | ...  
1738 | 0 | 1 | 0 | 0  
1739 | 0 | 1 | 0 | 0  
1740 | 0 | 1 | 0 | 0  
1741 | 0 | 1 | 0 | 0  
1742 | 0 | 1 | 0 | 0  

要使用獨熱編碼的品種作為輸入訓練線性回歸，我們只需要正確初始化 `X` 和 `y` 數據：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

其餘代碼與我們之前用於訓練線性回歸的代碼相同。如果你嘗試一下，你會發現均方誤差差不多，但我們的決定係數大幅提高（約 77%）。為了獲得更準確的預測，我們可以考慮更多的類別特徵，以及數值特徵，例如 `Month` 或 `DayOfYear`。要獲得一個大的特徵數組，我們可以使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

在這裡，我們還考慮了 `City` 和 `Package` 類型，這使得 MSE 降至 2.84（10%），決定係數提高到 0.94！

## 整合所有內容

為了創建最佳模型，我們可以使用上述示例中的結合數據（獨熱編碼的類別特徵 + 數值特徵）以及多項式回歸。以下是完整代碼供你參考：

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```  

這應該能讓我們的決定係數達到接近 97%，MSE=2.23（約 8% 的預測誤差）。

| 模型 | MSE | 決定係數 |  
|-------|-----|---------------|  
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |  
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |  
| 所有特徵線性 | 2.84 (10.5%) | 0.94 |  
| 所有特徵多項式 | 2.23 (8.25%) | 0.97 |  

🏆 做得好！你在一節課中創建了四個回歸模型，並將模型品質提升到 97%。在回歸的最後一部分中，你將學習如何使用邏輯回歸來確定類別。

---

## 🚀挑戰

在此筆記本中測試幾個不同的變量，看看相關性如何影響模型準確性。

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學

在本課中，我們學習了線性回歸。還有其他重要的回歸類型。閱讀有關逐步回歸、嶺回歸、套索回歸和彈性網技術的資料。一門很好的課程是 [斯坦福統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)。

## 作業

[建立模型](assignment.md)  

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對於因使用此翻譯而產生的任何誤解或錯誤解讀概不負責。