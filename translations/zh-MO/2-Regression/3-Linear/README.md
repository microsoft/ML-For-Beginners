# 使用 Scikit-learn 建立迴歸模型：四種迴歸方式

## 初學者筆記

當我們想要預測一個<strong>數值</strong>（例如房價、溫度或銷售額）時，會使用線性迴歸。它透過尋找一條最能代表輸入特徵與輸出之間關係的直線來運作。

在本課程中，我們將先著重理解這個概念，再進一步探討更進階的迴歸技術。  
![線性與多項式迴歸圖解](../../../../translated_images/zh-MO/linear-polynomial.5523c7cb6576ccab.webp)  
> 圖解由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 提供  

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 語言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### 介紹

到目前為止，你已經探索過什麼是迴歸，並使用整個課程將會用到的南瓜價格資料集做範例。你也已經運用 Matplotlib 視覺化這些資料。

現在你已經準備深入了解為機器學習用的迴歸。雖然視覺化讓你可以理解資料，但機器學習的真正威力在於_訓練模型_。模型是透過歷史資料訓練，以自動捕捉資料間的依賴關係，並讓你得以對未見過的新資料進行結果預測。

本課程會介紹兩種迴歸方法：_基本線性迴歸_與_多項式迴歸_，並略談其背後的數學原理。這些模型將讓我們依據不同輸入資料預測南瓜價格。

[![機器學習入門－理解線性迴歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "機器學習入門－理解線性迴歸")

> 🎥 點擊上方圖片，觀看線性迴歸的短片概述。

> 本課程假設讀者數學基礎不深，並致力於讓來自其他背景的學生也能輕鬆理解，請留意筆記、🧮 計算提示、圖示與其他助學工具。

### 前置條件

你現在應該已熟悉我們所探討的南瓜資料結構。本課程的 _notebook.ipynb_ 檔案中已預載並預先清理好資料。該檔案將南瓜價格以每蒲式耳價格顯示於新的資料框中。請確保可於 Visual Studio Code 的內核中順利執行這些筆記本。

### 準備工作

提醒你，載入這些資料是為了向它提出問題：

- 什麼時候買南瓜最好？  
- 一箱迷你南瓜價格會是多少？  
- 我應該買半蒲式耳籃裝還是一箱 1 1/9 蒲式耳盒裝的？  
讓我們繼續深入挖掘這些資料。

前一課你建立了一個 Pandas 資料框，並用原始資料集的一部分填充，將價格標準化為每蒲式耳價格。但這樣只能得到約 400 筆資料，且只涵蓋秋季的幾個月。

請查看本課程隨課程提供筆記本中預載的資料。資料已預載，且已繪製初始散點圖以顯示月份。也許我們可以透過更進一步的清理，來了解資料的本質。

## 線性迴歸線

正如你在第一課學到的，線性迴歸的目標是能繪製一條線：

- <strong>展現變數關係</strong>。展現變數間的關係  
- <strong>做出預測</strong>。準確預測新資料點相對於此線的位置  

最常用的做法是<strong>最小平方法迴歸</strong>（Least-Squares Regression）畫出這樣的一條線。“最小平方法”意指在模型中使誤差總和最小化的過程。對每一資料點，我們會測量該點與迴歸線間的垂直距離（稱為殘差）。

我們將這些距離平方，有兩個主要原因：

1. **只看大小，不看方向：** 我們想讓漏誤為 -5 與 +5 變成同樣的錯誤大小。平方可將所有數值轉為正數。

2. **懲罰離群點：** 平方會使離誤差較大的點權重更重，迫使線盡可能接近這些較遠的點。

接著，我們加總所有平方後的誤差，我們的目標是找到使此和達最小值的那條直線（故名「最小平方法」）。

> **🧮 數學說明**  
>  
> 這條線，稱為_最佳擬合線_，其表達式是[一個等式](https://en.wikipedia.org/wiki/Simple_linear_regression)：  
>  
> ```
> Y = a + bX
> ```
>  
> `X` 是「解釋變數」，`Y` 是「依變數」。線的斜率是 `b`，而 `a` 是 y 截距，也就是當 `X = 0` 時 `Y` 的值。  
>  
>![計算斜率](../../../../translated_images/zh-MO/slope.f3c9d5910ddbfcf9.webp)  
>  
> 首先計算斜率 `b`。圖解由 [Jen Looper](https://twitter.com/jenlooper) 提供  
>  
> 換句話說，針對我們南瓜資料的原始問題：「依照月份預測每蒲式耳南瓜價格」，`X` 是價格，`Y` 則是銷售的月份。  
>  
>![完成等式](../../../../translated_images/zh-MO/calculation.a209813050a1ddb1.webp)  
>  
> 計算 Y 的值。如果你付出約 $4，那一定是四月！圖解由 [Jen Looper](https://twitter.com/jenlooper) 提供  
>  
> 計算線的數學必須顯示線的斜率，同時取決於截距，即 `X = 0` 時 `Y` 的位置。  
>  
> 你可以在 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站看到這些數值的計算方式，也可拜訪[這個最小二乘計算器](https://www.mathsisfun.com/data/least-squares-calculator.html)來觀察數字如何影響線的表現。

## 相關性

另一個要理解的詞是給定 X 和 Y 變數間的<strong>相關係數</strong>。透過散佈圖，你可以快速視覺化此係數。點呈現筆直排列的散佈圖表示高度相關，但若資料點在 X 與 Y 間亂散則表示相關性低。

一個良好的線性迴歸模型會在最小平方法與迴歸線下，呈現較高（接近 1 而非 0）的相關係數。

✅ 執行本課程附帶的筆記本並查看「月份 vs 價格」的散佈圖。依據你對散佈圖的視覺判斷，南瓜銷售的月份與價格看來關聯性高還是低？如果不用 `Month` 而改用更細緻的量度指標，例如_一年中的日數_（即自年初後流逝的天數），這狀況會改變嗎？

下方程式碼中，我們假設已整理好資料，取得名為 `new_pumpkins` 的資料框，如下所示：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理資料的程式碼位於 [`notebook.ipynb`](notebook.ipynb) 檔案中。我們採用了前課的相同步驟，並用以下表達式計算 `DayOfYear` 欄位：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```
  
了解線性迴歸背後的數學後，我們來建立一個迴歸模型，看看是否可以預測哪種南瓜包裝擁有最佳南瓜價格。採買節慶南瓜的人可能想要這些資訊，以便優化購買南瓜包裝的決策。

## 尋找相關性

[![機器學習入門－尋找相關性：線性迴歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "機器學習入門－尋找相關性：線性迴歸的關鍵")

> 🎥 點擊上方影片縮圖，觀看相關性簡介。

從前一課你可能已看到不同月份的平均價格長這樣：

<img alt="月份平均價格" src="../../../../translated_images/zh-MO/barchart.a833ea9194346d76.webp" width="50%"/>

這暗示著一定存在某種相關性，我們可以嘗試訓練線性迴歸模型來預測 `Month` 與 `Price`，或是 `DayOfYear` 與 `Price` 的關係。下圖為後者的散點圖：

<img alt="價格與一年中日數的散佈圖" src="../../../../translated_images/zh-MO/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

讓我們用 `corr` 函式檢視是否存在相關性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```
  
看來相關性相當小，按 `Month` 計算是 -0.15，按 `DayOfYear` 是 -0.17，但可能有另一個重要關係—不同南瓜品種似乎對價格產生價格群聚現象。為了驗證此猜想，我們以不同顏色標示每種南瓜品種。藉由在 `scatter` 繪圖函式中傳入 `ax` 參數，我們能將所有點畫在同一張圖上：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```
  
<img alt="價格與一年中日數的散佈圖（彩色標示）" src="../../../../translated_images/zh-MO/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

研判發現品種對整體價格影響較大，而非實際銷售日期。我們可以用長條圖來看：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```
  
<img alt="品種與價格的長條圖" src="../../../../translated_images/zh-MO/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

目前我們先專注於一種南瓜品種，『派型』，看看日期對價格的影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格與一年中日數的散佈圖（派型）" src="../../../../translated_images/zh-MO/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

如果用 `corr` 函式計算 `Price` 與 `DayOfYear` 的相關性，大概會得到 `-0.27`，這代表訓練預測模型是有意義的。

> 在訓練線性迴歸模型之前，重要的是確保資料清潔。線性迴歸對缺失值表現不佳，因此建議先剔除所有空值：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```
  
另一個作法是用對應欄位的平均值填補這些空缺。

## 簡單線性迴歸

[![機器學習入門－使用 Scikit-learn 作線性與多項式迴歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "機器學習入門－使用 Scikit-learn 作線性與多項式迴歸")

> 🎥 點擊上方圖片，觀看線性與多項式迴歸簡介。

我們將採用<strong>Scikit-learn</strong>套件訓練線性迴歸模型。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```
  
首先將輸入特徵與預期輸出（標籤）分別存入 numpy 陣列：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```
  
> 注意，我們對輸入資料做了 `reshape`，讓線性迴歸套件能正確識別。線性迴歸期望輸入為 2D 陣列，其中每一列代表一組輸入特徵向量，因為此處只有一個特徵，故需一個大小為 N×1 的陣列，N 為資料筆數。

接著，我們需要將資料拆分為訓練與測試集，好在訓練後驗證模型成效：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
  
最後，訓練線性迴歸模型僅需兩行程式碼。我們宣告 `LinearRegression` 物件，並使用 `fit` 方法訓練：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 物件在進行 `fit` 後包含了所有回歸的係數，可以用 `.coef_` 屬性存取。在我們的例子中只有一個係數，大約是 `-0.017`。這表示價格隨時間似乎有輕微下降，但幅度不大，大約每天下降 2 分錢。我們也可以用 `lin_reg.intercept_` 存取回歸線與 Y 軸的交叉點——在我們的例子中約為 `21`，表示年初的價格。

要看看我們的模型有多準確，我們可以在測試資料集上預測價格，然後衡量預測值與期望值的接近程度。這可以用均方根誤差 (RMSE) 指標來完成，該指標是所有期望值與預測值平方差的平均後開根號。

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

我們的誤差大約是 2 點，約為 ~17%。這不算太好。另一個衡量模型品質的指標是 <strong>決定係數</strong>，可用以下程式取得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
當該值為 0 時，表示模型不考慮輸入資料，僅當作<em>最差線性預測</em>，即結果的平均值。當該值為 1 時，表示我們可以完美預測所有期望輸出。在我們的例子中，決定係數約為 0.06，偏低。

我們也可以將測試資料與回歸線繪圖，以更好地觀察回歸的效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-MO/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式回歸

線性回歸的另一種類型是多項式回歸。有時候變數之間呈線性關係——體積越大的南瓜價格越高——但有時這種關係無法用平面或直線描述。

✅ 這裡有[更多示例](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples)可以使用多項式回歸的資料

再看一次日期與價格的關係。這個散點圖看起來一定要用直線分析嗎？價格難道不會波動？這時可以嘗試多項式回歸。

✅ 多項式是可能包含一個或多個變數與係數的數學表達式

多項式回歸會產生一條曲線，以更貼合非線性資料。在我們的情況下，如果將平方的 `DayOfYear` 變數包含進輸入資料，應該能以一條抛物線擬合資料，該曲線在年中的某個點有一個最小值。

Scikit-learn 包含方便的[管道 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)來結合不同的資料處理步驟。<strong>管道</strong>是一連串的<strong>估計器</strong>。在我們的例子中，我們會建立一個先增加多項式特徵，再訓練迴歸模型的管道：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 表示我們會包含所有二次多項式。就我們例子來說，就是 `DayOfYear`<sup>2</sup>，但對兩個輸入變數 X 和 Y，會加入 X<sup>2</sup>、XY 和 Y<sup>2</sup>。如有需要，也可用更高次方多項式。

管道可像原始的 `LinearRegression` 物件一樣使用，也就是我們可以在管道上呼叫 `fit`，然後用 `predict` 取得預測結果：

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

要繪製平滑的擬合曲線，我們用 `np.linspace` 產生均勻間距的輸入值範圍，而非直接在無序的測試資料上繪製（那樣會呈現鋸齒狀線）：

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

這是包含測試資料和擬合曲線的圖：

<img alt="Polynomial regression" src="../../../../translated_images/zh-MO/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多項式回歸，我們能得到稍低的 RMSE 與稍高的決定係數，但差異不大。我們需要考慮其他特徵！

> 你可以看到，最低的南瓜價格大約出現在萬聖節左右。你怎麼解釋這個現象？

🎃 恭喜，你剛剛建立了一個可以幫助預測派南瓜價格的模型。你或許也可以用相同程序針對其他南瓜品種建立模型，但那會很繁瑣。現在讓我們學習如何在模型中考慮南瓜品種！

## 類別特徵

理想狀況下，我們希望用同一個模型預測不同南瓜品種的價格。然而，`Variety` 欄位與 `Month` 等欄位不同，它包含非數值資料。這類欄位稱為<strong>類別特徵</strong>。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上方圖片觀看使用類別特徵的線性回歸短片概述。

這裡顯示平均價格如何依品種變化：

<img alt="Average price by variety" src="../../../../translated_images/zh-MO/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考慮品種，我們首先需要將其轉換為數字形式，稱作<strong>編碼</strong>。有幾種方法：

* 簡單的<strong>數值編碼</strong>會建立一張各品種的表，然後用該表的索引取代品種名稱。這對線性回歸來說並不理想，因為線性回歸會直接將索引數值乘以係數相加，但在我們案例中索引號碼與價格的關係很明顯是非線性的，即使我們嘗試將索引做某種特定排序。
* <strong>獨熱編碼</strong>則會用 4 個不同欄位取代 `Variety`，每個欄位對應一個品種。如果該列是特定品種，該欄位為 `1`，反之為 `0`。這表示線性回歸會有四個係數，分別對應四個品種，代表該品種的“起始價”（或者說“額外價格”）。

以下程式示範如何對品種做獨熱編碼：

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

要用獨熱編碼品種作為輸入，訓練線性回歸，只要初始化 `X` 和 `y` 的資料正確即可：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其餘程式碼與前面訓練線性回歸時相同。你試試看就會發現均方誤差差不多，但決定係數大幅提高到約 77%。若想更精確，可以同時考慮更多類別特徵及數值特徵，例如 `Month` 或 `DayOfYear`。要合併成一個大特徵陣列，可以用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡我們也納入 `City` 和 `Package` 類別，得出的 RMSE 是 2.84（10.5%），決定係數為 0.94！

## 綜合應用

要建立最好的模型，我們可以用上述案例的合併（獨熱編碼的類別 + 數值）特徵資料，並搭配多項式回歸。以下是完整程式，方便參考：

```python
# 設定訓練數據
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 製作訓練-測試分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設定並訓練流程
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試數據結果
pred = pipeline.predict(X_test)

# 計算均方根誤差及決定係數
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

這應該會給我們接近 97% 的最高決定係數與 RMSE=2.23（約 8% 預測誤差）。

| 模型 | RMSE | 決定係數 |
|-------|-----|---------------|
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |
| 所有特徵 線性 | 2.84 (10.5%) | 0.94 |
| 所有特徵 多項式 | 2.23 (8.25%) | 0.97 |

🏆 幹得好！在一堂課中建立了四個回歸模型，並將模型品質提升到 97%。在回歸的最後一節，你將學習用於分類判斷的邏輯迴歸。

---
## 🚀挑戰

在此筆記本中測試不同變數，觀察其相關性與模型準確性的關係。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

本課涵蓋了線性回歸。其他重要的回歸類型還包括階層回歸、嶺回歸、Lasso 與彈性網路。想深入學習可以參考 [Stanford 統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作業 

[建立模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應視為權威來源。對於重要資訊，建議尋求專業人工作翻譯。我們對因使用本翻譯而產生的任何誤解或誤釋不承擔任何責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->