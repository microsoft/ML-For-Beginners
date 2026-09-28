# 使用 Scikit-learn 建立迴歸模型：迴歸四種方法

## 初學者提醒

當我們想要預測<strong>數值型的值</strong>（例如房價、溫度或銷售額）時，就會用到線性迴歸。
線性迴歸透過尋找一條最佳代表輸入特徵與輸出之間關係的直線來工作。

本課程將先著重了解概念，再探索更進階的迴歸技術。
![線性與多項式迴歸資訊圖](../../../../translated_images/zh-TW/linear-polynomial.5523c7cb6576ccab.webp)
> 資訊圖表作者 [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 語言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### 介紹

到目前為止，你已經使用我們將在本課中持續使用的南瓜價格資料集探索了迴歸是什麼。你也使用 Matplotlib 視覺化了資料。

現在你已準備好深入探討機器學習中的迴歸。雖然視覺化幫助理解資料，機器學習的真正強大在於_訓練模型_。模型會基於歷史資料自動捕捉資料依賴關係，並允許你對模型未曾見過的新資料進行預測。

本課將進一步介紹兩種迴歸：_基本線性迴歸_和_多項式迴歸_，並說明這些技術背後的部分數學理論。這些模型將讓我們能依據不同的輸入資料預測南瓜價格。

[![初學者機器學習 - 理解線性迴歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "初學者機器學習 - 理解線性迴歸")

> 🎥 點擊上方圖片觀看線性迴歸簡介短片。

> 在整個課程中，我們假設數學知識最低限度，並致力於讓來自不同領域的學生都能理解，因此請留意注記、🧮 數學提示、圖解與其他學習工具。

### 先備知識

你現在應該對我們檢視的南瓜資料結構有所熟悉。本課將在 _notebook.ipynb_ 檔案中預先載入與預先清理該資料。此檔中，南瓜價格會以每蒲式耳價格顯示在新的資料框中。請確保你能在 Visual Studio Code 的 kernel 中執行這些 notebooks。

### 準備

提醒你，我們載入這些資料是為了向它提出問題：

- 何時是買南瓜的最佳時機？
- 我能預期迷你南瓜整箱的價格是多少？
- 我應該買半蒲式耳籃裝，還是 1又1/9蒲式耳箱裝？
讓我們繼續探究這些資料。

上一課你建立了 Pandas 資料框，並使用了原始資料部分，只蒐集了秋季月份且標準化至每蒲式耳價格，因而僅取得約 400 筆資料。

請參考本課附帶 notebook 中預載的資料。資料已預先載入並繪出初始散點圖以顯示月份資料。也許透過更細部的清理，可以更了解該資料特性。

## 線性迴歸線

正如你在第一課所學，線性迴歸的目標是繪製一條直線，以：

- <strong>展示變數關聯性</strong>：展示變數間的關係
- <strong>做出預測</strong>：準確預測新資料點相對於該條線的位置

<strong>最小平方法迴歸</strong> 會繪製這類線。術語「最小平方法」指的是我們透過最小化模型中總誤差的過程。對每個資料點，我們測量實際點和迴歸線間的垂直距離（稱為殘差）。

這些距離的平方有兩大理由：

1. <strong>誤差大小重於方向</strong>：我們希望將-5與+5的誤差視同。平方會把所有值變為正數。

2. <strong>懲罰離群點</strong>：平方會給予較大的誤差更高權重，迫使直線貼近遠離的點。

接著加總所有平方值。我們目標是找出使此總和最小的那條直線，故稱「最小平方法」。

> **🧮 數學說明**
> 
> 此直線稱為_最佳擬合線_，可用[方程式](https://en.wikipedia.org/wiki/Simple_linear_regression)表示：
> 
> ```
> Y = a + bX
> ```
>
> `X` 為「解釋變數」，`Y` 為「依賴變數」。直線斜率為 `b`，`a` 是截距，指當 `X=0` 時 `Y` 的值。
>
>![計算斜率](../../../../translated_images/zh-TW/slope.f3c9d5910ddbfcf9.webp)
>
> 首先計算斜率 `b`。資訊圖表作者 [Jen Looper](https://twitter.com/jenlooper)
>
> 換句話說，針對我們南瓜資料的問題：「依月份預測每蒲式耳南瓜價格」，`X` 代表價格，`Y` 代表銷售月份。
>
>![完成方程式](../../../../translated_images/zh-TW/calculation.a209813050a1ddb1.webp)
>
> 計算 Y 的值。如果價格約為 4 美元，那應該是四月！資訊圖表作者 [Jen Looper](https://twitter.com/jenlooper)
>
> 計算該直線的數學公式必須反映斜率，且斜率依賴截距，即當 `X=0` 時 `Y` 的位置。
>
> 你可至 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站查看這些值的計算方法，亦可使用[此最小平方法計算器](https://www.mathsisfun.com/data/least-squares-calculator.html)觀看數字值如何影響直線。

## 相關性

另一句需要了解的術語是給定 X 和 Y 變數間的<strong>相關係數</strong>。使用散點圖，你可直觀了解此係數。資料點排列整齊成一條線的散點圖代表高度相關，而資料點四散無序的散點圖則相關性較低。

良好的線性迴歸模型將具有用最小平方法計算出的較高（接近 1 而非 0）的相關係數。

✅ 執行本課附帶的 notebook，觀察「月份對價格」的散點圖。依你對散點圖的視覺判斷，月份與南瓜價格關聯性是高還是低？如果改用更精細的時間測量（例如一年中某天，表示自年初以來的天數）相關性是否改變？

以下程式中，我們假設已清理資料，並取得名為 `new_pumpkins` 的資料框，格式類似如下：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理資料的程式碼請見 [`notebook.ipynb`](notebook.ipynb)。我們已執行和上一課相同的清理步驟，並用以下表達式計算了 `DayOfYear` 欄位： 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

現在你了解線性迴歸背後的數學，我們來建立一個迴歸模型，看看是否能預測哪種包裝的南瓜價格最佳。買南瓜裝飾節慶專用的人可能會希望藉此資訊優化他們的購買配置。

## 尋找相關性

[![初學者機器學習 - 尋找相關性：線性迴歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "初學者機器學習 - 尋找相關性：線性迴歸的關鍵")

> 🎥 點擊上方圖片觀看相關性簡介短片。

從上一課你可能看過不同月份的平均價格如下：

<img alt="依月份顯示的平均價格" src="../../../../translated_images/zh-TW/barchart.a833ea9194346d76.webp" width="50%"/>

這暗示應該存在某種相關性，我們可以嘗試用線性迴歸模型預測`Month`與`Price`、或`DayOfYear`與`Price`的關係。以下是後者的散點圖：

<img alt="價格對應一年中的天數散點圖" src="../../../../translated_images/zh-TW/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

我們來用 `corr` 函數檢視相關性：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

似乎相關係數很小，`Month` 為 -0.15，`DayOfYear` 為 -0.17，但可能存在另一重要關係。看起來不同南瓜品種的價格形成了不同群集。為驗證此假設，我們將用不同顏色繪製每個南瓜類別。並透過傳入 `ax` 參數給 `scatter` 函數，將所有點繪在同一圖表：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格對應一年中的天數並以顏色分品種散點圖" src="../../../../translated_images/zh-TW/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

調查顯示品種比銷售日期對價格影響更大。這點可見於長條圖：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="以南瓜品種為分類的價格長條圖" src="../../../../translated_images/zh-TW/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

暫時我們只聚焦於「派型」南瓜，觀察日期對價格的影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格對應一年中的天數（派型南瓜）的散點圖" src="../../../../translated_images/zh-TW/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

若計算 `Price` 與 `DayOfYear` 的相關係數，結果約為 `-0.27`——這表示訓練預測模型是合理的。

> 在訓練線性迴歸模型之前，請務必確保資料乾淨。線性迴歸對缺失值較不穩定，因此刪除所有空值較為適宜：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種方法是用對應欄位的平均值取代空值。

## 簡單線性迴歸

[![初學者機器學習 - 使用 Scikit-learn 的線性與多項式迴歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "初學者機器學習 - 使用 Scikit-learn 的線性與多項式迴歸")

> 🎥 點擊上方圖片觀看線性與多項式迴歸簡介短片。

我們使用 **Scikit-learn** 函式庫來訓練線性迴歸模型。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

我們先把輸入值（特徵）與期望輸出（標籤）分開，放入獨立的 numpy 陣列：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意需對輸入資料做 `reshape`，讓線性迴歸套件正確理解。線性迴歸期望輸入為二維陣列，陣列每列為一組輸入特徵向量。我們此例僅有單一輸入，故需為 N×1 的矩陣，其中 N 是資料集大小。

接著，我們要將資料分割成訓練集與測試集，以便訓練後驗證模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練線性迴歸模型僅要兩行程式。定義 `LinearRegression` 物件，並用 `fit` 方法對資料做擬合：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 物件在完成 `fit` 後，會包含迴歸的所有係數，可以透過 `.coef_` 屬性取得。在我們的例子中，只有一個係數，該係數應該約為 `-0.017`。這表示價格似乎隨時間略微下降，但幅度不大，約為每天 2 分錢。我們也可以透過 `lin_reg.intercept_` 取得迴歸線與 Y 軸的截距點——在我們的例子中，約為 `21`，代表一年的開始時的價格。

為了檢視我們模型的準確度，我們可以在測試資料集上預測價格，然後衡量預測值與期望值的接近程度。這可以使用均方根誤差 (RMSE) 指標來完成，它是期望值與預測值所有平方差的平均值的平方根。

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

我們的誤差約為 2 點，約佔 17%。表現不是很好。另一個模型品質指標是<strong>決定係數</strong>，可以這樣取得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```

如果該值為 0，表示模型不考慮輸入資料，並以<em>最差線性預測器</em>運作，簡單來說就是結果的平均值。值為 1 表示我們能完全準確預測所有期望輸出。在我們的例子中，決定係數約為 0.06，十分低。

我們也可以將測試資料與迴歸線一起繪圖，來更好地觀察迴歸結果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-TW/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式迴歸

另一種線性迴歸稱為多項式迴歸。有時候變數間是線性關係——例如南瓜的體積越大，價格越高，但有時這種關係無法用平面或直線來描述。

✅ 這裡有[更多範例](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples)適合使用多項式迴歸的資料

再看一次日期與價格的關係。這個散佈圖看起來一定要用直線來分析嗎？價格不會波動的嗎？在這種情況下，你可以嘗試多項式迴歸。

✅ 多項式是可能含有一個或多個變數與係數的數學表達式

多項式迴歸會創建曲線，以更好地擬合非線性資料。在我們的例子中，如果將 `DayOfYear` 的平方加入輸入資料，我們應該可以用拋物線擬合數據，該曲線會在年度某個點達到最低點。

Scikit-learn 提供了一個方便的[pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)用來串接不同的資料處理步驟。**pipeline** 是一連串的<strong>估計器</strong>。在本例中，我們將建立一個 pipeline，先加入多項式特徵，再訓練迴歸：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 表示我們將包含所有二次多項式特徵。對於本例，就是 `DayOfYear`<sup>2</sup>，但若有兩個輸入變數 X 和 Y，則包含 X<sup>2</sup>、XY 及 Y<sup>2</sup>。若需要，也可使用更高階多項式。

pipeline 使用方式跟原本的 `LinearRegression` 物件相同，也就是可以 `fit` pipeline，然後用 `predict` 取得預測結果：

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

為了繪製平滑的近似曲線，我們使用 `np.linspace` 創建均勻的輸入值範圍，而不是直接在無序的測試資料上繪圖（後者會產生鋸齒狀線條）：

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

下圖顯示測試資料和近似曲線：

<img alt="Polynomial regression" src="../../../../translated_images/zh-TW/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多項式迴歸，我們可獲得稍低的 RMSE 與較高的決定係數，但改進不大。我們還需要考慮其他特徵！

> 你可以看到南瓜價格最低點大約出現在萬聖節附近。你怎麼解釋這個現象？

🎃 恭喜，你已經建立了一個能預測派形南瓜價格的模型。你或許可以用相同程序來處理所有南瓜品種，不過那很繁瑣。接著讓我們學習如何在模型中考量南瓜品種！

## 類別特徵

理想情況下，我們希望用相同模型預測不同南瓜品種的價格。然而 `Variety` 欄位有別於像 `Month` 這類的欄位，因為它包含非數字值。這類欄位稱為<strong>類別特徵</strong>。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上方圖片可觀看關於如何使用類別特徵的短影片。

這裡可以看到平均價格如何依品種而異：

<img alt="Average price by variety" src="../../../../translated_images/zh-TW/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

要考慮品種，我們首先需要將其轉為數值形式，也就是<strong>編碼</strong>。有幾種方法：

* 簡單的<strong>數字編碼</strong>會建立一張品種表，並將品種名稱替換為表中的索引。在線性迴歸中這不是最佳方案，因為線性迴歸使用索引的數值，乘以某個係數後再加入結果。以我們的例子來說，品種索引與價格的關係明顯非線性，即便我們確保索引依某個有意義的順序排列亦然。
* **獨熱編碼 (One-hot encoding)** 會將 `Variety` 欄替換成 4 個欄位，分別對應每個品種。每個欄位若該行屬於對應品種，值為 `1`，否則為 `0`。這表示線性迴歸中會有四個係數，分別代表各品種的「基本價格」（或可理解為「額外價格」）。

以下程式碼示範如何將品種獨熱編碼：

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

要用獨熱編碼品種作為輸入訓練線性迴歸，只要正確初始化 `X` 與 `y` 資料即可：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其餘程式碼同前面訓練線性迴歸時使用的。如果你嘗試執行，會發現均方誤差差不多，但決定係數大幅提升至約 77%。要讓預測更準確，我們能考慮更多類別特徵和數值特徵，例如 `Month` 或 `DayOfYear`。若要得到一個包含所有特徵的大陣列，可以用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡我們還加入了 `City` 和包裝方式 `Package`，結果 RMSE 降至 2.84（10.5%），決定係數達到 0.94！

## 組合應用

為了打造最佳模型，我們可以將上述例子的組合數據（獨熱編碼類別加上數值特徵）與多項式迴歸結合。以下是完整程式碼：

```python
# 設置訓練數據
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 進行訓練測試分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設置並訓練流程
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試數據結果
pred = pipeline.predict(X_test)

# 計算均方根誤差和決定係數
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

這會給出幾乎 97% 的最佳決定係數，以及 2.23 的 RMSE（約 8% 預測誤差）。

| 模型 | RMSE | 決定係數 |
|-------|-----|---------------|
| `DayOfYear` 線性 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性 | 5.24 (19.7%) | 0.77 |
| 全特徵 線性 | 2.84 (10.5%) | 0.94 |
| 全特徵 多項式 | 2.23 (8.25%) | 0.97 |

🏆 表現非常好！你在一堂課中建立了四個迴歸模型，並將模型質量提升到 97%。在迴歸的最後一個章節，你將學習邏輯迴歸，用來判斷類別。

---
## 🚀挑戰

在這個筆記本中測試多個不同變數，觀察相關性與模型準確度間的關係。

## [課後小考](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

這堂課我們學習了線性迴歸。還有其他重要的迴歸類型。請閱讀 Stepwise、Ridge、Lasso 與 Elasticnet 技術。推薦的進階課程是[史丹佛大學統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作業

[建立模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意自動翻譯可能包含錯誤或不準確之處。原始文件的本地語言版本應視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯所產生的任何誤解或曲解負責。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->