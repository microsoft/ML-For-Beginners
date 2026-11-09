# 使用 Scikit-learn 建立迴歸模型：四種迴歸方式

## 初學者筆記

當我們想要預測<strong>數值</strong>（例如房屋價格、溫度或銷售額）時，會使用線性迴歸。它是透過尋找最能代表輸入特徵與輸出之間關係的直線來進行預測。

本課程將先專注於理解概念，稍後再探討更進階的迴歸技術。  
![線性與多項式迴歸資訊圖](../../../../translated_images/zh-HK/linear-polynomial.5523c7cb6576ccab.webp)  
> 資訊圖表由 [Dasani Madipalli](https://twitter.com/dasani_decoded) 製作

## [課前小測驗](https://ff-quizzes.netlify.app/en/ml/)

> ### [本課程也提供 R 語言版本！](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)  
### 導言

到目前為止，你已經用南瓜定價資料集探討了迴歸概念，並使用 Matplotlib 進行視覺化。

現在你準備深入了解 ML 中的迴歸應用。視覺化讓你理解資料，真正的機器學習威力來自於「訓練模型」。模型會基於歷史資料自動捕捉資料關聯，並能對未見過的新資料進行預測。

本課程將介紹兩種迴歸方式：＿基本線性迴歸＿和＿多項式迴歸＿，並涵蓋這些技術背後的數學原理。這些模型將幫助我們依據不同輸入資料預測南瓜價格。

[![初學者機器學習 - 理解線性迴歸](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "初學者機器學習 - 理解線性迴歸")

> 🎥 點擊上方圖片觀看線性迴歸的簡短影片介紹。

> 本課程假設極少的數學背景，並致力讓非相關領域的學生也能理解，請留意課堂筆記、🧮 數學補充、圖表等學習輔助工具。

### 先修知識

你現在應熟悉我們分析的南瓜資料結構。資料已預先載入並清理，存於本課程的 _notebook.ipynb_。在該檔案中，南瓜價格以「每蒲式耳」價格呈現。請確定你能在 Visual Studio Code 的 kernel 執行這些 notebook。

### 準備工作

提醒你，載入資料是為了向它提出問題：

- 什麼時候買南瓜最划算？
- 一箱迷你南瓜大約多少錢？
- 我該買半蒲式耳籃裝還是 1 1/9 蒲式耳箱裝？
讓我們持續深入挖掘資料。

上一課中，你建立了 Pandas dataframe 並用原始數據部分填充，還根據蒲式耳標準化了價格。不過，那時僅蒐集約 400 筆資料，也只涵蓋秋季月份。

請看看本課 notebook 中預載的資料。資料已預先載入，並藉由散佈圖初步描繪出月份資料。也許我們可以進一步清理，以更細緻了解資料本質。

## 線性迴歸直線

如第一課所學，線性迴歸的目的是畫出一條線，用以：

- <strong>顯示變數關係</strong>。展示變數間的關聯。
- <strong>進行預測</strong>。對新數據點相較這條線的落點做出精準預測。

此類線通常由<strong>最小平方法回歸</strong>繪出。「最小平方法」指的是最小化模型總誤差的過程。對每筆資料點，測量實際點與迴歸線間的垂直距離（稱作殘差）。

這些距離會被平方，理由有二：

1. <strong>大小勝過方向</strong>：我們希望把 -5 與 +5 的誤差同等對待，平方後皆轉為正值。

2. <strong>懲罰離群點</strong>：平方強化較大誤差權重，迫使線條更貼近遠離的點。

接著將所有平方值加總。目標是找到能使平方誤差加總最低（最小值）的線，因此稱為「最小平方法」。

> **🧮 數學公式展現**  
>  
> 此線稱為＿最佳擬合線＿，可用[以下方程式表達](https://en.wikipedia.org/wiki/Simple_linear_regression)：  
>  
> ```
> Y = a + bX
> ```
>
> `X` 是「解釋變數」，`Y` 是「被解釋變數」。線的斜率是 `b`，`a` 是 y 軸截距，即當 `X = 0` 時的 `Y` 值。  
>
>![計算斜率](../../../../translated_images/zh-HK/slope.f3c9d5910ddbfcf9.webp)
>
> 先計算斜率 `b`。資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作。
>
> 換句話說，根據我們的南瓜數據的問題：「依月份預測每蒲式耳南瓜價格」，`X` 指價格，`Y` 指銷售月份。
>
>![完成公式](../../../../translated_images/zh-HK/calculation.a209813050a1ddb1.webp)
>
> 計算 Y 的值。如果你付了約 4 美元，那一定是四月！資訊圖表由 [Jen Looper](https://twitter.com/jenlooper) 製作。
>
> 計算線條的數學必須反映出斜率，也取決於截距，即當 `X = 0` 時 `Y` 的位置。
>
> 你可參考 [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) 網站了解計算方法，也可以使用[這個最小平方法計算器](https://www.mathsisfun.com/data/least-squares-calculator.html) 觀察數值如何影響線條。

## 相關係數

另一重要概念是給定 X 和 Y 變數間的<strong>相關係數</strong>。利用散佈圖，能快速看出相關輕重。若資料點沿一條整齊線排列，相關程度高；若資料點在 X、Y 間散布成零散狀，則相關度低。

一個良好的線性迴歸模型，會有接近 1（比 0 大得多）的高相關係數，並採用最小平方法找到最適回歸線。

✅ 執行本課附帶的 notebook，觀察「月份對價格」的散佈圖。依你目測，南瓜銷售的「月份／價格」資料是高相關還是低相關？換用更細膩的時間單位，如「年中第幾天」（Day of Year），結果會改變嗎？

下面程式碼示範假設資料已清理乾淨，並取得名為 `new_pumpkins` 的資料框，內容大致如下：

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> 清理資料的程式碼可在 [`notebook.ipynb`](notebook.ipynb) 中找到。我們做了和前一課相同的清理步驟，`DayOfYear` 欄位是用以下公式計算的：

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

理解線性迴歸背後的數學後，讓我們來建立一個迴歸模型，看看能否預測哪種包裝的南瓜價格較低。要在節慶南瓜園採購南瓜時，這資訊有助於優化選購方案。

## 尋找相關性

[![初學者機器學習 - 尋找相關性：線性迴歸的關鍵](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "初學者機器學習 - 尋找相關性：線性迴歸的關鍵")

> 🎥 點擊圖片觀看相關性的簡短影片介紹。

前一課你可能已見過不同月份平均價格如下圖：

<img alt="各月份平均價格" src="../../../../translated_images/zh-HK/barchart.a833ea9194346d76.webp" width="50%"/>

這代表應該會有某種相關性，我們可以嘗試訓練線性迴歸模型，去預測 `Month` 與 `Price` 或 `DayOfYear` 與 `Price` 之間的關係。下圖是後者的散佈圖：

<img alt="價格與年中日散佈圖" src="../../../../translated_images/zh-HK/scatter-dayofyear.bc171c189c9fd553.webp" width="50%" /> 

讓我們用 `corr` 函式檢視相關係數：

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

看起來相關性偏低，`Month` 約為 -0.15，`DayOfYear` 約 -0.17，但似乎有其他重要因素。不同南瓜品種對應不同價格群聚。為驗證，將每種南瓜用不同顏色繪圖。傳入 `ax` 參數至 `scatter` 繪圖函式可同時在同張圖上繪出所有點：

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="價格與年中日散佈圖（彩色品種區分）" src="../../../../translated_images/zh-HK/scatter-dayofyear-color.65790faefbb9d54f.webp" width="50%" /> 

結果顯示品種對價格影響較大，勝過實際銷售日期。下圖柱狀圖呈現此結果：

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="不同品種價格柱狀圖" src="../../../../translated_images/zh-HK/price-by-variety.744a2f9925d9bcb4.webp" width="50%" /> 

暫時只聚焦於南瓜品種中的「派用型」（pie type），觀察日期對價格的影響：

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="價格與年中日散佈圖（派用型）" src="../../../../translated_images/zh-HK/pie-pumpkins-scatter.d14f9804a53f927e.webp" width="50%" /> 

接著以 `corr` 函式計算 `Price` 與 `DayOfYear` 的相關係數，可能得出約 `-0.27`，顯示訓練預測模型是有意義的。

> 訓練線性迴歸模型前，務必確認資料是乾淨的。線性迴歸不適用於遺失值，因此應排除所有空白欄位：

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

另一種做法是用該欄位平均值填補空白資料。

## 簡單線性迴歸

[![初學者機器學習 - 使用 Scikit-learn 的線性與多項式迴歸](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "初學者機器學習 - 使用 Scikit-learn 的線性與多項式迴歸")

> 🎥 點擊圖片觀看線性及多項式迴歸的簡短影片介紹。

訓練線性迴歸模型時，我們會使用 **Scikit-learn** 函式庫。

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

我們先將輸入特徵（features）與預期輸出（label）分別存為兩個 numpy 陣列：

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> 注意我們必須對輸入資料做 `reshape`，讓 Linear Regression 套件能識別。線性迴歸期待輸入為二維陣列，每列為一組特徵向量。因為我們只有一個輸入變數，須將陣列形狀調整為 N×1，其中 N 是資料筆數。

接著，將資料切分成訓練集與測試集，方便訓練後評估模型：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

最後，訓練線性迴歸模型只要兩行程式碼。定義 `LinearRegression` 物件，並用 `fit` 方法套用資料：

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` 對象在 `fit` 之後包含了迴歸的所有係數，可以通過 `.coef_` 屬性訪問。在我們的例子中，只有一個係數，應該約為 `-0.017`。這意味著價格隨時間略有下降，但不多，大約每天下降 2 分錢。我們還可以通過 `lin_reg.intercept_` 訪問迴歸與 Y 軸的截距 — 在我們的例子中約為 `21`，表示年初的價格。

為了檢視模型的準確度，我們可以在測試數據集上預測價格，然後測量預測結果與期望值的接近程度。這可以使用均方根誤差 (RMSE) 指標計算，即期望值與預測值所有平方差均值的平方根。

```python
pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')
```

我們的誤差約為 2 點，約為 17%。表現不是很好。另一個評估模型質量的指標是<strong>決定係數</strong>，可以這樣獲得：

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
 如果值為 0，表示模型沒有考慮輸入數據，表現為<em>最差線性預測器</em>，即結果的平均值。值為 1 表示我們可以完美預測所有預期輸出。在我們的例子中，係數約為 0.06，相當低。

我們還可以將測試數據與迴歸線一起繪圖，更清楚地觀察迴歸效果：

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linear regression" src="../../../../translated_images/zh-HK/linear-results.f7c3552c85b0ed1c.webp" width="50%" />

## 多項式迴歸

另一種線性迴歸類型是多項式迴歸。有時候變量之間是線性關係 — 譬如南瓜體積越大，價格越高 — 但有時這些關係無法用平面或直線表示。

✅ 這裡有一些[更多範例](https://online.stat.psu.edu/stat501/Lesson09#polynomial-regression-examples)適合使用多項式迴歸的資料

再看看日期與價格的關係。散點圖是否一定要用直線分析？價格難道不會波動？此時可以嘗試多項式迴歸。

✅ 多項式是可能包含一個或多個變量及係數的數學表達式

多項式迴歸會擬合曲線以更好地適應非線性資料。在我們的例子中，如果將平方的 `DayOfYear` 變量加入輸入資料，就能用拋物線擬合數據，且拋物線會在年度中的某點有最小值。

Scikit-learn 提供了實用的[管道 API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline)，用來串聯資料處理的各步驟。<strong>管道</strong>是一連串的<strong>估計器</strong>。在此，我們會先產生多項式特徵，再培訓迴歸模型：

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

使用 `PolynomialFeatures(2)` 表示我們將包括輸入數據中所有二次多項式。對我們來說，只有 `DayOfYear`<sup>2</sup>，但如果有兩個輸入變量 X 與 Y，則會加上 X<sup>2</sup>、XY 與 Y<sup>2</sup>。如果想，也可以使用更高次數的多項式。

管道的用法與原本的 `LinearRegression` 物件相同，即先 `fit` 管道，接著用 `predict` 得到預測結果：

```python
pred = pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,pred))
print(f'RMSE: {rmse:3.3} ({rmse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

為了繪製平滑擬合曲線，我們使用 `np.linspace` 產生均勻分布的輸入值，而非直接在無序的測試資料中繪圖（否則線條會鋸齒狀）：

```python
X_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
y_range = pipeline.predict(X_range)

plt.scatter(X_test, y_test)
plt.plot(X_range, y_range)
```

以下是測試資料與擬合曲線的圖示：

<img alt="Polynomial regression" src="../../../../translated_images/zh-HK/poly-results.ee587348f0f1f60b.webp" width="50%" />

使用多項式迴歸，我們可以得到稍低的 RMSE 與較高的決定係數，但幅度有限。我們還需要考慮其他特徵！

> 你可以看到最低的南瓜價格出現在萬聖節附近。你怎麼解釋這個現象？ 

🎃 恭喜你剛完成可以預測派用南瓜價格的模型。你大概也可以用同樣流程對其他南瓜種類進行建模，但會很費時。接下來讓我們學習如何將南瓜品種納入模型！

## 類別特徵

理想情況下，我們希望能用同一模型預測不同南瓜品種的價格。但 `Variety` 欄位與 `Month` 欄位不同，因為它包含非數字的值。這類欄位稱為<strong>類別欄位(categorical)</strong>。

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 點擊上圖觀看使用類別特徵的簡短影片。

這裡可以看到不同品種的平均價格：

<img alt="Average price by variety" src="../../../../translated_images/zh-HK/price-by-variety.744a2f9925d9bcb4.webp" width="50%" />

為了考慮品種，我們首先要將其轉換成數值形式，即<strong>編碼</strong>。有幾種方法：

* 簡單的 <strong>數值編碼</strong> 是建立一張不同品種的索引表，再將品種名以該表的索引數替代。這對線性迴歸不算好，因為線性迴歸直接用該數字乘上係數，然後加到結果中。我們的例子中，索引與價格的關係明顯非線性，即使我們確保索引有特定順序。
* **One-hot 編碼** 會將 `Variety` 欄位拆成四個不同欄位，對應四種品種。每個欄位若該行是此品種則為 `1`，否則為 `0`。這會產生四個線性迴歸係數，分別代表每種南瓜品種的「基準價格」（或者更準確來說，是「額外價格」）。

下面的程式碼展示如何對品種使用 one-hot 編碼：

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

要用 one-hot 編碼品種的資料訓練線性迴歸，我們只要正確初始化 `X` 和 `y`：

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

其餘程式碼與我們之前用來訓練線性迴歸的相同。如果你嘗試，你會發現均方誤差差不多，但決定係數大幅提升到約 77%。要得到更準確的預測，我們還可以考慮更多類別型特徵及數值型特徵，如 `Month` 或 `DayOfYear`。為了合併成一個特徵陣列，可使用 `join`：

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

這裡我們還加入了 `City` 和 `Package` 類型，結果是 RMSE 2.84（10.5%）與決定係數 0.94！

## 全面整合

要做出最佳模型，我們可以用上述合併的（one-hot 編碼類別 + 數值）特徵資料，同時套用多項式迴歸。方便起見，這是完整程式碼：

```python
# 設置訓練數據
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# 製作訓練和測試數據分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 設置及訓練管道
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# 預測測試數據結果
pred = pipeline.predict(X_test)

# 計算均方根誤差及判定係數
rmse = mean_squared_error(y_test, pred, squared=False)
print(f'RMSE: {rmse:3.3} ({rmse/pred.mean()*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

這應該會得到接近 97% 的最高決定係數，RMSE=2.23（約 8% 的預測誤差）。

| 模型 | RMSE | 決定係數 |
|-------|-----|---------|
| `DayOfYear` 線性迴歸 | 2.77 (17.2%) | 0.07 |
| `DayOfYear` 多項式迴歸 | 2.73 (17.0%) | 0.08 |
| `Variety` 線性迴歸 | 5.24 (19.7%) | 0.77 |
| 所有特徵 線性迴歸 | 2.84 (10.5%) | 0.94 |
| 所有特徵 多項式迴歸 | 2.23 (8.25%) | 0.97 |

🏆 做得好！你在一課中創建了四個迴歸模型，並將模型質量提升至 97%。在迴歸的最後部分，你將學習用來判別類別的邏輯迴歸。

---
## 🚀 挑戰

在這個筆記本中嘗試幾個不同變量，看看它們的相關性如何對模型準確度造成影響。

## [課後小測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學

本課我們學到了線性迴歸。還有其他重要類型的迴歸。可以閱讀關於逐步、嶺回歸 (Ridge)、套索 (Lasso) 和彈性網路 (Elasticnet) 的技術。推薦研習的課程是[史丹佛統計學習課程](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)

## 作業

[建立模型](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：  
本文件係使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保翻譯準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件之原文版本應被視為權威來源。對於重要資訊，建議採用專業人工翻譯。我們不對因使用本翻譯而引致的任何誤解或誤譯承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->