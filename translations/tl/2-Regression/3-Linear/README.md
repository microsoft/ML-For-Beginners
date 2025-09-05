<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T18:08:24+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "tl"
}
-->
# Gumawa ng regression model gamit ang Scikit-learn: regression sa apat na paraan

![Linear vs polynomial regression infographic](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ang araling ito ay available sa R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Panimula

Sa ngayon, napag-aralan mo na kung ano ang regression gamit ang sample data mula sa dataset ng presyo ng kalabasa na gagamitin natin sa buong araling ito. Na-visualize mo na rin ito gamit ang Matplotlib.

Ngayon, handa ka nang mas lumalim sa regression para sa ML. Bagama't ang visualization ay tumutulong upang maunawaan ang data, ang tunay na lakas ng Machine Learning ay nagmumula sa _pagsasanay ng mga modelo_. Ang mga modelo ay sinasanay gamit ang makasaysayang data upang awtomatikong makuha ang mga dependency ng data, at nagbibigay-daan ito upang mahulaan ang mga resulta para sa bagong data na hindi pa nakikita ng modelo.

Sa araling ito, matututunan mo ang higit pa tungkol sa dalawang uri ng regression: _basic linear regression_ at _polynomial regression_, kasama ang ilang math na nasa likod ng mga teknik na ito. Ang mga modelong ito ay magbibigay-daan sa atin upang mahulaan ang presyo ng kalabasa batay sa iba't ibang input data.

[![ML para sa mga nagsisimula - Pag-unawa sa Linear Regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML para sa mga nagsisimula - Pag-unawa sa Linear Regression")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng linear regression.

> Sa buong kurikulum na ito, inaasahan namin ang minimal na kaalaman sa math, at sinisikap naming gawing accessible ito para sa mga estudyanteng nagmumula sa ibang larangan, kaya't maghanap ng mga tala, 🧮 callouts, diagram, at iba pang learning tools upang makatulong sa pag-unawa.

### Paunang Kaalaman

Dapat ay pamilyar ka na sa istruktura ng data ng kalabasa na ating sinusuri. Makikita mo itong preloaded at pre-cleaned sa _notebook.ipynb_ file ng araling ito. Sa file, ang presyo ng kalabasa ay ipinapakita kada bushel sa isang bagong data frame. Siguraduhing maipatakbo mo ang mga notebook na ito sa kernels sa Visual Studio Code.

### Paghahanda

Bilang paalala, niloload mo ang data na ito upang magtanong tungkol dito.

- Kailan ang pinakamagandang oras para bumili ng kalabasa?
- Anong presyo ang maaasahan ko para sa isang case ng miniature pumpkins?
- Dapat ko bang bilhin ang mga ito sa half-bushel baskets o sa 1 1/9 bushel box?
Patuloy nating tuklasin ang data na ito.

Sa nakaraang aralin, gumawa ka ng Pandas data frame at pinunan ito ng bahagi ng orihinal na dataset, na-standardize ang presyo kada bushel. Gayunpaman, sa paggawa nito, nakakuha ka lamang ng humigit-kumulang 400 datapoints at para lamang sa mga buwan ng taglagas.

Tingnan ang data na preloaded sa notebook na kasama ng araling ito. Ang data ay preloaded at isang paunang scatterplot ang na-chart upang ipakita ang data ng buwan. Marahil makakakuha tayo ng mas detalyadong impormasyon tungkol sa kalikasan ng data sa pamamagitan ng mas malalim na paglilinis nito.

## Isang linear regression line

Tulad ng natutunan mo sa Lesson 1, ang layunin ng isang linear regression exercise ay makapag-plot ng linya upang:

- **Ipakita ang relasyon ng mga variable**. Ipakita ang relasyon sa pagitan ng mga variable
- **Gumawa ng mga prediksyon**. Gumawa ng tumpak na prediksyon kung saan mahuhulog ang isang bagong datapoint kaugnay ng linya.

Karaniwan sa **Least-Squares Regression** ang pagguhit ng ganitong uri ng linya. Ang terminong 'least-squares' ay nangangahulugan na ang lahat ng datapoints na nakapalibot sa regression line ay pinapangkat at pagkatapos ay ina-add. Sa ideal na sitwasyon, ang huling kabuuan ay dapat na kasing liit hangga't maaari, dahil gusto natin ng mababang bilang ng mga error, o `least-squares`.

Ginagawa natin ito dahil gusto nating magmodelo ng linya na may pinakamaliit na kabuuang distansya mula sa lahat ng ating data points. Pinapangkat din natin ang mga termino bago i-add ang mga ito dahil mas mahalaga sa atin ang magnitude kaysa sa direksyon nito.

> **🧮 Ipakita ang math**
>
> Ang linyang ito, na tinatawag na _line of best fit_, ay maaaring ipahayag sa pamamagitan ng [isang equation](https://en.wikipedia.org/wiki/Simple_linear_regression):
>
> ```
> Y = a + bX
> ```
>
> Ang `X` ay ang 'explanatory variable'. Ang `Y` ay ang 'dependent variable'. Ang slope ng linya ay `b` at ang `a` ay ang y-intercept, na tumutukoy sa halaga ng `Y` kapag `X = 0`.
>
>![calculate the slope](../../../../2-Regression/3-Linear/images/slope.png)
>
> Una, kalkulahin ang slope `b`. Infographic ni [Jen Looper](https://twitter.com/jenlooper)
>
> Sa madaling salita, at tumutukoy sa orihinal na tanong ng data ng kalabasa: "hulaan ang presyo ng kalabasa kada bushel ayon sa buwan", ang `X` ay tumutukoy sa presyo at ang `Y` ay tumutukoy sa buwan ng pagbebenta.
>
>![complete the equation](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Kalkulahin ang halaga ng Y. Kung nagbabayad ka ng humigit-kumulang $4, malamang Abril ito! Infographic ni [Jen Looper](https://twitter.com/jenlooper)
>
> Ang math na nagkakalkula ng linya ay dapat ipakita ang slope ng linya, na nakadepende rin sa intercept, o kung saan nakaposisyon ang `Y` kapag `X = 0`.
>
> Maaari mong obserbahan ang paraan ng pagkalkula para sa mga halagang ito sa [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) web site. Bisitahin din ang [Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) upang makita kung paano nakakaapekto ang mga halaga ng numero sa linya.

## Correlation

Isa pang terminong dapat maunawaan ay ang **Correlation Coefficient** sa pagitan ng mga X at Y variables. Gamit ang scatterplot, mabilis mong ma-visualize ang coefficient na ito. Ang plot na may datapoints na maayos na nakahanay ay may mataas na correlation, ngunit ang plot na may datapoints na kalat-kalat sa pagitan ng X at Y ay may mababang correlation.

Ang isang mahusay na linear regression model ay ang may mataas (mas malapit sa 1 kaysa sa 0) Correlation Coefficient gamit ang Least-Squares Regression method na may linya ng regression.

✅ Patakbuhin ang notebook na kasama ng araling ito at tingnan ang scatterplot ng Month to Price. Ang data ba na nag-uugnay sa Month to Price para sa pagbebenta ng kalabasa ay mukhang may mataas o mababang correlation, ayon sa iyong visual na interpretasyon ng scatterplot? Nagbabago ba ito kung gumamit ka ng mas detalyadong sukat sa halip na `Month`, halimbawa *day of the year* (i.e. bilang ng mga araw mula sa simula ng taon)?

Sa code sa ibaba, ipagpalagay natin na nalinis na natin ang data, at nakakuha ng data frame na tinatawag na `new_pumpkins`, na katulad ng sumusunod:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Ang code para linisin ang data ay makikita sa [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Ginawa namin ang parehong mga hakbang sa paglilinis tulad ng sa nakaraang aralin, at kinakalkula ang `DayOfYear` column gamit ang sumusunod na expression:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Ngayon na nauunawaan mo ang math sa likod ng linear regression, gumawa tayo ng Regression model upang makita kung kaya nating hulaan kung aling package ng kalabasa ang may pinakamagandang presyo. Ang isang taong bumibili ng kalabasa para sa holiday pumpkin patch ay maaaring gusto ng impormasyong ito upang ma-optimize ang kanilang pagbili ng mga package ng kalabasa para sa patch.

## Paghahanap ng Correlation

[![ML para sa mga nagsisimula - Paghahanap ng Correlation: Ang Susi sa Linear Regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML para sa mga nagsisimula - Paghahanap ng Correlation: Ang Susi sa Linear Regression")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng correlation.

Mula sa nakaraang aralin, marahil nakita mo na ang average na presyo para sa iba't ibang buwan ay ganito:

<img alt="Average price by month" src="../2-Data/images/barchart.png" width="50%"/>

Ipinapakita nito na maaaring may correlation, at maaari nating subukang sanayin ang linear regression model upang hulaan ang relasyon sa pagitan ng `Month` at `Price`, o sa pagitan ng `DayOfYear` at `Price`. Narito ang scatter plot na nagpapakita ng huling relasyon:

<img alt="Scatter plot of Price vs. Day of Year" src="images/scatter-dayofyear.png" width="50%" /> 

Tingnan natin kung may correlation gamit ang `corr` function:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Mukhang maliit ang correlation, -0.15 sa `Month` at -0.17 sa `DayOfMonth`, ngunit maaaring may isa pang mahalagang relasyon. Mukhang may iba't ibang clusters ng presyo na tumutugma sa iba't ibang pumpkin varieties. Upang kumpirmahin ang hypothesis na ito, i-plot natin ang bawat kategorya ng kalabasa gamit ang iba't ibang kulay. Sa pamamagitan ng pagpasa ng `ax` parameter sa `scatter` plotting function, maaari nating i-plot ang lahat ng puntos sa parehong graph:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatter plot of Price vs. Day of Year" src="images/scatter-dayofyear-color.png" width="50%" /> 

Ang ating pagsisiyasat ay nagpapahiwatig na ang variety ay may mas malaking epekto sa kabuuang presyo kaysa sa aktwal na petsa ng pagbebenta. Makikita natin ito gamit ang bar graph:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Bar graph of price vs variety" src="images/price-by-variety.png" width="50%" /> 

Tumutok muna tayo sa isang pumpkin variety, ang 'pie type', at tingnan kung anong epekto ang petsa sa presyo:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatter plot of Price vs. Day of Year" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Kung kalkulahin natin ngayon ang correlation sa pagitan ng `Price` at `DayOfYear` gamit ang `corr` function, makakakuha tayo ng humigit-kumulang `-0.27` - na nangangahulugang may saysay ang pagsasanay ng predictive model.

> Bago sanayin ang linear regression model, mahalagang tiyakin na malinis ang ating data. Hindi mahusay gumagana ang linear regression sa mga nawawalang halaga, kaya't makatuwiran na alisin ang lahat ng empty cells:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Ang isa pang paraan ay punan ang mga nawawalang halaga gamit ang mean values mula sa kaukulang column.

## Simple Linear Regression

[![ML para sa mga nagsisimula - Linear at Polynomial Regression gamit ang Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML para sa mga nagsisimula - Linear at Polynomial Regression gamit ang Scikit-learn")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng linear at polynomial regression.

Upang sanayin ang ating Linear Regression model, gagamit tayo ng **Scikit-learn** library.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Magsimula tayo sa paghihiwalay ng input values (features) at ang inaasahang output (label) sa magkahiwalay na numpy arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Tandaan na kailangan nating i-perform ang `reshape` sa input data upang maunawaan ito nang tama ng Linear Regression package. Ang Linear Regression ay inaasahan ang isang 2D-array bilang input, kung saan ang bawat row ng array ay tumutugma sa isang vector ng input features. Sa ating kaso, dahil mayroon lamang tayong isang input - kailangan natin ng array na may hugis N×1, kung saan ang N ay ang laki ng dataset.

Pagkatapos, kailangan nating hatiin ang data sa train at test datasets, upang ma-validate natin ang ating modelo pagkatapos ng pagsasanay:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Sa wakas, ang pagsasanay ng aktwal na Linear Regression model ay nangangailangan lamang ng dalawang linya ng code. I-define natin ang `LinearRegression` object, at i-fit ito sa ating data gamit ang `fit` method:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Ang `LinearRegression` object pagkatapos ng `fit`-ting ay naglalaman ng lahat ng coefficients ng regression, na maaaring ma-access gamit ang `.coef_` property. Sa ating kaso, mayroon lamang isang coefficient, na dapat ay nasa paligid ng `-0.017`. Ibig sabihin, ang mga presyo ay tila bumababa nang kaunti sa paglipas ng panahon, ngunit hindi masyado, humigit-kumulang 2 cents kada araw. Maaari rin nating ma-access ang intersection point ng regression sa Y-axis gamit ang `lin_reg.intercept_` - ito ay nasa paligid ng `21` sa ating kaso, na nagpapahiwatig ng presyo sa simula ng taon.

Upang makita kung gaano katumpak ang ating modelo, maaari nating hulaan ang mga presyo sa test dataset, at pagkatapos ay sukatin kung gaano kalapit ang ating mga prediksyon sa inaasahang mga halaga. Magagawa ito gamit ang mean square error (MSE) metrics, na siyang mean ng lahat ng squared differences sa pagitan ng inaasahan at prediktadong halaga.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Ang ating error ay tila nasa paligid ng 2 puntos, na ~17%. Hindi masyadong maganda. Isa pang indikasyon ng kalidad ng modelo ay ang **coefficient of determination**, na maaaring makuha sa ganitong paraan:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Kung ang halaga ay 0, nangangahulugan ito na ang modelo ay hindi isinasaalang-alang ang input data, at kumikilos bilang *pinakamasamang linear predictor*, na simpleng mean value ng resulta. Ang halaga na 1 ay nangangahulugan na maaari nating perpektong mahulaan ang lahat ng inaasahang output. Sa ating kaso, ang coefficient ay nasa paligid ng 0.06, na medyo mababa.

Maaari rin nating i-plot ang test data kasama ang regression line upang mas makita kung paano gumagana ang regression sa ating kaso:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Linear regression" src="images/linear-results.png" width="50%" />

## Polynomial Regression

Isa pang uri ng Linear Regression ay ang Polynomial Regression. Bagamat minsan may linear na relasyon sa pagitan ng mga variable - mas malaki ang volume ng kalabasa, mas mataas ang presyo - minsan ang mga relasyong ito ay hindi maaaring i-plot bilang isang plane o tuwid na linya.

✅ Narito ang [ilang mga halimbawa](https://online.stat.psu.edu/stat501/lesson/9/9.8) ng data na maaaring gumamit ng Polynomial Regression.

Tingnan muli ang relasyon sa pagitan ng Petsa at Presyo. Mukha bang ang scatterplot na ito ay dapat na suriin gamit ang tuwid na linya? Hindi ba maaaring magbago-bago ang mga presyo? Sa ganitong kaso, maaari mong subukan ang polynomial regression.

✅ Ang mga polynomial ay mga matematikal na ekspresyon na maaaring binubuo ng isa o higit pang mga variable at coefficients.

Ang Polynomial Regression ay lumilikha ng kurbadong linya upang mas maayos na magkasya sa nonlinear na data. Sa ating kaso, kung isasama natin ang squared na `DayOfYear` variable sa input data, dapat nating magawang i-fit ang ating data gamit ang isang parabolic curve, na magkakaroon ng minimum sa isang tiyak na punto sa loob ng taon.

Ang Scikit-learn ay may kasamang kapaki-pakinabang na [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) upang pagsamahin ang iba't ibang hakbang ng data processing. Ang **pipeline** ay isang chain ng **estimators**. Sa ating kaso, gagawa tayo ng pipeline na unang magdadagdag ng polynomial features sa ating modelo, at pagkatapos ay magsasanay ng regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Ang paggamit ng `PolynomialFeatures(2)` ay nangangahulugan na isasama natin ang lahat ng second-degree polynomials mula sa input data. Sa ating kaso, ito ay nangangahulugan lamang ng `DayOfYear`<sup>2</sup>, ngunit kung may dalawang input variables na X at Y, ito ay magdadagdag ng X<sup>2</sup>, XY, at Y<sup>2</sup>. Maaari rin tayong gumamit ng mas mataas na degree na polynomials kung nais natin.

Ang mga pipeline ay maaaring gamitin sa parehong paraan tulad ng orihinal na `LinearRegression` object, ibig sabihin maaari nating `fit` ang pipeline, at pagkatapos ay gamitin ang `predict` upang makuha ang mga resulta ng prediksyon. Narito ang graph na nagpapakita ng test data, at ang approximation curve:

<img alt="Polynomial regression" src="images/poly-results.png" width="50%" />

Sa paggamit ng Polynomial Regression, maaari tayong makakuha ng bahagyang mas mababang MSE at mas mataas na determination, ngunit hindi gaanong malaki. Kailangan nating isaalang-alang ang iba pang mga features!

> Makikita mo na ang pinakamababang presyo ng kalabasa ay napapansin sa paligid ng Halloween. Paano mo ito maipapaliwanag?

🎃 Binabati kita, nakagawa ka ng modelo na makakatulong sa paghulaan ang presyo ng pie pumpkins. Maaari mong ulitin ang parehong proseso para sa lahat ng uri ng kalabasa, ngunit magiging nakakapagod iyon. Alamin natin ngayon kung paano isama ang iba't ibang uri ng kalabasa sa ating modelo!

## Categorical Features

Sa ideal na mundo, nais nating mahulaan ang mga presyo para sa iba't ibang uri ng kalabasa gamit ang parehong modelo. Gayunpaman, ang column na `Variety` ay medyo naiiba sa mga column tulad ng `Month`, dahil naglalaman ito ng non-numeric na mga halaga. Ang ganitong mga column ay tinatawag na **categorical**.

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")

> 🎥 I-click ang imahe sa itaas para sa maikling video overview ng paggamit ng categorical features.

Narito makikita mo kung paano nakadepende ang average na presyo sa variety:

<img alt="Average price by variety" src="images/price-by-variety.png" width="50%" />

Upang isama ang variety, kailangan muna nating i-convert ito sa numeric form, o **encode** ito. May ilang paraan upang gawin ito:

* Ang simpleng **numeric encoding** ay gagawa ng table ng iba't ibang varieties, at pagkatapos ay papalitan ang pangalan ng variety ng index sa table na iyon. Hindi ito ang pinakamahusay na ideya para sa linear regression, dahil ang linear regression ay gumagamit ng aktwal na numeric value ng index, at idinadagdag ito sa resulta, na pinararami ng ilang coefficient. Sa ating kaso, ang relasyon sa pagitan ng index number at presyo ay malinaw na hindi linear, kahit na tiyakin natin na ang mga indices ay nakaayos sa isang partikular na paraan.
* Ang **one-hot encoding** ay papalitan ang column na `Variety` ng 4 na magkakaibang column, isa para sa bawat variety. Ang bawat column ay maglalaman ng `1` kung ang kaukulang row ay ng isang partikular na variety, at `0` kung hindi. Nangangahulugan ito na magkakaroon ng apat na coefficients sa linear regression, isa para sa bawat uri ng kalabasa, na responsable para sa "starting price" (o "additional price") para sa partikular na variety.

Ang code sa ibaba ay nagpapakita kung paano natin mai-one-hot encode ang variety:

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

Upang magsanay ng linear regression gamit ang one-hot encoded variety bilang input, kailangan lang nating i-initialize ang `X` at `y` data nang tama:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

Ang natitirang code ay pareho sa ginamit natin sa itaas upang magsanay ng Linear Regression. Kung susubukan mo ito, makikita mo na ang mean squared error ay halos pareho, ngunit nakakakuha tayo ng mas mataas na coefficient of determination (~77%). Upang makakuha ng mas tumpak na prediksyon, maaari nating isama ang mas maraming categorical features, pati na rin ang numeric features, tulad ng `Month` o `DayOfYear`. Upang makakuha ng isang malaking array ng features, maaari nating gamitin ang `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Dito isinama rin natin ang `City` at `Package` type, na nagbibigay sa atin ng MSE 2.84 (10%), at determination 0.94!

## Pagsasama-sama ng Lahat

Upang makagawa ng pinakamahusay na modelo, maaari nating gamitin ang pinagsamang (one-hot encoded categorical + numeric) data mula sa halimbawa sa itaas kasama ang Polynomial Regression. Narito ang kumpletong code para sa iyong kaginhawahan:

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

Dapat nitong ibigay sa atin ang pinakamahusay na determination coefficient na halos 97%, at MSE=2.23 (~8% prediction error).

| Modelo | MSE | Determination |  
|-------|-----|---------------|  
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |  
| `Variety` Linear | 5.24 (19.7%) | 0.77 |  
| All features Linear | 2.84 (10.5%) | 0.94 |  
| All features Polynomial | 2.23 (8.25%) | 0.97 |  

🏆 Magaling! Nakagawa ka ng apat na Regression models sa isang aralin, at napabuti ang kalidad ng modelo sa 97%. Sa huling seksyon ng Regression, matututo ka tungkol sa Logistic Regression upang matukoy ang mga kategorya.

---

## 🚀Hamunin

Subukan ang ilang iba't ibang mga variable sa notebook na ito upang makita kung paano tumutugma ang correlation sa accuracy ng modelo.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Sa araling ito natutunan natin ang tungkol sa Linear Regression. May iba pang mahahalagang uri ng Regression. Basahin ang tungkol sa Stepwise, Ridge, Lasso, at Elasticnet techniques. Isang magandang kurso na maaaring pag-aralan upang matuto pa ay ang [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Takdang Aralin

[Magbuo ng Modelo](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.