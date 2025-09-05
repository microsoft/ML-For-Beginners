<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T18:38:05+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "nl"
}
-->
# Bouw een regressiemodel met Scikit-learn: regressie op vier manieren

![Lineaire vs polynomiale regressie infographic](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is beschikbaar in R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introductie 

Tot nu toe heb je onderzocht wat regressie is met voorbeeldgegevens uit de dataset over pompoenprijzen die we gedurende deze les zullen gebruiken. Je hebt het ook gevisualiseerd met behulp van Matplotlib.

Nu ben je klaar om dieper in te gaan op regressie voor machine learning. Hoewel visualisatie je helpt om gegevens te begrijpen, komt de echte kracht van machine learning voort uit _het trainen van modellen_. Modellen worden getraind op historische gegevens om automatisch afhankelijkheden in gegevens vast te leggen, en ze stellen je in staat om uitkomsten te voorspellen voor nieuwe gegevens die het model nog niet eerder heeft gezien.

In deze les leer je meer over twee soorten regressie: _basis lineaire regressie_ en _polynomiale regressie_, samen met enkele wiskundige principes die aan deze technieken ten grondslag liggen. Met deze modellen kunnen we pompoenprijzen voorspellen op basis van verschillende invoergegevens.

[![ML voor beginners - Begrip van lineaire regressie](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML voor beginners - Begrip van lineaire regressie")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van lineaire regressie.

> Door deze hele cursus gaan we uit van minimale wiskundige kennis en proberen we het toegankelijk te maken voor studenten uit andere vakgebieden. Let daarom op notities, 🧮 oproepen, diagrammen en andere leermiddelen om het begrip te vergemakkelijken.

### Vereisten

Je zou inmiddels bekend moeten zijn met de structuur van de pompoengegevens die we onderzoeken. Je kunt deze gegevens vooraf geladen en schoongemaakt vinden in het _notebook.ipynb_-bestand van deze les. In het bestand wordt de pompoenprijs per bushel weergegeven in een nieuwe data frame. Zorg ervoor dat je deze notebooks kunt uitvoeren in kernels in Visual Studio Code.

### Voorbereiding

Ter herinnering: je laadt deze gegevens in om er vragen over te stellen. 

- Wanneer is de beste tijd om pompoenen te kopen? 
- Welke prijs kan ik verwachten voor een doos miniatuurpompoenen?
- Moet ik ze kopen in halve bushelmanden of in een 1 1/9 busheldoos?
Laten we verder graven in deze gegevens.

In de vorige les heb je een Pandas data frame gemaakt en gevuld met een deel van de oorspronkelijke dataset, waarbij je de prijzen standaardiseerde per bushel. Door dat te doen, kon je echter slechts ongeveer 400 datapunten verzamelen en alleen voor de herfstmaanden.

Bekijk de gegevens die we vooraf hebben geladen in het notebook dat bij deze les hoort. De gegevens zijn vooraf geladen en een eerste spreidingsdiagram is gemaakt om maandgegevens te tonen. Misschien kunnen we iets meer detail krijgen over de aard van de gegevens door ze verder te schonen.

## Een lineaire regressielijn

Zoals je hebt geleerd in Les 1, is het doel van een lineaire regressieoefening om een lijn te kunnen plotten om:

- **Variabele relaties te tonen**. De relatie tussen variabelen te tonen
- **Voorspellingen te maken**. Nauwkeurige voorspellingen te maken over waar een nieuw datapunt zou vallen in relatie tot die lijn. 
 
Het is typisch voor **Least-Squares Regression** om dit type lijn te tekenen. De term 'least-squares' betekent dat alle datapunten rondom de regressielijn worden gekwadrateerd en vervolgens opgeteld. Idealiter is die uiteindelijke som zo klein mogelijk, omdat we een laag aantal fouten willen, of `least-squares`. 

We doen dit omdat we een lijn willen modelleren die de minste cumulatieve afstand heeft tot al onze datapunten. We kwadrateren de termen voordat we ze optellen, omdat we ons bezighouden met de grootte ervan en niet met de richting.

> **🧮 Laat me de wiskunde zien** 
> 
> Deze lijn, de _lijn van beste fit_, kan worden uitgedrukt door [een vergelijking](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` is de 'verklarende variabele'. `Y` is de 'afhankelijke variabele'. De helling van de lijn is `b` en `a` is het snijpunt met de y-as, wat verwijst naar de waarde van `Y` wanneer `X = 0`. 
>
>![bereken de helling](../../../../2-Regression/3-Linear/images/slope.png)
>
> Bereken eerst de helling `b`. Infographic door [Jen Looper](https://twitter.com/jenlooper)
>
> Met andere woorden, en verwijzend naar de oorspronkelijke vraag over pompoengegevens: "voorspel de prijs van een pompoen per bushel per maand", zou `X` verwijzen naar de prijs en `Y` naar de verkoopmaand. 
>
>![voltooi de vergelijking](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Bereken de waarde van Y. Als je ongeveer $4 betaalt, moet het april zijn! Infographic door [Jen Looper](https://twitter.com/jenlooper)
>
> De wiskunde die de lijn berekent, moet de helling van de lijn aantonen, die ook afhankelijk is van het snijpunt, of waar `Y` zich bevindt wanneer `X = 0`.
>
> Je kunt de methode van berekening voor deze waarden bekijken op de [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) website. Bezoek ook [deze Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) om te zien hoe de waarden van de getallen de lijn beïnvloeden.

## Correlatie

Een andere term om te begrijpen is de **correlatiecoëfficiënt** tussen gegeven X- en Y-variabelen. Met een spreidingsdiagram kun je deze coëfficiënt snel visualiseren. Een diagram met datapunten die netjes in een lijn liggen, heeft een hoge correlatie, maar een diagram met datapunten die overal tussen X en Y verspreid zijn, heeft een lage correlatie.

Een goed lineair regressiemodel is er een met een hoge (dichter bij 1 dan bij 0) correlatiecoëfficiënt, gebruikmakend van de Least-Squares Regression-methode met een regressielijn.

✅ Voer het notebook uit dat bij deze les hoort en bekijk het spreidingsdiagram van Maand naar Prijs. Lijkt de data die Maand aan Prijs koppelt voor pompoenverkopen een hoge of lage correlatie te hebben, volgens jouw visuele interpretatie van het spreidingsdiagram? Verandert dat als je een meer verfijnde maatstaf gebruikt in plaats van `Maand`, bijvoorbeeld *dag van het jaar* (d.w.z. aantal dagen sinds het begin van het jaar)?

In de onderstaande code gaan we ervan uit dat we de gegevens hebben opgeschoond en een data frame hebben verkregen genaamd `new_pumpkins`, vergelijkbaar met het volgende:

ID | Maand | DagVanJaar | Soort | Stad | Verpakking | Lage Prijs | Hoge Prijs | Prijs
---|-------|------------|-------|------|------------|------------|------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel dozen | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel dozen | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel dozen | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel dozen | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel dozen | 15.0 | 15.0 | 13.636364

> De code om de gegevens te schonen is beschikbaar in [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). We hebben dezelfde schoonmaakstappen uitgevoerd als in de vorige les en hebben de kolom `DagVanJaar` berekend met behulp van de volgende expressie: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nu je een begrip hebt van de wiskunde achter lineaire regressie, laten we een regressiemodel maken om te zien of we kunnen voorspellen welk pakket pompoenen de beste pompoenprijzen zal hebben. Iemand die pompoenen koopt voor een feestelijke pompoenveld wil deze informatie misschien om zijn aankopen van pompoenpakketten voor het veld te optimaliseren.

## Op zoek naar correlatie

[![ML voor beginners - Op zoek naar correlatie: De sleutel tot lineaire regressie](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML voor beginners - Op zoek naar correlatie: De sleutel tot lineaire regressie")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van correlatie.

Uit de vorige les heb je waarschijnlijk gezien dat de gemiddelde prijs voor verschillende maanden er als volgt uitziet:

<img alt="Gemiddelde prijs per maand" src="../2-Data/images/barchart.png" width="50%"/>

Dit suggereert dat er enige correlatie zou moeten zijn, en we kunnen proberen een lineair regressiemodel te trainen om de relatie tussen `Maand` en `Prijs`, of tussen `DagVanJaar` en `Prijs` te voorspellen. Hier is het spreidingsdiagram dat de laatste relatie toont:

<img alt="Spreidingsdiagram van Prijs vs. Dag van het Jaar" src="images/scatter-dayofyear.png" width="50%" /> 

Laten we kijken of er een correlatie is met behulp van de `corr`-functie:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Het lijkt erop dat de correlatie vrij klein is, -0.15 bij `Maand` en -0.17 bij `DagVanJaar`, maar er kan een andere belangrijke relatie zijn. Het lijkt erop dat er verschillende clusters van prijzen zijn die overeenkomen met verschillende pompoensoorten. Om deze hypothese te bevestigen, laten we elke pompoencategorie plotten met een andere kleur. Door een `ax`-parameter door te geven aan de `scatter`-plotfunctie kunnen we alle punten op hetzelfde diagram plotten:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Spreidingsdiagram van Prijs vs. Dag van het Jaar" src="images/scatter-dayofyear-color.png" width="50%" /> 

Ons onderzoek suggereert dat de soort meer invloed heeft op de totale prijs dan de daadwerkelijke verkoopdatum. We kunnen dit zien met een staafdiagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Staafdiagram van prijs vs soort" src="images/price-by-variety.png" width="50%" /> 

Laten we ons voorlopig alleen richten op één pompoensoort, de 'pie type', en kijken welk effect de datum heeft op de prijs:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Spreidingsdiagram van Prijs vs. Dag van het Jaar" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Als we nu de correlatie tussen `Prijs` en `DagVanJaar` berekenen met behulp van de `corr`-functie, krijgen we iets als `-0.27` - wat betekent dat het trainen van een voorspellend model zinvol is.

> Voordat je een lineair regressiemodel traint, is het belangrijk om ervoor te zorgen dat onze gegevens schoon zijn. Lineaire regressie werkt niet goed met ontbrekende waarden, dus het is logisch om alle lege cellen te verwijderen:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Een andere aanpak zou zijn om die lege waarden in te vullen met gemiddelde waarden uit de corresponderende kolom.

## Eenvoudige lineaire regressie

[![ML voor beginners - Lineaire en polynomiale regressie met Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML voor beginners - Lineaire en polynomiale regressie met Scikit-learn")

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van lineaire en polynomiale regressie.

Om ons lineaire regressiemodel te trainen, gebruiken we de **Scikit-learn** bibliotheek.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

We beginnen met het scheiden van invoerwaarden (features) en de verwachte output (label) in afzonderlijke numpy-arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Merk op dat we `reshape` moesten uitvoeren op de invoergegevens zodat het lineaire regressiepakket ze correct begrijpt. Lineaire regressie verwacht een 2D-array als invoer, waarbij elke rij van de array overeenkomt met een vector van invoerfeatures. In ons geval, aangezien we slechts één invoer hebben, hebben we een array nodig met vorm N×1, waarbij N de datasetgrootte is.

Vervolgens moeten we de gegevens splitsen in trainings- en testdatasets, zodat we ons model na training kunnen valideren:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Het trainen van het daadwerkelijke lineaire regressiemodel kost uiteindelijk slechts twee regels code. We definiëren het `LinearRegression`-object en passen het aan onze gegevens aan met de `fit`-methode:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Het `LinearRegression`-object bevat na het toepassen van `fit` alle coëfficiënten van de regressie, die toegankelijk zijn via de `.coef_`-eigenschap. In ons geval is er slechts één coëfficiënt, die ongeveer `-0.017` zou moeten zijn. Dit betekent dat de prijzen met de tijd lijken te dalen, maar niet veel, ongeveer 2 cent per dag. We kunnen ook het snijpunt van de regressie met de Y-as benaderen met `lin_reg.intercept_` - dit zal ongeveer `21` zijn in ons geval, wat de prijs aan het begin van het jaar aangeeft.

Om te zien hoe nauwkeurig ons model is, kunnen we prijzen voorspellen op een testdataset en vervolgens meten hoe dicht onze voorspellingen bij de verwachte waarden liggen. Dit kan worden gedaan met behulp van de mean square error (MSE)-metriek, die het gemiddelde is van alle gekwadrateerde verschillen tussen de verwachte en voorspelde waarde.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Onze fout lijkt te liggen rond 2 punten, wat ongeveer 17% is. Niet al te goed. Een andere indicator van modelkwaliteit is de **coëfficiënt van determinatie**, die als volgt kan worden verkregen:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Als de waarde 0 is, betekent dit dat het model geen rekening houdt met invoergegevens en fungeert als de *slechtste lineaire voorspeller*, wat simpelweg een gemiddelde waarde van het resultaat is. De waarde 1 betekent dat we alle verwachte uitkomsten perfect kunnen voorspellen. In ons geval is de coëfficiënt ongeveer 0,06, wat vrij laag is.

We kunnen ook de testgegevens samen met de regressielijn plotten om beter te zien hoe regressie in ons geval werkt:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Lineaire regressie" src="images/linear-results.png" width="50%" />

## Polynomiale Regressie  

Een ander type lineaire regressie is polynomiale regressie. Hoewel er soms een lineaire relatie is tussen variabelen - hoe groter de pompoen in volume, hoe hoger de prijs - kunnen deze relaties soms niet worden weergegeven als een vlak of rechte lijn.  

✅ Hier zijn [enkele voorbeelden](https://online.stat.psu.edu/stat501/lesson/9/9.8) van gegevens die polynomiale regressie kunnen gebruiken  

Bekijk de relatie tussen Datum en Prijs nog eens. Lijkt deze scatterplot noodzakelijkerwijs te moeten worden geanalyseerd met een rechte lijn? Kunnen prijzen niet fluctueren? In dit geval kun je polynomiale regressie proberen.  

✅ Polynomen zijn wiskundige uitdrukkingen die kunnen bestaan uit één of meer variabelen en coëfficiënten  

Polynomiale regressie creëert een gebogen lijn om niet-lineaire gegevens beter te passen. In ons geval, als we een kwadratische `DayOfYear`-variabele toevoegen aan de invoergegevens, zouden we onze gegevens moeten kunnen passen met een parabool, die een minimum heeft op een bepaald punt in het jaar.  

Scikit-learn bevat een handige [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) om verschillende stappen van gegevensverwerking samen te voegen. Een **pipeline** is een keten van **schatters**. In ons geval zullen we een pipeline maken die eerst polynomiale kenmerken toevoegt aan ons model en vervolgens de regressie traint:  

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Het gebruik van `PolynomialFeatures(2)` betekent dat we alle tweedegraads polynomen uit de invoergegevens zullen opnemen. In ons geval betekent dit gewoon `DayOfYear`<sup>2</sup>, maar gegeven twee invoervariabelen X en Y, zal dit X<sup>2</sup>, XY en Y<sup>2</sup> toevoegen. We kunnen ook polynomen van hogere graden gebruiken als we dat willen.  

Pipelines kunnen op dezelfde manier worden gebruikt als het oorspronkelijke `LinearRegression`-object, d.w.z. we kunnen de pipeline `fitten` en vervolgens `predict` gebruiken om de voorspelde resultaten te krijgen. Hier is de grafiek die testgegevens en de benaderingscurve toont:  

<img alt="Polynomiale regressie" src="images/poly-results.png" width="50%" />  

Met polynomiale regressie kunnen we een iets lagere MSE en hogere determinatie krijgen, maar niet significant. We moeten rekening houden met andere kenmerken!  

> Je kunt zien dat de minimale pompoenprijzen ergens rond Halloween worden waargenomen. Hoe kun je dit verklaren?  

🎃 Gefeliciteerd, je hebt zojuist een model gemaakt dat kan helpen de prijs van taartpompoenen te voorspellen. Je kunt waarschijnlijk dezelfde procedure herhalen voor alle pompoensoorten, maar dat zou tijdrovend zijn. Laten we nu leren hoe we pompoenvariëteiten in ons model kunnen opnemen!  

## Categorische Kenmerken  

In een ideale wereld willen we prijzen voor verschillende pompoenvariëteiten kunnen voorspellen met hetzelfde model. De kolom `Variety` is echter enigszins anders dan kolommen zoals `Month`, omdat deze niet-numerieke waarden bevat. Dergelijke kolommen worden **categorisch** genoemd.  

[![ML voor beginners - Categorische Kenmerken Voorspellingen met Lineaire Regressie](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML voor beginners - Categorische Kenmerken Voorspellingen met Lineaire Regressie")  

> 🎥 Klik op de afbeelding hierboven voor een korte video-overzicht van het gebruik van categorische kenmerken.  

Hier kun je zien hoe de gemiddelde prijs afhankelijk is van de variëteit:  

<img alt="Gemiddelde prijs per variëteit" src="images/price-by-variety.png" width="50%" />  

Om rekening te houden met variëteit, moeten we deze eerst omzetten naar numerieke vorm, of **coderen**. Er zijn verschillende manieren waarop we dit kunnen doen:  

* Eenvoudige **numerieke codering** zal een tabel maken van verschillende variëteiten en vervolgens de variëteitsnaam vervangen door een index in die tabel. Dit is niet de beste optie voor lineaire regressie, omdat lineaire regressie de werkelijke numerieke waarde van de index neemt en deze toevoegt aan het resultaat, vermenigvuldigd met een coëfficiënt. In ons geval is de relatie tussen het indexnummer en de prijs duidelijk niet-lineair, zelfs als we ervoor zorgen dat indices op een specifieke manier worden gerangschikt.  
* **One-hot encoding** zal de kolom `Variety` vervangen door 4 verschillende kolommen, één voor elke variëteit. Elke kolom bevat `1` als de corresponderende rij van een bepaalde variëteit is, en `0` anders. Dit betekent dat er vier coëfficiënten zullen zijn in lineaire regressie, één voor elke pompoenvariëteit, verantwoordelijk voor de "startprijs" (of eerder "extra prijs") voor die specifieke variëteit.  

De onderstaande code laat zien hoe we een variëteit kunnen one-hot encoden:  

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

Om lineaire regressie te trainen met one-hot encoded variëteit als invoer, hoeven we alleen de `X`- en `y`-gegevens correct te initialiseren:  

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

De rest van de code is hetzelfde als wat we hierboven hebben gebruikt om lineaire regressie te trainen. Als je het probeert, zul je zien dat de gemiddelde kwadratische fout ongeveer hetzelfde is, maar we krijgen een veel hogere coëfficiënt van determinatie (~77%). Om nog nauwkeurigere voorspellingen te krijgen, kunnen we meer categorische kenmerken in rekening brengen, evenals numerieke kenmerken zoals `Month` of `DayOfYear`. Om één grote array van kenmerken te krijgen, kunnen we `join` gebruiken:  

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Hier nemen we ook `City` en `Package` type in rekening, wat ons een MSE van 2,84 (10%) en een determinatie van 0,94 geeft!  

## Alles samenvoegen  

Om het beste model te maken, kunnen we gecombineerde (one-hot encoded categorische + numerieke) gegevens uit het bovenstaande voorbeeld gebruiken samen met polynomiale regressie. Hier is de volledige code voor je gemak:  

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

Dit zou ons de beste determinatiecoëfficiënt van bijna 97% moeten geven, en MSE=2,23 (~8% voorspellingsfout).  

| Model | MSE | Determinatie |  
|-------|-----|---------------|  
| `DayOfYear` Lineair | 2,77 (17,2%) | 0,07 |  
| `DayOfYear` Polynomiaal | 2,73 (17,0%) | 0,08 |  
| `Variety` Lineair | 5,24 (19,7%) | 0,77 |  
| Alle kenmerken Lineair | 2,84 (10,5%) | 0,94 |  
| Alle kenmerken Polynomiaal | 2,23 (8,25%) | 0,97 |  

🏆 Goed gedaan! Je hebt vier regressiemodellen gemaakt in één les en de modelkwaliteit verbeterd tot 97%. In het laatste deel over regressie leer je over logistische regressie om categorieën te bepalen.  

---  
## 🚀Uitdaging  

Test verschillende variabelen in dit notebook om te zien hoe correlatie overeenkomt met modelnauwkeurigheid.  

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)  

## Review & Zelfstudie  

In deze les hebben we geleerd over lineaire regressie. Er zijn andere belangrijke soorten regressie. Lees over Stepwise, Ridge, Lasso en Elasticnet technieken. Een goede cursus om meer te leren is de [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)  

## Opdracht  

[Maak een model](assignment.md)  

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.