<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T21:06:20+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "sv"
}
-->
# Bygg en regressionsmodell med Scikit-learn: regression på fyra sätt

![Infografik om linjär vs polynomisk regression](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Den här lektionen finns tillgänglig i R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduktion 

Hittills har du utforskat vad regression är med exempeldata från pumpaprisdatamängden som vi kommer att använda genom hela denna lektion. Du har också visualiserat den med hjälp av Matplotlib.

Nu är du redo att fördjupa dig i regression för maskininlärning. Medan visualisering hjälper dig att förstå data, ligger den verkliga styrkan i maskininlärning i _att träna modeller_. Modeller tränas på historisk data för att automatiskt fånga datadependenser, och de gör det möjligt att förutsäga resultat för ny data som modellen inte har sett tidigare.

I denna lektion kommer du att lära dig mer om två typer av regression: _grundläggande linjär regression_ och _polynomisk regression_, tillsammans med lite av matematiken bakom dessa tekniker. Dessa modeller kommer att hjälpa oss att förutsäga pumpapriser baserat på olika indata.

[![ML för nybörjare - Förstå linjär regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML för nybörjare - Förstå linjär regression")

> 🎥 Klicka på bilden ovan för en kort videogenomgång av linjär regression.

> Genom hela detta läroplan antar vi minimal kunskap om matematik och strävar efter att göra det tillgängligt för studenter från andra områden, så håll utkik efter anteckningar, 🧮 matematiska inslag, diagram och andra lärverktyg för att underlätta förståelsen.

### Förkunskaper

Du bör nu vara bekant med strukturen för pumpadatan som vi undersöker. Du hittar den förladdad och förberedd i denna lektions _notebook.ipynb_-fil. I filen visas pumpapriset per skäppa i en ny dataram. Se till att du kan köra dessa notebooks i kärnor i Visual Studio Code.

### Förberedelse

Som en påminnelse laddar du denna data för att kunna ställa frågor om den. 

- När är det bästa tillfället att köpa pumpor? 
- Vilket pris kan jag förvänta mig för en låda med miniatyrpumpor?
- Ska jag köpa dem i halvskäppakorgar eller i lådor med 1 1/9 skäppa?
Låt oss fortsätta att gräva i denna data.

I den föregående lektionen skapade du en Pandas-dataram och fyllde den med en del av den ursprungliga datamängden, standardiserade prissättningen per skäppa. Genom att göra det kunde du dock bara samla in cirka 400 datapunkter och endast för höstmånaderna.

Ta en titt på datan som vi har förladdat i denna lektions medföljande notebook. Datan är förladdad och ett initialt spridningsdiagram är skapat för att visa månatlig data. Kanske kan vi få lite mer detaljer om datans natur genom att rengöra den ytterligare.

## En linjär regressionslinje

Som du lärde dig i Lektion 1 är målet med en linjär regressionsövning att kunna rita en linje för att:

- **Visa variabelrelationer**. Visa relationen mellan variabler
- **Göra förutsägelser**. Göra korrekta förutsägelser om var en ny datapunkt skulle hamna i förhållande till den linjen. 
 
Det är typiskt för **Least-Squares Regression** att dra denna typ av linje. Termen 'least-squares' betyder att alla datapunkter runt regressionslinjen kvadreras och sedan adderas. Idealiskt sett är den slutliga summan så liten som möjligt, eftersom vi vill ha ett lågt antal fel, eller `least-squares`. 

Vi gör detta eftersom vi vill modellera en linje som har den minsta kumulativa avståndet från alla våra datapunkter. Vi kvadrerar också termerna innan vi adderar dem eftersom vi är intresserade av dess storlek snarare än dess riktning.

> **🧮 Visa mig matematiken** 
> 
> Denna linje, kallad _linjen för bästa passform_, kan uttryckas med [en ekvation](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` är den 'förklarande variabeln'. `Y` är den 'beroende variabeln'. Lutningen på linjen är `b` och `a` är y-interceptet, vilket hänvisar till värdet på `Y` när `X = 0`. 
>
>![beräkna lutningen](../../../../2-Regression/3-Linear/images/slope.png)
>
> Först, beräkna lutningen `b`. Infografik av [Jen Looper](https://twitter.com/jenlooper)
>
> Med andra ord, och med hänvisning till vår ursprungliga fråga om pumpadata: "förutsäg priset på en pumpa per skäppa efter månad", skulle `X` hänvisa till priset och `Y` hänvisa till försäljningsmånaden. 
>
>![slutför ekvationen](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Beräkna värdet på Y. Om du betalar runt $4, måste det vara april! Infografik av [Jen Looper](https://twitter.com/jenlooper)
>
> Matematiken som beräknar linjen måste visa lutningen på linjen, vilket också beror på interceptet, eller var `Y` är placerad när `X = 0`.
>
> Du kan observera metoden för beräkning av dessa värden på [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) webbplatsen. Besök också [denna Least-squares kalkylator](https://www.mathsisfun.com/data/least-squares-calculator.html) för att se hur värdena påverkar linjen.

## Korrelation

Ett annat begrepp att förstå är **korrelationskoefficienten** mellan givna X- och Y-variabler. Med ett spridningsdiagram kan du snabbt visualisera denna koefficient. Ett diagram med datapunkter som är snyggt uppradade har hög korrelation, men ett diagram med datapunkter spridda överallt mellan X och Y har låg korrelation.

En bra linjär regressionsmodell är en som har en hög (närmare 1 än 0) korrelationskoefficient med hjälp av Least-Squares Regression-metoden med en regressionslinje.

✅ Kör notebooken som följer med denna lektion och titta på spridningsdiagrammet för månad till pris. Verkar datan som associerar månad till pris för pumpaförsäljning ha hög eller låg korrelation, enligt din visuella tolkning av spridningsdiagrammet? Förändras det om du använder en mer detaljerad mätning istället för `Månad`, t.ex. *dag på året* (dvs. antal dagar sedan årets början)?

I koden nedan antar vi att vi har rengjort datan och fått en dataram kallad `new_pumpkins`, liknande följande:

ID | Månad | DagPåÅret | Sort | Stad | Förpackning | Lägsta Pris | Högsta Pris | Pris
---|-------|-----------|------|------|-------------|-------------|-------------|-----
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 skäppakartonger | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 skäppakartonger | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 skäppakartonger | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 skäppakartonger | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 skäppakartonger | 15.0 | 15.0 | 13.636364

> Koden för att rengöra datan finns i [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Vi har utfört samma rengöringssteg som i föregående lektion och har beräknat kolumnen `DagPåÅret` med hjälp av följande uttryck: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nu när du har en förståelse för matematiken bakom linjär regression, låt oss skapa en regressionsmodell för att se om vi kan förutsäga vilken pumpaförpackning som kommer att ha de bästa pumpapriserna. Någon som köper pumpor för en högtidspumpaplats kanske vill ha denna information för att optimera sina inköp av pumpaförpackningar för platsen.

## Söka efter korrelation

[![ML för nybörjare - Söka efter korrelation: Nyckeln till linjär regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML för nybörjare - Söka efter korrelation: Nyckeln till linjär regression")

> 🎥 Klicka på bilden ovan för en kort videogenomgång av korrelation.

Från föregående lektion har du förmodligen sett att det genomsnittliga priset för olika månader ser ut så här:

<img alt="Genomsnittligt pris per månad" src="../2-Data/images/barchart.png" width="50%"/>

Detta antyder att det borde finnas någon korrelation, och vi kan försöka träna en linjär regressionsmodell för att förutsäga relationen mellan `Månad` och `Pris`, eller mellan `DagPåÅret` och `Pris`. Här är spridningsdiagrammet som visar den senare relationen:

<img alt="Spridningsdiagram av Pris vs. Dag på året" src="images/scatter-dayofyear.png" width="50%" /> 

Låt oss se om det finns en korrelation med hjälp av funktionen `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Det verkar som att korrelationen är ganska liten, -0.15 för `Månad` och -0.17 för `DagPåÅret`, men det kan finnas en annan viktig relation. Det verkar som att det finns olika kluster av priser som motsvarar olika pumpasorter. För att bekräfta denna hypotes, låt oss plotta varje pumpakategori med en annan färg. Genom att skicka en `ax`-parameter till spridningsplotfunktionen kan vi plotta alla punkter på samma diagram:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Spridningsdiagram av Pris vs. Dag på året" src="images/scatter-dayofyear-color.png" width="50%" /> 

Vår undersökning antyder att sorten har större effekt på det övergripande priset än det faktiska försäljningsdatumet. Vi kan se detta med ett stapeldiagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stapeldiagram av pris vs sort" src="images/price-by-variety.png" width="50%" /> 

Låt oss för tillfället fokusera endast på en pumpasort, 'pie type', och se vilken effekt datumet har på priset:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Spridningsdiagram av Pris vs. Dag på året" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Om vi nu beräknar korrelationen mellan `Pris` och `DagPåÅret` med hjälp av funktionen `corr`, kommer vi att få något som `-0.27` - vilket betyder att det är meningsfullt att träna en prediktiv modell.

> Innan du tränar en linjär regressionsmodell är det viktigt att se till att vår data är ren. Linjär regression fungerar inte bra med saknade värden, så det är vettigt att ta bort alla tomma celler:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Ett annat tillvägagångssätt skulle vara att fylla dessa tomma värden med medelvärden från motsvarande kolumn.

## Enkel linjär regression

[![ML för nybörjare - Linjär och polynomisk regression med Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML för nybörjare - Linjär och polynomisk regression med Scikit-learn")

> 🎥 Klicka på bilden ovan för en kort videogenomgång av linjär och polynomisk regression.

För att träna vår linjära regressionsmodell kommer vi att använda biblioteket **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Vi börjar med att separera indata (funktioner) och det förväntade resultatet (etikett) i separata numpy-arrayer:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Observera att vi var tvungna att utföra `reshape` på indata för att linjär regressionspaketet skulle förstå det korrekt. Linjär regression förväntar sig en 2D-array som indata, där varje rad i arrayen motsvarar en vektor av indatafunktioner. I vårt fall, eftersom vi bara har en indata - behöver vi en array med formen N×1, där N är datamängdens storlek.

Sedan behöver vi dela upp datan i tränings- och testdatamängder, så att vi kan validera vår modell efter träning:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Slutligen tar det bara två rader kod att träna den faktiska linjära regressionsmodellen. Vi definierar `LinearRegression`-objektet och anpassar det till vår data med hjälp av metoden `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objektet efter att ha anpassats innehåller alla koefficienter för regressionen, som kan nås med egenskapen `.coef_`. I vårt fall finns det bara en koefficient, som bör vara runt `-0.017`. Det betyder att priser verkar sjunka lite med tiden, men inte mycket, runt 2 cent per dag. Vi kan också nå skärningspunkten för regressionen med Y-axeln med `lin_reg.intercept_` - det kommer att vara runt `21` i vårt fall, vilket indikerar priset i början av året.

För att se hur exakt vår modell är kan vi förutsäga priser på en testdatamängd och sedan mäta hur nära våra förutsägelser är de förväntade värdena. Detta kan göras med hjälp av metrik för medelkvadratfel (MSE), vilket är medelvärdet av alla kvadrerade skillnader mellan förväntat och förutspått värde.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Vårt fel verkar ligga runt 2 punkter, vilket är ~17 %. Inte så bra. En annan indikator på modellkvalitet är **determinationskoefficienten**, som kan erhållas så här:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Om värdet är 0 betyder det att modellen inte tar hänsyn till indata och fungerar som den *sämsta linjära prediktorn*, vilket helt enkelt är medelvärdet av resultatet. Värdet 1 betyder att vi kan förutsäga alla förväntade utdata perfekt. I vårt fall är koefficienten runt 0,06, vilket är ganska lågt.

Vi kan också plotta testdata tillsammans med regressionslinjen för att bättre se hur regression fungerar i vårt fall:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Linjär regression" src="images/linear-results.png" width="50%" />

## Polynomregression

En annan typ av linjär regression är polynomregression. Även om det ibland finns ett linjärt samband mellan variabler - ju större pumpan är i volym, desto högre pris - kan dessa samband ibland inte plottas som ett plan eller en rak linje.

✅ Här är [några fler exempel](https://online.stat.psu.edu/stat501/lesson/9/9.8) på data som kan använda polynomregression.

Titta en gång till på sambandet mellan datum och pris. Ser detta spridningsdiagram ut som om det nödvändigtvis borde analyseras med en rak linje? Kan inte priser fluktuera? I detta fall kan du prova polynomregression.

✅ Polynom är matematiska uttryck som kan bestå av en eller flera variabler och koefficienter.

Polynomregression skapar en kurvad linje för att bättre passa icke-linjär data. I vårt fall, om vi inkluderar en kvadrerad `DayOfYear`-variabel i indata, bör vi kunna passa vår data med en parabolisk kurva, som kommer att ha ett minimum vid en viss punkt under året.

Scikit-learn inkluderar ett användbart [pipeline-API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) för att kombinera olika steg i databehandlingen. En **pipeline** är en kedja av **estimators**. I vårt fall kommer vi att skapa en pipeline som först lägger till polynomfunktioner till vår modell och sedan tränar regressionen:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Att använda `PolynomialFeatures(2)` betyder att vi kommer att inkludera alla andragradspolynom från indata. I vårt fall innebär det bara `DayOfYear`<sup>2</sup>, men med två indata X och Y kommer detta att lägga till X<sup>2</sup>, XY och Y<sup>2</sup>. Vi kan också använda polynom av högre grad om vi vill.

Pipelines kan användas på samma sätt som det ursprungliga `LinearRegression`-objektet, dvs. vi kan `fit`-a pipelinen och sedan använda `predict` för att få prediktionsresultaten. Här är grafen som visar testdata och approximationskurvan:

<img alt="Polynomregression" src="images/poly-results.png" width="50%" />

Med polynomregression kan vi få något lägre MSE och högre determination, men inte signifikant. Vi behöver ta hänsyn till andra funktioner!

> Du kan se att de lägsta pumpapriserna observeras någonstans runt Halloween. Hur kan du förklara detta?

🎃 Grattis, du har just skapat en modell som kan hjälpa till att förutsäga priset på pajpumpor. Du kan förmodligen upprepa samma procedur för alla pumpatyper, men det skulle vara tidskrävande. Låt oss nu lära oss hur man tar pumpasort i beaktande i vår modell!

## Kategoriska funktioner

I den ideala världen vill vi kunna förutsäga priser för olika pumpasorter med samma modell. Kolumnen `Variety` är dock något annorlunda än kolumner som `Month`, eftersom den innehåller icke-numeriska värden. Sådana kolumner kallas **kategoriska**.

[![ML för nybörjare - Kategoriska funktioner med linjär regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML för nybörjare - Kategoriska funktioner med linjär regression")

> 🎥 Klicka på bilden ovan för en kort videoöversikt om att använda kategoriska funktioner.

Här kan du se hur medelpriset beror på sort:

<img alt="Medelpris per sort" src="images/price-by-variety.png" width="50%" />

För att ta sort i beaktande måste vi först konvertera den till numerisk form, eller **koda** den. Det finns flera sätt att göra detta:

* Enkel **numerisk kodning** bygger en tabell över olika sorter och ersätter sedan sortnamnet med ett index i den tabellen. Detta är inte den bästa idén för linjär regression, eftersom linjär regression tar det faktiska numeriska värdet av indexet och lägger till det i resultatet, multiplicerat med någon koefficient. I vårt fall är sambandet mellan indexnummer och pris klart icke-linjärt, även om vi ser till att indexen är ordnade på något specifikt sätt.
* **One-hot-kodning** ersätter kolumnen `Variety` med 4 olika kolumner, en för varje sort. Varje kolumn innehåller `1` om den motsvarande raden är av en given sort och `0` annars. Detta innebär att det kommer att finnas fyra koefficienter i linjär regression, en för varje pumpasort, som ansvarar för "startpris" (eller snarare "tilläggspris") för den specifika sorten.

Koden nedan visar hur vi kan one-hot-koda en sort:

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

För att träna linjär regression med one-hot-kodad sort som indata behöver vi bara initiera `X` och `y`-data korrekt:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Resten av koden är densamma som den vi använde ovan för att träna linjär regression. Om du provar det kommer du att se att medelkvadratfelet är ungefär detsamma, men vi får mycket högre determinationskoefficient (~77 %). För att få ännu mer exakta förutsägelser kan vi ta fler kategoriska funktioner i beaktande, samt numeriska funktioner, såsom `Month` eller `DayOfYear`. För att få en stor array av funktioner kan vi använda `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Här tar vi också hänsyn till `City` och `Package`-typ, vilket ger oss MSE 2,84 (10 %) och determination 0,94!

## Sätta ihop allt

För att skapa den bästa modellen kan vi använda kombinerad (one-hot-kodad kategorisk + numerisk) data från exemplet ovan tillsammans med polynomregression. Här är den kompletta koden för din bekvämlighet:

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

Detta bör ge oss den bästa determinationskoefficienten på nästan 97 % och MSE=2,23 (~8 % prediktionsfel).

| Modell | MSE | Determination |
|--------|-----|---------------|
| `DayOfYear` Linjär | 2,77 (17,2 %) | 0,07 |
| `DayOfYear` Polynom | 2,73 (17,0 %) | 0,08 |
| `Variety` Linjär | 5,24 (19,7 %) | 0,77 |
| Alla funktioner Linjär | 2,84 (10,5 %) | 0,94 |
| Alla funktioner Polynom | 2,23 (8,25 %) | 0,97 |

🏆 Bra jobbat! Du skapade fyra regressionsmodeller i en lektion och förbättrade modellkvaliteten till 97 %. I det sista avsnittet om regression kommer du att lära dig om logistisk regression för att bestämma kategorier.

---
## 🚀Utmaning

Testa flera olika variabler i denna notebook för att se hur korrelationen motsvarar modellens noggrannhet.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

I denna lektion lärde vi oss om linjär regression. Det finns andra viktiga typer av regression. Läs om Stepwise, Ridge, Lasso och Elasticnet-tekniker. En bra kurs att studera för att lära sig mer är [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Uppgift

[Bygg en modell](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiserade översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.