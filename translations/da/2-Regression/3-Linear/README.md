<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-04T23:21:44+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "da"
}
-->
# Byg en regressionsmodel med Scikit-learn: regression på fire måder

![Infografik om lineær vs. polynomisk regression](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografik af [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne lektion er også tilgængelig i R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Introduktion 

Indtil nu har du udforsket, hvad regression er, med eksempeldata fra græskarpris-datasættet, som vi vil bruge gennem hele denne lektion. Du har også visualiseret det med Matplotlib.

Nu er du klar til at dykke dybere ned i regression for maskinlæring. Mens visualisering hjælper dig med at forstå data, ligger den virkelige styrke i maskinlæring i _træning af modeller_. Modeller trænes på historiske data for automatisk at fange dataafhængigheder, og de giver dig mulighed for at forudsige resultater for nye data, som modellen ikke har set før.

I denne lektion vil du lære mere om to typer regression: _grundlæggende lineær regression_ og _polynomisk regression_, sammen med noget af den matematik, der ligger bag disse teknikker. Disse modeller vil give os mulighed for at forudsige græskarpriser baseret på forskellige inputdata. 

[![ML for begyndere - Forståelse af lineær regression](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML for begyndere - Forståelse af lineær regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over lineær regression.

> Gennem hele dette pensum antager vi minimal viden om matematik og søger at gøre det tilgængeligt for studerende fra andre områder, så hold øje med noter, 🧮 callouts, diagrammer og andre læringsværktøjer, der kan hjælpe med forståelsen.

### Forudsætninger

Du bør nu være bekendt med strukturen af græskardataene, som vi undersøger. Du kan finde det forudindlæst og forudrenset i denne lektions _notebook.ipynb_-fil. I filen vises græskarprisen pr. bushel i en ny data frame. Sørg for, at du kan køre disse notebooks i kerner i Visual Studio Code.

### Forberedelse

Som en påmindelse indlæser du disse data for at stille spørgsmål til dem. 

- Hvornår er det bedste tidspunkt at købe græskar? 
- Hvilken pris kan jeg forvente for en kasse med miniaturegræskar?
- Skal jeg købe dem i halv-bushel kurve eller i 1 1/9 bushel kasser?
Lad os fortsætte med at grave i disse data.

I den forrige lektion oprettede du en Pandas data frame og fyldte den med en del af det originale datasæt, hvor du standardiserede prisen pr. bushel. Ved at gøre det var du dog kun i stand til at samle omkring 400 datapunkter og kun for efterårsmånederne. 

Tag et kig på de data, der er forudindlæst i denne lektions tilhørende notebook. Dataene er forudindlæst, og et indledende scatterplot er oprettet for at vise månedsdata. Måske kan vi få lidt mere detaljeret information om dataenes natur ved at rense dem yderligere.

## En lineær regressionslinje

Som du lærte i Lektion 1, er målet med en lineær regressionsøvelse at kunne plotte en linje for at:

- **Vis variableforhold**. Vis forholdet mellem variabler
- **Lav forudsigelser**. Lav præcise forudsigelser om, hvor et nyt datapunkt vil falde i forhold til den linje. 
 
Det er typisk for **Least-Squares Regression** at tegne denne type linje. Udtrykket 'least-squares' betyder, at alle datapunkterne omkring regressionslinjen kvadreres og derefter lægges sammen. Ideelt set er den endelige sum så lille som muligt, fordi vi ønsker et lavt antal fejl, eller `least-squares`. 

Vi gør dette, fordi vi ønsker at modellere en linje, der har den mindste kumulative afstand fra alle vores datapunkter. Vi kvadrerer også termerne, før vi lægger dem sammen, da vi er interesserede i deres størrelse snarere end deres retning.

> **🧮 Vis mig matematikken** 
> 
> Denne linje, kaldet _linjen for bedste pasform_, kan udtrykkes ved [en ligning](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` er den 'forklarende variabel'. `Y` er den 'afhængige variabel'. Hældningen af linjen er `b`, og `a` er y-skæringen, som refererer til værdien af `Y`, når `X = 0`. 
>
>![beregn hældningen](../../../../2-Regression/3-Linear/images/slope.png)
>
> Først beregnes hældningen `b`. Infografik af [Jen Looper](https://twitter.com/jenlooper)
>
> Med andre ord, og med henvisning til det oprindelige spørgsmål om græskardata: "forudsige prisen på et græskar pr. bushel efter måned", ville `X` referere til prisen, og `Y` ville referere til salgsdatoen. 
>
>![fuldfør ligningen](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Beregn værdien af Y. Hvis du betaler omkring $4, må det være april! Infografik af [Jen Looper](https://twitter.com/jenlooper)
>
> Matematikken, der beregner linjen, skal demonstrere hældningen af linjen, som også afhænger af skæringspunktet, eller hvor `Y` er placeret, når `X = 0`.
>
> Du kan se metoden til beregning af disse værdier på [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) webstedet. Besøg også [denne Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) for at se, hvordan talværdierne påvirker linjen.

## Korrelation

Et andet begreb, du skal forstå, er **korrelationskoefficienten** mellem givne X- og Y-variabler. Ved hjælp af et scatterplot kan du hurtigt visualisere denne koefficient. Et plot med datapunkter spredt i en pæn linje har høj korrelation, men et plot med datapunkter spredt overalt mellem X og Y har lav korrelation.

En god lineær regressionsmodel vil være en, der har en høj (nærmere 1 end 0) korrelationskoefficient ved hjælp af Least-Squares Regression-metoden med en regressionslinje.

✅ Kør notebooken, der ledsager denne lektion, og kig på scatterplottet for måned til pris. Ser dataene, der forbinder måned til pris for græskarsalg, ud til at have høj eller lav korrelation ifølge din visuelle fortolkning af scatterplottet? Ændrer det sig, hvis du bruger en mere detaljeret måling i stedet for `Month`, fx *dag i året* (dvs. antal dage siden årets begyndelse)?

I koden nedenfor antager vi, at vi har renset dataene og opnået en data frame kaldet `new_pumpkins`, der ligner følgende:

ID | Month | DayOfYear | Variety | City | Package | Low Price | High Price | Price
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Koden til at rense dataene er tilgængelig i [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Vi har udført de samme rengøringsskridt som i den forrige lektion og har beregnet `DayOfYear`-kolonnen ved hjælp af følgende udtryk: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nu hvor du har en forståelse af matematikken bag lineær regression, lad os oprette en regressionsmodel for at se, om vi kan forudsige, hvilken pakke græskar der vil have de bedste græskarpriser. En person, der køber græskar til en feriegræskarplads, vil måske have denne information for at optimere sine køb af græskarpakker til pladsen.

## Søger efter korrelation

[![ML for begyndere - Søger efter korrelation: Nøglen til lineær regression](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML for begyndere - Søger efter korrelation: Nøglen til lineær regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over korrelation.

Fra den forrige lektion har du sandsynligvis set, at gennemsnitsprisen for forskellige måneder ser sådan ud:

<img alt="Gennemsnitspris pr. måned" src="../2-Data/images/barchart.png" width="50%"/>

Dette antyder, at der bør være en vis korrelation, og vi kan prøve at træne en lineær regressionsmodel til at forudsige forholdet mellem `Month` og `Price`, eller mellem `DayOfYear` og `Price`. Her er scatterplottet, der viser sidstnævnte forhold:

<img alt="Scatterplot af pris vs. dag i året" src="images/scatter-dayofyear.png" width="50%" /> 

Lad os se, om der er en korrelation ved hjælp af funktionen `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Det ser ud til, at korrelationen er ret lille, -0.15 for `Month` og -0.17 for `DayOfMonth`, men der kunne være et andet vigtigt forhold. Det ser ud til, at der er forskellige prisgrupper, der svarer til forskellige græskartyper. For at bekræfte denne hypotese, lad os plotte hver græskarkategori med en anden farve. Ved at sende en `ax`-parameter til scatter-plotfunktionen kan vi plotte alle punkter på samme graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Scatterplot af pris vs. dag i året" src="images/scatter-dayofyear-color.png" width="50%" /> 

Vores undersøgelse antyder, at sorten har større effekt på den samlede pris end den faktiske salgsdato. Vi kan se dette med et søjlediagram:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Søjlediagram af pris vs. sort" src="images/price-by-variety.png" width="50%" /> 

Lad os fokusere for øjeblikket kun på én græskarsort, 'pie type', og se, hvilken effekt datoen har på prisen:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Scatterplot af pris vs. dag i året" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Hvis vi nu beregner korrelationen mellem `Price` og `DayOfYear` ved hjælp af funktionen `corr`, får vi noget som `-0.27` - hvilket betyder, at det giver mening at træne en forudsigelsesmodel.

> Før du træner en lineær regressionsmodel, er det vigtigt at sikre, at vores data er rene. Lineær regression fungerer ikke godt med manglende værdier, så det giver mening at fjerne alle tomme celler:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

En anden tilgang ville være at udfylde de tomme værdier med gennemsnitsværdier fra den tilsvarende kolonne.

## Simpel lineær regression

[![ML for begyndere - Lineær og polynomisk regression med Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML for begyndere - Lineær og polynomisk regression med Scikit-learn")

> 🎥 Klik på billedet ovenfor for en kort videooversigt over lineær og polynomisk regression.

For at træne vores lineære regressionsmodel vil vi bruge **Scikit-learn**-biblioteket.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Vi starter med at adskille inputværdier (features) og det forventede output (label) i separate numpy-arrays:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Bemærk, at vi var nødt til at udføre `reshape` på inputdataene, for at Linear Regression-pakken kunne forstå det korrekt. Lineær regression forventer et 2D-array som input, hvor hver række i arrayet svarer til en vektor af inputfeatures. I vores tilfælde, da vi kun har én input, har vi brug for et array med formen N×1, hvor N er datasættets størrelse.

Derefter skal vi opdele dataene i trænings- og testdatasæt, så vi kan validere vores model efter træning:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Endelig tager det kun to linjer kode at træne den faktiske lineære regressionsmodel. Vi definerer `LinearRegression`-objektet og tilpasser det til vores data ved hjælp af metoden `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objektet indeholder efter `fit`-processen alle koefficienterne for regressionen, som kan tilgås ved hjælp af `.coef_`-egenskaben. I vores tilfælde er der kun én koefficient, som bør være omkring `-0.017`. Det betyder, at priserne ser ud til at falde lidt med tiden, men ikke meget, omkring 2 cent pr. dag. Vi kan også tilgå skæringspunktet for regressionen med Y-aksen ved hjælp af `lin_reg.intercept_` - det vil være omkring `21` i vores tilfælde, hvilket indikerer prisen i begyndelsen af året.

For at se, hvor præcis vores model er, kan vi forudsige priser på et testdatasæt og derefter måle, hvor tæt vores forudsigelser er på de forventede værdier. Dette kan gøres ved hjælp af mean square error (MSE)-metrikken, som er gennemsnittet af alle kvadrerede forskelle mellem forventet og forudsagt værdi.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Vores fejl ser ud til at ligge omkring 2 punkter, hvilket svarer til ~17%. Ikke så godt. En anden indikator for modelkvalitet er **bestemmelseskoefficienten**, som kan beregnes sådan her:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Hvis værdien er 0, betyder det, at modellen ikke tager inputdata i betragtning og fungerer som den *dårligste lineære forudsigelse*, hvilket simpelthen er gennemsnitsværdien af resultatet. Værdien 1 betyder, at vi kan forudsige alle forventede output perfekt. I vores tilfælde er koefficienten omkring 0,06, hvilket er ret lavt.

Vi kan også plotte testdata sammen med regressionslinjen for bedre at se, hvordan regression fungerer i vores tilfælde:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineær regression" src="images/linear-results.png" width="50%" />

## Polynomisk Regression

En anden type lineær regression er polynomisk regression. Mens der nogle gange er en lineær sammenhæng mellem variabler - jo større græskar i volumen, jo højere pris - kan disse sammenhænge nogle gange ikke plottes som et plan eller en lige linje.

✅ Her er [nogle flere eksempler](https://online.stat.psu.edu/stat501/lesson/9/9.8) på data, der kunne bruge polynomisk regression.

Tag et nyt kig på sammenhængen mellem dato og pris. Ser dette scatterplot ud som om det nødvendigvis skal analyseres med en lige linje? Kan priser ikke svinge? I dette tilfælde kan du prøve polynomisk regression.

✅ Polynomier er matematiske udtryk, der kan bestå af en eller flere variabler og koefficienter.

Polynomisk regression skaber en buet linje for bedre at tilpasse sig ikke-lineære data. I vores tilfælde, hvis vi inkluderer en kvadreret `DayOfYear`-variabel i inputdata, burde vi kunne tilpasse vores data med en parabolsk kurve, som vil have et minimum på et bestemt tidspunkt i løbet af året.

Scikit-learn inkluderer en nyttig [pipeline-API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) til at kombinere forskellige trin i databehandlingen. En **pipeline** er en kæde af **estimators**. I vores tilfælde vil vi oprette en pipeline, der først tilføjer polynomiske funktioner til vores model og derefter træner regressionen:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Ved at bruge `PolynomialFeatures(2)` betyder det, at vi vil inkludere alle anden-gradspolynomier fra inputdata. I vores tilfælde vil det blot betyde `DayOfYear`<sup>2</sup>, men givet to inputvariabler X og Y, vil dette tilføje X<sup>2</sup>, XY og Y<sup>2</sup>. Vi kan også bruge polynomier af højere grad, hvis vi ønsker det.

Pipelines kan bruges på samme måde som det originale `LinearRegression`-objekt, dvs. vi kan `fit` pipelinen og derefter bruge `predict` til at få forudsigelsesresultater. Her er grafen, der viser testdata og tilnærmningskurven:

<img alt="Polynomisk regression" src="images/poly-results.png" width="50%" />

Ved at bruge polynomisk regression kan vi få en lidt lavere MSE og højere bestemmelseskoefficient, men ikke markant. Vi skal tage andre funktioner i betragtning!

> Du kan se, at de laveste græskarpriser observeres omkring Halloween. Hvordan kan du forklare dette?

🎃 Tillykke, du har lige oprettet en model, der kan hjælpe med at forudsige prisen på tærtegræskar. Du kan sandsynligvis gentage den samme procedure for alle græskartyper, men det ville være tidskrævende. Lad os nu lære, hvordan man tager græskarsort i betragtning i vores model!

## Kategoriske Funktioner

I en ideel verden ønsker vi at kunne forudsige priser for forskellige græskarsorter ved hjælp af den samme model. Dog er kolonnen `Variety` noget anderledes end kolonner som `Month`, fordi den indeholder ikke-numeriske værdier. Sådanne kolonner kaldes **kategoriske**.

[![ML for begyndere - Kategoriske funktioner med lineær regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for begyndere - Kategoriske funktioner med lineær regression")

> 🎥 Klik på billedet ovenfor for en kort videooversigt om brug af kategoriske funktioner.

Her kan du se, hvordan gennemsnitsprisen afhænger af sorten:

<img alt="Gennemsnitspris efter sort" src="images/price-by-variety.png" width="50%" />

For at tage sorten i betragtning skal vi først konvertere den til numerisk form, eller **kode** den. Der er flere måder, vi kan gøre det på:

* Enkel **numerisk kodning** vil oprette en tabel over forskellige sorter og derefter erstatte sortsnavnet med et indeks i den tabel. Dette er ikke den bedste idé for lineær regression, fordi lineær regression tager den faktiske numeriske værdi af indekset og tilføjer det til resultatet, multipliceret med en koefficient. I vores tilfælde er forholdet mellem indeksnummeret og prisen klart ikke-lineært, selv hvis vi sørger for, at indeksene er ordnet på en bestemt måde.
* **One-hot kodning** vil erstatte kolonnen `Variety` med 4 forskellige kolonner, én for hver sort. Hver kolonne vil indeholde `1`, hvis den tilsvarende række er af en given sort, og `0` ellers. Dette betyder, at der vil være fire koefficienter i lineær regression, én for hver græskarsort, ansvarlig for "startpris" (eller rettere "ekstrapris") for den pågældende sort.

Koden nedenfor viser, hvordan vi kan one-hot kode en sort:

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

For at træne lineær regression ved hjælp af one-hot kodet sort som input skal vi blot initialisere `X` og `y` data korrekt:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Resten af koden er den samme som den, vi brugte ovenfor til at træne lineær regression. Hvis du prøver det, vil du se, at den gennemsnitlige kvadratiske fejl er omtrent den samme, men vi får en meget højere bestemmelseskoefficient (~77%). For at få endnu mere præcise forudsigelser kan vi tage flere kategoriske funktioner i betragtning samt numeriske funktioner som `Month` eller `DayOfYear`. For at få én stor array af funktioner kan vi bruge `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Her tager vi også `City` og `Package` type i betragtning, hvilket giver os MSE 2.84 (10%) og bestemmelse 0.94!

## Samlet set

For at lave den bedste model kan vi bruge kombinerede (one-hot kodede kategoriske + numeriske) data fra ovenstående eksempel sammen med polynomisk regression. Her er den komplette kode for din bekvemmelighed:

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

Dette burde give os den bedste bestemmelseskoefficient på næsten 97% og MSE=2.23 (~8% forudsigelsesfejl).

| Model | MSE | Bestemmelse |
|-------|-----|-------------|
| `DayOfYear` Lineær | 2.77 (17.2%) | 0.07 |
| `DayOfYear` Polynomisk | 2.73 (17.0%) | 0.08 |
| `Variety` Lineær | 5.24 (19.7%) | 0.77 |
| Alle funktioner Lineær | 2.84 (10.5%) | 0.94 |
| Alle funktioner Polynomisk | 2.23 (8.25%) | 0.97 |

🏆 Godt gået! Du har oprettet fire regressionsmodeller i én lektion og forbedret modelkvaliteten til 97%. I den sidste sektion om regression vil du lære om logistisk regression til at bestemme kategorier.

---
## 🚀Udfordring

Test flere forskellige variabler i denne notebook for at se, hvordan korrelation svarer til modelnøjagtighed.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

I denne lektion lærte vi om lineær regression. Der er andre vigtige typer af regression. Læs om Stepwise, Ridge, Lasso og Elasticnet teknikker. Et godt kursus at studere for at lære mere er [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Opgave 

[Byg en model](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.