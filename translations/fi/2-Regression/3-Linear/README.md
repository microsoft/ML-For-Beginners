<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-04T23:25:03+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "fi"
}
-->
# Rakenna regressiomalli Scikit-learnilla: neljä tapaa regressioon

![Lineaarinen vs polynominen regressio infografiikka](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografiikka: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tämä oppitunti on saatavilla myös R-kielellä!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Johdanto 

Tähän mennessä olet tutustunut regressioon käyttäen esimerkkidataa kurpitsan hinnoitteludatasta, jota käytämme koko tämän oppitunnin ajan. Olet myös visualisoinut dataa Matplotlibin avulla.

Nyt olet valmis sukeltamaan syvemmälle koneoppimisen regressioon. Vaikka visualisointi auttaa ymmärtämään dataa, koneoppimisen todellinen voima tulee _mallien kouluttamisesta_. Mallit koulutetaan historiallisella datalla, jotta ne voivat automaattisesti tunnistaa datan riippuvuuksia, ja niiden avulla voidaan ennustaa tuloksia uudelle datalle, jota malli ei ole aiemmin nähnyt.

Tässä oppitunnissa opit lisää kahdesta regressiotyypistä: _perus lineaarisesta regressiosta_ ja _polynomisesta regressiosta_, sekä näiden tekniikoiden taustalla olevasta matematiikasta. Näiden mallien avulla voimme ennustaa kurpitsan hintoja eri syöttödatasta riippuen.

[![Koneoppimisen perusteet - Lineaarisen regression ymmärtäminen](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Koneoppimisen perusteet - Lineaarisen regression ymmärtäminen")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyttä videota varten lineaarisesta regressiosta.

> Tässä oppimateriaalissa oletamme vain vähäistä matematiikan osaamista ja pyrimme tekemään sen helposti lähestyttäväksi opiskelijoille, jotka tulevat muilta aloilta. Huomioi muistiinpanot, 🧮 matemaattiset esimerkit, diagrammit ja muut oppimistyökalut, jotka auttavat ymmärtämisessä.

### Esitiedot

Sinun tulisi nyt olla perehtynyt kurpitsadatan rakenteeseen, jota tutkimme. Löydät sen esiladattuna ja esipuhdistettuna tämän oppitunnin _notebook.ipynb_-tiedostosta. Tiedostossa kurpitsan hinta on esitetty per bushel uudessa dataframessa. Varmista, että voit ajaa nämä notebookit Visual Studio Coden kernelleissä.

### Valmistelu

Muistutuksena, lataat tätä dataa voidaksesi esittää kysymyksiä siitä. 

- Milloin on paras aika ostaa kurpitsoja? 
- Minkä hinnan voin odottaa miniatyyrikurpitsojen laatikolle?
- Pitäisikö minun ostaa ne puolibushelin koreissa vai 1 1/9 bushelin laatikoissa?
Jatketaan datan tutkimista.

Edellisessä oppitunnissa loit Pandas-dataframen ja täytit sen osalla alkuperäistä datasettiä, standardisoiden hinnoittelun bushelin mukaan. Tällä tavalla pystyit kuitenkin keräämään vain noin 400 datapistettä ja vain syksyn kuukausilta.

Tutustu dataan, joka on esiladattu tämän oppitunnin mukana tulevassa notebookissa. Data on esiladattu ja alkuperäinen hajontakaavio on piirretty näyttämään kuukausidataa. Ehkä voimme saada hieman enemmän yksityiskohtia datan luonteesta puhdistamalla sitä lisää.

## Lineaarinen regressioviiva

Kuten opit oppitunnilla 1, lineaarisen regressioharjoituksen tavoitteena on pystyä piirtämään viiva, joka:

- **Näyttää muuttujien suhteet**. Näyttää muuttujien välisen suhteen
- **Tekee ennusteita**. Tekee tarkkoja ennusteita siitä, mihin uusi datapiste sijoittuisi suhteessa viivaan. 
 
On tyypillistä käyttää **Least-Squares Regression** -menetelmää tämän tyyppisen viivan piirtämiseen. Termi 'least-squares' tarkoittaa, että kaikki regressioviivan ympärillä olevat datapisteet neliöidään ja sitten summataan. Ihanteellisesti lopullinen summa on mahdollisimman pieni, koska haluamme vähän virheitä eli `least-squares`.

Teemme näin, koska haluamme mallintaa viivan, jolla on pienin kumulatiivinen etäisyys kaikista datapisteistämme. Neliöimme termit ennen niiden yhteenlaskemista, koska olemme kiinnostuneita niiden suuruudesta, emmekä suunnasta.

> **🧮 Näytä matematiikka** 
> 
> Tämä viiva, jota kutsutaan _parhaiten sopivaksi viivaksi_, voidaan ilmaista [yhtälöllä](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` on 'selittävä muuttuja'. `Y` on 'riippuva muuttuja'. Viivan kulmakerroin on `b` ja `a` on y-akselin leikkauspiste, joka viittaa `Y`:n arvoon, kun `X = 0`. 
>
>![kulmakertoimen laskeminen](../../../../2-Regression/3-Linear/images/slope.png)
>
> Ensin lasketaan kulmakerroin `b`. Infografiikka: [Jen Looper](https://twitter.com/jenlooper)
>
> Toisin sanoen, viitaten alkuperäiseen kurpitsadataan liittyvään kysymykseen: "ennusta kurpitsan hinta per bushel kuukauden mukaan", `X` viittaa hintaan ja `Y` viittaa myyntikuukauteen. 
>
>![yhtälön täydentäminen](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Laske `Y`:n arvo. Jos maksat noin $4, sen täytyy olla huhtikuu! Infografiikka: [Jen Looper](https://twitter.com/jenlooper)
>
> Matematiikka, joka laskee viivan, täytyy osoittaa viivan kulmakerroin, joka riippuu myös leikkauspisteestä, eli siitä, missä `Y` sijaitsee, kun `X = 0`.
>
> Voit tarkastella näiden arvojen laskentamenetelmää [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) -sivustolla. Käy myös [Least-squares calculator](https://www.mathsisfun.com/data/least-squares-calculator.html) -sivustolla nähdäksesi, miten lukuarvot vaikuttavat viivaan.

## Korrelaatio

Yksi termi, joka on hyvä ymmärtää, on **korrelaatiokerroin** annettujen X- ja Y-muuttujien välillä. Hajontakaavion avulla voit nopeasti visualisoida tämän kertoimen. Kaavio, jossa datapisteet ovat siistissä linjassa, omaa korkean korrelaation, mutta kaavio, jossa datapisteet ovat hajallaan X:n ja Y:n välillä, omaa matalan korrelaation.

Hyvä lineaarinen regressiomalli on sellainen, jolla on korkea (lähempänä 1 kuin 0) korrelaatiokerroin käyttäen Least-Squares Regression -menetelmää ja regressioviivaa.

✅ Aja tämän oppitunnin mukana tuleva notebook ja katso Kuukausi-Hinta hajontakaaviota. Vaikuttaako data, joka yhdistää Kuukauden ja Hinnan kurpitsamyynnissä, olevan korkea vai matala korrelaatio visuaalisen tulkintasi mukaan hajontakaaviosta? Muuttuuko tämä, jos käytät tarkempaa mittaa kuin `Kuukausi`, esim. *vuoden päivä* (eli päivien lukumäärä vuoden alusta)?

Alla olevassa koodissa oletamme, että olemme puhdistaneet datan ja saaneet dataframen nimeltä `new_pumpkins`, joka näyttää seuraavalta:

ID | Kuukausi | VuodenPäivä | Lajike | Kaupunki | Pakkaus | Alin Hinta | Korkein Hinta | Hinta
---|----------|-------------|--------|----------|---------|------------|---------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Koodi datan puhdistamiseen löytyy tiedostosta [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Olemme suorittaneet samat puhdistusvaiheet kuin edellisessä oppitunnissa ja laskeneet `VuodenPäivä`-sarakkeen seuraavalla lausekkeella:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nyt kun ymmärrät lineaarisen regression taustalla olevan matematiikan, luodaan regressiomalli nähdäksesi, voimmeko ennustaa, mikä kurpitsapaketti tarjoaa parhaat hinnat. Joku, joka ostaa kurpitsoja juhlapäivän kurpitsapellolle, saattaa haluta tätä tietoa voidakseen optimoida kurpitsapakettien ostot pellolle.

## Korrelaation etsiminen

[![Koneoppimisen perusteet - Korrelaation etsiminen: Lineaarisen regression avain](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Koneoppimisen perusteet - Korrelaation etsiminen: Lineaarisen regression avain")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyttä videota varten korrelaatiosta.

Edellisestä oppitunnista olet todennäköisesti nähnyt, että keskimääräinen hinta eri kuukausina näyttää tältä:

<img alt="Keskimääräinen hinta kuukauden mukaan" src="../2-Data/images/barchart.png" width="50%"/>

Tämä viittaa siihen, että korrelaatiota saattaa olla, ja voimme yrittää kouluttaa lineaarisen regressiomallin ennustamaan suhdetta `Kuukausi` ja `Hinta` välillä tai `VuodenPäivä` ja `Hinta` välillä. Tässä on hajontakaavio, joka näyttää jälkimmäisen suhteen:

<img alt="Hajontakaavio Hinta vs. Vuoden Päivä" src="images/scatter-dayofyear.png" width="50%" /> 

Katsotaan, onko korrelaatiota käyttämällä `corr`-funktiota:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Näyttää siltä, että korrelaatio on melko pieni, -0.15 `Kuukauden` mukaan ja -0.17 `VuodenPäivän` mukaan, mutta saattaa olla toinen tärkeä suhde. Näyttää siltä, että eri kurpitsalajikkeiden hinnat muodostavat erilaisia klustereita. Vahvistaaksemme tämän hypoteesin, piirretään jokainen kurpitsakategoria eri värillä. Käyttämällä `ax`-parametria `scatter`-piirtofunktiossa voimme piirtää kaikki pisteet samaan kaavioon:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Hajontakaavio Hinta vs. Vuoden Päivä" src="images/scatter-dayofyear-color.png" width="50%" /> 

Tutkimuksemme viittaa siihen, että lajikkeella on suurempi vaikutus kokonaishintaan kuin varsinaisella myyntipäivällä. Voimme nähdä tämän pylväsdiagrammilla:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Pylväsdiagrammi hinta vs lajike" src="images/price-by-variety.png" width="50%" /> 

Keskitytään hetkeksi vain yhteen kurpitsalajikkeeseen, 'pie type', ja katsotaan, mitä vaikutusta päivämäärällä on hintaan:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Hajontakaavio Hinta vs. Vuoden Päivä" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Jos nyt laskemme korrelaation `Hinta` ja `VuodenPäivä` välillä käyttäen `corr`-funktiota, saamme jotain kuten `-0.27` - mikä tarkoittaa, että ennustavan mallin kouluttaminen on järkevää.

> Ennen lineaarisen regressiomallin kouluttamista on tärkeää varmistaa, että datamme on puhdasta. Lineaarinen regressio ei toimi hyvin puuttuvien arvojen kanssa, joten on järkevää poistaa kaikki tyhjät solut:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Toinen lähestymistapa olisi täyttää tyhjät arvot vastaavan sarakkeen keskiarvoilla.

## Yksinkertainen lineaarinen regressio

[![Koneoppimisen perusteet - Lineaarinen ja polynominen regressio Scikit-learnilla](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Koneoppimisen perusteet - Lineaarinen ja polynominen regressio Scikit-learnilla")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyttä videota varten lineaarisesta ja polynomisesta regressiosta.

Lineaarisen regressiomallin kouluttamiseen käytämme **Scikit-learn**-kirjastoa.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Aloitamme erottamalla syöttöarvot (ominaisuudet) ja odotetut tulokset (label) erillisiin numpy-taulukoihin:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Huomaa, että meidän täytyi suorittaa `reshape` syöttödatalle, jotta lineaarisen regression paketti ymmärtäisi sen oikein. Lineaarinen regressio odottaa 2D-taulukkoa syötteenä, jossa taulukon jokainen rivi vastaa syöttöominaisuuksien vektoria. Meidän tapauksessamme, koska meillä on vain yksi syöte, tarvitsemme taulukon, jonka muoto on N×1, missä N on datasetin koko.

Seuraavaksi meidän täytyy jakaa data koulutus- ja testidatasettiin, jotta voimme validoida mallimme koulutuksen jälkeen:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Lopuksi varsinaisen lineaarisen regressiomallin kouluttaminen vie vain kaksi koodiriviä. Määrittelemme `LinearRegression`-objektin ja sovitamme sen dataamme käyttämällä `fit`-metodia:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression`-objekti sisältää `fit`-vaiheen jälkeen kaikki regression kertoimet, jotka voidaan hakea `.coef_`-ominaisuuden avulla. Meidän tapauksessamme on vain yksi kerroin, jonka pitäisi olla noin `-0.017`. Tämä tarkoittaa, että hinnat näyttävät laskevan hieman ajan myötä, mutta eivät kovin paljon, noin 2 senttiä päivässä. Voimme myös hakea regressioviivan y-akselin leikkauspisteen `lin_reg.intercept_`-ominaisuuden avulla - se on noin `21` meidän tapauksessamme, mikä osoittaa hinnan vuoden alussa.

Mallimme tarkkuuden näkemiseksi voimme ennustaa hinnat testidatasetilla ja mitata, kuinka lähellä ennusteemme ovat odotettuja arvoja. Tämä voidaan tehdä käyttämällä keskimääräisen neliövirheen (MSE) mittaria, joka on kaikkien odotettujen ja ennustettujen arvojen neliöityjen erojen keskiarvo.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Virheemme näyttää olevan noin 2 pisteen kohdalla, mikä on ~17 %. Ei kovin hyvä. Toinen indikaattori mallin laadusta on **determinointikerroin**, joka voidaan laskea näin:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Jos arvo on 0, se tarkoittaa, että malli ei ota syöttötietoja huomioon ja toimii *huonoimpana lineaarisena ennustajana*, joka on yksinkertaisesti tuloksen keskiarvo. Arvo 1 tarkoittaa, että voimme täydellisesti ennustaa kaikki odotetut tulokset. Meidän tapauksessamme kerroin on noin 0.06, mikä on melko alhainen.

Voimme myös piirtää testidatan yhdessä regressioviivan kanssa, jotta näemme paremmin, miten regressio toimii meidän tapauksessamme:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineaarinen regressio" src="images/linear-results.png" width="50%" />

## Polynominen regressio

Toinen lineaarisen regression tyyppi on polynominen regressio. Vaikka joskus muuttujien välillä on lineaarinen suhde – mitä suurempi kurpitsa tilavuudeltaan, sitä korkeampi hinta – joskus näitä suhteita ei voida kuvata tasolla tai suoralla viivalla.

✅ Tässä on [joitakin esimerkkejä](https://online.stat.psu.edu/stat501/lesson/9/9.8) datasta, joka voisi hyödyntää polynomista regressiota.

Katso uudelleen suhdetta Päivämäärän ja Hinnan välillä. Vaikuttaako tämä hajontakuvio siltä, että sitä pitäisi välttämättä analysoida suoralla viivalla? Eivätkö hinnat voi vaihdella? Tässä tapauksessa voit kokeilla polynomista regressiota.

✅ Polynomit ovat matemaattisia lausekkeita, jotka voivat koostua yhdestä tai useammasta muuttujasta ja kertoimesta.

Polynominen regressio luo kaarevan viivan, joka sopii paremmin epälineaariseen dataan. Meidän tapauksessamme, jos sisällytämme neliöidyn `DayOfYear`-muuttujan syöttötietoihin, meidän pitäisi pystyä sovittamaan datamme parabolisella käyrällä, jolla on minimi tiettynä ajankohtana vuoden aikana.

Scikit-learn sisältää hyödyllisen [pipeline-rajapinnan](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), joka yhdistää eri datankäsittelyvaiheet yhteen. **Pipeline** on ketju **estimaattoreita**. Meidän tapauksessamme luomme pipelinen, joka ensin lisää polynomisia ominaisuuksia malliin ja sitten kouluttaa regression:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Käyttämällä `PolynomialFeatures(2)` tarkoittaa, että sisällytämme kaikki toisen asteen polynomit syöttötiedoista. Meidän tapauksessamme tämä tarkoittaa vain `DayOfYear`<sup>2</sup>, mutta kahden syöttömuuttujan X ja Y tapauksessa tämä lisää X<sup>2</sup>, XY ja Y<sup>2</sup>. Voimme myös käyttää korkeamman asteen polynomeja, jos haluamme.

Pipelinea voidaan käyttää samalla tavalla kuin alkuperäistä `LinearRegression`-objektia, eli voimme `fit` pipelinea ja sitten käyttää `predict` saadaksemme ennustetulokset. Tässä on graafi, joka näyttää testidatan ja approksimaatiokäyrän:

<img alt="Polynominen regressio" src="images/poly-results.png" width="50%" />

Polynomista regressiota käyttämällä voimme saada hieman pienemmän MSE:n ja korkeamman determinointikertoimen, mutta ei merkittävästi. Meidän täytyy ottaa huomioon muita ominaisuuksia!

> Voit nähdä, että kurpitsan hinnat ovat alhaisimmillaan jossain Halloweenin tienoilla. Miten selittäisit tämän?

🎃 Onnittelut, loit juuri mallin, joka voi auttaa ennustamaan piirakkakurpitsojen hinnan. Voit todennäköisesti toistaa saman prosessin kaikille kurpitsatyypeille, mutta se olisi työlästä. Opitaan nyt, miten ottaa kurpitsan lajike huomioon mallissamme!

## Kategoriset ominaisuudet

Ihanteellisessa maailmassa haluaisimme pystyä ennustamaan hinnat eri kurpitsalajikkeille käyttämällä samaa mallia. Kuitenkin `Variety`-sarake on hieman erilainen kuin sarakkeet kuten `Month`, koska se sisältää ei-numeerisia arvoja. Tällaisia sarakkeita kutsutaan **kategorisiksi**.

[![ML aloittelijoille - Kategoristen ominaisuuksien ennustaminen lineaarisella regressiolla](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML aloittelijoille - Kategoristen ominaisuuksien ennustaminen lineaarisella regressiolla")

> 🎥 Klikkaa yllä olevaa kuvaa saadaksesi lyhyen videokatsauksen kategoristen ominaisuuksien käytöstä.

Tässä näet, miten keskimääräinen hinta riippuu lajikkeesta:

<img alt="Keskimääräinen hinta lajikkeen mukaan" src="images/price-by-variety.png" width="50%" />

Jotta voimme ottaa lajikkeen huomioon, meidän täytyy ensin muuntaa se numeeriseen muotoon eli **koodata** se. On olemassa useita tapoja tehdä tämä:

* Yksinkertainen **numeerinen koodaus** rakentaa taulukon eri lajikkeista ja korvaa lajikenimen taulukon indeksillä. Tämä ei ole paras idea lineaariselle regressiolle, koska lineaarinen regressio käyttää indeksin todellista numeerista arvoa ja lisää sen tulokseen, kertomalla sen jollain kertoimella. Meidän tapauksessamme indeksin numeron ja hinnan välinen suhde on selvästi epälineaarinen, vaikka varmistaisimme, että indeksit ovat järjestetty jollain erityisellä tavalla.
* **One-hot-koodaus** korvaa `Variety`-sarakkeen neljällä eri sarakkeella, yksi kullekin lajikkeelle. Jokainen sarake sisältää `1`, jos vastaava rivi on tiettyä lajiketta, ja `0` muuten. Tämä tarkoittaa, että lineaarisessa regressiossa on neljä kerrointa, yksi kullekin kurpitsalajikkeelle, jotka vastaavat kyseisen lajikkeen "aloitushintaa" (tai pikemminkin "lisähintaa").

Alla oleva koodi näyttää, miten voimme tehdä one-hot-koodauksen lajikkeelle:

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

Jotta voimme kouluttaa lineaarisen regression käyttäen one-hot-koodattua lajiketta syötteenä, meidän täytyy vain alustaa `X` ja `y` data oikein:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Loppu koodi on sama kuin mitä käytimme aiemmin kouluttaaksemme lineaarisen regression. Jos kokeilet sitä, huomaat, että keskimääräinen neliövirhe (MSE) on suunnilleen sama, mutta saamme paljon korkeamman determinointikertoimen (~77 %). Jotta ennusteet olisivat vielä tarkempia, voimme ottaa huomioon enemmän kategorisia ominaisuuksia sekä numeerisia ominaisuuksia, kuten `Month` tai `DayOfYear`. Jotta saamme yhden suuren ominaisuusjoukon, voimme käyttää `join`-toimintoa:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Tässä otamme myös huomioon `City` ja `Package`-tyypin, mikä antaa meille MSE:n 2.84 (10 %) ja determinointikertoimen 0.94!

## Yhdistäminen

Parhaan mallin luomiseksi voimme käyttää yhdistettyä (one-hot-koodattua kategorista + numeerista) dataa yllä olevasta esimerkistä yhdessä polynomisen regression kanssa. Tässä on täydellinen koodi käteväksi viitteeksi:

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

Tämän pitäisi antaa meille paras determinointikerroin, lähes 97 %, ja MSE=2.23 (~8 % ennustevirhe).

| Malli | MSE | Determinointi |
|-------|-----|---------------|
| `DayOfYear` Lineaarinen | 2.77 (17.2 %) | 0.07 |
| `DayOfYear` Polynominen | 2.73 (17.0 %) | 0.08 |
| `Variety` Lineaarinen | 5.24 (19.7 %) | 0.77 |
| Kaikki ominaisuudet Lineaarinen | 2.84 (10.5 %) | 0.94 |
| Kaikki ominaisuudet Polynominen | 2.23 (8.25 %) | 0.97 |

🏆 Hyvin tehty! Loit neljä regressiomallia yhdessä oppitunnissa ja paransit mallin laatua 97 %:iin. Regressiota käsittelevän osion viimeisessä osassa opit logistisesta regressiosta kategorioiden määrittämiseksi.

---
## 🚀Haaste

Testaa useita eri muuttujia tässä muistikirjassa ja katso, miten korrelaatio vastaa mallin tarkkuutta.

## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Tässä oppitunnissa opimme lineaarisesta regressiosta. On olemassa muita tärkeitä regressiotyyppejä. Lue Stepwise-, Ridge-, Lasso- ja Elasticnet-tekniikoista. Hyvä kurssi lisäopiskeluun on [Stanfordin tilastollisen oppimisen kurssi](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Tehtävä

[Rakenna malli](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.