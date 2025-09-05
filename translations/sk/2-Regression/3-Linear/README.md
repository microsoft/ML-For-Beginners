<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T15:11:10+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "sk"
}
-->
# Vytvorenie regresného modelu pomocou Scikit-learn: štyri spôsoby regresie

![Infografika lineárna vs polynomiálna regresia](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná v R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Úvod 

Doteraz ste preskúmali, čo je regresia, na vzorových údajoch zo súboru údajov o cenách tekvíc, ktorý budeme používať počas celej tejto lekcie. Vizualizovali ste ich pomocou Matplotlibu.

Teraz ste pripravení ponoriť sa hlbšie do regresie pre strojové učenie. Zatiaľ čo vizualizácia vám umožňuje pochopiť údaje, skutočná sila strojového učenia spočíva v _tréningu modelov_. Modely sú trénované na historických údajoch, aby automaticky zachytili závislosti medzi údajmi, a umožňujú vám predpovedať výsledky pre nové údaje, ktoré model predtým nevidel.

V tejto lekcii sa dozviete viac o dvoch typoch regresie: _základná lineárna regresia_ a _polynomiálna regresia_, spolu s niektorými matematickými základmi týchto techník. Tieto modely nám umožnia predpovedať ceny tekvíc na základe rôznych vstupných údajov.

[![ML pre začiatočníkov - Pochopenie lineárnej regresie](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pre začiatočníkov - Pochopenie lineárnej regresie")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o lineárnej regresii.

> Počas celého kurzu predpokladáme minimálne znalosti matematiky a snažíme sa sprístupniť obsah študentom z iných odborov, preto sledujte poznámky, 🧮 výpočty, diagramy a ďalšie nástroje na učenie, ktoré vám pomôžu pochopiť obsah.

### Predpoklady

Teraz by ste mali byť oboznámení so štruktúrou údajov o tekviciach, ktoré skúmame. Nájdete ich prednahrané a predčistené v súbore _notebook.ipynb_ tejto lekcie. V súbore je cena tekvíc zobrazená za bušel v novom dátovom rámci. Uistite sa, že dokážete spustiť tieto notebooky v jadrách vo Visual Studio Code.

### Príprava

Pripomeňme si, že tieto údaje načítavate, aby ste mohli klásť otázky:

- Kedy je najlepší čas na kúpu tekvíc? 
- Akú cenu môžem očakávať za balenie miniatúrnych tekvíc?
- Mám ich kúpiť v polovičných bušlových košoch alebo v 1 1/9 bušlových škatuliach?
Poďme sa hlbšie pozrieť na tieto údaje.

V predchádzajúcej lekcii ste vytvorili Pandas dátový rámec a naplnili ho časťou pôvodného súboru údajov, štandardizujúc ceny podľa bušlu. Týmto spôsobom ste však dokázali zhromaždiť iba približne 400 dátových bodov a iba pre jesenné mesiace.

Pozrite sa na údaje, ktoré sme prednahrali v notebooku tejto lekcie. Údaje sú prednahrané a úvodný bodový graf je vytvorený na zobrazenie údajov podľa mesiacov. Možno môžeme získať trochu viac detailov o povahe údajov ich ďalším čistením.

## Lineárna regresná čiara

Ako ste sa naučili v Lekcii 1, cieľom cvičenia lineárnej regresie je byť schopný nakresliť čiaru na:

- **Ukázanie vzťahov medzi premennými**. Ukázať vzťah medzi premennými
- **Predpovedanie**. Urobiť presné predpovede, kde by nový dátový bod spadol vo vzťahu k tejto čiare. 
 
Typické pre **regresiu metódou najmenších štvorcov** je nakresliť tento typ čiary. Termín 'najmenšie štvorce' znamená, že všetky dátové body obklopujúce regresnú čiaru sú umocnené na druhú a potom sčítané. Ideálne je, aby tento konečný súčet bol čo najmenší, pretože chceme nízky počet chýb, alebo `najmenšie štvorce`. 

Robíme to, pretože chceme modelovať čiaru, ktorá má najmenšiu kumulatívnu vzdialenosť od všetkých našich dátových bodov. Tiež umocňujeme hodnoty na druhú pred ich sčítaním, pretože nás zaujíma ich veľkosť, nie ich smer.

> **🧮 Ukážte mi matematiku** 
> 
> Táto čiara, nazývaná _čiara najlepšieho prispôsobenia_, môže byť vyjadrená [rovnicou](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'vysvetľujúca premenná'. `Y` je 'závislá premenná'. Sklon čiary je `b` a `a` je y-priesečník, ktorý odkazuje na hodnotu `Y`, keď `X = 0`. 
>
>![výpočet sklonu](../../../../2-Regression/3-Linear/images/slope.png)
>
> Najprv vypočítajte sklon `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Inými slovami, a odkazujúc na pôvodnú otázku o údajoch o tekviciach: "predpovedajte cenu tekvice za bušel podľa mesiaca", `X` by odkazovalo na cenu a `Y` by odkazovalo na mesiac predaja. 
>
>![dokončenie rovnice](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Vypočítajte hodnotu Y. Ak platíte okolo $4, musí byť apríl! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika, ktorá vypočítava čiaru, musí demonštrovať sklon čiary, ktorý tiež závisí od priesečníka, alebo kde sa `Y` nachádza, keď `X = 0`.
>
> Metódu výpočtu týchto hodnôt si môžete pozrieť na webovej stránke [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Navštívte tiež [tento kalkulátor metódy najmenších štvorcov](https://www.mathsisfun.com/data/least-squares-calculator.html), aby ste videli, ako hodnoty čísel ovplyvňujú čiaru.

## Korelácia

Ďalší termín, ktorý je potrebné pochopiť, je **koeficient korelácie** medzi danými premennými X a Y. Pomocou bodového grafu môžete rýchlo vizualizovať tento koeficient. Graf s dátovými bodmi rozmiestnenými v úhľadnej čiare má vysokú koreláciu, ale graf s dátovými bodmi rozmiestnenými všade medzi X a Y má nízku koreláciu.

Dobrý model lineárnej regresie bude taký, ktorý má vysoký (bližšie k 1 ako k 0) koeficient korelácie pomocou metódy najmenších štvorcov s regresnou čiarou.

✅ Spustite notebook, ktorý sprevádza túto lekciu, a pozrite sa na bodový graf Mesiac vs Cena. Zdá sa, že údaje spájajúce Mesiac s Cenou za predaj tekvíc majú podľa vašej vizuálnej interpretácie bodového grafu vysokú alebo nízku koreláciu? Zmení sa to, ak použijete jemnejšie meranie namiesto `Mesiac`, napr. *deň v roku* (t. j. počet dní od začiatku roka)?

V nasledujúcom kóde predpokladáme, že sme vyčistili údaje a získali dátový rámec nazvaný `new_pumpkins`, podobný nasledujúcemu:

ID | Mesiac | DeňVroku | Typ | Mesto | Balenie | Nízka cena | Vysoká cena | Cena
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bušlové kartóny | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bušlové kartóny | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bušlové kartóny | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bušlové kartóny | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bušlové kartóny | 15.0 | 15.0 | 13.636364

> Kód na čistenie údajov je dostupný v [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Vykonali sme rovnaké kroky čistenia ako v predchádzajúcej lekcii a vypočítali sme stĺpec `DeňVroku` pomocou nasledujúceho výrazu: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Teraz, keď rozumiete matematike za lineárnou regresiou, poďme vytvoriť regresný model, aby sme zistili, ktorý balík tekvíc bude mať najlepšie ceny tekvíc. Niekto, kto kupuje tekvice na sviatočnú tekvicovú záhradu, by mohol chcieť tieto informácie, aby optimalizoval svoje nákupy balíkov tekvíc pre záhradu.

## Hľadanie korelácie

[![ML pre začiatočníkov - Hľadanie korelácie: Kľúč k lineárnej regresii](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pre začiatočníkov - Hľadanie korelácie: Kľúč k lineárnej regresii")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o korelácii.

Z predchádzajúcej lekcie ste pravdepodobne videli, že priemerná cena za rôzne mesiace vyzerá takto:

<img alt="Priemerná cena podľa mesiaca" src="../2-Data/images/barchart.png" width="50%"/>

To naznačuje, že by mala existovať nejaká korelácia, a môžeme skúsiť trénovať model lineárnej regresie na predpovedanie vzťahu medzi `Mesiac` a `Cena`, alebo medzi `DeňVroku` a `Cena`. Tu je bodový graf, ktorý ukazuje druhý vzťah:

<img alt="Bodový graf Cena vs. Deň v roku" src="images/scatter-dayofyear.png" width="50%" /> 

Pozrime sa, či existuje korelácia pomocou funkcie `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Zdá sa, že korelácia je pomerne malá, -0.15 podľa `Mesiac` a -0.17 podľa `DeňVroku`, ale mohol by existovať iný dôležitý vzťah. Zdá sa, že existujú rôzne zhluky cien zodpovedajúce rôznym odrodám tekvíc. Na potvrdenie tejto hypotézy nakreslime každú kategóriu tekvíc pomocou inej farby. Pri prechode parametra `ax` do funkcie `scatter` môžeme nakresliť všetky body na rovnaký graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Bodový graf Cena vs. Deň v roku" src="images/scatter-dayofyear-color.png" width="50%" /> 

Naše vyšetrovanie naznačuje, že odroda má väčší vplyv na celkovú cenu ako skutočný dátum predaja. Môžeme to vidieť na stĺpcovom grafe:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stĺpcový graf cena vs odroda" src="images/price-by-variety.png" width="50%" /> 

Zamerajme sa na chvíľu iba na jednu odrodu tekvíc, 'pie type', a pozrime sa, aký vplyv má dátum na cenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Bodový graf Cena vs. Deň v roku" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Ak teraz vypočítame koreláciu medzi `Cena` a `DeňVroku` pomocou funkcie `corr`, dostaneme hodnotu okolo `-0.27` - čo znamená, že trénovanie prediktívneho modelu má zmysel.

> Pred trénovaním modelu lineárnej regresie je dôležité zabezpečiť, aby naše údaje boli čisté. Lineárna regresia nefunguje dobre s chýbajúcimi hodnotami, preto má zmysel zbaviť sa všetkých prázdnych buniek:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Ďalším prístupom by bolo vyplniť tieto prázdne hodnoty priemernými hodnotami z príslušného stĺpca.

## Jednoduchá lineárna regresia

[![ML pre začiatočníkov - Lineárna a polynomiálna regresia pomocou Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pre začiatočníkov - Lineárna a polynomiálna regresia pomocou Scikit-learn")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o lineárnej a polynomiálnej regresii.

Na trénovanie nášho modelu lineárnej regresie použijeme knižnicu **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Začíname oddelením vstupných hodnôt (príznakov) a očakávaného výstupu (označenia) do samostatných numpy polí:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Všimnite si, že sme museli vykonať `reshape` na vstupných údajoch, aby ich balík lineárnej regresie správne pochopil. Lineárna regresia očakáva 2D pole ako vstup, kde každý riadok poľa zodpovedá vektoru vstupných príznakov. V našom prípade, keďže máme iba jeden vstup, potrebujeme pole s tvarom N×1, kde N je veľkosť súboru údajov.

Potom musíme rozdeliť údaje na trénovací a testovací súbor údajov, aby sme mohli po tréningu overiť náš model:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Nakoniec, samotné trénovanie modelu lineárnej regresie trvá iba dva riadky kódu. Definujeme objekt `LinearRegression` a prispôsobíme ho našim údajom pomocou metódy `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` po prispôsobení obsahuje všetky koeficienty regresie, ku ktorým je možné pristupovať pomocou vlastnosti `.coef_`. V našom prípade existuje iba jeden koeficient, ktorý by mal byť okolo `-0.017`. To znamená, že ceny sa zdajú klesať trochu s časom, ale nie príliš, približne o 2 centy za deň. Môžeme tiež pristupovať k priesečníku regresie s Y-osou pomocou `lin_reg.intercept_` - bude to okolo `21` v našom prípade, čo naznačuje cenu na začiatku roka.

Aby sme videli, aký presný je náš model, môžeme predpovedať ceny na testovacom súbore údajov a potom zmerať, ako blízko sú naše predpovede k očakávaným hodnotám. To sa dá urobiť pomocou metriky strednej kvadratickej chyby (MSE), ktorá je priemerom všetkých kvadratických rozdielov medzi očakávanou a predpovedanou hodnotou.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Naša chyba sa zdá byť okolo 2 bodov, čo je ~17 %. Nie je to príliš dobré. Ďalším indikátorom kvality modelu je **koeficient determinácie**, ktorý môžeme získať takto:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Ak je hodnota 0, znamená to, že model neberie do úvahy vstupné údaje a funguje ako *najhorší lineárny prediktor*, čo je jednoducho priemerná hodnota výsledku. Hodnota 1 znamená, že môžeme dokonale predpovedať všetky očakávané výstupy. V našom prípade je koeficient okolo 0.06, čo je pomerne nízke.

Môžeme tiež vykresliť testovacie údaje spolu s regresnou čiarou, aby sme lepšie videli, ako regresia funguje v našom prípade:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Lineárna regresia" src="images/linear-results.png" width="50%" />

## Polynomická regresia

Ďalším typom lineárnej regresie je polynomická regresia. Zatiaľ čo niekedy existuje lineárny vzťah medzi premennými – čím väčší objem tekvice, tým vyššia cena – niekedy tieto vzťahy nemožno vykresliť ako rovinu alebo priamku.

✅ Tu sú [niektoré ďalšie príklady](https://online.stat.psu.edu/stat501/lesson/9/9.8) údajov, ktoré by mohli využiť polynomickú regresiu.

Pozrite sa znova na vzťah medzi dátumom a cenou. Zdá sa, že tento bodový graf by mal byť nevyhnutne analyzovaný priamkou? Nemôžu ceny kolísať? V tomto prípade môžete vyskúšať polynomickú regresiu.

✅ Polynómy sú matematické výrazy, ktoré môžu pozostávať z jednej alebo viacerých premenných a koeficientov.

Polynomická regresia vytvára zakrivenú čiaru, ktorá lepšie zodpovedá nelineárnym údajom. V našom prípade, ak do vstupných údajov zahrnieme premennú `DayOfYear` na druhú, mali by sme byť schopní prispôsobiť naše údaje parabolickou krivkou, ktorá bude mať minimum v určitom bode počas roka.

Scikit-learn obsahuje užitočné [API pre pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), ktoré kombinuje rôzne kroky spracovania údajov. **Pipeline** je reťaz **odhadovateľov**. V našom prípade vytvoríme pipeline, ktorá najskôr pridá polynomické prvky do nášho modelu a potom trénuje regresiu:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Použitie `PolynomialFeatures(2)` znamená, že zahrnieme všetky polynómy druhého stupňa zo vstupných údajov. V našom prípade to bude znamenať len `DayOfYear`<sup>2</sup>, ale pri dvoch vstupných premenných X a Y to pridá X<sup>2</sup>, XY a Y<sup>2</sup>. Môžeme tiež použiť polynómy vyššieho stupňa, ak chceme.

Pipeline môžeme používať rovnakým spôsobom ako pôvodný objekt `LinearRegression`, t.j. môžeme pipeline `fit` a potom použiť `predict` na získanie výsledkov predikcie. Tu je graf zobrazujúci testovacie údaje a aproximačnú krivku:

<img alt="Polynomická regresia" src="images/poly-results.png" width="50%" />

Použitím polynomickej regresie môžeme dosiahnuť mierne nižšie MSE a vyššiu determináciu, ale nie významne. Musíme zohľadniť ďalšie prvky!

> Vidíte, že minimálne ceny tekvíc sú pozorované niekde okolo Halloweenu. Ako to môžete vysvetliť?

🎃 Gratulujeme, práve ste vytvorili model, ktorý môže pomôcť predpovedať cenu tekvíc na koláče. Pravdepodobne môžete zopakovať rovnaký postup pre všetky typy tekvíc, ale to by bolo zdĺhavé. Poďme sa teraz naučiť, ako zohľadniť odrodu tekvíc v našom modeli!

## Kategorické prvky

V ideálnom svete chceme byť schopní predpovedať ceny pre rôzne odrody tekvíc pomocou rovnakého modelu. Stĺpec `Variety` je však trochu odlišný od stĺpcov ako `Month`, pretože obsahuje nenumerické hodnoty. Takéto stĺpce sa nazývajú **kategorické**.

[![ML pre začiatočníkov - Predikcia kategorických prvkov pomocou lineárnej regresie](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pre začiatočníkov - Predikcia kategorických prvkov pomocou lineárnej regresie")

> 🎥 Kliknite na obrázok vyššie pre krátky video prehľad o používaní kategorických prvkov.

Tu môžete vidieť, ako priemerná cena závisí od odrody:

<img alt="Priemerná cena podľa odrody" src="images/price-by-variety.png" width="50%" />

Aby sme zohľadnili odrodu, musíme ju najskôr previesť na numerickú formu, alebo ju **zakódovať**. Existuje niekoľko spôsobov, ako to môžeme urobiť:

* Jednoduché **numerické kódovanie** vytvorí tabuľku rôznych odrôd a potom nahradí názov odrody indexom v tejto tabuľke. Toto nie je najlepší nápad pre lineárnu regresiu, pretože lineárna regresia berie skutočnú numerickú hodnotu indexu a pridáva ju k výsledku, násobiac ju nejakým koeficientom. V našom prípade je vzťah medzi číslom indexu a cenou jasne nelineárny, aj keď zabezpečíme, že indexy sú usporiadané určitým spôsobom.
* **One-hot kódovanie** nahradí stĺpec `Variety` štyrmi rôznymi stĺpcami, jeden pre každú odrodu. Každý stĺpec bude obsahovať `1`, ak príslušný riadok patrí danej odrode, a `0` inak. To znamená, že v lineárnej regresii budú štyri koeficienty, jeden pre každú odrodu tekvíc, zodpovedné za "východiskovú cenu" (alebo skôr "dodatočnú cenu") pre danú odrodu.

Nižšie uvedený kód ukazuje, ako môžeme zakódovať odrodu pomocou one-hot kódovania:

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

Na trénovanie lineárnej regresie pomocou one-hot zakódovanej odrody ako vstupu stačí správne inicializovať údaje `X` a `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

Zvyšok kódu je rovnaký ako ten, ktorý sme použili vyššie na trénovanie lineárnej regresie. Ak to vyskúšate, uvidíte, že stredná kvadratická chyba je približne rovnaká, ale získame oveľa vyšší koeficient determinácie (~77 %). Na získanie ešte presnejších predpovedí môžeme zohľadniť viac kategorických prvkov, ako aj numerické prvky, ako `Month` alebo `DayOfYear`. Na získanie jedného veľkého poľa prvkov môžeme použiť `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Tu tiež zohľadňujeme `City` a typ balenia `Package`, čo nám dáva MSE 2.84 (10 %) a determináciu 0.94!

## Spojenie všetkého dohromady

Na vytvorenie najlepšieho modelu môžeme použiť kombinované (one-hot zakódované kategorické + numerické) údaje z vyššie uvedeného príkladu spolu s polynomickou regresiou. Tu je kompletný kód pre vaše pohodlie:

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

Toto by nám malo dať najlepší koeficient determinácie takmer 97 % a MSE=2.23 (~8 % predikčná chyba).

| Model | MSE | Determinácia |  
|-------|-----|-------------|  
| `DayOfYear` Lineárny | 2.77 (17.2 %) | 0.07 |  
| `DayOfYear` Polynomický | 2.73 (17.0 %) | 0.08 |  
| `Variety` Lineárny | 5.24 (19.7 %) | 0.77 |  
| Všetky prvky Lineárny | 2.84 (10.5 %) | 0.94 |  
| Všetky prvky Polynomický | 2.23 (8.25 %) | 0.97 |  

🏆 Výborne! Vytvorili ste štyri regresné modely v jednej lekcii a zlepšili kvalitu modelu na 97 %. V poslednej časti o regresii sa naučíte o logistickej regresii na určenie kategórií.

---

## 🚀Výzva

Otestujte niekoľko rôznych premenných v tomto notebooku, aby ste videli, ako korelácia zodpovedá presnosti modelu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samoštúdium

V tejto lekcii sme sa naučili o lineárnej regresii. Existujú aj ďalšie dôležité typy regresie. Prečítajte si o technikách Stepwise, Ridge, Lasso a Elasticnet. Dobrý kurz na štúdium je [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Zadanie

[Postavte model](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.