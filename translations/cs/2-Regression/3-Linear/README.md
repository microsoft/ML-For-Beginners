<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-04T23:20:43+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "cs"
}
-->
# Vytvoření regresního modelu pomocí Scikit-learn: čtyři způsoby regresí

![Infografika lineární vs. polynomiální regrese](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tato lekce je dostupná v R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Úvod 

Doposud jste prozkoumali, co je regrese, na vzorových datech získaných z datasetu cen dýní, který budeme používat v celé této lekci. Také jste ji vizualizovali pomocí Matplotlibu.

Nyní jste připraveni ponořit se hlouběji do regresí pro strojové učení. Zatímco vizualizace vám umožňuje pochopit data, skutečná síla strojového učení spočívá v _trénování modelů_. Modely jsou trénovány na historických datech, aby automaticky zachytily závislosti mezi daty, a umožňují vám předpovídat výsledky pro nová data, která model dosud neviděl.

V této lekci se dozvíte více o dvou typech regresí: _základní lineární regrese_ a _polynomiální regrese_, spolu s některými matematickými základy těchto technik. Tyto modely nám umožní předpovídat ceny dýní na základě různých vstupních dat.

[![ML pro začátečníky - Porozumění lineární regresi](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pro začátečníky - Porozumění lineární regresi")

> 🎥 Klikněte na obrázek výše pro krátký video přehled o lineární regresi.

> V celém tomto kurzu předpokládáme minimální znalosti matematiky a snažíme se ji zpřístupnit studentům z jiných oborů, takže sledujte poznámky, 🧮 výpočty, diagramy a další učební nástroje, které vám pomohou s pochopením.

### Předpoklady

Nyní byste měli být obeznámeni se strukturou dat o dýních, která zkoumáme. Najdete je předem načtená a předem vyčištěná v souboru _notebook.ipynb_ této lekce. V souboru je cena dýní zobrazena za bušl v novém datovém rámci. Ujistěte se, že můžete tyto notebooky spustit v jádrech ve Visual Studio Code.

### Příprava

Připomeňme si, že tato data načítáte, abyste si mohli klást otázky:

- Kdy je nejlepší čas na nákup dýní? 
- Jakou cenu mohu očekávat za balení mini dýní?
- Mám je koupit v polovičních bušlových koších nebo v krabici o velikosti 1 1/9 bušlu?
Pokračujme v prozkoumávání těchto dat.

V předchozí lekci jste vytvořili datový rámec Pandas a naplnili jej částí původního datasetu, standardizovali ceny podle bušlu. Tímto způsobem jste však byli schopni shromáždit pouze asi 400 datových bodů a pouze pro podzimní měsíce.

Podívejte se na data, která jsme předem načetli v doprovodném notebooku této lekce. Data jsou předem načtena a počáteční bodový graf je vytvořen, aby ukázal data podle měsíců. Možná můžeme získat trochu více detailů o povaze dat jejich dalším čištěním.

## Lineární regresní přímka

Jak jste se naučili v lekci 1, cílem cvičení lineární regrese je být schopen vykreslit přímku, která:

- **Ukazuje vztahy mezi proměnnými**. Ukazuje vztah mezi proměnnými
- **Dělá předpovědi**. Umožňuje přesně předpovědět, kde by nový datový bod spadal ve vztahu k této přímce. 
 
Je typické pro **regresi metodou nejmenších čtverců**, že se kreslí tento typ přímky. Termín 'nejmenší čtverce' znamená, že všechny datové body obklopující regresní přímku jsou umocněny na druhou a poté sečteny. Ideálně je tento konečný součet co nejmenší, protože chceme nízký počet chyb, tedy `nejmenší čtverce`. 

Děláme to proto, že chceme modelovat přímku, která má nejmenší kumulativní vzdálenost od všech našich datových bodů. Také umocňujeme hodnoty na druhou před jejich sečtením, protože nás zajímá jejich velikost, nikoli směr.

> **🧮 Ukažte mi matematiku** 
> 
> Tato přímka, nazývaná _přímka nejlepšího přizpůsobení_, může být vyjádřena [rovnicí](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'vysvětlující proměnná'. `Y` je 'závislá proměnná'. Sklon přímky je `b` a `a` je průsečík s osou Y, což odkazuje na hodnotu `Y`, když `X = 0`. 
>
>![výpočet sklonu](../../../../2-Regression/3-Linear/images/slope.png)
>
> Nejprve vypočítejte sklon `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Jinými slovy, a odkazujíc na původní otázku o datech dýní: "předpovězte cenu dýně za bušl podle měsíce", `X` by odkazovalo na cenu a `Y` by odkazovalo na měsíc prodeje. 
>
>![dokončení rovnice](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Vypočítejte hodnotu Y. Pokud platíte kolem $4, musí být duben! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika, která vypočítává přímku, musí ukázat sklon přímky, který také závisí na průsečíku, tedy na tom, kde se `Y` nachází, když `X = 0`.
>
> Metodu výpočtu těchto hodnot můžete pozorovat na webu [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Navštivte také [tento kalkulátor nejmenších čtverců](https://www.mathsisfun.com/data/least-squares-calculator.html), abyste viděli, jak hodnoty čísel ovlivňují přímku.

## Korelace

Ještě jeden termín, který je třeba pochopit, je **koeficient korelace** mezi danými proměnnými X a Y. Pomocí bodového grafu můžete rychle vizualizovat tento koeficient. Graf s datovými body rozptýlenými v úhledné přímce má vysokou korelaci, ale graf s datovými body rozptýlenými všude mezi X a Y má nízkou korelaci.

Dobrý model lineární regrese bude takový, který má vysoký (blíže k 1 než k 0) koeficient korelace pomocí metody nejmenších čtverců s regresní přímkou.

✅ Spusťte notebook doprovázející tuto lekci a podívejte se na bodový graf Měsíc vs. Cena. Zdá se, že data spojující Měsíc s Cenou za prodej dýní mají podle vašeho vizuálního hodnocení bodového grafu vysokou nebo nízkou korelaci? Změní se to, pokud použijete jemnější měřítko místo `Měsíc`, např. *den v roce* (tj. počet dní od začátku roku)?

V níže uvedeném kódu předpokládáme, že jsme data vyčistili a získali datový rámec nazvaný `new_pumpkins`, podobný následujícímu:

ID | Měsíc | DenVRoce | Druh | Město | Balení | Nízká cena | Vysoká cena | Cena
---|-------|-----------|---------|------|---------|-----------|------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kód pro čištění dat je dostupný v [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Provedli jsme stejné kroky čištění jako v předchozí lekci a vypočítali sloupec `DenVRoce` pomocí následujícího výrazu: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Nyní, když máte pochopení matematiky za lineární regresí, vytvořme regresní model, abychom zjistili, zda můžeme předpovědět, které balení dýní bude mít nejlepší ceny dýní. Někdo, kdo kupuje dýně pro sváteční dýňovou zahradu, by mohl chtít tyto informace, aby mohl optimalizovat své nákupy balení dýní pro zahradu.

## Hledání korelace

[![ML pro začátečníky - Hledání korelace: Klíč k lineární regresi](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pro začátečníky - Hledání korelace: Klíč k lineární regresi")

> 🎥 Klikněte na obrázek výše pro krátký video přehled o korelaci.

Z předchozí lekce jste pravděpodobně viděli, že průměrná cena pro různé měsíce vypadá takto:

<img alt="Průměrná cena podle měsíce" src="../2-Data/images/barchart.png" width="50%"/>

To naznačuje, že by měla existovat nějaká korelace, a můžeme zkusit trénovat model lineární regrese, abychom předpověděli vztah mezi `Měsíc` a `Cena`, nebo mezi `DenVRoce` a `Cena`. Zde je bodový graf, který ukazuje druhý vztah:

<img alt="Bodový graf Cena vs. Den v roce" src="images/scatter-dayofyear.png" width="50%" /> 

Podívejme se, zda existuje korelace pomocí funkce `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Zdá se, že korelace je poměrně malá, -0.15 podle `Měsíc` a -0.17 podle `DenVRoce`, ale mohlo by existovat jiné důležité spojení. Zdá se, že existují různé shluky cen odpovídající různým druhům dýní. Abychom tuto hypotézu potvrdili, vykresleme každou kategorii dýní pomocí jiné barvy. Předáním parametru `ax` funkci pro vykreslení bodového grafu můžeme vykreslit všechny body na stejný graf:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Bodový graf Cena vs. Den v roce" src="images/scatter-dayofyear-color.png" width="50%" /> 

Naše zkoumání naznačuje, že druh má větší vliv na celkovou cenu než skutečné datum prodeje. Můžeme to vidět na sloupcovém grafu:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Sloupcový graf cena vs. druh" src="images/price-by-variety.png" width="50%" /> 

Zaměřme se nyní pouze na jeden druh dýní, 'pie type', a podívejme se, jaký vliv má datum na cenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Bodový graf Cena vs. Den v roce" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Pokud nyní vypočítáme korelaci mezi `Cena` a `DenVRoce` pomocí funkce `corr`, dostaneme něco jako `-0.27` - což znamená, že trénování prediktivního modelu má smysl.

> Před trénováním modelu lineární regrese je důležité zajistit, že naše data jsou čistá. Lineární regrese nefunguje dobře s chybějícími hodnotami, proto má smysl zbavit se všech prázdných buněk:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Dalším přístupem by bylo vyplnit tyto prázdné hodnoty průměrnými hodnotami z odpovídajícího sloupce.

## Jednoduchá lineární regrese

[![ML pro začátečníky - Lineární a polynomiální regrese pomocí Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pro začátečníky - Lineární a polynomiální regrese pomocí Scikit-learn")

> 🎥 Klikněte na obrázek výše pro krátký video přehled o lineární a polynomiální regresi.

Pro trénování našeho modelu lineární regrese použijeme knihovnu **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Začneme oddělením vstupních hodnot (features) a očekávaného výstupu (label) do samostatných numpy polí:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Všimněte si, že jsme museli provést `reshape` na vstupních datech, aby je balíček lineární regrese správně pochopil. Lineární regrese očekává 2D pole jako vstup, kde každý řádek pole odpovídá vektoru vstupních vlastností. V našem případě, protože máme pouze jeden vstup, potřebujeme pole s tvarem N×1, kde N je velikost datasetu.

Poté musíme data rozdělit na trénovací a testovací dataset, abychom mohli po trénování ověřit náš model:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Nakonec samotné trénování modelu lineární regrese zabere pouze dva řádky kódu. Definujeme objekt `LinearRegression` a přizpůsobíme ho našim datům pomocí metody `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` po přizpůsobení obsahuje všechny koeficienty regrese, které lze získat pomocí vlastnosti `.coef_`. V našem případě je pouze jeden koeficient, který by měl být kolem `-0.017`. To znamená, že ceny se zdají s časem mírně klesat, ale ne příliš, asi o 2 centy za den. Průsečík regrese s osou Y můžeme také získat pomocí `lin_reg.intercept_` - bude kolem `21` v našem případě, což naznačuje cenu na začátku roku.

Abychom viděli, jak přesný je náš model, můžeme předpovědět ceny na testovacím datasetu a poté změřit, jak blízko jsou naše předpovědi očekávaným hodnotám. To lze provést pomocí metriky střední kvadratické chyby (MSE), což je průměr všech kvadratických rozdílů mezi očekávanou a předpovězenou hodnotou.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Naše chyba se zdá být kolem 2 bodů, což je ~17 %. Nic moc. Dalším ukazatelem kvality modelu je **koeficient determinace**, který lze získat takto:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Pokud je hodnota 0, znamená to, že model nebere v úvahu vstupní data a funguje jako *nejhorší lineární prediktor*, což je jednoduše průměrná hodnota výsledku. Hodnota 1 znamená, že můžeme dokonale předpovědět všechny očekávané výstupy. V našem případě je koeficient kolem 0,06, což je poměrně nízké.

Můžeme také vykreslit testovací data spolu s regresní přímkou, abychom lépe viděli, jak regresní analýza v našem případě funguje:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Lineární regrese" src="images/linear-results.png" width="50%" />

## Polynomická regrese

Dalším typem lineární regrese je polynomická regrese. Zatímco někdy existuje lineární vztah mezi proměnnými – čím větší je objem dýně, tím vyšší je cena – někdy tyto vztahy nelze vykreslit jako rovinu nebo přímku.

✅ Zde jsou [některé další příklady](https://online.stat.psu.edu/stat501/lesson/9/9.8) dat, která by mohla využít polynomickou regresi.

Podívejte se znovu na vztah mezi datem a cenou. Zdá se, že tento bodový graf by měl být nutně analyzován přímkou? Nemohou ceny kolísat? V tomto případě můžete zkusit polynomickou regresi.

✅ Polynomy jsou matematické výrazy, které mohou obsahovat jednu nebo více proměnných a koeficientů.

Polynomická regrese vytváří zakřivenou čáru, která lépe odpovídá nelineárním datům. V našem případě, pokud do vstupních dat zahrneme kvadratickou proměnnou `DayOfYear`, měli bychom být schopni přizpůsobit naše data parabolické křivce, která bude mít minimum v určitém bodě během roku.

Scikit-learn obsahuje užitečné [API pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) pro kombinaci různých kroků zpracování dat dohromady. **Pipeline** je řetězec **odhadovačů**. V našem případě vytvoříme pipeline, která nejprve přidá polynomické prvky do našeho modelu a poté provede trénink regrese:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

Použití `PolynomialFeatures(2)` znamená, že zahrneme všechny polynomy druhého stupně ze vstupních dat. V našem případě to bude jednoduše `DayOfYear`<sup>2</sup>, ale pokud máme dvě vstupní proměnné X a Y, přidá to X<sup>2</sup>, XY a Y<sup>2</sup>. Můžeme také použít polynomy vyššího stupně, pokud chceme.

Pipeline lze použít stejným způsobem jako původní objekt `LinearRegression`, tj. můžeme pipeline `fit` a poté použít `predict` k získání výsledků predikce. Zde je graf zobrazující testovací data a aproximační křivku:

<img alt="Polynomická regrese" src="images/poly-results.png" width="50%" />

Použitím polynomické regrese můžeme dosáhnout mírně nižší MSE a vyšší determinace, ale ne výrazně. Musíme vzít v úvahu další prvky!

> Vidíte, že minimální ceny dýní jsou pozorovány někde kolem Halloweenu. Jak to můžete vysvětlit?

🎃 Gratulujeme, právě jste vytvořili model, který může pomoci předpovědět cenu dýní na koláče. Pravděpodobně můžete stejný postup zopakovat pro všechny typy dýní, ale to by bylo zdlouhavé. Naučme se nyní, jak vzít do úvahy odrůdu dýní v našem modelu!

## Kategorické prvky

V ideálním světě bychom chtěli být schopni předpovědět ceny pro různé odrůdy dýní pomocí stejného modelu. Sloupec `Variety` je však poněkud odlišný od sloupců jako `Month`, protože obsahuje nenumerické hodnoty. Takové sloupce se nazývají **kategorické**.

[![ML pro začátečníky - Predikce kategorických prvků pomocí lineární regrese](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pro začátečníky - Predikce kategorických prvků pomocí lineární regrese")

> 🎥 Klikněte na obrázek výše pro krátký přehled o použití kategorických prvků.

Zde můžete vidět, jak průměrná cena závisí na odrůdě:

<img alt="Průměrná cena podle odrůdy" src="images/price-by-variety.png" width="50%" />

Abychom vzali odrůdu v úvahu, musíme ji nejprve převést na číselnou formu, nebo ji **zakódovat**. Existuje několik způsobů, jak to udělat:

* Jednoduché **číselné kódování** vytvoří tabulku různých odrůd a poté nahradí název odrůdy indexem v této tabulce. To není nejlepší nápad pro lineární regresi, protože lineární regrese bere skutečnou číselnou hodnotu indexu a přidává ji k výsledku, násobí ji nějakým koeficientem. V našem případě je vztah mezi číslem indexu a cenou zjevně nelineární, i když zajistíme, že indexy budou seřazeny nějakým specifickým způsobem.
* **One-hot kódování** nahradí sloupec `Variety` čtyřmi různými sloupci, jeden pro každou odrůdu. Každý sloupec bude obsahovat `1`, pokud odpovídající řádek patří dané odrůdě, a `0` jinak. To znamená, že v lineární regresi budou čtyři koeficienty, jeden pro každou odrůdu dýní, odpovědné za "výchozí cenu" (nebo spíše "dodatečnou cenu") pro danou odrůdu.

Níže uvedený kód ukazuje, jak můžeme provést one-hot kódování odrůdy:

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

Abychom provedli trénink lineární regrese s použitím one-hot kódované odrůdy jako vstupu, stačí správně inicializovat data `X` a `y`:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Zbytek kódu je stejný jako ten, který jsme použili výše pro trénink lineární regrese. Pokud to vyzkoušíte, uvidíte, že střední kvadratická chyba je přibližně stejná, ale získáme mnohem vyšší koeficient determinace (~77 %). Pro ještě přesnější predikce můžeme vzít v úvahu více kategorických prvků, stejně jako numerické prvky, jako `Month` nebo `DayOfYear`. Abychom získali jedno velké pole prvků, můžeme použít `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Zde také bereme v úvahu `City` a typ balení `Package`, což nám dává MSE 2,84 (10 %) a determinaci 0,94!

## Spojení všeho dohromady

Abychom vytvořili nejlepší model, můžeme použít kombinovaná data (one-hot kódované kategorické + numerické) z výše uvedeného příkladu spolu s polynomickou regresí. Zde je kompletní kód pro vaše pohodlí:

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

To by nám mělo dát nejlepší koeficient determinace téměř 97 % a MSE=2,23 (~8 % chybovost predikce).

| Model | MSE | Determinace |
|-------|-----|-------------|
| `DayOfYear` Lineární | 2,77 (17,2 %) | 0,07 |
| `DayOfYear` Polynomická | 2,73 (17,0 %) | 0,08 |
| `Variety` Lineární | 5,24 (19,7 %) | 0,77 |
| Všechny prvky Lineární | 2,84 (10,5 %) | 0,94 |
| Všechny prvky Polynomická | 2,23 (8,25 %) | 0,97 |

🏆 Skvělá práce! Vytvořili jste čtyři regresní modely v jedné lekci a zlepšili kvalitu modelu na 97 %. V poslední části o regresi se naučíte o logistické regresi pro určení kategorií.

---
## 🚀Výzva

Otestujte několik různých proměnných v tomto notebooku a zjistěte, jak korelace odpovídá přesnosti modelu.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

V této lekci jsme se naučili o lineární regresi. Existují další důležité typy regrese. Přečtěte si o technikách Stepwise, Ridge, Lasso a Elasticnet. Dobrou možností pro další studium je [kurz statistického učení Stanfordu](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Úkol 

[Postavte model](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.