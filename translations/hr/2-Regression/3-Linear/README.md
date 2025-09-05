<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T11:27:31+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "hr"
}
-->
# Izgradnja regresijskog modela koristeći Scikit-learn: četiri načina regresije

![Infografika linearne i polinomne regresije](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ova lekcija je dostupna u R!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Uvod 

Do sada ste istražili što je regresija koristeći uzorke podataka iz skupa podataka o cijenama bundeva, koji ćemo koristiti tijekom ove lekcije. Također ste vizualizirali podatke koristeći Matplotlib.

Sada ste spremni dublje zaroniti u regresiju za strojno učenje. Dok vizualizacija omogućuje razumijevanje podataka, prava snaga strojnog učenja dolazi iz _treniranja modela_. Modeli se treniraju na povijesnim podacima kako bi automatski uhvatili ovisnosti podataka i omogućili predviđanje ishoda za nove podatke koje model nije prethodno vidio.

U ovoj lekciji naučit ćete više o dvije vrste regresije: _osnovnoj linearnoj regresiji_ i _polinomnoj regresiji_, zajedno s nekim matematičkim osnovama ovih tehnika. Ti modeli omogućit će nam predviđanje cijena bundeva ovisno o različitim ulaznim podacima.

[![ML za početnike - Razumijevanje linearne regresije](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML za početnike - Razumijevanje linearne regresije")

> 🎥 Kliknite na sliku iznad za kratki video pregled linearne regresije.

> Tijekom ovog kurikuluma pretpostavljamo minimalno znanje matematike i nastojimo ga učiniti dostupnim studentima iz drugih područja, pa obratite pažnju na bilješke, 🧮 matematičke primjere, dijagrame i druge alate za učenje koji pomažu u razumijevanju.

### Preduvjeti

Do sada biste trebali biti upoznati sa strukturom podataka o bundevama koje analiziramo. Možete ih pronaći unaprijed učitane i očišćene u datoteci _notebook.ipynb_ ove lekcije. U datoteci je cijena bundeve prikazana po bušelu u novom podatkovnom okviru. Provjerite možete li pokrenuti ove bilježnice u kernelima u Visual Studio Codeu.

### Priprema

Podsjetnik: učitavate ove podatke kako biste postavili pitanja o njima.

- Kada je najbolje vrijeme za kupnju bundeva? 
- Koju cijenu mogu očekivati za kutiju minijaturnih bundeva?
- Trebam li ih kupiti u košarama od pola bušela ili u kutijama od 1 1/9 bušela?
Nastavimo istraživati ove podatke.

U prethodnoj lekciji kreirali ste Pandas podatkovni okvir i popunili ga dijelom izvornog skupa podataka, standardizirajući cijene po bušelu. Međutim, na taj način uspjeli ste prikupiti samo oko 400 podatkovnih točaka i to samo za jesenske mjesece.

Pogledajte podatke koje smo unaprijed učitali u bilježnici koja prati ovu lekciju. Podaci su unaprijed učitani, a početni dijagram raspršenja je nacrtan kako bi prikazao podatke po mjesecima. Možda možemo dobiti malo više detalja o prirodi podataka ako ih dodatno očistimo.

## Linija linearne regresije

Kao što ste naučili u Lekciji 1, cilj vježbe linearne regresije je moći nacrtati liniju kako bi:

- **Prikazali odnose varijabli**. Prikazali odnos između varijabli
- **Napravili predviđanja**. Napravili točna predviđanja o tome gdje bi nova podatkovna točka mogla pasti u odnosu na tu liniju.

Tipično je za **regresiju metodom najmanjih kvadrata** nacrtati ovu vrstu linije. Pojam 'najmanji kvadrati' znači da su sve podatkovne točke oko regresijske linije kvadrirane i zatim zbrojene. Idealno, taj konačni zbroj je što manji, jer želimo mali broj pogrešaka, ili `najmanje kvadrate`.

To radimo jer želimo modelirati liniju koja ima najmanju kumulativnu udaljenost od svih naših podatkovnih točaka. Također kvadriramo vrijednosti prije zbrajanja jer nas zanima njihova veličina, a ne smjer.

> **🧮 Pokaži mi matematiku** 
> 
> Ova linija, nazvana _linija najboljeg pristajanja_, može se izraziti [jednadžbom](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` je 'objašnjavajuća varijabla'. `Y` je 'ovisna varijabla'. Nagib linije je `b`, a `a` je presjek s y-osom, koji se odnosi na vrijednost `Y` kada je `X = 0`. 
>
>![izračunaj nagib](../../../../2-Regression/3-Linear/images/slope.png)
>
> Prvo, izračunajte nagib `b`. Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Drugim riječima, i referirajući se na izvorno pitanje o podacima o bundevama: "predvidite cijenu bundeve po bušelu po mjesecu", `X` bi se odnosio na cijenu, a `Y` na mjesec prodaje. 
>
>![dovršite jednadžbu](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Izračunajte vrijednost Y. Ako plaćate oko $4, mora da je travanj! Infografika od [Jen Looper](https://twitter.com/jenlooper)
>
> Matematika koja izračunava liniju mora pokazati nagib linije, koji također ovisi o presjeku, odnosno gdje se `Y` nalazi kada je `X = 0`.
>
> Metodu izračuna ovih vrijednosti možete vidjeti na web stranici [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Također posjetite [ovaj kalkulator najmanjih kvadrata](https://www.mathsisfun.com/data/least-squares-calculator.html) kako biste vidjeli kako vrijednosti brojeva utječu na liniju.

## Korelacija

Još jedan pojam koji treba razumjeti je **koeficijent korelacije** između danih X i Y varijabli. Koristeći dijagram raspršenja, možete brzo vizualizirati ovaj koeficijent. Dijagram s podatkovnim točkama raspoređenim u urednu liniju ima visoku korelaciju, dok dijagram s podatkovnim točkama raspršenim svugdje između X i Y ima nisku korelaciju.

Dobar model linearne regresije bit će onaj koji ima visok (bliži 1 nego 0) koeficijent korelacije koristeći metodu najmanjih kvadrata s regresijskom linijom.

✅ Pokrenite bilježnicu koja prati ovu lekciju i pogledajte dijagram raspršenja Mjesec prema Cijeni. Čini li se da podaci koji povezuju Mjesec s Cijenom za prodaju bundeva imaju visoku ili nisku korelaciju, prema vašoj vizualnoj interpretaciji dijagrama raspršenja? Mijenja li se to ako koristite precizniju mjeru umjesto `Mjesec`, npr. *dan u godini* (tj. broj dana od početka godine)?

U kodu ispod pretpostavit ćemo da smo očistili podatke i dobili podatkovni okvir nazvan `new_pumpkins`, sličan sljedećem:

ID | Mjesec | DanUGodini | Vrsta | Grad | Paket | Najniža cijena | Najviša cijena | Cijena
---|--------|------------|-------|------|-------|----------------|----------------|-------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Kod za čišćenje podataka dostupan je u [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Proveli smo iste korake čišćenja kao u prethodnoj lekciji i izračunali stupac `DanUGodini` koristeći sljedeći izraz: 

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Sada kada razumijete matematiku iza linearne regresije, kreirajmo regresijski model kako bismo vidjeli možemo li predvidjeti koji paket bundeva će imati najbolje cijene bundeva. Netko tko kupuje bundeve za blagdanski vrt bundeva možda želi ove informacije kako bi optimizirao svoje kupnje paketa bundeva za vrt.

## Traženje korelacije

[![ML za početnike - Traženje korelacije: Ključ za linearnu regresiju](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML za početnike - Traženje korelacije: Ključ za linearnu regresiju")

> 🎥 Kliknite na sliku iznad za kratki video pregled korelacije.

Iz prethodne lekcije vjerojatno ste vidjeli da prosječna cijena za različite mjesece izgleda ovako:

<img alt="Prosječna cijena po mjesecu" src="../2-Data/images/barchart.png" width="50%"/>

To sugerira da bi mogla postojati neka korelacija, i možemo pokušati trenirati model linearne regresije kako bismo predvidjeli odnos između `Mjesec` i `Cijena`, ili između `DanUGodini` i `Cijena`. Evo dijagrama raspršenja koji pokazuje potonji odnos:

<img alt="Dijagram raspršenja Cijena vs. Dan u godini" src="images/scatter-dayofyear.png" width="50%" /> 

Pogledajmo postoji li korelacija koristeći funkciju `corr`:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Čini se da je korelacija prilično mala, -0.15 za `Mjesec` i -0.17 za `DanUGodini`, ali mogla bi postojati druga važna veza. Izgleda da postoje različiti klasteri cijena koji odgovaraju različitim vrstama bundeva. Da bismo potvrdili ovu hipotezu, nacrtajmo svaku kategoriju bundeva koristeći različitu boju. Prosljeđivanjem parametra `ax` funkciji za crtanje raspršenja možemo nacrtati sve točke na istom grafikonu:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Dijagram raspršenja Cijena vs. Dan u godini" src="images/scatter-dayofyear-color.png" width="50%" /> 

Naša istraga sugerira da vrsta bundeve ima veći utjecaj na ukupnu cijenu nego stvarni datum prodaje. To možemo vidjeti s dijagramom stupaca:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Dijagram stupaca cijena vs vrsta" src="images/price-by-variety.png" width="50%" /> 

Usredotočimo se za trenutak samo na jednu vrstu bundeve, 'pie type', i pogledajmo kakav učinak datum ima na cijenu:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Dijagram raspršenja Cijena vs. Dan u godini" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Ako sada izračunamo korelaciju između `Cijena` i `DanUGodini` koristeći funkciju `corr`, dobit ćemo nešto poput `-0.27` - što znači da treniranje prediktivnog modela ima smisla.

> Prije treniranja modela linearne regresije, važno je osigurati da su naši podaci čisti. Linearna regresija ne funkcionira dobro s nedostajućim vrijednostima, stoga ima smisla riješiti se svih praznih ćelija:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Drugi pristup bio bi popuniti te prazne vrijednosti srednjim vrijednostima iz odgovarajućeg stupca.

## Jednostavna linearna regresija

[![ML za početnike - Linearna i polinomna regresija koristeći Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML za početnike - Linearna i polinomna regresija koristeći Scikit-learn")

> 🎥 Kliknite na sliku iznad za kratki video pregled linearne i polinomne regresije.

Za treniranje našeg modela linearne regresije koristit ćemo biblioteku **Scikit-learn**.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Počinjemo razdvajanjem ulaznih vrijednosti (značajki) i očekivanog izlaza (oznaka) u zasebne numpy nizove:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Napomena: morali smo izvršiti `reshape` na ulaznim podacima kako bi paket za linearnu regresiju ispravno razumio podatke. Linearna regresija očekuje 2D-niz kao ulaz, gdje svaki redak niza odgovara vektoru ulaznih značajki. U našem slučaju, budući da imamo samo jedan ulaz, trebamo niz oblika N×1, gdje je N veličina skupa podataka.

Zatim trebamo podijeliti podatke na skupove za treniranje i testiranje, kako bismo mogli validirati naš model nakon treniranja:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Na kraju, treniranje stvarnog modela linearne regresije traje samo dva retka koda. Definiramo objekt `LinearRegression` i prilagodimo ga našim podacima koristeći metodu `fit`:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

Objekt `LinearRegression` nakon prilagodbe (`fit`) sadrži sve koeficijente regresije, koji se mogu pristupiti koristeći svojstvo `.coef_`. U našem slučaju postoji samo jedan koeficijent, koji bi trebao biti oko `-0.017`. To znači da se cijene čini da malo padaju s vremenom, ali ne previše, oko 2 centa dnevno. Također možemo pristupiti točki presjeka regresije s Y-osom koristeći `lin_reg.intercept_` - ona će biti oko `21` u našem slučaju, što ukazuje na cijenu na početku godine.

Kako bismo vidjeli koliko je naš model točan, možemo predvidjeti cijene na testnom skupu podataka, a zatim izmjeriti koliko su naše predikcije blizu očekivanim vrijednostima. To se može učiniti koristeći metriku srednje kvadratne pogreške (MSE), koja je srednja vrijednost svih kvadriranih razlika između očekivane i predviđene vrijednosti.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Naša pogreška čini se da se kreće oko 2 točke, što je ~17%. Nije baš dobro. Još jedan pokazatelj kvalitete modela je **koeficijent determinacije**, koji se može dobiti ovako:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Ako je vrijednost 0, to znači da model ne uzima ulazne podatke u obzir i ponaša se kao *najgori linearni prediktor*, što je jednostavno prosječna vrijednost rezultata. Vrijednost 1 znači da možemo savršeno predvidjeti sve očekivane izlaze. U našem slučaju, koeficijent je oko 0.06, što je prilično nisko.

Također možemo prikazati testne podatke zajedno s regresijskom linijom kako bismo bolje vidjeli kako regresija funkcionira u našem slučaju:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Linear regression" src="images/linear-results.png" width="50%" />

## Polinomijalna regresija  

Druga vrsta linearne regresije je polinomijalna regresija. Iako ponekad postoji linearna veza između varijabli - što je veća bundeva po volumenu, to je viša cijena - ponekad te veze ne mogu biti prikazane kao ravnina ili ravna linija.  

✅ Evo [nekoliko primjera](https://online.stat.psu.edu/stat501/lesson/9/9.8) podataka koji bi mogli koristiti polinomijalnu regresiju.  

Pogledajte ponovno vezu između datuma i cijene. Čini li se ovaj dijagram raspršenosti kao da bi nužno trebao biti analiziran ravnom linijom? Zar cijene ne mogu fluktuirati? U ovom slučaju možete pokušati s polinomijalnom regresijom.  

✅ Polinomi su matematički izrazi koji mogu sadržavati jednu ili više varijabli i koeficijenata.  

Polinomijalna regresija stvara zakrivljenu liniju kako bi bolje odgovarala nelinearnim podacima. U našem slučaju, ako uključimo kvadratnu varijablu `DayOfYear` u ulazne podatke, trebali bismo moći prilagoditi naše podatke paraboličnoj krivulji, koja će imati minimum u određenom trenutku unutar godine.  

Scikit-learn uključuje koristan [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) za kombiniranje različitih koraka obrade podataka. **Pipeline** je lanac **procjenitelja**. U našem slučaju, stvorit ćemo pipeline koji prvo dodaje polinomijalne značajke našem modelu, a zatim trenira regresiju:  

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Korištenje `PolynomialFeatures(2)` znači da ćemo uključiti sve polinome drugog stupnja iz ulaznih podataka. U našem slučaju to će jednostavno značiti `DayOfYear`<sup>2</sup>, ali s obzirom na dvije ulazne varijable X i Y, to će dodati X<sup>2</sup>, XY i Y<sup>2</sup>. Također možemo koristiti polinome višeg stupnja ako želimo.  

Pipeline se može koristiti na isti način kao i originalni objekt `LinearRegression`, tj. možemo koristiti `fit` za treniranje pipelinea, a zatim `predict` za dobivanje rezultata predikcije. Evo grafikona koji prikazuje testne podatke i aproksimacijsku krivulju:  

<img alt="Polynomial regression" src="images/poly-results.png" width="50%" />  

Korištenjem polinomijalne regresije možemo dobiti nešto niži MSE i viši koeficijent determinacije, ali ne značajno. Moramo uzeti u obzir i druge značajke!  

> Možete vidjeti da su minimalne cijene bundeva zabilježene negdje oko Noći vještica. Kako to možete objasniti?  

🎃 Čestitamo, upravo ste stvorili model koji može pomoći u predviđanju cijene bundeva za pite. Vjerojatno možete ponoviti isti postupak za sve vrste bundeva, ali to bi bilo zamorno. Sada naučimo kako uzeti u obzir raznolikost bundeva u našem modelu!  

## Kategorijalne značajke  

U idealnom svijetu želimo moći predvidjeti cijene za različite vrste bundeva koristeći isti model. Međutim, stupac `Variety` je donekle drugačiji od stupaca poput `Month`, jer sadrži nenumeričke vrijednosti. Takvi stupci nazivaju se **kategorijalni**.  

[![ML for beginners - Categorical Feature Predictions with Linear Regression](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML for beginners - Categorical Feature Predictions with Linear Regression")  

> 🎥 Kliknite na sliku iznad za kratki video pregled korištenja kategorijalnih značajki.  

Ovdje možete vidjeti kako prosječna cijena ovisi o vrsti:  

<img alt="Average price by variety" src="images/price-by-variety.png" width="50%" />  

Kako bismo uzeli u obzir vrstu, prvo je moramo pretvoriti u numerički oblik, odnosno **kodirati**. Postoji nekoliko načina kako to možemo učiniti:  

* Jednostavno **numeričko kodiranje** će izgraditi tablicu različitih vrsta, a zatim zamijeniti naziv vrste indeksom u toj tablici. Ovo nije najbolja ideja za linearnu regresiju, jer linearna regresija uzima stvarnu numeričku vrijednost indeksa i dodaje je rezultatu, množeći je nekim koeficijentom. U našem slučaju, veza između broja indeksa i cijene očito nije linearna, čak i ako osiguramo da su indeksi poredani na neki specifičan način.  
* **One-hot kodiranje** će zamijeniti stupac `Variety` s 4 različita stupca, po jedan za svaku vrstu. Svaki stupac će sadržavati `1` ako odgovarajući redak pripada određenoj vrsti, a `0` inače. To znači da će u linearnom regresijskom modelu postojati četiri koeficijenta, po jedan za svaku vrstu bundeve, odgovorna za "početnu cijenu" (ili radije "dodatnu cijenu") za tu određenu vrstu.  

Kod ispod pokazuje kako možemo one-hot kodirati vrstu:  

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

Kako bismo trenirali linearnu regresiju koristeći one-hot kodiranu vrstu kao ulaz, samo trebamo ispravno inicijalizirati podatke `X` i `y`:  

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

Ostatak koda je isti kao što smo koristili gore za treniranje linearne regresije. Ako ga isprobate, vidjet ćete da je srednja kvadratna pogreška otprilike ista, ali dobivamo puno viši koeficijent determinacije (~77%). Kako bismo dobili još točnija predviđanja, možemo uzeti u obzir više kategorijalnih značajki, kao i numeričke značajke poput `Month` ili `DayOfYear`. Kako bismo dobili jedan veliki niz značajki, možemo koristiti `join`:  

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Ovdje također uzimamo u obzir `City` i vrstu `Package`, što nam daje MSE 2.84 (10%) i determinaciju 0.94!  

## Sve zajedno  

Kako bismo napravili najbolji model, možemo koristiti kombinirane (one-hot kodirane kategorijalne + numeričke) podatke iz gornjeg primjera zajedno s polinomijalnom regresijom. Evo kompletnog koda za vašu praktičnost:  

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

Ovo bi nam trebalo dati najbolji koeficijent determinacije od gotovo 97% i MSE=2.23 (~8% pogreške u predviđanju).  

| Model | MSE | Determinacija |  
|-------|-----|---------------|  
| `DayOfYear` Linear | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polynomial | 2.73 (17.0%) | 0.08 |  
| `Variety` Linear | 5.24 (19.7%) | 0.77 |  
| Sve značajke Linear | 2.84 (10.5%) | 0.94 |  
| Sve značajke Polynomial | 2.23 (8.25%) | 0.97 |  

🏆 Bravo! Stvorili ste četiri regresijska modela u jednoj lekciji i poboljšali kvalitetu modela na 97%. U završnom dijelu o regresiji naučit ćete o logističkoj regresiji za određivanje kategorija.  

---  
## 🚀Izazov  

Testirajte nekoliko različitih varijabli u ovom notebooku kako biste vidjeli kako korelacija odgovara točnosti modela.  

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)  

## Pregled i samostalno učenje  

U ovoj lekciji naučili smo o linearnoj regresiji. Postoje i druge važne vrste regresije. Pročitajte o tehnikama Stepwise, Ridge, Lasso i Elasticnet. Dobar tečaj za daljnje učenje je [Stanford Statistical Learning course](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).  

## Zadatak  

[Izgradite model](assignment.md)  

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.