<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-05T07:43:58+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "lt"
}
-->
# Sukurkite regresijos modelį naudodami Scikit-learn: keturi regresijos būdai

![Linijinės ir polinominės regresijos infografika](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Prieš paskaitą atlikite testą](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ši pamoka taip pat prieinama R kalba!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Įvadas

Iki šiol jūs tyrinėjote, kas yra regresija, naudodami pavyzdinius duomenis iš moliūgų kainų duomenų rinkinio, kurį naudosime visoje šioje pamokoje. Taip pat vizualizavote duomenis naudodami Matplotlib.

Dabar esate pasiruošę giliau pasinerti į regresiją mašininio mokymosi (ML) kontekste. Nors vizualizacija leidžia suprasti duomenis, tikroji mašininio mokymosi galia slypi _modelių treniravime_. Modeliai yra treniruojami naudojant istorinius duomenis, kad automatiškai užfiksuotų duomenų priklausomybes, ir jie leidžia prognozuoti rezultatus naujiems duomenims, kurių modelis dar nematė.

Šioje pamokoje sužinosite daugiau apie du regresijos tipus: _paprastą linijinę regresiją_ ir _polinominę regresiją_, kartu su kai kuriais šių metodų matematiniais pagrindais. Šie modeliai leis mums prognozuoti moliūgų kainas, atsižvelgiant į skirtingus įvesties duomenis.

[![ML pradedantiesiems - Linijinės regresijos supratimas](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "ML pradedantiesiems - Linijinės regresijos supratimas")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie linijinę regresiją.

> Visame šioje mokymo programoje darome prielaidą, kad turite minimalias matematikos žinias, ir siekiame, kad ji būtų prieinama studentams iš kitų sričių. Atkreipkite dėmesį į pastabas, 🧮 išnašas, diagramas ir kitus mokymosi įrankius, kurie padės suprasti.

### Prieš pradedant

Iki šiol turėtumėte būti susipažinę su moliūgų duomenų struktūra, kurią nagrinėjame. Šioje pamokoje galite rasti iš anksto įkeltus ir išvalytus duomenis _notebook.ipynb_ faile. Faile moliūgų kaina pateikiama už bušelį naujame duomenų rėmelyje. Įsitikinkite, kad galite paleisti šiuos užrašus naudodami Visual Studio Code branduolius.

### Pasiruošimas

Primename, kad įkeliame šiuos duomenis tam, kad galėtume užduoti klausimus:

- Kada geriausia pirkti moliūgus?
- Kokią kainą galiu tikėtis už miniatiūrinių moliūgų dėžę?
- Ar turėčiau juos pirkti pusės bušelio krepšiuose ar 1 1/9 bušelio dėžėse?
Pažiūrėkime giliau į šiuos duomenis.

Ankstesnėje pamokoje sukūrėte Pandas duomenų rėmelį ir užpildėte jį dalimi pradinio duomenų rinkinio, standartizuodami kainas pagal bušelį. Tačiau taip galėjote surinkti tik apie 400 duomenų taškų ir tik rudens mėnesiams.

Pažvelkite į duomenis, kurie yra iš anksto įkelti šios pamokos pridedamame užrašų faile. Duomenys yra iš anksto įkelti, o pradinis sklaidos grafikas yra sudarytas, kad parodytų mėnesio duomenis. Galbūt galime gauti šiek tiek daugiau informacijos apie duomenų pobūdį, juos dar labiau išvalydami.

## Linijinės regresijos linija

Kaip sužinojote 1-oje pamokoje, linijinės regresijos tikslas yra nubrėžti liniją, kuri:

- **Parodo kintamųjų ryšius**. Parodo ryšį tarp kintamųjų
- **Leidžia prognozuoti**. Leidžia tiksliai prognozuoti, kur naujas duomenų taškas atsidurs santykyje su ta linija.

Įprasta **mažiausių kvadratų regresija** naudoti šio tipo linijai braižyti. Terminas „mažiausi kvadratai“ reiškia, kad visi duomenų taškai aplink regresijos liniją yra pakelti kvadratu ir tada sudedami. Idealiu atveju, galutinė suma yra kuo mažesnė, nes norime kuo mažiau klaidų arba `mažiausių kvadratų`.

Mes tai darome, nes norime modeliuoti liniją, kuri turi mažiausią bendrą atstumą nuo visų mūsų duomenų taškų. Taip pat kvadratuojame terminus prieš juos sudėdami, nes mums rūpi jų dydis, o ne kryptis.

> **🧮 Parodykite matematiką**
> 
> Ši linija, vadinama _geriausiai tinkančia linija_, gali būti išreikšta [lygtimi](https://en.wikipedia.org/wiki/Simple_linear_regression): 
> 
> ```
> Y = a + bX
> ```
>
> `X` yra „paaiškinamasis kintamasis“. `Y` yra „priklausomas kintamasis“. Linijos nuolydis yra `b`, o `a` yra y-ašies perėmimas, kuris nurodo `Y` reikšmę, kai `X = 0`. 
>
>![apskaičiuoti nuolydį](../../../../2-Regression/3-Linear/images/slope.png)
>
> Pirmiausia apskaičiuokite nuolydį `b`. Infografika: [Jen Looper](https://twitter.com/jenlooper)
>
> Kitaip tariant, remiantis mūsų moliūgų duomenų pradiniu klausimu: „prognozuokite moliūgo kainą už bušelį pagal mėnesį“, `X` reikštų kainą, o `Y` reikštų pardavimo mėnesį. 
>
>![užbaigti lygtį](../../../../2-Regression/3-Linear/images/calculation.png)
>
> Apskaičiuokite `Y` reikšmę. Jei mokate apie 4 dolerius, tai turi būti balandis! Infografika: [Jen Looper](https://twitter.com/jenlooper)
>
> Matematikos, kuri apskaičiuoja liniją, tikslas yra parodyti linijos nuolydį, kuris taip pat priklauso nuo perėmimo, arba kur `Y` yra, kai `X = 0`.
>
> Galite stebėti šių reikšmių skaičiavimo metodą svetainėje [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html). Taip pat apsilankykite [mažiausių kvadratų skaičiuoklėje](https://www.mathsisfun.com/data/least-squares-calculator.html), kad pamatytumėte, kaip skaičių reikšmės veikia liniją.

## Koreliacija

Dar vienas terminas, kurį reikia suprasti, yra **koreliacijos koeficientas** tarp duotų X ir Y kintamųjų. Naudodami sklaidos grafiką, galite greitai vizualizuoti šį koeficientą. Grafikas, kuriame duomenų taškai išsidėstę tvarkinga linija, turi aukštą koreliaciją, tačiau grafikas, kuriame duomenų taškai išmėtyti tarp X ir Y, turi žemą koreliaciją.

Geras linijinės regresijos modelis bus tas, kuris turi aukštą (artimesnį 1 nei 0) koreliacijos koeficientą, naudojant mažiausių kvadratų regresijos metodą su regresijos linija.

✅ Paleiskite šios pamokos pridedamą užrašų failą ir pažiūrėkite į mėnesio ir kainos sklaidos grafiką. Ar duomenys, siejantys mėnesį su moliūgų pardavimo kaina, atrodo turintys aukštą ar žemą koreliaciją, remiantis jūsų vizualine sklaidos grafiko interpretacija? Ar tai pasikeičia, jei naudojate tikslesnį matą, pvz., *metų dieną* (t. y. dienų skaičių nuo metų pradžios)?

Toliau pateiktame kode darome prielaidą, kad išvalėme duomenis ir gavome duomenų rėmelį, vadinamą `new_pumpkins`, panašų į šį:

ID | Mėnuo | MetųDiena | Rūšis | Miestas | Pakuotė | Mažiausia Kaina | Didžiausia Kaina | Kaina
---|-------|-----------|-------|---------|---------|-----------------|------------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bušelio dėžės | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bušelio dėžės | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bušelio dėžės | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bušelio dėžės | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bušelio dėžės | 15.0 | 15.0 | 13.636364

> Duomenų valymo kodas yra pateiktas [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb). Atlikome tuos pačius valymo veiksmus kaip ir ankstesnėje pamokoje, ir apskaičiavome `MetųDiena` stulpelį naudodami šią išraišką:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Dabar, kai suprantate linijinės regresijos matematiką, sukurkime regresijos modelį, kad pamatytume, ar galime prognozuoti, kuri moliūgų pakuotė turės geriausias kainas. Kažkas, perkantis moliūgus šventinei moliūgų aikštelei, galėtų norėti šios informacijos, kad optimizuotų savo pirkinius.

## Ieškome koreliacijos

[![ML pradedantiesiems - Koreliacijos paieška: Linijinės regresijos raktas](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "ML pradedantiesiems - Koreliacijos paieška: Linijinės regresijos raktas")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie koreliaciją.

Iš ankstesnės pamokos tikriausiai matėte, kad vidutinė kaina skirtingais mėnesiais atrodo taip:

<img alt="Vidutinė kaina pagal mėnesį" src="../2-Data/images/barchart.png" width="50%"/>

Tai rodo, kad turėtų būti tam tikra koreliacija, ir galime pabandyti treniruoti linijinį regresijos modelį, kad prognozuotume ryšį tarp `Mėnuo` ir `Kaina`, arba tarp `MetųDiena` ir `Kaina`. Štai sklaidos grafikas, rodantis pastarąjį ryšį:

<img alt="Sklaidos grafikas: Kaina vs. Metų Diena" src="images/scatter-dayofyear.png" width="50%" /> 

Pažiūrėkime, ar yra koreliacija, naudodami `corr` funkciją:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Atrodo, kad koreliacija yra gana maža: -0.15 pagal `Mėnuo` ir -0.17 pagal `MetųDiena`, tačiau gali būti kita svarbi priklausomybė. Atrodo, kad yra skirtingos kainų grupės, atitinkančios skirtingas moliūgų rūšis. Norėdami patvirtinti šią hipotezę, nubrėžkime kiekvieną moliūgų kategoriją skirtinga spalva. Perduodami `ax` parametrą `scatter` braižymo funkcijai, galime nubrėžti visus taškus tame pačiame grafike:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Sklaidos grafikas: Kaina vs. Metų Diena" src="images/scatter-dayofyear-color.png" width="50%" /> 

Mūsų tyrimas rodo, kad rūšis turi didesnę įtaką bendrai kainai nei faktinė pardavimo data. Tai galime pamatyti stulpeline diagrama:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Stulpelinė diagrama: Kaina pagal rūšį" src="images/price-by-variety.png" width="50%" /> 

Dabar sutelkime dėmesį tik į vieną moliūgų rūšį, „pie type“, ir pažiūrėkime, kokią įtaką data turi kainai:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Sklaidos grafikas: Kaina vs. Metų Diena" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Jei dabar apskaičiuosime koreliaciją tarp `Kaina` ir `MetųDiena` naudodami `corr` funkciją, gausime maždaug `-0.27` - tai reiškia, kad treniruoti prognozavimo modelį yra prasminga.

> Prieš treniruojant linijinį regresijos modelį, svarbu įsitikinti, kad mūsų duomenys yra švarūs. Linijinė regresija neveikia gerai su trūkstamomis reikšmėmis, todėl verta pašalinti visas tuščias ląsteles:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Kitas požiūris būtų užpildyti tas tuščias reikšmes vidutinėmis reikšmėmis iš atitinkamo stulpelio.

## Paprasta linijinė regresija

[![ML pradedantiesiems - Linijinė ir polinominė regresija naudojant Scikit-learn](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "ML pradedantiesiems - Linijinė ir polinominė regresija naudojant Scikit-learn")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie linijinę ir polinominę regresiją.

Norėdami treniruoti mūsų linijinį regresijos modelį, naudosime **Scikit-learn** biblioteką.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Pradėsime atskirdami įvesties reikšmes (savybes) ir laukiamą rezultatą (etiketę) į atskirus numpy masyvus:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Atkreipkite dėmesį, kad turėjome atlikti `reshape` įvesties duomenims, kad linijinės regresijos paketas juos suprastų teisingai. Linijinė regresija tikisi 2D masyvo kaip įvesties, kur kiekviena masyvo eilutė atitinka įvesties savybių vektorių. Mūsų atveju, kadangi turime tik vieną įvestį, mums reikia masyvo su forma N×1, kur N yra duomenų rinkinio dydis.

Tada turime padalyti duomenis į treniravimo ir testavimo duomenų rinkinius, kad galėtume patikrinti savo modelį po treniravimo:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Galiausiai, treniruoti faktinį linijinį regresijos modelį užtrunka tik dvi kodo eilutes. Apibrėžiame `LinearRegression` objektą ir pritaikome jį mūsų duomenims naudodami `fit` metodą:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` objektas po `fit`-inimo turi visus regresijos koeficientus, kuriuos galima pasiekti naudojant `.coef_` savybę. Mūsų atveju yra tik vienas koeficientas, kuris turėtų būti apie `-0.017`. Tai reiškia, kad kainos atrodo šiek tiek mažėjančios laikui bėgant, bet ne per daug - apie 2 centus per dieną. Taip pat galime pasiekti regresijos susikirtimo tašką su Y ašimi naudodami `lin_reg.intercept_` - mūsų atveju
Mūsų klaida atrodo susijusi su 2 taškais, tai yra ~17%. Nelabai gerai. Kitas modelio kokybės rodiklis yra **determinacijos koeficientas**, kurį galima gauti taip:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```  
Jei reikšmė yra 0, tai reiškia, kad modelis neatsižvelgia į įvesties duomenis ir veikia kaip *blogiausias linijinis prognozuotojas*, kuris tiesiog yra rezultatų vidutinė reikšmė. Reikšmė 1 reiškia, kad galime tobulai prognozuoti visus numatomus rezultatus. Mūsų atveju determinacijos koeficientas yra apie 0.06, kas yra gana žema reikšmė.

Taip pat galime nubraižyti testinius duomenis kartu su regresijos linija, kad geriau suprastume, kaip regresija veikia mūsų atveju:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```  

<img alt="Linijinė regresija" src="images/linear-results.png" width="50%" />

## Polinominė regresija  

Kitas linijinės regresijos tipas yra polinominė regresija. Nors kartais tarp kintamųjų yra linijinis ryšys – kuo didesnis moliūgas pagal tūrį, tuo didesnė kaina – kartais šių ryšių negalima pavaizduoti kaip plokštumos ar tiesės.

✅ Štai [keletas pavyzdžių](https://online.stat.psu.edu/stat501/lesson/9/9.8) duomenų, kuriems galėtų būti taikoma polinominė regresija.

Pažvelkite dar kartą į ryšį tarp datos ir kainos. Ar šis sklaidos grafikas atrodo taip, kad jį būtinai reikėtų analizuoti tiesės pagalba? Ar kainos negali svyruoti? Tokiu atveju galite išbandyti polinominę regresiją.

✅ Polinomai yra matematinės išraiškos, kurios gali susidėti iš vieno ar daugiau kintamųjų ir koeficientų.

Polinominė regresija sukuria kreivę, kuri geriau atitinka nelinijinius duomenis. Mūsų atveju, jei įvesties duomenyse įtrauksime kvadratinį `DayOfYear` kintamąjį, turėtume galėti pritaikyti savo duomenis parabolinei kreivei, kuri turės minimumą tam tikru metų momentu.

Scikit-learn turi naudingą [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline), leidžiantį sujungti skirtingus duomenų apdorojimo žingsnius. **Pipeline** yra **vertintojų** grandinė. Mūsų atveju sukursime pipeline, kuris pirmiausia prideda polinomines savybes prie mūsų modelio, o tada treniruoja regresiją:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```  

Naudojant `PolynomialFeatures(2)` reiškia, kad įtrauksime visus antro laipsnio polinomus iš įvesties duomenų. Mūsų atveju tai tiesiog reikš `DayOfYear`<sup>2</sup>, bet turint du įvesties kintamuosius X ir Y, tai pridės X<sup>2</sup>, XY ir Y<sup>2</sup>. Jei norime, galime naudoti aukštesnio laipsnio polinomus.

Pipeline galima naudoti taip pat, kaip ir originalų `LinearRegression` objektą, t.y. galime `fit` pipeline, o tada naudoti `predict`, kad gautume prognozės rezultatus. Štai grafikas, rodantis testinius duomenis ir aproksimacijos kreivę:

<img alt="Polinominė regresija" src="images/poly-results.png" width="50%" />

Naudojant polinominę regresiją, galime gauti šiek tiek mažesnį MSE ir aukštesnį determinacijos koeficientą, bet ne žymiai. Turime atsižvelgti į kitas savybes!

> Galite pastebėti, kad mažiausios moliūgų kainos stebimos kažkur apie Heloviną. Kaip tai paaiškintumėte?

🎃 Sveikiname, ką tik sukūrėte modelį, kuris gali padėti prognozuoti pyraginių moliūgų kainą. Tikriausiai galite pakartoti tą pačią procedūrą visiems moliūgų tipams, bet tai būtų varginantis darbas. Dabar išmokime, kaip įtraukti moliūgų veislę į mūsų modelį!

## Kategorinės savybės  

Idealiame pasaulyje norėtume sugebėti prognozuoti kainas skirtingoms moliūgų veislėms naudodami tą patį modelį. Tačiau `Variety` stulpelis šiek tiek skiriasi nuo tokių stulpelių kaip `Month`, nes jame yra ne skaitinės reikšmės. Tokie stulpeliai vadinami **kategoriniais**.

[![ML pradedantiesiems - Kategorinių savybių prognozės naudojant linijinę regresiją](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "ML pradedantiesiems - Kategorinių savybių prognozės naudojant linijinę regresiją")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte trumpą vaizdo įrašą apie kategorinių savybių naudojimą.

Čia galite pamatyti, kaip vidutinė kaina priklauso nuo veislės:

<img alt="Vidutinė kaina pagal veislę" src="images/price-by-variety.png" width="50%" />

Norėdami atsižvelgti į veislę, pirmiausia turime ją konvertuoti į skaitinę formą, arba **užkoduoti**. Yra keli būdai, kaip tai padaryti:

* Paprastas **skaitinis kodavimas** sukurs skirtingų veislių lentelę, o tada pakeis veislės pavadinimą indeksu toje lentelėje. Tai nėra geriausia idėja linijinei regresijai, nes linijinė regresija naudoja faktinę indekso skaitinę reikšmę ir prideda ją prie rezultato, padaugindama iš tam tikro koeficiento. Mūsų atveju ryšys tarp indekso numerio ir kainos yra aiškiai nelinijinis, net jei užtikrinsime, kad indeksai būtų išdėstyti tam tikra tvarka.
* **Vieno karšto kodavimo (One-hot encoding)** pakeis `Variety` stulpelį 4 skirtingais stulpeliais, po vieną kiekvienai veislei. Kiekviename stulpelyje bus `1`, jei atitinkama eilutė priklauso tam tikrai veislei, ir `0` kitu atveju. Tai reiškia, kad linijinėje regresijoje bus keturi koeficientai, po vieną kiekvienai moliūgų veislei, atsakingi už "pradinę kainą" (arba "papildomą kainą") tai konkrečiai veislei.

Žemiau pateiktas kodas rodo, kaip galime vieno karšto kodavimo būdu užkoduoti veislę:

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

Norėdami treniruoti linijinę regresiją, naudodami vieno karšto kodavimo būdu užkoduotą veislę kaip įvestį, tiesiog turime tinkamai inicializuoti `X` ir `y` duomenis:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```  

Likęs kodas yra toks pat, kaip tas, kurį naudojome aukščiau linijinei regresijai treniruoti. Jei tai išbandysite, pamatysite, kad vidutinė kvadratinė klaida yra maždaug tokia pati, tačiau gauname daug aukštesnį determinacijos koeficientą (~77%). Norėdami gauti dar tikslesnes prognozes, galime atsižvelgti į daugiau kategorinių savybių, taip pat skaitines savybes, tokias kaip `Month` ar `DayOfYear`. Norėdami gauti vieną didelį savybių masyvą, galime naudoti `join`:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```  

Čia taip pat atsižvelgiame į `City` ir `Package` tipą, kas suteikia MSE 2.84 (10%) ir determinaciją 0.94!

## Viskas kartu  

Norėdami sukurti geriausią modelį, galime naudoti kombinuotus (vieno karšto kodavimo kategorinius + skaitinius) duomenis iš aukščiau pateikto pavyzdžio kartu su polinomine regresija. Štai visas kodas jūsų patogumui:

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

Tai turėtų suteikti geriausią determinacijos koeficientą beveik 97% ir MSE=2.23 (~8% prognozės klaida).

| Modelis | MSE | Determinacija |  
|---------|-----|---------------|  
| `DayOfYear` Linijinis | 2.77 (17.2%) | 0.07 |  
| `DayOfYear` Polinominis | 2.73 (17.0%) | 0.08 |  
| `Variety` Linijinis | 5.24 (19.7%) | 0.77 |  
| Visos savybės Linijinis | 2.84 (10.5%) | 0.94 |  
| Visos savybės Polinominis | 2.23 (8.25%) | 0.97 |  

🏆 Puikiai padirbėta! Jūs sukūrėte keturis regresijos modelius per vieną pamoką ir pagerinote modelio kokybę iki 97%. Paskutinėje regresijos dalyje išmoksite apie logistinę regresiją, skirtą kategorijoms nustatyti.

---  
## 🚀Iššūkis  

Išbandykite kelis skirtingus kintamuosius šiame užrašų knygelėje, kad pamatytumėte, kaip koreliacija atitinka modelio tikslumą.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)  

## Apžvalga ir savarankiškas mokymasis  

Šioje pamokoje išmokome apie linijinę regresiją. Yra ir kitų svarbių regresijos tipų. Perskaitykite apie Stepwise, Ridge, Lasso ir Elasticnet metodus. Geras kursas, norint sužinoti daugiau, yra [Stanfordo statistinio mokymosi kursas](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning).

## Užduotis  

[Sukurkite modelį](assignment.md)  

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.