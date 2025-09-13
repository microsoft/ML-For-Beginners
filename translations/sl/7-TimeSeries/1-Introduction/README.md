<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T11:59:59+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "sl"
}
-->
# Uvod v napovedovanje časovnih vrst

![Povzetek časovnih vrst v sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

V tej lekciji in naslednji boste spoznali osnove napovedovanja časovnih vrst, zanimivega in dragocenega dela repertoarja znanstvenika za strojno učenje, ki je nekoliko manj poznan kot drugi tematski sklopi. Napovedovanje časovnih vrst je nekakšna 'kristalna krogla': na podlagi preteklega delovanja spremenljivke, kot je cena, lahko napovemo njeno prihodnjo potencialno vrednost.

[![Uvod v napovedovanje časovnih vrst](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Uvod v napovedovanje časovnih vrst")

> 🎥 Kliknite zgornjo sliko za video o napovedovanju časovnih vrst

## [Predlekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

Gre za uporabno in zanimivo področje z resnično vrednostjo za poslovanje, saj ima neposredno aplikacijo pri reševanju težav s cenami, zalogami in dobavnimi verigami. Čeprav se za pridobivanje boljših vpogledov v prihodnje delovanje vse bolj uporabljajo tehnike globokega učenja, napovedovanje časovnih vrst ostaja področje, ki ga močno zaznamujejo klasične tehnike strojnega učenja.

> Koristno gradivo o časovnih vrstah univerze Penn State najdete [tukaj](https://online.stat.psu.edu/stat510/lesson/1)

## Uvod

Predstavljajte si, da upravljate mrežo pametnih parkirnih števcev, ki zagotavljajo podatke o tem, kako pogosto in kako dolgo se uporabljajo skozi čas.

> Kaj če bi lahko na podlagi preteklega delovanja števca napovedali njegovo prihodnjo vrednost v skladu z zakoni ponudbe in povpraševanja?

Natančno napovedovanje, kdaj ukrepati za dosego cilja, je izziv, ki ga je mogoče rešiti z napovedovanjem časovnih vrst. Morda ljudje ne bi bili navdušeni, če bi morali plačati več v času največje zasedenosti, ko iščejo parkirno mesto, vendar bi bil to zanesljiv način za ustvarjanje prihodkov za čiščenje ulic!

Raziskali bomo nekaj vrst algoritmov časovnih vrst in začeli z beležko za čiščenje in pripravo podatkov. Podatki, ki jih boste analizirali, so vzeti iz tekmovanja GEFCom2014 za napovedovanje. Vključujejo 3 leta urnih vrednosti porabe električne energije in temperature med letoma 2012 in 2014. Glede na zgodovinske vzorce porabe električne energije in temperature lahko napoveste prihodnje vrednosti porabe električne energije.

V tem primeru se boste naučili, kako napovedati eno časovno točko vnaprej, pri čemer boste uporabili le zgodovinske podatke o porabi. Preden začnete, pa je koristno razumeti, kaj se dogaja v ozadju.

## Nekatere definicije

Ko naletite na izraz 'časovne vrste', morate razumeti njegovo uporabo v različnih kontekstih.

🎓 **Časovne vrste**

V matematiki so "časovne vrste zaporedje podatkovnih točk, ki so indeksirane (ali navedene ali narisane) v časovnem zaporedju. Najpogosteje so časovne vrste zaporedje, zajeto v enakih časovnih intervalih." Primer časovnih vrst je dnevna zaključna vrednost [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Uporaba grafov časovnih vrst in statističnega modeliranja je pogosto prisotna pri obdelavi signalov, vremenskih napovedih, napovedovanju potresov in drugih področjih, kjer se dogodki pojavljajo in podatkovne točke lahko narišemo skozi čas.

🎓 **Analiza časovnih vrst**

Analiza časovnih vrst je analiza zgoraj omenjenih podatkov časovnih vrst. Podatki časovnih vrst lahko zavzamejo različne oblike, vključno z 'prekinjenimi časovnimi vrstami', ki zaznavajo vzorce v razvoju časovnih vrst pred in po prekinjajočem dogodku. Vrsta analize, ki je potrebna za časovne vrste, je odvisna od narave podatkov. Podatki časovnih vrst sami lahko zavzamejo obliko zaporedja številk ali znakov.

Analiza uporablja različne metode, vključno s frekvenčno domeno in časovno domeno, linearne in nelinearne metode ter druge. [Preberite več](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) o številnih načinih analize te vrste podatkov.

🎓 **Napovedovanje časovnih vrst**

Napovedovanje časovnih vrst je uporaba modela za napovedovanje prihodnjih vrednosti na podlagi vzorcev, ki jih prikazujejo prej zbrani podatki, kot so se pojavili v preteklosti. Čeprav je mogoče uporabiti regresijske modele za raziskovanje podatkov časovnih vrst, pri čemer so časovni indeksi x spremenljivke na grafu, je takšne podatke najbolje analizirati s posebnimi vrstami modelov.

Podatki časovnih vrst so seznam urejenih opazovanj, za razliko od podatkov, ki jih je mogoče analizirati z linearno regresijo. Najpogostejši model je ARIMA, kratica za "Avtoregresivno Integrirano Premično Povprečje".

[ARIMA modeli](https://online.stat.psu.edu/stat510/lesson/1/1.1) "povezujejo trenutno vrednost serije s preteklimi vrednostmi in preteklimi napakami napovedi." Najbolj so primerni za analizo podatkov v časovni domeni, kjer so podatki urejeni skozi čas.

> Obstaja več vrst ARIMA modelov, o katerih lahko preberete [tukaj](https://people.duke.edu/~rnau/411arim.htm) in jih boste obravnavali v naslednji lekciji.

V naslednji lekciji boste zgradili ARIMA model z uporabo [Univariatnih časovnih vrst](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), ki se osredotočajo na eno spremenljivko, ki spreminja svojo vrednost skozi čas. Primer te vrste podatkov je [ta podatkovni niz](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), ki beleži mesečno koncentracijo CO2 na observatoriju Mauna Loa:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Prepoznajte spremenljivko, ki se spreminja skozi čas v tem podatkovnem nizu.

## Značilnosti podatkov časovnih vrst, ki jih je treba upoštevati

Ko opazujete podatke časovnih vrst, lahko opazite, da imajo [določene značilnosti](https://online.stat.psu.edu/stat510/lesson/1/1.1), ki jih morate upoštevati in omiliti, da bi bolje razumeli njihove vzorce. Če podatke časovnih vrst obravnavate kot potencialni 'signal', ki ga želite analizirati, lahko te značilnosti obravnavate kot 'šum'. Pogosto boste morali zmanjšati ta 'šum' z uporabo nekaterih statističnih tehnik.

Tukaj je nekaj konceptov, ki jih morate poznati, da lahko delate s časovnimi vrstami:

🎓 **Trendi**

Trendi so opredeljeni kot merljive rasti in padci skozi čas. [Preberite več](https://machinelearningmastery.com/time-series-trends-in-python). V kontekstu časovnih vrst gre za to, kako uporabiti in, če je potrebno, odstraniti trende iz časovnih vrst.

🎓 **[Sezonskost](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sezonskost je opredeljena kot periodična nihanja, kot so na primer praznični nakupovalni vrhovi, ki lahko vplivajo na prodajo. [Oglejte si](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), kako različne vrste grafov prikazujejo sezonskost v podatkih.

🎓 **Izstopajoče vrednosti**

Izstopajoče vrednosti so daleč od standardne variance podatkov.

🎓 **Dolgotrajni cikli**

Neodvisno od sezonskosti lahko podatki prikazujejo dolgotrajne cikle, kot je gospodarska recesija, ki traja dlje kot eno leto.

🎓 **Konstanta varianca**

Skozi čas nekateri podatki prikazujejo konstantna nihanja, kot je poraba energije podnevi in ponoči.

🎓 **Nenadne spremembe**

Podatki lahko prikazujejo nenadno spremembo, ki jo je treba dodatno analizirati. Nenadno zaprtje podjetij zaradi COVID-a, na primer, je povzročilo spremembe v podatkih.

✅ Tukaj je [primer grafa časovnih vrst](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), ki prikazuje dnevno porabo valute v igri skozi nekaj let. Ali lahko v teh podatkih prepoznate katere koli od zgoraj navedenih značilnosti?

![Poraba valute v igri](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Naloga - začetek z podatki o porabi energije

Začnimo z ustvarjanjem modela časovnih vrst za napovedovanje prihodnje porabe energije glede na preteklo porabo.

> Podatki v tem primeru so vzeti iz tekmovanja GEFCom2014 za napovedovanje. Vključujejo 3 leta urnih vrednosti porabe električne energije in temperature med letoma 2012 in 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli in Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, julij-september, 2016.

1. V mapi `working` te lekcije odprite datoteko _notebook.ipynb_. Začnite z dodajanjem knjižnic, ki vam bodo pomagale naložiti in vizualizirati podatke.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Upoštevajte, da uporabljate datoteke iz priložene mape `common`, ki nastavijo vaše okolje in poskrbijo za prenos podatkov.

2. Nato preglejte podatke kot podatkovni okvir z uporabo `load_data()` in `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Vidite lahko, da sta dva stolpca, ki predstavljata datum in porabo:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Zdaj narišite podatke z uporabo `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![graf energije](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Zdaj narišite prvi teden julija 2014, tako da ga podate kot vhod v `energy` v vzorcu `[od datuma]: [do datuma]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julij](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Čudovit graf! Oglejte si te grafe in preverite, ali lahko določite katere koli od zgoraj navedenih značilnosti. Kaj lahko sklepamo z vizualizacijo podatkov?

V naslednji lekciji boste ustvarili ARIMA model za izdelavo napovedi.

---

## 🚀Izziv

Naredite seznam vseh industrij in področij raziskovanja, ki bi lahko imeli koristi od napovedovanja časovnih vrst. Ali lahko pomislite na aplikacijo teh tehnik v umetnosti? V ekonometriji? Ekologiji? Maloprodaji? Industriji? Financah? Kje še?

## [Po-lekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Čeprav jih tukaj ne bomo obravnavali, se včasih za izboljšanje klasičnih metod napovedovanja časovnih vrst uporabljajo nevronske mreže. Preberite več o njih [v tem članku](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Naloga

[Vizualizirajte še več časovnih vrst](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.