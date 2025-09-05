<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T11:59:10+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "hr"
}
-->
# Uvod u predviđanje vremenskih serija

![Sažetak vremenskih serija u sketchnoteu](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

U ovoj lekciji i sljedećoj, naučit ćete nešto o predviđanju vremenskih serija, zanimljivom i vrijednom dijelu repertoara ML znanstvenika koji je nešto manje poznat od drugih tema. Predviđanje vremenskih serija je poput 'kristalne kugle': na temelju prošlih performansi varijable, poput cijene, možete predvidjeti njezinu buduću potencijalnu vrijednost.

[![Uvod u predviđanje vremenskih serija](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Uvod u predviđanje vremenskih serija")

> 🎥 Kliknite na sliku iznad za video o predviđanju vremenskih serija

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

To je korisno i zanimljivo područje s pravom vrijednošću za poslovanje, s obzirom na njegovu izravnu primjenu na probleme cijena, inventara i pitanja opskrbnog lanca. Iako su tehnike dubokog učenja počele pružati više uvida za bolje predviđanje budućih performansi, predviđanje vremenskih serija ostaje područje koje se uvelike oslanja na klasične ML tehnike.

> Korisni kurikulum o vremenskim serijama sa Sveučilišta Penn State možete pronaći [ovdje](https://online.stat.psu.edu/stat510/lesson/1)

## Uvod

Pretpostavimo da održavate niz pametnih parkirnih mjerača koji pružaju podatke o tome koliko često se koriste i koliko dugo tijekom vremena.

> Što ako biste mogli predvidjeti, na temelju prošlih performansi mjerača, njegovu buduću vrijednost prema zakonima ponude i potražnje?

Točno predviđanje kada djelovati kako biste postigli svoj cilj izazov je koji se može riješiti predviđanjem vremenskih serija. Ne bi bilo popularno naplaćivati više u prometnim vremenima kada ljudi traže parkirno mjesto, ali to bi bio siguran način za generiranje prihoda za čišćenje ulica!

Istražimo neke vrste algoritama vremenskih serija i započnimo rad u bilježnici kako bismo očistili i pripremili podatke. Podaci koje ćete analizirati preuzeti su iz natjecanja za predviđanje GEFCom2014. Sastoje se od 3 godine satnih vrijednosti potrošnje električne energije i temperature između 2012. i 2014. Na temelju povijesnih obrazaca potrošnje električne energije i temperature, možete predvidjeti buduće vrijednosti potrošnje električne energije.

U ovom primjeru naučit ćete kako predvidjeti jedan korak unaprijed, koristeći samo povijesne podatke o potrošnji. Međutim, prije nego što započnete, korisno je razumjeti što se događa iza kulisa.

## Neke definicije

Kada naiđete na pojam 'vremenske serije', trebate razumjeti njegovu upotrebu u nekoliko različitih konteksta.

🎓 **Vremenske serije**

U matematici, "vremenska serija je niz podatkovnih točaka indeksiranih (ili navedenih ili grafički prikazanih) u vremenskom redoslijedu. Najčešće, vremenska serija je sekvenca uzeta u sukcesivnim jednako razmaknutim vremenskim točkama." Primjer vremenske serije je dnevna završna vrijednost [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Upotreba grafova vremenskih serija i statističkog modeliranja često se susreće u obradi signala, prognozi vremena, predviđanju potresa i drugim područjima gdje se događaji odvijaju i podatkovne točke mogu biti prikazane tijekom vremena.

🎓 **Analiza vremenskih serija**

Analiza vremenskih serija odnosi se na analizu gore spomenutih podataka vremenskih serija. Podaci vremenskih serija mogu imati različite oblike, uključujući 'prekinute vremenske serije' koje otkrivaju obrasce u evoluciji vremenske serije prije i nakon prekidnog događaja. Vrsta analize potrebna za vremensku seriju ovisi o prirodi podataka. Sami podaci vremenskih serija mogu biti niz brojeva ili znakova.

Analiza se provodi koristeći razne metode, uključujući frekvencijsku domenu i vremensku domenu, linearne i nelinearne metode i druge. [Saznajte više](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) o mnogim načinima analize ove vrste podataka.

🎓 **Predviđanje vremenskih serija**

Predviđanje vremenskih serija je upotreba modela za predviđanje budućih vrijednosti na temelju obrazaca prikazanih prethodno prikupljenim podacima kako su se događali u prošlosti. Iako je moguće koristiti regresijske modele za istraživanje podataka vremenskih serija, s vremenskim indeksima kao x varijablama na grafu, takvi podaci najbolje se analiziraju pomoću posebnih vrsta modela.

Podaci vremenskih serija su popis uređenih opažanja, za razliku od podataka koji se mogu analizirati linearnom regresijom. Najčešći model je ARIMA, akronim koji označava "Autoregresivni Integrirani Pokretni Prosjek".

[ARIMA modeli](https://online.stat.psu.edu/stat510/lesson/1/1.1) "povezuju trenutnu vrijednost serije s prošlim vrijednostima i prošlim pogreškama predviđanja." Najprikladniji su za analizu podataka vremenske domene, gdje su podaci poredani tijekom vremena.

> Postoji nekoliko vrsta ARIMA modela, o kojima možete saznati [ovdje](https://people.duke.edu/~rnau/411arim.htm) i koje ćete obraditi u sljedećoj lekciji.

U sljedećoj lekciji izgradit ćete ARIMA model koristeći [Univarijantne Vremenske Serije](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), koje se fokusiraju na jednu varijablu koja mijenja svoju vrijednost tijekom vremena. Primjer ove vrste podataka je [ovaj skup podataka](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) koji bilježi mjesečnu koncentraciju CO2 na opservatoriju Mauna Loa:

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

✅ Identificirajte varijablu koja se mijenja tijekom vremena u ovom skupu podataka.

## Karakteristike podataka vremenskih serija koje treba uzeti u obzir

Kada promatrate podatke vremenskih serija, možda ćete primijetiti da imaju [određene karakteristike](https://online.stat.psu.edu/stat510/lesson/1/1.1) koje trebate uzeti u obzir i ublažiti kako biste bolje razumjeli njihove obrasce. Ako smatrate da podaci vremenskih serija potencijalno pružaju 'signal' koji želite analizirati, ove karakteristike mogu se smatrati 'šumom'. Često ćete morati smanjiti taj 'šum' koristeći neke statističke tehnike.

Evo nekoliko pojmova koje biste trebali znati kako biste mogli raditi s vremenskim serijama:

🎓 **Trendovi**

Trendovi su definirani kao mjerljiva povećanja i smanjenja tijekom vremena. [Pročitajte više](https://machinelearningmastery.com/time-series-trends-in-python). U kontekstu vremenskih serija, radi se o tome kako koristiti i, ako je potrebno, ukloniti trendove iz vremenskih serija.

🎓 **[Sezonalnost](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sezonalnost je definirana kao periodične fluktuacije, poput blagdanskih navala koje mogu utjecati na prodaju, na primjer. [Pogledajte](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) kako različite vrste grafova prikazuju sezonalnost u podacima.

🎓 **Izvanredne vrijednosti**

Izvanredne vrijednosti su daleko od standardne varijance podataka.

🎓 **Dugoročni ciklus**

Neovisno o sezonalnosti, podaci mogu pokazivati dugoročni ciklus, poput ekonomskog pada koji traje dulje od godine dana.

🎓 **Konstantna varijanca**

Tijekom vremena, neki podaci pokazuju konstantne fluktuacije, poput potrošnje energije po danu i noći.

🎓 **Nagla promjena**

Podaci mogu pokazivati naglu promjenu koja može zahtijevati daljnju analizu. Naglo zatvaranje poslovanja zbog COVID-a, na primjer, uzrokovalo je promjene u podacima.

✅ Ovdje je [primjer grafičkog prikaza vremenskih serija](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) koji prikazuje dnevnu potrošnju valute u igri tijekom nekoliko godina. Možete li identificirati neke od gore navedenih karakteristika u ovim podacima?

![Potrošnja valute u igri](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Vježba - početak rada s podacima o potrošnji energije

Započnimo stvaranje modela vremenskih serija za predviđanje buduće potrošnje energije na temelju prošle potrošnje.

> Podaci u ovom primjeru preuzeti su iz natjecanja za predviđanje GEFCom2014. Sastoje se od 3 godine satnih vrijednosti potrošnje električne energije i temperature između 2012. i 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli i Rob J. Hyndman, "Probabilističko energetsko predviđanje: Globalno natjecanje za energetsko predviđanje 2014 i dalje", International Journal of Forecasting, vol.32, no.3, str. 896-913, srpanj-rujan, 2016.

1. U mapi `working` ove lekcije, otvorite datoteku _notebook.ipynb_. Započnite dodavanjem biblioteka koje će vam pomoći učitati i vizualizirati podatke.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Napomena, koristite datoteke iz uključene mape `common` koje postavljaju vaše okruženje i upravljaju preuzimanjem podataka.

2. Zatim, pregledajte podatke kao dataframe pozivajući `load_data()` i `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Možete vidjeti da postoje dva stupca koji predstavljaju datum i potrošnju:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Sada, grafički prikažite podatke pozivajući `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![graf potrošnje energije](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Sada, grafički prikažite prvi tjedan srpnja 2014., pružajući ga kao ulaz u `energy` u obrascu `[od datuma]: [do datuma]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![srpanj](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Prekrasan graf! Pogledajte ove grafove i pokušajte odrediti neke od gore navedenih karakteristika. Što možemo zaključiti vizualizacijom podataka?

U sljedećoj lekciji, izradit ćete ARIMA model za stvaranje nekih predviđanja.

---

## 🚀Izazov

Napravite popis svih industrija i područja istraživanja koja bi mogla imati koristi od predviđanja vremenskih serija. Možete li smisliti primjenu ovih tehnika u umjetnosti? U ekonometriji? Ekologiji? Maloprodaji? Industriji? Financijama? Gdje još?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Iako ih ovdje nećemo obraditi, neuronske mreže ponekad se koriste za poboljšanje klasičnih metoda predviđanja vremenskih serija. Pročitajte više o njima [u ovom članku](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Zadatak

[Vizualizirajte još vremenskih serija](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.