<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T21:21:53+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "no"
}
-->
# Introduksjon til tidsserieprognoser

![Oppsummering av tidsserier i en sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

I denne leksjonen og den neste vil du lære litt om tidsserieprognoser, en interessant og verdifull del av en ML-forskers verktøykasse som er litt mindre kjent enn andre emner. Tidsserieprognoser er som en slags "krystallkule": basert på tidligere ytelse av en variabel, som pris, kan du forutsi dens fremtidige potensielle verdi.

[![Introduksjon til tidsserieprognoser](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introduksjon til tidsserieprognoser")

> 🎥 Klikk på bildet over for en video om tidsserieprognoser

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

Det er et nyttig og interessant felt med reell verdi for næringslivet, gitt dets direkte anvendelse på problemer som prissetting, lagerstyring og forsyningskjedeutfordringer. Selv om dype læringsteknikker har begynt å bli brukt for å få mer innsikt og bedre forutsi fremtidig ytelse, forblir tidsserieprognoser et felt som i stor grad er informert av klassiske ML-teknikker.

> Penn States nyttige tidsseriepensum finner du [her](https://online.stat.psu.edu/stat510/lesson/1)

## Introduksjon

Anta at du administrerer en rekke smarte parkeringsmålere som gir data om hvor ofte de brukes og hvor lenge over tid.

> Hva om du kunne forutsi, basert på målerens tidligere ytelse, dens fremtidige verdi i henhold til lovene om tilbud og etterspørsel?

Å forutsi nøyaktig når man skal handle for å oppnå sitt mål er en utfordring som kan takles med tidsserieprognoser. Det ville ikke gjøre folk glade å bli belastet mer i travle tider når de leter etter en parkeringsplass, men det ville være en sikker måte å generere inntekter for å rengjøre gatene!

La oss utforske noen av typene tidsseriealgoritmer og starte en notebook for å rense og forberede noen data. Dataene du vil analysere er hentet fra GEFCom2014 prognosekonkurransen. Det består av 3 år med timebaserte verdier for elektrisitetsforbruk og temperatur mellom 2012 og 2014. Gitt de historiske mønstrene for elektrisitetsforbruk og temperatur, kan du forutsi fremtidige verdier for elektrisitetsforbruk.

I dette eksemplet vil du lære hvordan du forutsier ett tidssteg fremover, ved å bruke kun historiske forbruksdata. Før du starter, er det imidlertid nyttig å forstå hva som skjer bak kulissene.

## Noen definisjoner

Når du møter begrepet "tidsserie", må du forstå dets bruk i flere forskjellige sammenhenger.

🎓 **Tidsserie**

I matematikk er "en tidsserie en serie med datapunkter indeksert (eller listet eller grafet) i tidsrekkefølge. Oftest er en tidsserie en sekvens tatt ved suksessive, jevnt fordelte punkter i tid." Et eksempel på en tidsserie er den daglige sluttverdien av [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Bruken av tidsserieplott og statistisk modellering er ofte sett i signalbehandling, værprognoser, jordskjelvforutsigelser og andre felt der hendelser oppstår og datapunkter kan plottes over tid.

🎓 **Tidsserieanalyse**

Tidsserieanalyse er analysen av den ovennevnte tidsseriedataen. Tidsseriedata kan ta forskjellige former, inkludert "avbrutte tidsserier" som oppdager mønstre i en tidsseries utvikling før og etter en avbrytende hendelse. Typen analyse som trengs for tidsserien, avhenger av naturen til dataene. Tidsseriedata i seg selv kan ta form av serier med tall eller tegn.

Analysen som skal utføres, bruker en rekke metoder, inkludert frekvensdomene og tidsdomene, lineære og ikke-lineære, og mer. [Lær mer](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) om de mange måtene å analysere denne typen data.

🎓 **Tidsserieprognoser**

Tidsserieprognoser er bruken av en modell for å forutsi fremtidige verdier basert på mønstre vist av tidligere innsamlede data slik de oppstod i fortiden. Selv om det er mulig å bruke regresjonsmodeller for å utforske tidsseriedata, med tidsindekser som x-variabler på et plott, analyseres slike data best ved hjelp av spesielle typer modeller.

Tidsseriedata er en liste med ordnede observasjoner, i motsetning til data som kan analyseres ved lineær regresjon. Den vanligste modellen er ARIMA, et akronym som står for "Autoregressive Integrated Moving Average".

[ARIMA-modeller](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relaterer den nåværende verdien av en serie til tidligere verdier og tidligere prediksjonsfeil." De er mest passende for å analysere tidsdomenedata, der data er ordnet over tid.

> Det finnes flere typer ARIMA-modeller, som du kan lære om [her](https://people.duke.edu/~rnau/411arim.htm) og som du vil berøre i neste leksjon.

I neste leksjon vil du bygge en ARIMA-modell ved hjelp av [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), som fokuserer på én variabel som endrer sin verdi over tid. Et eksempel på denne typen data er [dette datasettet](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) som registrerer den månedlige CO2-konsentrasjonen ved Mauna Loa-observatoriet:

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

✅ Identifiser variabelen som endrer seg over tid i dette datasettet

## Tidsseriedataegenskaper å vurdere

Når du ser på tidsseriedata, kan du legge merke til at det har [visse egenskaper](https://online.stat.psu.edu/stat510/lesson/1/1.1) som du må ta hensyn til og redusere for bedre å forstå mønstrene. Hvis du ser på tidsseriedata som potensielt gir et "signal" du vil analysere, kan disse egenskapene betraktes som "støy". Du må ofte redusere denne "støyen" ved å kompensere for noen av disse egenskapene ved hjelp av statistiske teknikker.

Her er noen begreper du bør kjenne til for å kunne jobbe med tidsserier:

🎓 **Trender**

Trender er definert som målbare økninger og reduksjoner over tid. [Les mer](https://machinelearningmastery.com/time-series-trends-in-python). I sammenheng med tidsserier handler det om hvordan man bruker og, hvis nødvendig, fjerner trender fra tidsserien.

🎓 **[Sesongvariasjon](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sesongvariasjon er definert som periodiske svingninger, som for eksempel høytidssalg som kan påvirke salget. [Ta en titt](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) på hvordan forskjellige typer plott viser sesongvariasjon i data.

🎓 **Avvik**

Avvik er datapunkter som ligger langt unna den normale variansen i dataene.

🎓 **Langsiktig syklus**

Uavhengig av sesongvariasjon kan data vise en langsiktig syklus, som for eksempel en økonomisk nedgang som varer lenger enn ett år.

🎓 **Konstant varians**

Over tid viser noen data konstante svingninger, som energiforbruk per dag og natt.

🎓 **Brå endringer**

Dataene kan vise en brå endring som kan kreve ytterligere analyse. For eksempel førte den plutselige nedstengningen av virksomheter på grunn av COVID til endringer i dataene.

✅ Her er et [eksempel på et tidsserieplott](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) som viser daglig bruk av in-game valuta over noen år. Kan du identifisere noen av egenskapene som er nevnt ovenfor i disse dataene?

![In-game valuta](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Øvelse - komme i gang med strømforbruksdata

La oss komme i gang med å lage en tidsseriemodell for å forutsi fremtidig strømforbruk basert på tidligere forbruk.

> Dataene i dette eksemplet er hentet fra GEFCom2014 prognosekonkurransen. Det består av 3 år med timebaserte verdier for elektrisitetsforbruk og temperatur mellom 2012 og 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli og Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, juli-september, 2016.

1. I `working`-mappen for denne leksjonen, åpne _notebook.ipynb_-filen. Start med å legge til biblioteker som vil hjelpe deg med å laste inn og visualisere data

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Merk at du bruker filene fra den inkluderte `common`-mappen som setter opp miljøet ditt og håndterer nedlasting av dataene.

2. Undersøk deretter dataene som en dataframe ved å kalle `load_data()` og `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Du kan se at det er to kolonner som representerer dato og forbruk:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Nå, plott dataene ved å kalle `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energiplott](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Nå, plott den første uken i juli 2014, ved å gi den som input til `energy` i `[fra dato]: [til dato]`-mønsteret:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![juli](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Et vakkert plott! Ta en titt på disse plottene og se om du kan identifisere noen av egenskapene som er nevnt ovenfor. Hva kan vi anta ved å visualisere dataene?

I neste leksjon vil du lage en ARIMA-modell for å lage noen prognoser.

---

## 🚀Utfordring

Lag en liste over alle bransjer og forskningsområder du kan tenke deg som vil ha nytte av tidsserieprognoser. Kan du tenke deg en anvendelse av disse teknikkene innen kunst? Innen økonometrikk? Økologi? Detaljhandel? Industri? Finans? Hvor ellers?

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Selv om vi ikke dekker dem her, brukes nevrale nettverk noen ganger for å forbedre klassiske metoder for tidsserieprognoser. Les mer om dem [i denne artikkelen](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Oppgave

[Visualiser flere tidsserier](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.