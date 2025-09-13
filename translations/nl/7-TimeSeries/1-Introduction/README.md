<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T19:03:03+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "nl"
}
-->
# Introductie tot tijdreeksvoorspelling

![Samenvatting van tijdreeksen in een sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

In deze les en de volgende leer je meer over tijdreeksvoorspelling, een interessant en waardevol onderdeel van het repertoire van een ML-wetenschapper dat minder bekend is dan andere onderwerpen. Tijdreeksvoorspelling is een soort 'kristallen bol': op basis van eerdere prestaties van een variabele, zoals prijs, kun je de toekomstige potentiële waarde voorspellen.

[![Introductie tot tijdreeksvoorspelling](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introductie tot tijdreeksvoorspelling")

> 🎥 Klik op de afbeelding hierboven voor een video over tijdreeksvoorspelling

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Het is een nuttig en interessant vakgebied met echte waarde voor bedrijven, gezien de directe toepassing op problemen zoals prijsstelling, voorraadbeheer en supply chain vraagstukken. Hoewel technieken voor deep learning steeds vaker worden gebruikt om meer inzichten te verkrijgen en toekomstige prestaties beter te voorspellen, blijft tijdreeksvoorspelling een vakgebied dat sterk wordt beïnvloed door klassieke ML-technieken.

> Penn State's nuttige curriculum over tijdreeksen is te vinden [hier](https://online.stat.psu.edu/stat510/lesson/1)

## Introductie

Stel je voor dat je een reeks slimme parkeermeters beheert die gegevens leveren over hoe vaak ze worden gebruikt en hoe lang, over een bepaalde periode.

> Wat als je, op basis van de eerdere prestaties van de meter, de toekomstige waarde zou kunnen voorspellen volgens de wetten van vraag en aanbod?

Het nauwkeurig voorspellen van wanneer je moet handelen om je doel te bereiken, is een uitdaging die kan worden aangepakt met tijdreeksvoorspelling. Het zou mensen niet blij maken om meer te betalen in drukke tijden wanneer ze op zoek zijn naar een parkeerplaats, maar het zou zeker een manier zijn om inkomsten te genereren om de straten schoon te maken!

Laten we enkele soorten tijdreeksalgoritmen verkennen en een notebook starten om enkele gegevens te reinigen en voor te bereiden. De gegevens die je gaat analyseren zijn afkomstig van de GEFCom2014 voorspellingscompetitie. Ze bestaan uit 3 jaar aan uurlijkse elektriciteitsbelasting en temperatuurwaarden tussen 2012 en 2014. Gegeven de historische patronen van elektriciteitsbelasting en temperatuur, kun je toekomstige waarden van elektriciteitsbelasting voorspellen.

In dit voorbeeld leer je hoe je één tijdstap vooruit kunt voorspellen, alleen met behulp van historische belastinggegevens. Voordat je begint, is het echter nuttig om te begrijpen wat er achter de schermen gebeurt.

## Enkele definities

Wanneer je de term 'tijdreeks' tegenkomt, moet je begrijpen hoe deze in verschillende contexten wordt gebruikt.

🎓 **Tijdreeks**

In de wiskunde is "een tijdreeks een reeks datapunten die zijn geïndexeerd (of opgesomd of geplot) in tijdsvolgorde. Meestal is een tijdreeks een reeks die is genomen op opeenvolgende, gelijkmatig verdeelde tijdspunten." Een voorbeeld van een tijdreeks is de dagelijkse slotwaarde van de [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Het gebruik van tijdreeksplots en statistische modellering wordt vaak aangetroffen in signaalverwerking, weersvoorspelling, aardbevingsvoorspelling en andere gebieden waar gebeurtenissen plaatsvinden en datapunten over tijd kunnen worden geplot.

🎓 **Tijdreeksanalyse**

Tijdreeksanalyse is de analyse van bovengenoemde tijdreeksgegevens. Tijdreeksgegevens kunnen verschillende vormen aannemen, waaronder 'onderbroken tijdreeksen' die patronen detecteren in de evolutie van een tijdreeks vóór en na een onderbrekende gebeurtenis. Het type analyse dat nodig is voor de tijdreeks hangt af van de aard van de gegevens. Tijdreeksgegevens zelf kunnen de vorm aannemen van reeksen getallen of tekens.

De analyse die wordt uitgevoerd, maakt gebruik van verschillende methoden, waaronder frequentiedomein en tijdsdomein, lineair en niet-lineair, en meer. [Lees meer](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) over de vele manieren om dit type gegevens te analyseren.

🎓 **Tijdreeksvoorspelling**

Tijdreeksvoorspelling is het gebruik van een model om toekomstige waarden te voorspellen op basis van patronen die worden weergegeven door eerder verzamelde gegevens zoals die in het verleden zijn opgetreden. Hoewel het mogelijk is om regressiemodellen te gebruiken om tijdreeksgegevens te verkennen, met tijdsindices als x-variabelen op een plot, worden dergelijke gegevens het best geanalyseerd met speciale soorten modellen.

Tijdreeksgegevens zijn een lijst van geordende observaties, in tegenstelling tot gegevens die kunnen worden geanalyseerd door lineaire regressie. Het meest voorkomende model is ARIMA, een acroniem dat staat voor "Autoregressive Integrated Moving Average".

[ARIMA-modellen](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relateren de huidige waarde van een reeks aan eerdere waarden en eerdere voorspellingsfouten." Ze zijn het meest geschikt voor het analyseren van tijdsdomeingegevens, waarbij gegevens in de tijd zijn geordend.

> Er zijn verschillende soorten ARIMA-modellen, waar je meer over kunt leren [hier](https://people.duke.edu/~rnau/411arim.htm) en die je in de volgende les zult behandelen.

In de volgende les bouw je een ARIMA-model met behulp van [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), dat zich richt op één variabele die in de tijd verandert. Een voorbeeld van dit type gegevens is [deze dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) die de maandelijkse CO2-concentratie bij het Mauna Loa Observatorium registreert:

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

✅ Identificeer de variabele die in de tijd verandert in deze dataset

## Kenmerken van tijdreeksgegevens om te overwegen

Wanneer je naar tijdreeksgegevens kijkt, kun je merken dat ze [bepaalde kenmerken](https://online.stat.psu.edu/stat510/lesson/1/1.1) hebben die je moet begrijpen en aanpakken om hun patronen beter te begrijpen. Als je tijdreeksgegevens beschouwt als mogelijk een 'signaal' dat je wilt analyseren, kunnen deze kenmerken worden gezien als 'ruis'. Je zult vaak deze 'ruis' moeten verminderen door enkele van deze kenmerken te compenseren met statistische technieken.

Hier zijn enkele concepten die je moet kennen om met tijdreeksen te werken:

🎓 **Trends**

Trends worden gedefinieerd als meetbare stijgingen en dalingen in de tijd. [Lees meer](https://machinelearningmastery.com/time-series-trends-in-python). In de context van tijdreeksen gaat het om hoe je trends kunt gebruiken en, indien nodig, verwijderen uit je tijdreeks.

🎓 **[Seizoensgebondenheid](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Seizoensgebondenheid wordt gedefinieerd als periodieke schommelingen, zoals bijvoorbeeld de drukte tijdens feestdagen die de verkoop kunnen beïnvloeden. [Bekijk](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) hoe verschillende soorten plots seizoensgebondenheid in gegevens weergeven.

🎓 **Uitschieters**

Uitschieters liggen ver buiten de standaard datavariatie.

🎓 **Langetermijncyclus**

Onafhankelijk van seizoensgebondenheid kunnen gegevens een langetermijncyclus vertonen, zoals een economische neergang die langer dan een jaar duurt.

🎓 **Constante variatie**

In de tijd vertonen sommige gegevens constante schommelingen, zoals energieverbruik per dag en nacht.

🎓 **Abrupte veranderingen**

De gegevens kunnen een abrupte verandering vertonen die verdere analyse vereist. Bijvoorbeeld de plotselinge sluiting van bedrijven door COVID veroorzaakte veranderingen in gegevens.

✅ Hier is een [voorbeeld van een tijdreeksplot](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) die dagelijks in-game valuta-uitgaven toont over een paar jaar. Kun je een van de hierboven genoemde kenmerken identificeren in deze gegevens?

![In-game valuta-uitgaven](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Oefening - aan de slag met stroomverbruiksgegevens

Laten we beginnen met het maken van een tijdreeksmodel om toekomstig stroomverbruik te voorspellen op basis van eerder verbruik.

> De gegevens in dit voorbeeld zijn afkomstig van de GEFCom2014 voorspellingscompetitie. Ze bestaan uit 3 jaar aan uurlijkse elektriciteitsbelasting en temperatuurwaarden tussen 2012 en 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli en Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, juli-september, 2016.

1. Open in de `working` map van deze les het bestand _notebook.ipynb_. Begin met het toevoegen van bibliotheken die je helpen gegevens te laden en te visualiseren:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Let op, je gebruikt de bestanden uit de meegeleverde `common` map die je omgeving instellen en het downloaden van de gegevens afhandelen.

2. Bekijk vervolgens de gegevens als een dataframe door `load_data()` en `head()` aan te roepen:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Je kunt zien dat er twee kolommen zijn die datum en belasting vertegenwoordigen:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Plot nu de gegevens door `plot()` aan te roepen:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![energieplot](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Plot vervolgens de eerste week van juli 2014 door deze als invoer te geven aan `energy` in het `[van datum]: [tot datum]` patroon:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![juli](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Een prachtige plot! Bekijk deze plots en kijk of je een van de hierboven genoemde kenmerken kunt bepalen. Wat kunnen we afleiden door de gegevens te visualiseren?

In de volgende les maak je een ARIMA-model om enkele voorspellingen te doen.

---

## 🚀Uitdaging

Maak een lijst van alle industrieën en onderzoeksgebieden die volgens jou zouden profiteren van tijdreeksvoorspelling. Kun je een toepassing van deze technieken bedenken in de kunsten? In econometrie? Ecologie? Detailhandel? Industrie? Financiën? Waar nog meer?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Hoewel we ze hier niet behandelen, worden neurale netwerken soms gebruikt om klassieke methoden van tijdreeksvoorspelling te verbeteren. Lees er meer over [in dit artikel](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Opdracht

[Visualiseer meer tijdreeksen](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen voor nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.