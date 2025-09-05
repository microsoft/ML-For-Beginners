<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T21:16:31+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "no"
}
-->
# Bygg en regresjonsmodell med Scikit-learn: forbered og visualiser data

![Infografikk for datavisualisering](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografikk av [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne leksjonen er tilgjengelig i R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introduksjon

Nå som du har satt opp verktøyene du trenger for å begynne å bygge maskinlæringsmodeller med Scikit-learn, er du klar til å begynne å stille spørsmål til dataene dine. Når du jobber med data og bruker ML-løsninger, er det svært viktig å forstå hvordan du stiller de riktige spørsmålene for å utnytte potensialet i datasettet ditt.

I denne leksjonen vil du lære:

- Hvordan forberede dataene dine for modellbygging.
- Hvordan bruke Matplotlib til datavisualisering.

## Stille de riktige spørsmålene til dataene dine

Spørsmålet du ønsker svar på vil avgjøre hvilken type ML-algoritmer du vil bruke. Og kvaliteten på svaret du får tilbake vil være sterkt avhengig av kvaliteten på dataene dine.

Ta en titt på [dataene](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) som er gitt for denne leksjonen. Du kan åpne denne .csv-filen i VS Code. Et raskt blikk viser umiddelbart at det finnes tomme felter og en blanding av tekst og numeriske data. Det er også en merkelig kolonne kalt 'Package' hvor dataene er en blanding av 'sacks', 'bins' og andre verdier. Dataene er faktisk litt rotete.

[![ML for nybegynnere - Hvordan analysere og rense et datasett](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML for nybegynnere - Hvordan analysere og rense et datasett")

> 🎥 Klikk på bildet over for en kort video som viser hvordan du forbereder dataene for denne leksjonen.

Det er faktisk ikke veldig vanlig å få et datasett som er helt klart til bruk for å lage en ML-modell rett ut av boksen. I denne leksjonen vil du lære hvordan du forbereder et rådatasett ved hjelp av standard Python-biblioteker. Du vil også lære ulike teknikker for å visualisere dataene.

## Case-studie: 'gresskarmarkedet'

I denne mappen finner du en .csv-fil i rotmappen `data` kalt [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) som inneholder 1757 linjer med data om markedet for gresskar, sortert i grupper etter by. Dette er rådata hentet fra [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuert av United States Department of Agriculture.

### Forberede data

Disse dataene er i det offentlige domene. De kan lastes ned i mange separate filer, per by, fra USDA-nettstedet. For å unngå for mange separate filer har vi slått sammen alle bydataene til ett regneark, så vi har allerede _forberedt_ dataene litt. La oss nå ta en nærmere titt på dataene.

### Gresskardata - tidlige konklusjoner

Hva legger du merke til med disse dataene? Du har allerede sett at det er en blanding av tekst, tall, tomme felter og merkelige verdier som du må forstå.

Hvilket spørsmål kan du stille til disse dataene ved hjelp av en regresjonsteknikk? Hva med "Forutsi prisen på et gresskar som selges i løpet av en gitt måned". Når du ser på dataene igjen, er det noen endringer du må gjøre for å skape den datastrukturen som er nødvendig for oppgaven.

## Øvelse - analyser gresskardataene

La oss bruke [Pandas](https://pandas.pydata.org/), (navnet står for `Python Data Analysis`) et verktøy som er svært nyttig for å forme data, til å analysere og forberede disse gresskardataene.

### Først, sjekk for manglende datoer

Du må først ta steg for å sjekke for manglende datoer:

1. Konverter datoene til et månedsformat (disse er amerikanske datoer, så formatet er `MM/DD/YYYY`).
2. Ekstraher måneden til en ny kolonne.

Åpne _notebook.ipynb_-filen i Visual Studio Code og importer regnearket til en ny Pandas dataframe.

1. Bruk funksjonen `head()` for å se de første fem radene.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Hvilken funksjon ville du brukt for å se de siste fem radene?

1. Sjekk om det er manglende data i den nåværende dataframen:

    ```python
    pumpkins.isnull().sum()
    ```

    Det er manglende data, men kanskje det ikke vil ha betydning for oppgaven.

1. For å gjøre dataframen din enklere å jobbe med, velg kun de kolonnene du trenger, ved å bruke funksjonen `loc` som henter ut en gruppe rader (gitt som første parameter) og kolonner (gitt som andre parameter) fra den originale dataframen. Uttrykket `:` i eksempelet nedenfor betyr "alle rader".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Deretter, bestem gjennomsnittsprisen på gresskar

Tenk på hvordan du kan bestemme gjennomsnittsprisen på et gresskar i en gitt måned. Hvilke kolonner ville du valgt for denne oppgaven? Hint: du trenger 3 kolonner.

Løsning: ta gjennomsnittet av kolonnene `Low Price` og `High Price` for å fylle den nye kolonnen Price, og konverter Date-kolonnen til kun å vise måneden. Heldigvis, ifølge sjekken ovenfor, er det ingen manglende data for datoer eller priser.

1. For å beregne gjennomsnittet, legg til følgende kode:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Du kan gjerne skrive ut data du ønsker å sjekke ved å bruke `print(month)`.

2. Kopier deretter de konverterte dataene til en ny Pandas dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Hvis du skriver ut dataframen din, vil du se et rent og ryddig datasett som du kan bruke til å bygge din nye regresjonsmodell.

### Men vent! Det er noe merkelig her

Hvis du ser på kolonnen `Package`, blir gresskar solgt i mange forskjellige konfigurasjoner. Noen blir solgt i '1 1/9 bushel'-mål, og noen i '1/2 bushel'-mål, noen per gresskar, noen per pund, og noen i store bokser med varierende bredder.

> Gresskar virker veldig vanskelig å veie konsekvent

Når du graver i de originale dataene, er det interessant at alt med `Unit of Sale` lik 'EACH' eller 'PER BIN' også har `Package`-typen per tomme, per bin, eller 'each'. Gresskar virker veldig vanskelig å veie konsekvent, så la oss filtrere dem ved å velge kun gresskar med strengen 'bushel' i kolonnen `Package`.

1. Legg til et filter øverst i filen, under den opprinnelige .csv-importen:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Hvis du skriver ut dataene nå, kan du se at du kun får de 415 eller så radene med data som inneholder gresskar per bushel.

### Men vent! Det er én ting til å gjøre

La du merke til at bushel-mengden varierer per rad? Du må normalisere prisingen slik at du viser prisen per bushel, så gjør litt matematikk for å standardisere det.

1. Legg til disse linjene etter blokken som oppretter new_pumpkins-dataframen:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Ifølge [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), avhenger vekten av en bushel av typen produkt, siden det er et volummål. "En bushel med tomater, for eksempel, skal veie 56 pund... Blader og grønnsaker tar opp mer plass med mindre vekt, så en bushel med spinat er bare 20 pund." Det er ganske komplisert! La oss ikke bry oss med å gjøre en bushel-til-pund-konvertering, og i stedet prise per bushel. All denne studien av bushels med gresskar viser imidlertid hvor viktig det er å forstå naturen til dataene dine!

Nå kan du analysere prisingen per enhet basert på deres bushel-mål. Hvis du skriver ut dataene en gang til, kan du se hvordan det er standardisert.

✅ La du merke til at gresskar som selges per halv-bushel er veldig dyre? Kan du finne ut hvorfor? Hint: små gresskar er mye dyrere enn store, sannsynligvis fordi det er så mange flere av dem per bushel, gitt det ubrukte rommet som tas opp av ett stort hullete pai-gresskar.

## Visualiseringsstrategier

En del av rollen til en dataforsker er å demonstrere kvaliteten og naturen til dataene de jobber med. For å gjøre dette lager de ofte interessante visualiseringer, eller diagrammer, grafer og tabeller, som viser ulike aspekter av dataene. På denne måten kan de visuelt vise relasjoner og mangler som ellers er vanskelig å avdekke.

[![ML for nybegynnere - Hvordan visualisere data med Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML for nybegynnere - Hvordan visualisere data med Matplotlib")

> 🎥 Klikk på bildet over for en kort video som viser hvordan du visualiserer dataene for denne leksjonen.

Visualiseringer kan også hjelpe med å avgjøre hvilken maskinlæringsteknikk som er mest passende for dataene. Et spredningsdiagram som ser ut til å følge en linje, for eksempel, indikerer at dataene er en god kandidat for en lineær regresjonsøvelse.

Et datavisualiseringsbibliotek som fungerer godt i Jupyter-notebooks er [Matplotlib](https://matplotlib.org/) (som du også så i forrige leksjon).

> Få mer erfaring med datavisualisering i [disse opplæringene](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Øvelse - eksperimenter med Matplotlib

Prøv å lage noen grunnleggende diagrammer for å vise den nye dataframen du nettopp opprettet. Hva ville et grunnleggende linjediagram vise?

1. Importer Matplotlib øverst i filen, under Pandas-importen:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Kjør hele notebooken på nytt for å oppdatere.
1. Nederst i notebooken, legg til en celle for å plotte dataene som en boks:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Et spredningsdiagram som viser pris til måned-forhold](../../../../2-Regression/2-Data/images/scatterplot.png)

    Er dette et nyttig diagram? Overrasker noe ved det deg?

    Det er ikke spesielt nyttig, da alt det gjør er å vise dataene dine som en spredning av punkter i en gitt måned.

### Gjør det nyttig

For å få diagrammer til å vise nyttige data, må du vanligvis gruppere dataene på en eller annen måte. La oss prøve å lage et diagram hvor y-aksen viser månedene og dataene demonstrerer distribusjonen av data.

1. Legg til en celle for å lage et gruppert stolpediagram:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Et stolpediagram som viser pris til måned-forhold](../../../../2-Regression/2-Data/images/barchart.png)

    Dette er en mer nyttig datavisualisering! Det ser ut til å indikere at den høyeste prisen for gresskar forekommer i september og oktober. Stemmer det med forventningene dine? Hvorfor eller hvorfor ikke?

---

## 🚀Utfordring

Utforsk de forskjellige typene visualiseringer som Matplotlib tilbyr. Hvilke typer er mest passende for regresjonsproblemer?

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang & Selvstudium

Ta en titt på de mange måtene å visualisere data på. Lag en liste over de ulike bibliotekene som er tilgjengelige og noter hvilke som er best for gitte typer oppgaver, for eksempel 2D-visualiseringer vs. 3D-visualiseringer. Hva oppdager du?

## Oppgave

[Utforsk visualisering](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.