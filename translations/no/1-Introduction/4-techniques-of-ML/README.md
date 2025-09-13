<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T21:41:24+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "no"
}
-->
# Teknikker for maskinlæring

Prosessen med å bygge, bruke og vedlikeholde maskinlæringsmodeller og dataene de bruker, er svært forskjellig fra mange andre utviklingsarbeidsflyter. I denne leksjonen vil vi avmystifisere prosessen og skissere de viktigste teknikkene du trenger å kjenne til. Du vil:

- Forstå prosessene som ligger til grunn for maskinlæring på et overordnet nivå.
- Utforske grunnleggende konsepter som 'modeller', 'prediksjoner' og 'treningsdata'.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

[![ML for nybegynnere - Teknikker for maskinlæring](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for nybegynnere - Teknikker for maskinlæring")

> 🎥 Klikk på bildet over for en kort video som går gjennom denne leksjonen.

## Introduksjon

På et overordnet nivå består håndverket med å lage maskinlæringsprosesser (ML) av flere steg:

1. **Bestem spørsmålet**. De fleste ML-prosesser starter med å stille et spørsmål som ikke kan besvares med et enkelt betingelsesbasert program eller regelbasert motor. Disse spørsmålene dreier seg ofte om prediksjoner basert på en samling data.
2. **Samle og forbered data**. For å kunne besvare spørsmålet ditt trenger du data. Kvaliteten og, noen ganger, mengden av dataene dine vil avgjøre hvor godt du kan besvare det opprinnelige spørsmålet. Visualisering av data er en viktig del av denne fasen. Denne fasen inkluderer også å dele dataene inn i en trenings- og testgruppe for å bygge en modell.
3. **Velg en treningsmetode**. Avhengig av spørsmålet ditt og naturen til dataene dine, må du velge hvordan du vil trene en modell for best å reflektere dataene og lage nøyaktige prediksjoner basert på dem. Dette er den delen av ML-prosessen som krever spesifikk ekspertise og ofte en betydelig mengde eksperimentering.
4. **Tren modellen**. Ved hjelp av treningsdataene dine bruker du ulike algoritmer for å trene en modell til å gjenkjenne mønstre i dataene. Modellen kan bruke interne vekter som kan justeres for å prioritere visse deler av dataene over andre for å bygge en bedre modell.
5. **Evaluer modellen**. Du bruker data som modellen aldri har sett før (testdataene dine) fra den innsamlede samlingen for å se hvordan modellen presterer.
6. **Parameterjustering**. Basert på modellens ytelse kan du gjenta prosessen med forskjellige parametere eller variabler som styrer oppførselen til algoritmene som brukes til å trene modellen.
7. **Prediksjon**. Bruk nye input for å teste modellens nøyaktighet.

## Hvilket spørsmål skal du stille?

Datamaskiner er spesielt dyktige til å oppdage skjulte mønstre i data. Denne egenskapen er svært nyttig for forskere som har spørsmål om et gitt område som ikke enkelt kan besvares ved å lage en betingelsesbasert regelmotor. Gitt en aktuariell oppgave, for eksempel, kan en dataforsker være i stand til å konstruere håndlagde regler rundt dødeligheten til røykere vs. ikke-røykere.

Når mange andre variabler tas med i ligningen, kan imidlertid en ML-modell vise seg å være mer effektiv til å forutsi fremtidige dødelighetsrater basert på tidligere helsehistorikk. Et mer oppløftende eksempel kan være å lage værprediksjoner for april måned på et gitt sted basert på data som inkluderer breddegrad, lengdegrad, klimaendringer, nærhet til havet, mønstre i jetstrømmen og mer.

✅ Denne [presentasjonen](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om værmodeller gir et historisk perspektiv på bruk av ML i væranalyse.  

## Oppgaver før bygging

Før du begynner å bygge modellen din, er det flere oppgaver du må fullføre. For å teste spørsmålet ditt og danne en hypotese basert på modellens prediksjoner, må du identifisere og konfigurere flere elementer.

### Data

For å kunne besvare spørsmålet ditt med en viss grad av sikkerhet, trenger du en god mengde data av riktig type. Det er to ting du må gjøre på dette tidspunktet:

- **Samle data**. Med tanke på den forrige leksjonen om rettferdighet i dataanalyse, samle dataene dine med omhu. Vær oppmerksom på kildene til disse dataene, eventuelle iboende skjevheter de kan ha, og dokumenter opprinnelsen.
- **Forbered data**. Det er flere steg i prosessen med databehandling. Du kan trenge å samle data og normalisere dem hvis de kommer fra ulike kilder. Du kan forbedre kvaliteten og mengden av dataene gjennom ulike metoder, som å konvertere strenger til tall (som vi gjør i [Clustering](../../5-Clustering/1-Visualize/README.md)). Du kan også generere nye data basert på de opprinnelige (som vi gjør i [Classification](../../4-Classification/1-Introduction/README.md)). Du kan rense og redigere dataene (som vi gjør før [Web App](../../3-Web-App/README.md)-leksjonen). Til slutt kan det hende du må randomisere og blande dem, avhengig av treningsmetodene dine.

✅ Etter å ha samlet og behandlet dataene dine, ta et øyeblikk for å se om formen deres vil tillate deg å adressere det tiltenkte spørsmålet. Det kan være at dataene ikke vil fungere godt for den gitte oppgaven, som vi oppdager i våre [Clustering](../../5-Clustering/1-Visualize/README.md)-leksjoner!

### Funksjoner og mål

En [funksjon](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) er en målbar egenskap ved dataene dine. I mange datasett uttrykkes det som en kolonneoverskrift som 'dato', 'størrelse' eller 'farge'. Funksjonsvariabelen din, vanligvis representert som `X` i kode, representerer inputvariabelen som vil bli brukt til å trene modellen.

Et mål er det du prøver å forutsi. Målet, vanligvis representert som `y` i kode, representerer svaret på spørsmålet du prøver å stille til dataene dine: i desember, hvilken **farge** vil gresskar være billigst? I San Francisco, hvilke nabolag vil ha de beste eiendoms**prisene**? Noen ganger refereres målet også til som etikettattributt.

### Velge funksjonsvariabelen din

🎓 **Funksjonsvalg og funksjonsekstraksjon** Hvordan vet du hvilken variabel du skal velge når du bygger en modell? Du vil sannsynligvis gå gjennom en prosess med funksjonsvalg eller funksjonsekstraksjon for å velge de riktige variablene for den mest effektive modellen. De er imidlertid ikke det samme: "Funksjonsekstraksjon lager nye funksjoner fra funksjoner av de opprinnelige funksjonene, mens funksjonsvalg returnerer et delsett av funksjonene." ([kilde](https://wikipedia.org/wiki/Feature_selection))

### Visualiser dataene dine

En viktig del av verktøysettet til en dataforsker er evnen til å visualisere data ved hjelp av flere utmerkede biblioteker som Seaborn eller MatPlotLib. Å representere dataene dine visuelt kan hjelpe deg med å avdekke skjulte korrelasjoner som du kan utnytte. Visualiseringene dine kan også hjelpe deg med å avdekke skjevheter eller ubalanserte data (som vi oppdager i [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Del opp datasettet ditt

Før trening må du dele datasettet ditt inn i to eller flere deler av ulik størrelse som fortsatt representerer dataene godt.

- **Trening**. Denne delen av datasettet tilpasses modellen din for å trene den. Dette settet utgjør majoriteten av det opprinnelige datasettet.
- **Testing**. Et testdatasett er en uavhengig gruppe data, ofte hentet fra de opprinnelige dataene, som du bruker for å bekrefte ytelsen til den bygde modellen.
- **Validering**. Et valideringssett er en mindre uavhengig gruppe eksempler som du bruker for å finjustere modellens hyperparametere eller arkitektur for å forbedre modellen. Avhengig av størrelsen på dataene dine og spørsmålet du stiller, trenger du kanskje ikke å bygge dette tredje settet (som vi bemerker i [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Bygge en modell

Ved hjelp av treningsdataene dine er målet ditt å bygge en modell, eller en statistisk representasjon av dataene dine, ved hjelp av ulike algoritmer for å **trene** den. Å trene en modell eksponerer den for data og lar den gjøre antakelser om oppdagede mønstre, validere dem og akseptere eller avvise dem.

### Bestem treningsmetoden

Avhengig av spørsmålet ditt og naturen til dataene dine, vil du velge en metode for å trene dem. Ved å gå gjennom [Scikit-learn's dokumentasjon](https://scikit-learn.org/stable/user_guide.html) - som vi bruker i dette kurset - kan du utforske mange måter å trene en modell på. Avhengig av erfaringen din, kan det hende du må prøve flere forskjellige metoder for å bygge den beste modellen. Du vil sannsynligvis gå gjennom en prosess der dataforskere evaluerer ytelsen til en modell ved å mate den med ukjente data, sjekke for nøyaktighet, skjevhet og andre kvalitetsreduserende problemer, og velge den mest passende treningsmetoden for oppgaven.

### Tren en modell

Med treningsdataene dine er du klar til å 'tilpasse' dem for å lage en modell. Du vil legge merke til at i mange ML-biblioteker finner du koden 'model.fit' - det er på dette tidspunktet du sender inn funksjonsvariabelen din som en matrise av verdier (vanligvis 'X') og en målvariabel (vanligvis 'y').

### Evaluer modellen

Når treningsprosessen er fullført (det kan ta mange iterasjoner, eller 'epoker', å trene en stor modell), vil du kunne evaluere modellens kvalitet ved å bruke testdata for å måle ytelsen. Disse dataene er et delsett av de opprinnelige dataene som modellen ikke tidligere har analysert. Du kan skrive ut en tabell med metrikker om modellens kvalitet.

🎓 **Modelltilpasning**

I sammenheng med maskinlæring refererer modelltilpasning til nøyaktigheten av modellens underliggende funksjon når den forsøker å analysere data den ikke er kjent med.

🎓 **Undertilpasning** og **overtilpasning** er vanlige problemer som reduserer modellens kvalitet, ettersom modellen enten tilpasser seg for dårlig eller for godt. Dette fører til at modellen lager prediksjoner som enten er for tett knyttet til eller for løst knyttet til treningsdataene. En overtilpasset modell forutsier treningsdataene for godt fordi den har lært detaljene og støyen i dataene for godt. En undertilpasset modell er ikke nøyaktig, da den verken kan analysere treningsdataene eller data den ikke har 'sett' på en korrekt måte.

![overtilpasset modell](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografikk av [Jen Looper](https://twitter.com/jenlooper)

## Parameterjustering

Når den første treningen er fullført, observer kvaliteten på modellen og vurder å forbedre den ved å justere dens 'hyperparametere'. Les mer om prosessen [i dokumentasjonen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksjon

Dette er øyeblikket hvor du kan bruke helt nye data for å teste modellens nøyaktighet. I en 'anvendt' ML-setting, der du bygger nettressurser for å bruke modellen i produksjon, kan denne prosessen innebære å samle brukerinput (for eksempel et knappetrykk) for å sette en variabel og sende den til modellen for inferens eller evaluering.

I disse leksjonene vil du oppdage hvordan du bruker disse stegene til å forberede, bygge, teste, evaluere og forutsi - alle oppgavene til en dataforsker og mer, mens du utvikler deg på reisen til å bli en 'full stack' ML-ingeniør.

---

## 🚀Utfordring

Lag et flytskjema som reflekterer stegene til en ML-praktiker. Hvor ser du deg selv akkurat nå i prosessen? Hvor tror du at du vil møte vanskeligheter? Hva virker enkelt for deg?

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Søk på nettet etter intervjuer med dataforskere som diskuterer sitt daglige arbeid. Her er [et](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Oppgave

[Intervju en dataforsker](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.