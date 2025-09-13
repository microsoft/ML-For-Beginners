<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T00:27:18+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "da"
}
-->
# Teknikker inden for maskinlæring

Processen med at opbygge, bruge og vedligeholde maskinlæringsmodeller og de data, de anvender, adskiller sig markant fra mange andre udviklingsarbejdsgange. I denne lektion vil vi afmystificere processen og skitsere de vigtigste teknikker, du skal kende. Du vil:

- Forstå de processer, der ligger til grund for maskinlæring på et overordnet niveau.
- Udforske grundlæggende begreber som 'modeller', 'forudsigelser' og 'træningsdata'.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

[![ML for begyndere - Teknikker inden for maskinlæring](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for begyndere - Teknikker inden for maskinlæring")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår denne lektion.

## Introduktion

På et overordnet niveau består håndværket med at skabe maskinlæringsprocesser af en række trin:

1. **Definér spørgsmålet**. De fleste ML-processer starter med at stille et spørgsmål, der ikke kan besvares med et simpelt betinget program eller en regelbaseret motor. Disse spørgsmål drejer sig ofte om forudsigelser baseret på en samling data.
2. **Indsaml og forbered data**. For at kunne besvare dit spørgsmål har du brug for data. Kvaliteten og, nogle gange, mængden af dine data vil afgøre, hvor godt du kan besvare dit oprindelige spørgsmål. Visualisering af data er en vigtig del af denne fase. Denne fase inkluderer også at opdele dataene i en trænings- og testgruppe for at opbygge en model.
3. **Vælg en træningsmetode**. Afhængigt af dit spørgsmål og karakteren af dine data skal du vælge, hvordan du vil træne en model, så den bedst afspejler dine data og giver præcise forudsigelser. Dette er den del af din ML-proces, der kræver specifik ekspertise og ofte en betydelig mængde eksperimentering.
4. **Træn modellen**. Ved hjælp af dine træningsdata vil du bruge forskellige algoritmer til at træne en model til at genkende mønstre i dataene. Modellen kan anvende interne vægte, der kan justeres for at prioritere visse dele af dataene frem for andre for at opbygge en bedre model.
5. **Evaluer modellen**. Du bruger data, som modellen aldrig har set før (dine testdata) fra din indsamlede samling for at se, hvordan modellen klarer sig.
6. **Parameterjustering**. Baseret på modellens ydeevne kan du gentage processen med forskellige parametre eller variabler, der styrer adfærden af de algoritmer, der bruges til at træne modellen.
7. **Forudsig**. Brug nye input til at teste modellens nøjagtighed.

## Hvilket spørgsmål skal du stille?

Computere er særligt dygtige til at opdage skjulte mønstre i data. Denne evne er meget nyttig for forskere, der har spørgsmål om et givet område, som ikke let kan besvares ved at oprette en betingelsesbaseret regelmotor. Givet en aktuarmæssig opgave kan en dataforsker for eksempel konstruere håndlavede regler omkring dødeligheden for rygere vs. ikke-rygere.

Når mange andre variabler bringes ind i ligningen, kan en ML-model imidlertid vise sig at være mere effektiv til at forudsige fremtidige dødelighedsrater baseret på tidligere sundhedshistorik. Et mere muntert eksempel kunne være at lave vejrudsigter for april måned på et givet sted baseret på data, der inkluderer breddegrad, længdegrad, klimaforandringer, nærhed til havet, jetstrømmønstre og mere.

✅ Denne [præsentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om vejrmodeller giver et historisk perspektiv på brugen af ML i vejranalyse.  

## Opgaver før opbygning

Før du begynder at opbygge din model, er der flere opgaver, du skal udføre. For at teste dit spørgsmål og danne en hypotese baseret på modellens forudsigelser skal du identificere og konfigurere flere elementer.

### Data

For at kunne besvare dit spørgsmål med nogen form for sikkerhed har du brug for en god mængde data af den rette type. Der er to ting, du skal gøre på dette tidspunkt:

- **Indsaml data**. Med tanke på den tidligere lektion om retfærdighed i dataanalyse skal du indsamle dine data med omhu. Vær opmærksom på kilderne til disse data, eventuelle iboende skævheder, de måtte have, og dokumentér deres oprindelse.
- **Forbered data**. Der er flere trin i dataforberedelsesprocessen. Du kan være nødt til at samle data og normalisere dem, hvis de kommer fra forskellige kilder. Du kan forbedre dataenes kvalitet og mængde gennem forskellige metoder, såsom at konvertere strenge til tal (som vi gør i [Clustering](../../5-Clustering/1-Visualize/README.md)). Du kan også generere nye data baseret på de oprindelige (som vi gør i [Classification](../../4-Classification/1-Introduction/README.md)). Du kan rense og redigere dataene (som vi gør før [Web App](../../3-Web-App/README.md)-lektionen). Endelig kan du også være nødt til at tilfældiggøre og blande dem, afhængigt af dine træningsteknikker.

✅ Efter at have indsamlet og behandlet dine data, tag et øjeblik til at se, om deres form vil tillade dig at adressere dit tilsigtede spørgsmål. Det kan være, at dataene ikke vil fungere godt i din givne opgave, som vi opdager i vores [Clustering](../../5-Clustering/1-Visualize/README.md)-lektioner!

### Features og mål

En [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) er en målbar egenskab ved dine data. I mange datasæt udtrykkes det som en kolonneoverskrift som 'dato', 'størrelse' eller 'farve'. Din feature-variabel, normalt repræsenteret som `X` i kode, repræsenterer inputvariablen, der vil blive brugt til at træne modellen.

Et mål er det, du forsøger at forudsige. Mål, normalt repræsenteret som `y` i kode, repræsenterer svaret på det spørgsmål, du forsøger at stille til dine data: i december, hvilken **farve** græskar vil være billigst? i San Francisco, hvilke kvarterer vil have de bedste ejendoms**priser**? Nogle gange omtales mål også som label-attribut.

### Valg af din feature-variabel

🎓 **Feature Selection og Feature Extraction** Hvordan ved du, hvilken variabel du skal vælge, når du opbygger en model? Du vil sandsynligvis gennemgå en proces med feature selection eller feature extraction for at vælge de rigtige variabler til den mest præstationsdygtige model. De er dog ikke det samme: "Feature extraction skaber nye features fra funktioner af de oprindelige features, mens feature selection returnerer et undersæt af features." ([kilde](https://wikipedia.org/wiki/Feature_selection))

### Visualiser dine data

En vigtig del af dataforskerens værktøjskasse er evnen til at visualisere data ved hjælp af flere fremragende biblioteker som Seaborn eller MatPlotLib. At repræsentere dine data visuelt kan give dig mulighed for at opdage skjulte korrelationer, som du kan udnytte. Dine visualiseringer kan også hjælpe dig med at opdage skævheder eller ubalancerede data (som vi opdager i [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Opdel dit datasæt

Før træning skal du opdele dit datasæt i to eller flere dele af ulige størrelse, der stadig repræsenterer dataene godt.

- **Træning**. Denne del af datasættet bruges til at træne din model. Dette sæt udgør størstedelen af det oprindelige datasæt.
- **Test**. Et testdatasæt er en uafhængig gruppe af data, ofte hentet fra de oprindelige data, som du bruger til at bekræfte ydeevnen af den opbyggede model.
- **Validering**. Et valideringssæt er en mindre uafhængig gruppe af eksempler, som du bruger til at finjustere modellens hyperparametre eller arkitektur for at forbedre modellen. Afhængigt af størrelsen på dine data og det spørgsmål, du stiller, behøver du måske ikke at opbygge dette tredje sæt (som vi bemærker i [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Opbygning af en model

Ved hjælp af dine træningsdata er dit mål at opbygge en model, eller en statistisk repræsentation af dine data, ved hjælp af forskellige algoritmer til at **træne** den. At træne en model udsætter den for data og giver den mulighed for at lave antagelser om opfattede mønstre, den opdager, validerer og accepterer eller afviser.

### Vælg en træningsmetode

Afhængigt af dit spørgsmål og karakteren af dine data vil du vælge en metode til at træne dem. Ved at gennemgå [Scikit-learns dokumentation](https://scikit-learn.org/stable/user_guide.html) - som vi bruger i dette kursus - kan du udforske mange måder at træne en model på. Afhængigt af din erfaring kan du være nødt til at prøve flere forskellige metoder for at opbygge den bedste model. Du vil sandsynligvis gennemgå en proces, hvor dataforskere evaluerer modellens ydeevne ved at fodre den med usete data, kontrollere for nøjagtighed, skævheder og andre kvalitetsforringende problemer og vælge den mest passende træningsmetode til den aktuelle opgave.

### Træn en model

Med dine træningsdata er du klar til at 'fitte' dem for at skabe en model. Du vil bemærke, at i mange ML-biblioteker vil du finde koden 'model.fit' - det er på dette tidspunkt, du sender din feature-variabel som en række værdier (normalt 'X') og en målvariabel (normalt 'y').

### Evaluer modellen

Når træningsprocessen er afsluttet (det kan tage mange iterationer, eller 'epochs', at træne en stor model), vil du kunne evaluere modellens kvalitet ved at bruge testdata til at vurdere dens ydeevne. Disse data er et undersæt af de oprindelige data, som modellen ikke tidligere har analyseret. Du kan udskrive en tabel med metrics om modellens kvalitet.

🎓 **Model fitting**

I maskinlæringskontekst refererer model fitting til modellens nøjagtighed i forhold til dens underliggende funktion, når den forsøger at analysere data, den ikke er bekendt med.

🎓 **Underfitting** og **overfitting** er almindelige problemer, der forringer modellens kvalitet, da modellen enten passer ikke godt nok eller for godt. Dette får modellen til at lave forudsigelser, der enten er for tæt på eller for løst forbundet med dens træningsdata. En overfit model forudsiger træningsdata for godt, fordi den har lært dataenes detaljer og støj for godt. En underfit model er ikke nøjagtig, da den hverken kan analysere sine træningsdata eller data, den endnu ikke har 'set', korrekt.

![overfitting model](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografik af [Jen Looper](https://twitter.com/jenlooper)

## Parameterjustering

Når din indledende træning er afsluttet, observer modellens kvalitet og overvej at forbedre den ved at justere dens 'hyperparametre'. Læs mere om processen [i dokumentationen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Forudsigelse

Dette er øjeblikket, hvor du kan bruge helt nye data til at teste modellens nøjagtighed. I en 'anvendt' ML-indstilling, hvor du opbygger webressourcer til at bruge modellen i produktion, kan denne proces involvere indsamling af brugerinput (et knaptryk, for eksempel) for at indstille en variabel og sende den til modellen for inferens eller evaluering.

I disse lektioner vil du opdage, hvordan du bruger disse trin til at forberede, opbygge, teste, evaluere og forudsige - alle dataforskerens bevægelser og mere, mens du skrider frem i din rejse mod at blive en 'full stack' ML-ingeniør.

---

## 🚀Udfordring

Tegn et flowdiagram, der afspejler trinnene for en ML-praktiker. Hvor ser du dig selv lige nu i processen? Hvor forudser du, at du vil finde vanskeligheder? Hvad virker nemt for dig?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Søg online efter interviews med dataforskere, der diskuterer deres daglige arbejde. Her er [et](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Opgave

[Interview en dataforsker](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.