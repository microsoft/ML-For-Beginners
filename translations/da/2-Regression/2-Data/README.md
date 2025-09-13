<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-04T23:41:33+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "da"
}
-->
# Byg en regressionsmodel med Scikit-learn: forbered og visualiser data

![Data visualisering infographic](../../../../2-Regression/2-Data/images/data-visualization.png)

Infographic af [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Denne lektion er også tilgængelig i R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introduktion

Nu hvor du har de nødvendige værktøjer til at begynde at bygge maskinlæringsmodeller med Scikit-learn, er du klar til at begynde at stille spørgsmål til dine data. Når du arbejder med data og anvender ML-løsninger, er det meget vigtigt at forstå, hvordan man stiller de rigtige spørgsmål for at udnytte potentialet i dit datasæt korrekt.

I denne lektion vil du lære:

- Hvordan du forbereder dine data til modelbygning.
- Hvordan du bruger Matplotlib til datavisualisering.

## Stille de rigtige spørgsmål til dine data

Det spørgsmål, du ønsker besvaret, vil afgøre, hvilken type ML-algoritmer du skal bruge. Og kvaliteten af det svar, du får, vil i høj grad afhænge af kvaliteten af dine data.

Tag et kig på [dataene](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), der er leveret til denne lektion. Du kan åbne denne .csv-fil i VS Code. En hurtig gennemgang viser straks, at der er tomme felter og en blanding af tekst og numeriske data. Der er også en mærkelig kolonne kaldet 'Package', hvor dataene er en blanding af 'sacks', 'bins' og andre værdier. Dataene er faktisk lidt rodede.

[![ML for begyndere - Hvordan analyserer og renser man et datasæt](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML for begyndere - Hvordan analyserer og renser man et datasæt")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår forberedelsen af dataene til denne lektion.

Det er faktisk ikke særlig almindeligt at få et datasæt, der er helt klar til brug til at skabe en ML-model direkte. I denne lektion vil du lære, hvordan du forbereder et råt datasæt ved hjælp af standard Python-biblioteker. Du vil også lære forskellige teknikker til at visualisere dataene.

## Case study: 'græskarmarkedet'

I denne mappe finder du en .csv-fil i roden af `data`-mappen kaldet [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), som indeholder 1757 linjer med data om markedet for græskar, sorteret i grupperinger efter by. Dette er rå data, der er hentet fra [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice), distribueret af United States Department of Agriculture.

### Forberedelse af data

Disse data er offentligt tilgængelige. De kan downloades i mange separate filer, per by, fra USDA's hjemmeside. For at undgå for mange separate filer har vi sammenføjet alle bydataene til ét regneark, så vi har allerede _forberedt_ dataene en smule. Lad os nu tage et nærmere kig på dataene.

### Græskardata - tidlige konklusioner

Hvad bemærker du ved disse data? Du har allerede set, at der er en blanding af tekst, tal, tomme felter og mærkelige værdier, som du skal finde mening i.

Hvilket spørgsmål kan du stille til disse data ved hjælp af en regressionsmetode? Hvad med "Forudsig prisen på et græskar til salg i en given måned". Når du ser på dataene igen, er der nogle ændringer, du skal foretage for at skabe den datastruktur, der er nødvendig for opgaven.

## Øvelse - analyser græskardataene

Lad os bruge [Pandas](https://pandas.pydata.org/) (navnet står for `Python Data Analysis`), et værktøj, der er meget nyttigt til at forme data, til at analysere og forberede disse græskardata.

### Først, tjek for manglende datoer

Du skal først tage skridt til at tjekke for manglende datoer:

1. Konverter datoerne til et månedsformat (disse er amerikanske datoer, så formatet er `MM/DD/YYYY`).
2. Uddrag måneden til en ny kolonne.

Åbn _notebook.ipynb_-filen i Visual Studio Code og importer regnearket til en ny Pandas dataframe.

1. Brug funktionen `head()` til at se de første fem rækker.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Hvilken funktion ville du bruge til at se de sidste fem rækker?

1. Tjek om der er manglende data i den aktuelle dataframe:

    ```python
    pumpkins.isnull().sum()
    ```

    Der er manglende data, men måske betyder det ikke noget for den aktuelle opgave.

1. For at gøre din dataframe lettere at arbejde med, vælg kun de kolonner, du har brug for, ved hjælp af funktionen `loc`, som udtrækker en gruppe af rækker (angivet som første parameter) og kolonner (angivet som anden parameter) fra den originale dataframe. Udtrykket `:` i nedenstående tilfælde betyder "alle rækker".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### For det andet, bestem gennemsnitsprisen på græskar

Tænk over, hvordan du kan bestemme gennemsnitsprisen på et græskar i en given måned. Hvilke kolonner ville du vælge til denne opgave? Hint: du skal bruge 3 kolonner.

Løsning: Tag gennemsnittet af kolonnerne `Low Price` og `High Price` for at udfylde den nye Price-kolonne, og konverter Date-kolonnen til kun at vise måneden. Heldigvis, ifølge ovenstående tjek, er der ingen manglende data for datoer eller priser.

1. For at beregne gennemsnittet, tilføj følgende kode:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Du er velkommen til at udskrive data, du gerne vil tjekke, ved hjælp af `print(month)`.

2. Kopier nu dine konverterede data til en ny Pandas dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Hvis du udskriver din dataframe, vil du se et rent, ryddeligt datasæt, som du kan bygge din nye regressionsmodel på.

### Men vent! Der er noget mærkeligt her

Hvis du ser på kolonnen `Package`, bliver græskar solgt i mange forskellige konfigurationer. Nogle bliver solgt i '1 1/9 bushel'-mål, og nogle i '1/2 bushel'-mål, nogle per græskar, nogle per pund, og nogle i store kasser med varierende bredder.

> Græskar synes at være meget svære at veje konsekvent

Når man dykker ned i de originale data, er det interessant, at alt med `Unit of Sale` lig med 'EACH' eller 'PER BIN' også har `Package`-typen per tomme, per bin eller 'each'. Græskar synes at være meget svære at veje konsekvent, så lad os filtrere dem ved kun at vælge græskar med strengen 'bushel' i deres `Package`-kolonne.

1. Tilføj et filter øverst i filen, under den oprindelige .csv-import:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Hvis du udskriver dataene nu, kan du se, at du kun får de cirka 415 rækker med data, der indeholder græskar per bushel.

### Men vent! Der er én ting mere at gøre

Bemærkede du, at bushel-mængden varierer per række? Du skal normalisere prissætningen, så du viser prisen per bushel, så lav nogle beregninger for at standardisere det.

1. Tilføj disse linjer efter blokken, der opretter new_pumpkins-dataframen:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Ifølge [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) afhænger en bushels vægt af typen af produkt, da det er en volumenmåling. "En bushel tomater, for eksempel, skal veje 56 pund... Blade og grøntsager fylder mere med mindre vægt, så en bushel spinat er kun 20 pund." Det er alt sammen ret kompliceret! Lad os ikke bekymre os om at lave en bushel-til-pund-konvertering, og i stedet prissætte per bushel. Al denne undersøgelse af bushels af græskar viser dog, hvor vigtigt det er at forstå naturen af dine data!

Nu kan du analysere prissætningen per enhed baseret på deres bushel-måling. Hvis du udskriver dataene en gang til, kan du se, hvordan det er standardiseret.

✅ Bemærkede du, at græskar solgt per halv bushel er meget dyre? Kan du finde ud af hvorfor? Hint: små græskar er meget dyrere end store, sandsynligvis fordi der er så mange flere af dem per bushel, givet den ubrugte plads, som et stort hul græskar til tærte optager.

## Visualiseringsstrategier

En del af dataforskerens rolle er at demonstrere kvaliteten og naturen af de data, de arbejder med. For at gøre dette skaber de ofte interessante visualiseringer, såsom grafer, diagrammer og plots, der viser forskellige aspekter af dataene. På denne måde kan de visuelt vise relationer og mangler, der ellers er svære at opdage.

[![ML for begyndere - Hvordan visualiserer man data med Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML for begyndere - Hvordan visualiserer man data med Matplotlib")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår visualisering af dataene til denne lektion.

Visualiseringer kan også hjælpe med at bestemme den maskinlæringsteknik, der er mest passende for dataene. Et scatterplot, der ser ud til at følge en linje, indikerer for eksempel, at dataene er en god kandidat til en lineær regressionsøvelse.

Et datavisualiseringsbibliotek, der fungerer godt i Jupyter notebooks, er [Matplotlib](https://matplotlib.org/) (som du også så i den forrige lektion).

> Få mere erfaring med datavisualisering i [disse tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Øvelse - eksperimentér med Matplotlib

Prøv at skabe nogle grundlæggende plots for at vise den nye dataframe, du lige har oprettet. Hvad ville et grundlæggende linjeplot vise?

1. Importér Matplotlib øverst i filen, under Pandas-importen:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Kør hele notebooken igen for at opdatere.
1. Nederst i notebooken, tilføj en celle for at plotte dataene som en boks:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Et scatterplot, der viser forholdet mellem pris og måned](../../../../2-Regression/2-Data/images/scatterplot.png)

    Er dette et nyttigt plot? Overrasker noget ved det dig?

    Det er ikke særlig nyttigt, da det blot viser dine data som en spredning af punkter i en given måned.

### Gør det nyttigt

For at få diagrammer til at vise nyttige data, skal du normalt gruppere dataene på en eller anden måde. Lad os prøve at skabe et plot, hvor y-aksen viser månederne, og dataene demonstrerer fordelingen af data.

1. Tilføj en celle for at skabe et grupperet søjlediagram:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Et søjlediagram, der viser forholdet mellem pris og måned](../../../../2-Regression/2-Data/images/barchart.png)

    Dette er en mere nyttig datavisualisering! Det ser ud til at indikere, at den højeste pris for græskar forekommer i september og oktober. Stemmer det overens med dine forventninger? Hvorfor eller hvorfor ikke?

---

## 🚀Udfordring

Udforsk de forskellige typer visualiseringer, som Matplotlib tilbyder. Hvilke typer er mest passende for regressionsproblemer?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Tag et kig på de mange måder at visualisere data på. Lav en liste over de forskellige biblioteker, der er tilgængelige, og notér hvilke der er bedst til bestemte typer opgaver, for eksempel 2D-visualiseringer vs. 3D-visualiseringer. Hvad opdager du?

## Opgave

[Udforsk visualisering](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.