<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T21:16:02+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "sv"
}
-->
# Bygg en regressionsmodell med Scikit-learn: förbered och visualisera data

![Infografik för datavisualisering](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografik av [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Den här lektionen finns tillgänglig i R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introduktion

Nu när du har verktygen du behöver för att börja bygga maskininlärningsmodeller med Scikit-learn, är du redo att börja ställa frågor till din data. När du arbetar med data och tillämpar ML-lösningar är det mycket viktigt att förstå hur man ställer rätt frågor för att verkligen utnyttja potentialen i din dataset.

I den här lektionen kommer du att lära dig:

- Hur du förbereder din data för modellbyggande.
- Hur du använder Matplotlib för datavisualisering.

## Ställa rätt frågor till din data

Frågan du vill ha svar på avgör vilken typ av ML-algoritmer du kommer att använda. Och kvaliteten på svaret du får tillbaka beror starkt på datans natur.

Ta en titt på [datan](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) som tillhandahålls för den här lektionen. Du kan öppna denna .csv-fil i VS Code. En snabb överblick visar direkt att det finns tomma fält och en blandning av strängar och numerisk data. Det finns också en märklig kolumn som heter 'Package' där datan är en blandning av 'sacks', 'bins' och andra värden. Datan är faktiskt lite rörig.

[![ML för nybörjare - Hur man analyserar och rensar en dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML för nybörjare - Hur man analyserar och rensar en dataset")

> 🎥 Klicka på bilden ovan för en kort video om hur man förbereder datan för den här lektionen.

Det är faktiskt inte särskilt vanligt att få en dataset som är helt redo att användas för att skapa en ML-modell direkt. I den här lektionen kommer du att lära dig hur man förbereder en rå dataset med hjälp av standardbibliotek i Python. Du kommer också att lära dig olika tekniker för att visualisera data.

## Fallstudie: 'pumpamarknaden'

I den här mappen hittar du en .csv-fil i rotmappen `data` som heter [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) som innehåller 1757 rader med data om marknaden för pumpor, sorterade i grupperingar efter stad. Detta är rådata som hämtats från [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) som distribueras av United States Department of Agriculture.

### Förbereda data

Denna data är offentlig. Den kan laddas ner i många separata filer, per stad, från USDA:s webbplats. För att undvika för många separata filer har vi sammanfogat all stadsdata till ett kalkylblad, så vi har redan _förberett_ datan lite. Låt oss nu ta en närmare titt på datan.

### Pumpadatan - tidiga slutsatser

Vad märker du om denna data? Du såg redan att det finns en blandning av strängar, siffror, tomma fält och märkliga värden som du behöver förstå.

Vilken fråga kan du ställa till denna data med hjälp av en regressionsmetod? Vad sägs om "Förutsäg priset på en pumpa som säljs under en viss månad". När du tittar på datan igen, finns det några ändringar du behöver göra för att skapa den datastruktur som krävs för uppgiften.

## Övning - analysera pumpadatan

Låt oss använda [Pandas](https://pandas.pydata.org/) (namnet står för `Python Data Analysis`), ett verktyg som är mycket användbart för att forma data, för att analysera och förbereda denna pumpadata.

### Först, kontrollera om det saknas datum

Du måste först ta steg för att kontrollera om det saknas datum:

1. Konvertera datumen till ett månadsformat (dessa är amerikanska datum, så formatet är `MM/DD/YYYY`).
2. Extrahera månaden till en ny kolumn.

Öppna _notebook.ipynb_-filen i Visual Studio Code och importera kalkylbladet till en ny Pandas-dataram.

1. Använd funktionen `head()` för att visa de första fem raderna.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Vilken funktion skulle du använda för att visa de sista fem raderna?

1. Kontrollera om det finns saknad data i den aktuella dataramen:

    ```python
    pumpkins.isnull().sum()
    ```

    Det finns saknad data, men kanske spelar det ingen roll för den aktuella uppgiften.

1. För att göra din dataram enklare att arbeta med, välj endast de kolumner du behöver med hjälp av funktionen `loc`, som extraherar en grupp rader (angivna som första parameter) och kolumner (angivna som andra parameter) från den ursprungliga dataramen. Uttrycket `:` i fallet nedan betyder "alla rader".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### För det andra, bestäm genomsnittligt pris på pumpor

Fundera på hur du kan bestämma det genomsnittliga priset på en pumpa under en viss månad. Vilka kolumner skulle du välja för denna uppgift? Tips: du behöver 3 kolumner.

Lösning: ta genomsnittet av kolumnerna `Low Price` och `High Price` för att fylla den nya kolumnen Price, och konvertera kolumnen Date till att endast visa månaden. Lyckligtvis, enligt kontrollen ovan, finns det ingen saknad data för datum eller priser.

1. För att beräkna genomsnittet, lägg till följande kod:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Känn dig fri att skriva ut vilken data du vill kontrollera med hjälp av `print(month)`.

2. Kopiera nu din konverterade data till en ny Pandas-dataram:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Om du skriver ut din dataram kommer du att se en ren och snygg dataset som du kan använda för att bygga din nya regressionsmodell.

### Men vänta! Det är något konstigt här

Om du tittar på kolumnen `Package`, säljs pumpor i många olika konfigurationer. Vissa säljs i mått som '1 1/9 bushel', andra i '1/2 bushel', vissa per pumpa, vissa per pound, och vissa i stora lådor med varierande bredd.

> Pumpor verkar vara väldigt svåra att väga konsekvent

När man gräver i den ursprungliga datan är det intressant att allt med `Unit of Sale` som är lika med 'EACH' eller 'PER BIN' också har `Package`-typen per tum, per bin eller 'each'. Pumpor verkar vara väldigt svåra att väga konsekvent, så låt oss filtrera dem genom att välja endast pumpor med strängen 'bushel' i deras `Package`-kolumn.

1. Lägg till ett filter högst upp i filen, under den initiala .csv-importen:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Om du skriver ut datan nu kan du se att du endast får de cirka 415 rader med data som innehåller pumpor per bushel.

### Men vänta! Det är en sak till att göra

Märkte du att bushel-mängden varierar per rad? Du behöver normalisera prissättningen så att du visar prissättningen per bushel, så gör lite matematik för att standardisera det.

1. Lägg till dessa rader efter blocket som skapar dataramen new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Enligt [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) beror en bushels vikt på typen av produkt, eftersom det är ett volymmått. "En bushel tomater, till exempel, ska väga 56 pounds... Blad och gröna tar upp mer plats med mindre vikt, så en bushel spenat är bara 20 pounds." Det är allt ganska komplicerat! Låt oss inte bry oss om att göra en bushel-till-pound-konvertering, och istället prissätta per bushel. All denna studie av bushels av pumpor visar dock hur viktigt det är att förstå naturen av din data!

Nu kan du analysera prissättningen per enhet baserat på deras bushel-mått. Om du skriver ut datan en gång till kan du se hur den är standardiserad.

✅ Märkte du att pumpor som säljs per halv-bushel är väldigt dyra? Kan du lista ut varför? Tips: små pumpor är mycket dyrare än stora, förmodligen eftersom det finns så många fler av dem per bushel, med tanke på det oanvända utrymmet som tas upp av en stor ihålig pajpumpa.

## Visualiseringsstrategier

En del av en dataspecialists roll är att demonstrera kvaliteten och naturen av datan de arbetar med. För att göra detta skapar de ofta intressanta visualiseringar, som diagram, grafer och tabeller, som visar olika aspekter av datan. På så sätt kan de visuellt visa relationer och luckor som annars är svåra att upptäcka.

[![ML för nybörjare - Hur man visualiserar data med Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML för nybörjare - Hur man visualiserar data med Matplotlib")

> 🎥 Klicka på bilden ovan för en kort video om hur man visualiserar datan för den här lektionen.

Visualiseringar kan också hjälpa till att avgöra vilken maskininlärningsteknik som är mest lämplig för datan. Ett spridningsdiagram som verkar följa en linje, till exempel, indikerar att datan är en bra kandidat för en linjär regressionsövning.

Ett datavisualiseringsbibliotek som fungerar bra i Jupyter notebooks är [Matplotlib](https://matplotlib.org/) (som du också såg i den föregående lektionen).

> Få mer erfarenhet av datavisualisering i [dessa tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Övning - experimentera med Matplotlib

Försök att skapa några grundläggande diagram för att visa den nya dataramen du just skapade. Vad skulle ett grundläggande linjediagram visa?

1. Importera Matplotlib högst upp i filen, under Pandas-importen:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Kör om hela notebooken för att uppdatera.
1. Lägg till en cell längst ner i notebooken för att plotta datan som en box:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Ett spridningsdiagram som visar pris-till-månad-relationen](../../../../2-Regression/2-Data/images/scatterplot.png)

    Är detta ett användbart diagram? Överraskar något dig?

    Det är inte särskilt användbart eftersom allt det gör är att visa din data som en spridning av punkter under en viss månad.

### Gör det användbart

För att få diagram att visa användbar data behöver du vanligtvis gruppera datan på något sätt. Låt oss försöka skapa ett diagram där y-axeln visar månaderna och datan demonstrerar fördelningen av data.

1. Lägg till en cell för att skapa ett grupperat stapeldiagram:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Ett stapeldiagram som visar pris-till-månad-relationen](../../../../2-Regression/2-Data/images/barchart.png)

    Detta är en mer användbar datavisualisering! Det verkar indikera att det högsta priset för pumpor inträffar i september och oktober. Motsvarar det dina förväntningar? Varför eller varför inte?

---

## 🚀Utmaning

Utforska de olika typerna av visualiseringar som Matplotlib erbjuder. Vilka typer är mest lämpliga för regressionsproblem?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Ta en titt på de många sätten att visualisera data. Gör en lista över de olika biblioteken som finns tillgängliga och notera vilka som är bäst för olika typer av uppgifter, till exempel 2D-visualiseringar kontra 3D-visualiseringar. Vad upptäcker du?

## Uppgift

[Utforska visualisering](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.