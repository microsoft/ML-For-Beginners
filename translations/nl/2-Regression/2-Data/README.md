<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T18:54:08+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "nl"
}
-->
# Bouw een regressiemodel met Scikit-learn: data voorbereiden en visualiseren

![Infographic over datavisualisatie](../../../../2-Regression/2-Data/images/data-visualization.png)

Infographic door [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

> ### [Deze les is beschikbaar in R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introductie

Nu je beschikt over de tools die je nodig hebt om te beginnen met het bouwen van machine learning-modellen met Scikit-learn, ben je klaar om vragen te stellen over je data. Het is erg belangrijk om te leren hoe je de juiste vragen stelt om de mogelijkheden van je dataset optimaal te benutten.

In deze les leer je:

- Hoe je je data voorbereidt voor het bouwen van modellen.
- Hoe je Matplotlib gebruikt voor datavisualisatie.

## De juiste vraag stellen over je data

De vraag die je wilt beantwoorden, bepaalt welk type ML-algoritme je zult gebruiken. De kwaliteit van het antwoord dat je terugkrijgt, hangt sterk af van de aard van je data.

Bekijk de [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) die voor deze les is verstrekt. Je kunt dit .csv-bestand openen in VS Code. Een snelle blik laat meteen zien dat er lege velden zijn en een mix van tekst en numerieke data. Er is ook een vreemde kolom genaamd 'Package' waarin de data varieert tussen 'sacks', 'bins' en andere waarden. De data is eigenlijk een beetje rommelig.

[![ML voor beginners - Hoe een dataset analyseren en opschonen](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML voor beginners - Hoe een dataset analyseren en opschonen")

> 🎥 Klik op de afbeelding hierboven voor een korte video over het voorbereiden van de data voor deze les.

Het is eigenlijk niet gebruikelijk om een dataset te krijgen die volledig klaar is om direct een ML-model mee te maken. In deze les leer je hoe je een ruwe dataset voorbereidt met standaard Python-bibliotheken. Je leert ook verschillende technieken om de data te visualiseren.

## Casestudy: 'de pompoenmarkt'

In deze map vind je een .csv-bestand in de root `data`-map genaamd [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), dat 1757 regels data bevat over de pompoenmarkt, gegroepeerd per stad. Dit is ruwe data afkomstig van de [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) van het Amerikaanse ministerie van Landbouw.

### Data voorbereiden

Deze data is openbaar beschikbaar. Het kan worden gedownload in veel afzonderlijke bestanden, per stad, van de USDA-website. Om te voorkomen dat er te veel afzonderlijke bestanden zijn, hebben we alle stadsdata samengevoegd in één spreadsheet. We hebben de data dus al een beetje _voorbereid_. Laten we nu eens beter naar de data kijken.

### De pompoendata - eerste conclusies

Wat valt je op aan deze data? Je hebt al gezien dat er een mix is van tekst, cijfers, lege velden en vreemde waarden die je moet interpreteren.

Welke vraag kun je stellen over deze data, met behulp van een regressietechniek? Wat dacht je van "Voorspel de prijs van een pompoen die te koop is in een bepaalde maand". Als je opnieuw naar de data kijkt, zijn er enkele wijzigingen die je moet aanbrengen om de datastructuur te creëren die nodig is voor deze taak.

## Oefening - analyseer de pompoendata

Laten we [Pandas](https://pandas.pydata.org/) gebruiken (de naam staat voor `Python Data Analysis`), een zeer handige tool voor het vormgeven van data, om deze pompoendata te analyseren en voor te bereiden.

### Eerst, controleer op ontbrekende datums

Je moet eerst stappen ondernemen om te controleren op ontbrekende datums:

1. Converteer de datums naar een maandformaat (dit zijn Amerikaanse datums, dus het formaat is `MM/DD/YYYY`).
2. Haal de maand uit de datum en plaats deze in een nieuwe kolom.

Open het _notebook.ipynb_-bestand in Visual Studio Code en importeer de spreadsheet in een nieuwe Pandas-dataframe.

1. Gebruik de functie `head()` om de eerste vijf rijen te bekijken.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Welke functie zou je gebruiken om de laatste vijf rijen te bekijken?

1. Controleer of er ontbrekende data is in de huidige dataframe:

    ```python
    pumpkins.isnull().sum()
    ```

    Er is ontbrekende data, maar misschien maakt dat niet uit voor de taak.

1. Om je dataframe gemakkelijker te maken om mee te werken, selecteer je alleen de kolommen die je nodig hebt, met behulp van de `loc`-functie. Deze functie haalt een groep rijen (eerste parameter) en kolommen (tweede parameter) uit de originele dataframe. De uitdrukking `:` hieronder betekent "alle rijen".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Vervolgens, bepaal de gemiddelde prijs van een pompoen

Denk na over hoe je de gemiddelde prijs van een pompoen in een bepaalde maand kunt bepalen. Welke kolommen zou je kiezen voor deze taak? Hint: je hebt 3 kolommen nodig.

Oplossing: neem het gemiddelde van de kolommen `Low Price` en `High Price` om de nieuwe kolom Price te vullen, en converteer de kolom Date zodat deze alleen de maand toont. Gelukkig is er volgens de bovenstaande controle geen ontbrekende data voor datums of prijzen.

1. Om het gemiddelde te berekenen, voeg je de volgende code toe:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Voel je vrij om data te printen die je wilt controleren met `print(month)`.

2. Kopieer nu je geconverteerde data naar een nieuwe Pandas-dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Als je je dataframe print, zie je een schone, overzichtelijke dataset waarop je je nieuwe regressiemodel kunt bouwen.

### Maar wacht! Er is iets vreemds hier

Als je naar de kolom `Package` kijkt, zie je dat pompoenen in veel verschillende configuraties worden verkocht. Sommige worden verkocht in '1 1/9 bushel'-maten, andere in '1/2 bushel'-maten, sommige per pompoen, sommige per pond, en sommige in grote dozen met verschillende breedtes.

> Pompoenen lijken erg moeilijk consistent te wegen

Als je in de originele data duikt, is het interessant dat alles met `Unit of Sale` gelijk aan 'EACH' of 'PER BIN' ook een `Package`-type heeft per inch, per bin, of 'each'. Pompoenen lijken erg moeilijk consistent te wegen, dus laten we ze filteren door alleen pompoenen te selecteren met de string 'bushel' in hun `Package`-kolom.

1. Voeg een filter toe bovenaan het bestand, onder de initiële .csv-import:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Als je de data nu print, zie je dat je alleen de ongeveer 415 rijen data krijgt die pompoenen per bushel bevatten.

### Maar wacht! Er is nog iets dat je moet doen

Heb je gemerkt dat de bushelhoeveelheid varieert per rij? Je moet de prijzen normaliseren zodat je de prijs per bushel toont. Doe wat wiskunde om dit te standaardiseren.

1. Voeg deze regels toe na het blok dat de nieuwe_pumpkins dataframe maakt:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Volgens [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) hangt het gewicht van een bushel af van het type product, omdat het een volumemeting is. "Een bushel tomaten, bijvoorbeeld, zou 56 pond moeten wegen... Bladeren en groenten nemen meer ruimte in met minder gewicht, dus een bushel spinazie weegt slechts 20 pond." Het is allemaal behoorlijk ingewikkeld! Laten we ons niet druk maken over het maken van een bushel-naar-pond-conversie, en in plaats daarvan prijzen per bushel. Al deze studie van bushels pompoenen laat echter zien hoe belangrijk het is om de aard van je data te begrijpen!

Nu kun je de prijs per eenheid analyseren op basis van hun bushelmeting. Als je de data nog een keer print, zie je hoe het is gestandaardiseerd.

✅ Heb je gemerkt dat pompoenen die per halve bushel worden verkocht erg duur zijn? Kun je achterhalen waarom? Hint: kleine pompoenen zijn veel duurder dan grote, waarschijnlijk omdat er veel meer van zijn per bushel, gezien de ongebruikte ruimte die wordt ingenomen door één grote holle taartpompoen.

## Visualisatiestrategieën

Een deel van de rol van een datawetenschapper is om de kwaliteit en aard van de data waarmee ze werken te demonstreren. Om dit te doen, maken ze vaak interessante visualisaties, zoals grafieken, diagrammen en plots, die verschillende aspecten van de data laten zien. Op deze manier kunnen ze visueel relaties en hiaten tonen die anders moeilijk te ontdekken zijn.

[![ML voor beginners - Hoe data visualiseren met Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML voor beginners - Hoe data visualiseren met Matplotlib")

> 🎥 Klik op de afbeelding hierboven voor een korte video over het visualiseren van de data voor deze les.

Visualisaties kunnen ook helpen bij het bepalen van de meest geschikte machine learning-techniek voor de data. Een scatterplot die een lijn lijkt te volgen, geeft bijvoorbeeld aan dat de data een goede kandidaat is voor een lineaire regressieoefening.

Een datavisualisatiebibliotheek die goed werkt in Jupyter-notebooks is [Matplotlib](https://matplotlib.org/) (die je ook in de vorige les hebt gezien).

> Krijg meer ervaring met datavisualisatie in [deze tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Oefening - experimenteren met Matplotlib

Probeer enkele basisplots te maken om de nieuwe dataframe die je net hebt gemaakt weer te geven. Wat zou een eenvoudige lijnplot laten zien?

1. Importeer Matplotlib bovenaan het bestand, onder de Pandas-import:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Voer het hele notebook opnieuw uit om te verversen.
1. Voeg onderaan het notebook een cel toe om de data als een boxplot weer te geven:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Een scatterplot die de relatie tussen prijs en maand toont](../../../../2-Regression/2-Data/images/scatterplot.png)

    Is dit een nuttige plot? Verrast iets je eraan?

    Het is niet bijzonder nuttig, omdat het alleen je data als een spreiding van punten in een bepaalde maand weergeeft.

### Maak het nuttig

Om grafieken nuttige data te laten weergeven, moet je de data meestal op een bepaalde manier groeperen. Laten we proberen een plot te maken waarbij de y-as de maanden toont en de data de verdeling van de data laat zien.

1. Voeg een cel toe om een gegroepeerde staafdiagram te maken:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Een staafdiagram die de relatie tussen prijs en maand toont](../../../../2-Regression/2-Data/images/barchart.png)

    Dit is een nuttigere datavisualisatie! Het lijkt erop dat de hoogste prijs voor pompoenen in september en oktober voorkomt. Voldoet dat aan je verwachting? Waarom wel of niet?

---

## 🚀Uitdaging

Verken de verschillende soorten visualisaties die Matplotlib biedt. Welke soorten zijn het meest geschikt voor regressieproblemen?

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Bekijk de vele manieren om data te visualiseren. Maak een lijst van de verschillende beschikbare bibliotheken en noteer welke het beste zijn voor bepaalde soorten taken, bijvoorbeeld 2D-visualisaties versus 3D-visualisaties. Wat ontdek je?

## Opdracht

[Visualisatie verkennen](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.