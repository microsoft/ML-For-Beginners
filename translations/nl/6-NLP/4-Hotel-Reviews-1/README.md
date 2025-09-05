<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T20:28:23+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "nl"
}
-->
# Sentimentanalyse met hotelbeoordelingen - gegevens verwerken

In deze sectie gebruik je de technieken uit de vorige lessen om een verkennende gegevensanalyse uit te voeren op een grote dataset. Zodra je een goed begrip hebt van de bruikbaarheid van de verschillende kolommen, leer je:

- hoe je onnodige kolommen kunt verwijderen
- hoe je nieuwe gegevens kunt berekenen op basis van bestaande kolommen
- hoe je de resulterende dataset kunt opslaan voor gebruik in de laatste uitdaging

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introductie

Tot nu toe heb je geleerd dat tekstgegevens heel anders zijn dan numerieke gegevens. Als het tekst is die door een mens is geschreven of gesproken, kan het worden geanalyseerd om patronen en frequenties, sentiment en betekenis te vinden. Deze les introduceert je in een echte dataset met een echte uitdaging: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, inclusief een [CC0: Public Domain-licentie](https://creativecommons.org/publicdomain/zero/1.0/). De gegevens zijn afkomstig van Booking.com en verzameld uit openbare bronnen. De maker van de dataset is Jiashen Liu.

### Voorbereiding

Wat je nodig hebt:

* De mogelijkheid om .ipynb-notebooks uit te voeren met Python 3
* pandas
* NLTK, [dat je lokaal moet installeren](https://www.nltk.org/install.html)
* De dataset, beschikbaar op Kaggle: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Het bestand is ongeveer 230 MB uitgepakt. Download het naar de root `/data` map die bij deze NLP-lessen hoort.

## Verkennende gegevensanalyse

Deze uitdaging gaat ervan uit dat je een hotelaanbevelingsbot bouwt met behulp van sentimentanalyse en gastbeoordelingsscores. De dataset die je gebruikt bevat beoordelingen van 1493 verschillende hotels in 6 steden.

Met Python, een dataset van hotelbeoordelingen en de sentimentanalyse van NLTK kun je ontdekken:

* Wat zijn de meest gebruikte woorden en zinnen in beoordelingen?
* Correleren de officiële *tags* die een hotel beschrijven met beoordelingsscores (bijvoorbeeld: zijn er meer negatieve beoordelingen voor een bepaald hotel van *Gezin met jonge kinderen* dan van *Solo reiziger*, wat misschien aangeeft dat het beter geschikt is voor *Solo reizigers*)?
* Stemmen de sentiment scores van NLTK overeen met de numerieke score van de hotelbeoordelaar?

#### Dataset

Laten we de dataset verkennen die je hebt gedownload en lokaal hebt opgeslagen. Open het bestand in een editor zoals VS Code of zelfs Excel.

De headers in de dataset zijn als volgt:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hier zijn ze gegroepeerd op een manier die gemakkelijker te onderzoeken is:  
##### Hotelkolommen

* `Hotel_Name`, `Hotel_Address`, `lat` (breedtegraad), `lng` (lengtegraad)
  * Met *lat* en *lng* kun je een kaart plotten met Python die de hotel locaties toont (misschien kleurgecodeerd voor negatieve en positieve beoordelingen)
  * Hotel_Address is niet direct nuttig voor ons, en we zullen dat waarschijnlijk vervangen door een land voor gemakkelijker sorteren en zoeken

**Hotel Meta-review kolommen**

* `Average_Score`
  * Volgens de datasetmaker is deze kolom de *Gemiddelde score van het hotel, berekend op basis van de laatste opmerking in het afgelopen jaar*. Dit lijkt een ongebruikelijke manier om de score te berekenen, maar het zijn de verzamelde gegevens, dus we nemen het voorlopig voor waar aan.

  ✅ Kun je op basis van de andere kolommen in deze dataset een andere manier bedenken om de gemiddelde score te berekenen?

* `Total_Number_of_Reviews`
  * Het totale aantal beoordelingen dat dit hotel heeft ontvangen - het is niet duidelijk (zonder code te schrijven) of dit verwijst naar de beoordelingen in de dataset.
* `Additional_Number_of_Scoring`
  * Dit betekent dat er een beoordelingsscore is gegeven, maar geen positieve of negatieve beoordeling is geschreven door de beoordelaar.

**Beoordelingskolommen**

- `Reviewer_Score`
  - Dit is een numerieke waarde met maximaal 1 decimaal tussen de minimum- en maximumwaarden 2.5 en 10
  - Het is niet uitgelegd waarom 2.5 de laagst mogelijke score is
- `Negative_Review`
  - Als een beoordelaar niets heeft geschreven, bevat dit veld "**No Negative**"
  - Merk op dat een beoordelaar een positieve beoordeling kan schrijven in de kolom Negative review (bijvoorbeeld: "er is niets slechts aan dit hotel")
- `Review_Total_Negative_Word_Counts`
  - Hogere negatieve woordenaantallen duiden op een lagere score (zonder de sentimentanalyse te controleren)
- `Positive_Review`
  - Als een beoordelaar niets heeft geschreven, bevat dit veld "**No Positive**"
  - Merk op dat een beoordelaar een negatieve beoordeling kan schrijven in de kolom Positive review (bijvoorbeeld: "er is helemaal niets goeds aan dit hotel")
- `Review_Total_Positive_Word_Counts`
  - Hogere positieve woordenaantallen duiden op een hogere score (zonder de sentimentanalyse te controleren)
- `Review_Date` en `days_since_review`
  - Een maatstaf voor versheid of veroudering kan worden toegepast op een beoordeling (oudere beoordelingen zijn mogelijk minder accuraat dan nieuwere omdat het hotelmanagement is veranderd, er renovaties zijn uitgevoerd, of er een zwembad is toegevoegd, etc.)
- `Tags`
  - Dit zijn korte beschrijvingen die een beoordelaar kan selecteren om het type gast dat ze waren te beschrijven (bijvoorbeeld solo of gezin), het type kamer dat ze hadden, de duur van het verblijf en hoe de beoordeling is ingediend.
  - Helaas is het gebruik van deze tags problematisch, zie de sectie hieronder die hun bruikbaarheid bespreekt.

**Beoordelaar kolommen**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dit kan een factor zijn in een aanbevelingsmodel, bijvoorbeeld als je kunt bepalen dat meer productieve beoordelaars met honderden beoordelingen eerder negatief dan positief zijn. Echter, de beoordelaar van een specifieke beoordeling wordt niet geïdentificeerd met een unieke code, en kan daarom niet worden gekoppeld aan een set beoordelingen. Er zijn 30 beoordelaars met 100 of meer beoordelingen, maar het is moeilijk te zien hoe dit het aanbevelingsmodel kan helpen.
- `Reviewer_Nationality`
  - Sommige mensen denken misschien dat bepaalde nationaliteiten eerder een positieve of negatieve beoordeling geven vanwege een nationale neiging. Wees voorzichtig met het opnemen van dergelijke anekdotische opvattingen in je modellen. Dit zijn nationale (en soms raciale) stereotypen, en elke beoordelaar was een individu die een beoordeling schreef op basis van hun ervaring. Dit kan zijn gefilterd door vele lenzen zoals hun eerdere hotelverblijven, de afgelegde afstand en hun persoonlijke temperament. Denken dat hun nationaliteit de reden was voor een beoordelingsscore is moeilijk te rechtvaardigen.

##### Voorbeelden

| Gemiddelde Score | Totaal aantal beoordelingen | Beoordelaar Score | Negatieve <br />Beoordeling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positieve Beoordeling             | Tags                                                                                      |
| ---------------- | --------------------------- | ----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8              | 1945                        | 2.5               | Dit is momenteel geen hotel maar een bouwplaats. Ik werd van vroeg in de ochtend tot de hele dag geterroriseerd door onacceptabel bouwlawaai terwijl ik probeerde uit te rusten na een lange reis en werken in de kamer. Mensen werkten de hele dag, bijvoorbeeld met drilboren in de aangrenzende kamers. Ik vroeg om een kamerwissel, maar er was geen stille kamer beschikbaar. Om het nog erger te maken, werd ik te veel in rekening gebracht. Ik checkte 's avonds uit omdat ik een vroege vlucht had en kreeg een passende rekening. Een dag later maakte het hotel een andere afschrijving zonder mijn toestemming bovenop de geboekte prijs. Het is een vreselijke plek. Straf jezelf niet door hier te boeken. | Niets. Vreselijke plek. Blijf weg. | Zakenreis                                Koppel Standaard Tweepersoonskamer Verbleef 2 nachten |

Zoals je kunt zien, had deze gast geen prettig verblijf in dit hotel. Het hotel heeft een goede gemiddelde score van 7.8 en 1945 beoordelingen, maar deze beoordelaar gaf het een 2.5 en schreef 115 woorden over hoe negatief hun verblijf was. Als ze niets hadden geschreven in de kolom Positive_Review, zou je kunnen concluderen dat er niets positiefs was, maar ze schreven toch 7 waarschuwende woorden. Als we alleen woorden zouden tellen in plaats van de betekenis of het sentiment van de woorden, zouden we een vertekend beeld kunnen krijgen van de intentie van de beoordelaar. Vreemd genoeg is hun score van 2.5 verwarrend, want als dat hotelverblijf zo slecht was, waarom dan überhaupt punten geven? Bij nader onderzoek van de dataset zul je zien dat de laagst mogelijke score 2.5 is, niet 0. De hoogst mogelijke score is 10.

##### Tags

Zoals hierboven vermeld, lijkt het idee om `Tags` te gebruiken om de gegevens te categoriseren in eerste instantie logisch. Helaas zijn deze tags niet gestandaardiseerd, wat betekent dat in een bepaald hotel de opties *Eenpersoonskamer*, *Tweepersoonskamer* en *Dubbele kamer* kunnen zijn, maar in het volgende hotel zijn ze *Deluxe Eenpersoonskamer*, *Classic Queen Kamer* en *Executive King Kamer*. Dit kunnen dezelfde dingen zijn, maar er zijn zoveel variaties dat de keuze wordt:

1. Proberen alle termen te veranderen naar een enkele standaard, wat erg moeilijk is, omdat het niet duidelijk is wat het conversiepad in elk geval zou zijn (bijvoorbeeld: *Classic eenpersoonskamer* wordt *Eenpersoonskamer*, maar *Superior Queen Kamer met Binnenplaats Tuin of Uitzicht op de Stad* is veel moeilijker te mappen)

1. We kunnen een NLP-aanpak nemen en de frequentie meten van bepaalde termen zoals *Solo*, *Zakelijke Reiziger* of *Gezin met jonge kinderen* zoals ze van toepassing zijn op elk hotel, en dat meenemen in de aanbeveling.

Tags zijn meestal (maar niet altijd) een enkel veld dat een lijst bevat van 5 tot 6 komma-gescheiden waarden die overeenkomen met *Type reis*, *Type gasten*, *Type kamer*, *Aantal nachten* en *Type apparaat waarop de beoordeling is ingediend*. Omdat sommige beoordelaars echter niet elk veld invullen (ze kunnen er een leeg laten), zijn de waarden niet altijd in dezelfde volgorde.

Als voorbeeld, neem *Type groep*. Er zijn 1025 unieke mogelijkheden in dit veld in de `Tags` kolom, en helaas verwijzen slechts enkele naar een groep (sommige zijn het type kamer, etc.). Als je alleen filtert op de waarden die familie vermelden, bevatten de resultaten veel *Familiekamer* type resultaten. Als je de term *met* opneemt, bijvoorbeeld de waarden *Gezin met*, zijn de resultaten beter, met meer dan 80.000 van de 515.000 resultaten die de zinsnede "Gezin met jonge kinderen" of "Gezin met oudere kinderen" bevatten.

Dit betekent dat de tags-kolom niet volledig nutteloos is, maar het zal wat werk kosten om het bruikbaar te maken.

##### Gemiddelde hotelscore

Er zijn een aantal eigenaardigheden of discrepanties in de dataset die ik niet kan verklaren, maar die hier worden geïllustreerd zodat je je ervan bewust bent bij het bouwen van je modellen. Als je het uitvindt, laat het ons weten in de discussiesectie!

De dataset heeft de volgende kolommen met betrekking tot de gemiddelde score en het aantal beoordelingen:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Het hotel met de meeste beoordelingen in deze dataset is *Britannia International Hotel Canary Wharf* met 4789 beoordelingen van de 515.000. Maar als we kijken naar de waarde `Total_Number_of_Reviews` voor dit hotel, is dat 9086. Je zou kunnen concluderen dat er veel meer scores zijn zonder beoordelingen, dus misschien moeten we de waarde in de kolom `Additional_Number_of_Scoring` toevoegen. Die waarde is 2682, en als we die optellen bij 4789 krijgen we 7471, wat nog steeds 1615 tekortschiet ten opzichte van `Total_Number_of_Reviews`.

Als je de kolom `Average_Score` neemt, zou je kunnen concluderen dat dit het gemiddelde is van de beoordelingen in de dataset, maar de beschrijving van Kaggle is "*Gemiddelde score van het hotel, berekend op basis van de laatste opmerking in het afgelopen jaar*". Dat lijkt niet zo nuttig, maar we kunnen ons eigen gemiddelde berekenen op basis van de beoordelingsscores in de dataset. Als we hetzelfde hotel als voorbeeld nemen, wordt de gemiddelde hotelscore gegeven als 7.1, maar de berekende score (gemiddelde beoordelaarsscore *in* de dataset) is 6.8. Dit is dichtbij, maar niet dezelfde waarde, en we kunnen alleen maar raden dat de scores in de `Additional_Number_of_Scoring` beoordelingen het gemiddelde verhoogden naar 7.1. Helaas, zonder een manier om die bewering te testen of te bewijzen, is het moeilijk om `Average_Score`, `Additional_Number_of_Scoring` en `Total_Number_of_Reviews` te gebruiken of te vertrouwen wanneer ze gebaseerd zijn op, of verwijzen naar, gegevens die we niet hebben.

Om het nog ingewikkelder te maken, heeft het hotel met het op één na hoogste aantal beoordelingen een berekende gemiddelde score van 8.12 en de dataset `Average_Score` is 8.1. Is deze correcte score toeval of is het eerste hotel een discrepantie?

Op basis van de mogelijkheid dat deze hotels een uitschieter zijn, en dat misschien de meeste waarden wel kloppen (maar sommige om een of andere reden niet), zullen we in de volgende stap een kort programma schrijven om de waarden in de dataset te verkennen en het juiste gebruik (of niet-gebruik) van de waarden te bepalen.
> 🚨 Een waarschuwing

> Bij het werken met deze dataset schrijf je code die iets berekent op basis van de tekst, zonder dat je de tekst zelf hoeft te lezen of te analyseren. Dit is de kern van NLP: betekenis of sentiment interpreteren zonder dat een mens dit hoeft te doen. Het is echter mogelijk dat je enkele negatieve beoordelingen zult lezen. Ik raad je aan dit niet te doen, omdat het niet nodig is. Sommige van deze beoordelingen zijn onzinnig of irrelevante negatieve hotelbeoordelingen, zoals "Het weer was niet goed", iets wat buiten de controle van het hotel, of eigenlijk van iedereen, ligt. Maar er is ook een donkere kant aan sommige beoordelingen. Soms zijn de negatieve beoordelingen racistisch, seksistisch of discriminerend op basis van leeftijd. Dit is jammer, maar te verwachten in een dataset die van een openbare website is gehaald. Sommige reviewers laten beoordelingen achter die je smakeloos, ongemakkelijk of verontrustend zou kunnen vinden. Het is beter om de code het sentiment te laten meten dan ze zelf te lezen en van streek te raken. Dat gezegd hebbende, het is een minderheid die dergelijke dingen schrijft, maar ze bestaan wel degelijk.
## Oefening - Data verkennen
### Laad de data

Dat is genoeg visueel onderzoek naar de data, nu ga je wat code schrijven en antwoorden krijgen! Dit gedeelte maakt gebruik van de pandas-bibliotheek. Je eerste taak is ervoor te zorgen dat je de CSV-data kunt laden en lezen. De pandas-bibliotheek heeft een snelle CSV-loader, en het resultaat wordt geplaatst in een dataframe, zoals in eerdere lessen. De CSV die we laden heeft meer dan een half miljoen rijen, maar slechts 17 kolommen. Pandas biedt veel krachtige manieren om met een dataframe te werken, waaronder de mogelijkheid om bewerkingen op elke rij uit te voeren.

Vanaf hier in deze les zullen er codefragmenten zijn, enkele uitleg over de code en discussies over wat de resultaten betekenen. Gebruik het meegeleverde _notebook.ipynb_ voor je code.

Laten we beginnen met het laden van het databestand dat je gaat gebruiken:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Nu de data is geladen, kunnen we enkele bewerkingen uitvoeren. Houd deze code bovenaan je programma voor het volgende deel.

## Verken de data

In dit geval is de data al *schoon*, wat betekent dat het klaar is om mee te werken en geen tekens in andere talen bevat die algoritmes, die alleen Engelse tekens verwachten, in de war kunnen brengen.

✅ Je zou kunnen werken met data die enige initiële verwerking vereist om het te formatteren voordat je NLP-technieken toepast, maar dat is deze keer niet nodig. Als je dat wel moest doen, hoe zou je omgaan met niet-Engelse tekens?

Neem even de tijd om ervoor te zorgen dat je, zodra de data is geladen, deze met code kunt verkennen. Het is heel verleidelijk om je te richten op de kolommen `Negative_Review` en `Positive_Review`. Ze zijn gevuld met natuurlijke tekst voor je NLP-algoritmes om te verwerken. Maar wacht! Voordat je begint met NLP en sentimentanalyse, moet je de onderstaande code volgen om te controleren of de waarden in de dataset overeenkomen met de waarden die je berekent met pandas.

## Dataframe-bewerkingen

De eerste taak in deze les is om te controleren of de volgende aannames correct zijn door code te schrijven die het dataframe onderzoekt (zonder het te wijzigen).

> Zoals bij veel programmeertaken zijn er meerdere manieren om dit te voltooien, maar een goed advies is om het op de eenvoudigste, gemakkelijkste manier te doen, vooral als het gemakkelijker te begrijpen is wanneer je later naar deze code terugkeert. Met dataframes is er een uitgebreide API die vaak een efficiënte manier biedt om te doen wat je wilt.

Behandel de volgende vragen als codetaken en probeer ze te beantwoorden zonder naar de oplossing te kijken.

1. Print de *shape* van het dataframe dat je zojuist hebt geladen (de shape is het aantal rijen en kolommen).
2. Bereken de frequentie van recensent-nationaliteiten:
   1. Hoeveel unieke waarden zijn er in de kolom `Reviewer_Nationality` en wat zijn ze?
   2. Welke recensent-nationaliteit komt het meest voor in de dataset (print land en aantal recensies)?
   3. Wat zijn de volgende top 10 meest voorkomende nationaliteiten en hun frequentie?
3. Wat was het meest beoordeelde hotel voor elk van de top 10 meest voorkomende recensent-nationaliteiten?
4. Hoeveel recensies zijn er per hotel (frequentie van hotel) in de dataset?
5. Hoewel er een kolom `Average_Score` is voor elk hotel in de dataset, kun je ook een gemiddelde score berekenen (het gemiddelde van alle recensent-scores in de dataset voor elk hotel). Voeg een nieuwe kolom toe aan je dataframe met de kolomkop `Calc_Average_Score` die dat berekende gemiddelde bevat.
6. Hebben hotels dezelfde (afgerond op 1 decimaal) `Average_Score` en `Calc_Average_Score`?
   1. Probeer een Python-functie te schrijven die een Series (rij) als argument neemt en de waarden vergelijkt, en een bericht print wanneer de waarden niet gelijk zijn. Gebruik vervolgens de `.apply()`-methode om elke rij met de functie te verwerken.
7. Bereken en print hoeveel rijen de waarde "No Negative" hebben in de kolom `Negative_Review`.
8. Bereken en print hoeveel rijen de waarde "No Positive" hebben in de kolom `Positive_Review`.
9. Bereken en print hoeveel rijen de waarde "No Positive" hebben in de kolom `Positive_Review` **en** de waarde "No Negative" in de kolom `Negative_Review`.

### Code-antwoorden

1. Print de *shape* van het dataframe dat je zojuist hebt geladen (de shape is het aantal rijen en kolommen).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Bereken de frequentie van recensent-nationaliteiten:

   1. Hoeveel unieke waarden zijn er in de kolom `Reviewer_Nationality` en wat zijn ze?
   2. Welke recensent-nationaliteit komt het meest voor in de dataset (print land en aantal recensies)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Wat zijn de volgende top 10 meest voorkomende nationaliteiten en hun frequentie?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Wat was het meest beoordeelde hotel voor elk van de top 10 meest voorkomende recensent-nationaliteiten?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Hoeveel recensies zijn er per hotel (frequentie van hotel) in de dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Je zult merken dat de *geteld in de dataset*-resultaten niet overeenkomen met de waarde in `Total_Number_of_Reviews`. Het is onduidelijk of deze waarde in de dataset het totale aantal recensies vertegenwoordigde dat het hotel had, maar niet allemaal werden gescraped, of een andere berekening. `Total_Number_of_Reviews` wordt niet gebruikt in het model vanwege deze onduidelijkheid.

5. Hoewel er een kolom `Average_Score` is voor elk hotel in de dataset, kun je ook een gemiddelde score berekenen (het gemiddelde van alle recensent-scores in de dataset voor elk hotel). Voeg een nieuwe kolom toe aan je dataframe met de kolomkop `Calc_Average_Score` die dat berekende gemiddelde bevat. Print de kolommen `Hotel_Name`, `Average_Score` en `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Je vraagt je misschien af over de waarde `Average_Score` en waarom deze soms verschilt van de berekende gemiddelde score. Omdat we niet kunnen weten waarom sommige waarden overeenkomen, maar andere een verschil hebben, is het in dit geval het veiligst om de recensiescores die we hebben te gebruiken om het gemiddelde zelf te berekenen. Dat gezegd hebbende, zijn de verschillen meestal erg klein. Hier zijn de hotels met de grootste afwijking tussen het datasetgemiddelde en het berekende gemiddelde:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Met slechts 1 hotel met een scoreverschil groter dan 1, betekent dit dat we het verschil waarschijnlijk kunnen negeren en de berekende gemiddelde score kunnen gebruiken.

6. Bereken en print hoeveel rijen de waarde "No Negative" hebben in de kolom `Negative_Review`.

7. Bereken en print hoeveel rijen de waarde "No Positive" hebben in de kolom `Positive_Review`.

8. Bereken en print hoeveel rijen de waarde "No Positive" hebben in de kolom `Positive_Review` **en** de waarde "No Negative" in de kolom `Negative_Review`.

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Een andere manier

Een andere manier om items te tellen zonder Lambdas, en gebruik sum om de rijen te tellen:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Je hebt misschien gemerkt dat er 127 rijen zijn die zowel "No Negative" als "No Positive" waarden hebben voor de kolommen `Negative_Review` en `Positive_Review` respectievelijk. Dat betekent dat de recensent het hotel een numerieke score heeft gegeven, maar ervoor heeft gekozen om geen positieve of negatieve recensie te schrijven. Gelukkig is dit een klein aantal rijen (127 van 515738, of 0,02%), dus het zal waarschijnlijk ons model of de resultaten niet in een bepaalde richting beïnvloeden, maar je had misschien niet verwacht dat een dataset met recensies rijen zou bevatten zonder recensies, dus het is de moeite waard om de data te verkennen om dergelijke rijen te ontdekken.

Nu je de dataset hebt verkend, ga je in de volgende les de data filteren en sentimentanalyse toevoegen.

---
## 🚀Uitdaging

Deze les laat zien, zoals we in eerdere lessen zagen, hoe cruciaal het is om je data en de eigenaardigheden ervan te begrijpen voordat je bewerkingen uitvoert. Tekstgebaseerde data, in het bijzonder, vereist zorgvuldig onderzoek. Graaf door verschillende tekstrijke datasets en kijk of je gebieden kunt ontdekken die vooringenomenheid of scheve sentimenten in een model kunnen introduceren.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Volg [dit leerpad over NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) om tools te ontdekken die je kunt proberen bij het bouwen van spraak- en tekstrijke modellen.

## Opdracht 

[NLTK](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.