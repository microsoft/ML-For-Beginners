<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T22:16:42+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "sv"
}
-->
# Sentimentanalys med hotellrecensioner - bearbetning av data

I den här sektionen kommer du att använda tekniker från tidigare lektioner för att utföra en utforskande dataanalys av en stor dataset. När du har fått en bra förståelse för användbarheten av de olika kolumnerna kommer du att lära dig:

- hur man tar bort onödiga kolumner
- hur man beräknar ny data baserat på befintliga kolumner
- hur man sparar den resulterande datasetet för användning i den slutliga utmaningen

## [Förtest innan lektionen](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

Hittills har du lärt dig att textdata skiljer sig mycket från numeriska datatyper. Om texten är skriven eller talad av en människa kan den analyseras för att hitta mönster, frekvenser, känslor och betydelser. Den här lektionen tar dig in i en verklig dataset med en verklig utmaning: **[515K hotellrecensioner i Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** som inkluderar en [CC0: Public Domain-licens](https://creativecommons.org/publicdomain/zero/1.0/). Datasetet har hämtats från Booking.com från offentliga källor. Skaparen av datasetet är Jiashen Liu.

### Förberedelser

Du behöver:

* Möjligheten att köra .ipynb-notebooks med Python 3
* pandas
* NLTK, [som du bör installera lokalt](https://www.nltk.org/install.html)
* Datasetet som finns tillgängligt på Kaggle [515K hotellrecensioner i Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Det är cirka 230 MB när det är uppackat. Ladda ner det till rotmappen `/data` som är kopplad till dessa NLP-lektioner.

## Utforskande dataanalys

Den här utmaningen utgår från att du bygger en hotellrekommendationsbot med hjälp av sentimentanalys och gästrecensioner. Datasetet du kommer att använda innehåller recensioner av 1493 olika hotell i 6 städer.

Med hjälp av Python, ett dataset med hotellrecensioner och NLTK:s sentimentanalys kan du ta reda på:

* Vilka är de mest frekvent använda orden och fraserna i recensionerna?
* Korrelerar de officiella *taggarna* som beskriver ett hotell med recensionsbetygen (t.ex. är de mer negativa recensionerna för ett visst hotell från *Familj med små barn* än från *Ensamresenär*, vilket kanske indikerar att det är bättre för *Ensamresenärer*)?
* Stämmer NLTK:s sentimentbetyg överens med hotellrecensentens numeriska betyg?

#### Dataset

Låt oss utforska datasetet som du har laddat ner och sparat lokalt. Öppna filen i en editor som VS Code eller till och med Excel.

Rubrikerna i datasetet är följande:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Här är de grupperade på ett sätt som kan vara lättare att granska:
##### Hotellkolumner

* `Hotel_Name`, `Hotel_Address`, `lat` (latitud), `lng` (longitud)
  * Med hjälp av *lat* och *lng* kan du plotta en karta med Python som visar hotellens platser (kanske färgkodade för negativa och positiva recensioner)
  * Hotel_Address är inte uppenbart användbar för oss, och vi kommer förmodligen att ersätta den med ett land för enklare sortering och sökning

**Meta-recensionskolumner för hotell**

* `Average_Score`
  * Enligt datasetets skapare är denna kolumn *Genomsnittsbetyget för hotellet, beräknat baserat på den senaste kommentaren under det senaste året*. Detta verkar vara ett ovanligt sätt att beräkna betyget, men eftersom det är data som hämtats kan vi för tillfället ta det för vad det är.
  
  ✅ Baserat på de andra kolumnerna i denna data, kan du tänka dig ett annat sätt att beräkna genomsnittsbetyget?

* `Total_Number_of_Reviews`
  * Det totala antalet recensioner som detta hotell har fått - det är inte klart (utan att skriva lite kod) om detta hänvisar till recensionerna i datasetet.
* `Additional_Number_of_Scoring`
  * Detta betyder att ett recensionsbetyg gavs men ingen positiv eller negativ recension skrevs av recensenten

**Recensionskolumner**

- `Reviewer_Score`
  - Detta är ett numeriskt värde med högst 1 decimal mellan min- och maxvärdena 2,5 och 10
  - Det förklaras inte varför 2,5 är det lägsta möjliga betyget
- `Negative_Review`
  - Om en recensent inte skrev något kommer detta fält att ha "**No Negative**"
  - Observera att en recensent kan skriva en positiv recension i kolumnen för negativa recensioner (t.ex. "there is nothing bad about this hotel")
- `Review_Total_Negative_Word_Counts`
  - Högre antal negativa ord indikerar ett lägre betyg (utan att kontrollera sentimentet)
- `Positive_Review`
  - Om en recensent inte skrev något kommer detta fält att ha "**No Positive**"
  - Observera att en recensent kan skriva en negativ recension i kolumnen för positiva recensioner (t.ex. "there is nothing good about this hotel at all")
- `Review_Total_Positive_Word_Counts`
  - Högre antal positiva ord indikerar ett högre betyg (utan att kontrollera sentimentet)
- `Review_Date` och `days_since_review`
  - Ett mått på färskhet eller ålder kan tillämpas på en recension (äldre recensioner kanske inte är lika exakta som nyare eftersom hotellledningen kan ha ändrats, renoveringar kan ha gjorts, en pool kan ha lagts till etc.)
- `Tags`
  - Dessa är korta beskrivningar som en recensent kan välja för att beskriva vilken typ av gäst de var (t.ex. ensam eller familj), vilken typ av rum de hade, vistelsens längd och hur recensionen skickades in.
  - Tyvärr är användningen av dessa taggar problematisk, se avsnittet nedan som diskuterar deras användbarhet

**Recensentkolumner**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Detta kan vara en faktor i en rekommendationsmodell, till exempel om du kan avgöra att mer produktiva recensenter med hundratals recensioner var mer benägna att vara negativa än positiva. Dock identifieras inte recensenten av en unik kod, och kan därför inte kopplas till en uppsättning recensioner. Det finns 30 recensenter med 100 eller fler recensioner, men det är svårt att se hur detta kan hjälpa rekommendationsmodellen.
- `Reviewer_Nationality`
  - Vissa kanske tror att vissa nationaliteter är mer benägna att ge en positiv eller negativ recension på grund av en nationell benägenhet. Var försiktig med att bygga in sådana anekdotiska uppfattningar i dina modeller. Dessa är nationella (och ibland rasliga) stereotyper, och varje recensent var en individ som skrev en recension baserad på sin upplevelse. Den kan ha filtrerats genom många linser som deras tidigare hotellvistelser, avståndet de rest och deras personliga temperament. Att tro att deras nationalitet var orsaken till ett recensionsbetyg är svårt att rättfärdiga.

##### Exempel

| Genomsnittligt betyg | Totalt antal recensioner | Recensentens betyg | Negativ <br />Recension                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positiv recension                 | Taggar                                                                                      |
| -------------------- | ------------------------ | ------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8                  | 1945                     | 2.5                | Detta är för närvarande inte ett hotell utan en byggarbetsplats Jag blev terroriserad från tidig morgon och hela dagen med oacceptabelt byggbuller medan jag vilade efter en lång resa och arbetade i rummet Folk arbetade hela dagen dvs med tryckluftsborrar i angränsande rum Jag bad om att få byta rum men inget tyst rum var tillgängligt För att göra saken värre blev jag överdebiterad Jag checkade ut på kvällen eftersom jag var tvungen att lämna mycket tidigt och fick en korrekt faktura En dag senare gjorde hotellet en annan debitering utan mitt samtycke över det bokade priset Det är ett hemskt ställe Straffa inte dig själv genom att boka här | Ingenting Hemskt ställe Håll er borta | Affärsresa                                Par Standard dubbelrum Bodde 2 nätter |

Som du kan se hade denna gäst inte en trevlig vistelse på detta hotell. Hotellet har ett bra genomsnittligt betyg på 7,8 och 1945 recensioner, men denna recensent gav det 2,5 och skrev 115 ord om hur negativ deras vistelse var. Om de inte skrev något alls i kolumnen för positiva recensioner kan man anta att det inte fanns något positivt, men de skrev trots allt 7 varningsord. Om vi bara räknade ord istället för betydelsen eller sentimentet av orden, skulle vi kunna få en skev bild av recensentens avsikt. Märkligt nog är deras betyg på 2,5 förvirrande, eftersom om hotellvistelsen var så dålig, varför ge några poäng alls? Vid närmare granskning av datasetet ser du att det lägsta möjliga betyget är 2,5, inte 0. Det högsta möjliga betyget är 10.

##### Taggar

Som nämnts ovan verkar idén att använda `Tags` för att kategorisera data vid första anblicken vettig. Tyvärr är dessa taggar inte standardiserade, vilket innebär att i ett givet hotell kan alternativen vara *Single room*, *Twin room* och *Double room*, men i nästa hotell är de *Deluxe Single Room*, *Classic Queen Room* och *Executive King Room*. Dessa kan vara samma saker, men det finns så många variationer att valet blir:

1. Försöka ändra alla termer till en enda standard, vilket är mycket svårt eftersom det inte är klart vad konverteringsvägen skulle vara i varje fall (t.ex. *Classic single room* motsvarar *Single room* men *Superior Queen Room with Courtyard Garden or City View* är mycket svårare att mappa)

1. Vi kan ta en NLP-ansats och mäta frekvensen av vissa termer som *Solo*, *Business Traveller* eller *Family with young kids* när de gäller varje hotell och inkludera det i rekommendationen  

Taggar är vanligtvis (men inte alltid) ett enda fält som innehåller en lista med 5 till 6 kommaseparerade värden som motsvarar *Typ av resa*, *Typ av gäster*, *Typ av rum*, *Antal nätter* och *Typ av enhet som recensionen skickades in från*. Eftersom vissa recensenter inte fyller i varje fält (de kan lämna ett tomt), är värdena dock inte alltid i samma ordning.

Som exempel, ta *Typ av grupp*. Det finns 1025 unika möjligheter i detta fält i kolumnen `Tags`, och tyvärr hänvisar endast några av dem till en grupp (vissa är typen av rum etc.). Om du filtrerar endast de som nämner familj, innehåller resultaten många *Family room*-typer av resultat. Om du inkluderar termen *with*, dvs. räknar *Family with*-värdena, blir resultaten bättre, med över 80 000 av de 515 000 resultaten som innehåller frasen "Family with young children" eller "Family with older children".

Detta innebär att kolumnen taggar inte är helt värdelös för oss, men det kommer att krävas lite arbete för att göra den användbar.

##### Genomsnittligt hotellbetyg

Det finns ett antal märkligheter eller avvikelser i datasetet som jag inte kan lista ut, men som illustreras här så att du är medveten om dem när du bygger dina modeller. Om du listar ut det, vänligen meddela oss i diskussionssektionen!

Datasetet har följande kolumner relaterade till genomsnittligt betyg och antal recensioner:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Det hotell med flest recensioner i detta dataset är *Britannia International Hotel Canary Wharf* med 4789 recensioner av 515 000. Men om vi tittar på värdet `Total_Number_of_Reviews` för detta hotell är det 9086. Du kan anta att det finns många fler betyg utan recensioner, så kanske vi bör lägga till värdet i kolumnen `Additional_Number_of_Scoring`. Det värdet är 2682, och att lägga till det till 4789 ger oss 7471, vilket fortfarande är 1615 kort från `Total_Number_of_Reviews`.

Om du tar kolumnen `Average_Score` kan du anta att det är genomsnittet av recensionerna i datasetet, men beskrivningen från Kaggle är "*Genomsnittsbetyget för hotellet, beräknat baserat på den senaste kommentaren under det senaste året*". Det verkar inte särskilt användbart, men vi kan beräkna vårt eget genomsnitt baserat på recensionsbetygen i datasetet. Med samma hotell som exempel är det genomsnittliga hotellbetyget angivet som 7,1 men det beräknade betyget (genomsnittligt recensionsbetyg *i* datasetet) är 6,8. Detta är nära, men inte samma värde, och vi kan bara gissa att betygen i recensionerna i kolumnen `Additional_Number_of_Scoring` ökade genomsnittet till 7,1. Tyvärr, utan något sätt att testa eller bevisa detta påstående, är det svårt att använda eller lita på `Average_Score`, `Additional_Number_of_Scoring` och `Total_Number_of_Reviews` när de är baserade på, eller hänvisar till, data vi inte har.

För att komplicera saker ytterligare har hotellet med näst flest recensioner ett beräknat genomsnittligt betyg på 8,12 och datasetets `Average_Score` är 8,1. Är detta korrekta betyg en slump eller är det första hotellet en avvikelse?

Med möjligheten att dessa hotell kan vara avvikelser, och att kanske de flesta värden stämmer (men vissa gör det inte av någon anledning) kommer vi att skriva ett kort program härnäst för att utforska värdena i datasetet och avgöra korrekt användning (eller icke-användning) av värdena.
> 🚨 En varning

> När du arbetar med denna dataset kommer du att skriva kod som beräknar något från texten utan att behöva läsa eller analysera texten själv. Detta är kärnan i NLP, att tolka mening eller känslor utan att en människa behöver göra det. Dock är det möjligt att du kommer att läsa några av de negativa recensionerna. Jag skulle starkt rekommendera att du låter bli, eftersom du inte behöver göra det. Vissa av dem är löjliga eller irrelevanta negativa hotellrecensioner, som "Vädret var inte bra", något som ligger utanför hotellets, eller någon annans, kontroll. Men det finns också en mörk sida med vissa recensioner. Ibland är de negativa recensionerna rasistiska, sexistiska eller åldersdiskriminerande. Detta är olyckligt men att förvänta sig i en dataset som hämtats från en offentlig webbplats. Vissa recensenter lämnar omdömen som du kan uppleva som smaklösa, obehagliga eller upprörande. Det är bättre att låta koden mäta känslan än att läsa dem själv och bli upprörd. Med det sagt är det en minoritet som skriver sådana saker, men de finns ändå.
## Övning - Datautforskning
### Ladda data

Nu har vi tittat på data visuellt, dags att skriva lite kod och få svar! Den här delen använder biblioteket pandas. Din första uppgift är att säkerställa att du kan ladda och läsa CSV-datan. Pandas har en snabb CSV-laddare, och resultatet placeras i en dataframe, precis som i tidigare lektioner. CSV-filen vi laddar har över en halv miljon rader, men bara 17 kolumner. Pandas ger dig många kraftfulla sätt att interagera med en dataframe, inklusive möjligheten att utföra operationer på varje rad.

Från och med nu i denna lektion kommer det att finnas kodsnuttar, förklaringar av koden och diskussioner om vad resultaten betyder. Använd den medföljande _notebook.ipynb_ för din kod.

Låt oss börja med att ladda datafilen du ska använda:

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

Nu när datan är laddad kan vi utföra några operationer på den. Behåll denna kod högst upp i ditt program för nästa del.

## Utforska data

I det här fallet är datan redan *ren*, vilket betyder att den är redo att arbeta med och inte innehåller tecken på andra språk som kan orsaka problem för algoritmer som förväntar sig endast engelska tecken.

✅ Du kan behöva arbeta med data som kräver viss initial bearbetning för att formatera den innan du tillämpar NLP-tekniker, men inte denna gång. Om du var tvungen, hur skulle du hantera tecken på andra språk?

Ta en stund för att säkerställa att när datan är laddad kan du utforska den med kod. Det är väldigt lätt att vilja fokusera på kolumnerna `Negative_Review` och `Positive_Review`. De är fyllda med naturlig text för dina NLP-algoritmer att bearbeta. Men vänta! Innan du hoppar in i NLP och sentimentanalys bör du följa koden nedan för att kontrollera om värdena i datasetet matchar de värden du beräknar med pandas.

## Dataframe-operationer

Den första uppgiften i denna lektion är att kontrollera om följande påståenden är korrekta genom att skriva kod som undersöker dataframen (utan att ändra den).

> Precis som många programmeringsuppgifter finns det flera sätt att lösa detta, men ett bra råd är att göra det på det enklaste och lättaste sättet du kan, särskilt om det blir lättare att förstå när du återvänder till koden i framtiden. Med dataframes finns det en omfattande API som ofta har ett sätt att göra det du vill effektivt.

Behandla följande frågor som kodningsuppgifter och försök att besvara dem utan att titta på lösningen.

1. Skriv ut *formen* på dataframen du just laddade (formen är antalet rader och kolumner).
2. Beräkna frekvensräkningen för recensenters nationaliteter:
   1. Hur många distinkta värden finns det för kolumnen `Reviewer_Nationality` och vilka är de?
   2. Vilken recensentnationalitet är den vanligaste i datasetet (skriv ut land och antal recensioner)?
   3. Vilka är de nästa 10 mest frekvent förekommande nationaliteterna och deras frekvensräkning?
3. Vilket var det mest recenserade hotellet för var och en av de 10 mest förekommande recensentnationaliteterna?
4. Hur många recensioner finns det per hotell (frekvensräkning av hotell) i datasetet?
5. Även om det finns en kolumn `Average_Score` för varje hotell i datasetet kan du också beräkna ett genomsnittligt betyg (genom att ta genomsnittet av alla recensenters betyg i datasetet för varje hotell). Lägg till en ny kolumn i din dataframe med kolumnrubriken `Calc_Average_Score` som innehåller det beräknade genomsnittet.
6. Har några hotell samma (avrundat till 1 decimal) `Average_Score` och `Calc_Average_Score`?
   1. Försök att skriva en Python-funktion som tar en Series (rad) som argument och jämför värdena, och skriver ut ett meddelande när värdena inte är lika. Använd sedan `.apply()`-metoden för att bearbeta varje rad med funktionen.
7. Beräkna och skriv ut hur många rader som har kolumnen `Negative_Review` med värdet "No Negative".
8. Beräkna och skriv ut hur många rader som har kolumnen `Positive_Review` med värdet "No Positive".
9. Beräkna och skriv ut hur många rader som har kolumnen `Positive_Review` med värdet "No Positive" **och** `Negative_Review` med värdet "No Negative".

### Kodlösningar

1. Skriv ut *formen* på dataframen du just laddade (formen är antalet rader och kolumner).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Beräkna frekvensräkningen för recensenters nationaliteter:

   1. Hur många distinkta värden finns det för kolumnen `Reviewer_Nationality` och vilka är de?
   2. Vilken recensentnationalitet är den vanligaste i datasetet (skriv ut land och antal recensioner)?

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

   3. Vilka är de nästa 10 mest frekvent förekommande nationaliteterna och deras frekvensräkning?

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

3. Vilket var det mest recenserade hotellet för var och en av de 10 mest förekommande recensentnationaliteterna?

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

4. Hur många recensioner finns det per hotell (frekvensräkning av hotell) i datasetet?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Namn                 | Totalt_Antal_Recensioner | Totalt_Recensioner_Funna |
   | :----------------------------------------: | :-----------------------: | :-----------------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789              |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169              |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578              |
   |                    ...                     |           ...            |         ...              |
   |       Mercure Paris Porte d Orleans        |           110            |         10               |
   |                Hotel Wagner                |           135            |         10               |
   |            Hotel Gallitzinberg             |           173            |          8               |

   Du kanske märker att resultaten *räknade i datasetet* inte matchar värdet i `Total_Number_of_Reviews`. Det är oklart om detta värde i datasetet representerade det totala antalet recensioner hotellet hade, men inte alla skrapades, eller någon annan beräkning. `Total_Number_of_Reviews` används inte i modellen på grund av denna oklarhet.

5. Även om det finns en kolumn `Average_Score` för varje hotell i datasetet kan du också beräkna ett genomsnittligt betyg (genom att ta genomsnittet av alla recensenters betyg i datasetet för varje hotell). Lägg till en ny kolumn i din dataframe med kolumnrubriken `Calc_Average_Score` som innehåller det beräknade genomsnittet. Skriv ut kolumnerna `Hotel_Name`, `Average_Score` och `Calc_Average_Score`.

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

   Du kanske också undrar över värdet `Average_Score` och varför det ibland skiljer sig från det beräknade genomsnittliga betyget. Eftersom vi inte kan veta varför vissa värden matchar, men andra har en skillnad, är det säkrast i detta fall att använda de betyg vi har för att beräkna genomsnittet själva. Med det sagt är skillnaderna vanligtvis mycket små, här är hotellen med den största avvikelsen mellan datasetets genomsnitt och det beräknade genomsnittet:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Namn |
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

   Med endast 1 hotell som har en skillnad i betyg större än 1 betyder det att vi förmodligen kan ignorera skillnaden och använda det beräknade genomsnittliga betyget.

6. Beräkna och skriv ut hur många rader som har kolumnen `Negative_Review` med värdet "No Negative".

7. Beräkna och skriv ut hur många rader som har kolumnen `Positive_Review` med värdet "No Positive".

8. Beräkna och skriv ut hur många rader som har kolumnen `Positive_Review` med värdet "No Positive" **och** `Negative_Review` med värdet "No Negative".

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

## Ett annat sätt

Ett annat sätt att räkna objekt utan Lambdas, och använda sum för att räkna rader:

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

   Du kanske har märkt att det finns 127 rader som har både "No Negative" och "No Positive" värden för kolumnerna `Negative_Review` och `Positive_Review` respektive. Det betyder att recensenten gav hotellet ett numeriskt betyg, men avstod från att skriva en positiv eller negativ recension. Lyckligtvis är detta en liten mängd rader (127 av 515738, eller 0,02%), så det kommer förmodligen inte att snedvrida vår modell eller resultat i någon särskild riktning, men du kanske inte har förväntat dig att ett dataset med recensioner skulle ha rader utan recensioner, så det är värt att utforska datan för att upptäcka sådana rader.

Nu när du har utforskat datasetet kommer du i nästa lektion att filtrera datan och lägga till lite sentimentanalys.

---
## 🚀Utmaning

Denna lektion demonstrerar, som vi såg i tidigare lektioner, hur otroligt viktigt det är att förstå din data och dess egenheter innan du utför operationer på den. Textbaserad data, i synnerhet, kräver noggrann granskning. Gräv igenom olika texttunga dataset och se om du kan upptäcka områden som kan introducera bias eller snedvriden sentiment i en modell.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Ta [denna lärväg om NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) för att upptäcka verktyg att prova när du bygger tal- och texttunga modeller.

## Uppgift

[NLTK](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiserade översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.