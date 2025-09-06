<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T22:18:22+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "no"
}
-->
# Sentimentanalyse med hotellanmeldelser - bearbeiding av data

I denne delen vil du bruke teknikkene fra de tidligere leksjonene til å utføre en utforskende dataanalyse av et stort datasett. Når du har fått en god forståelse av nytten av de ulike kolonnene, vil du lære:

- hvordan du fjerner unødvendige kolonner
- hvordan du beregner ny data basert på eksisterende kolonner
- hvordan du lagrer det resulterende datasettet for bruk i den endelige utfordringen

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

### Introduksjon

Så langt har du lært hvordan tekstdata er ganske annerledes enn numeriske datatyper. Hvis det er tekst skrevet eller sagt av et menneske, kan det analyseres for å finne mønstre, frekvenser, sentiment og mening. Denne leksjonen tar deg inn i et ekte datasett med en ekte utfordring: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** som inkluderer en [CC0: Public Domain-lisens](https://creativecommons.org/publicdomain/zero/1.0/). Det ble hentet fra Booking.com fra offentlige kilder. Skaperen av datasettet er Jiashen Liu.

### Forberedelse

Du vil trenge:

* Muligheten til å kjøre .ipynb-notatbøker med Python 3
* pandas
* NLTK, [som du bør installere lokalt](https://www.nltk.org/install.html)
* Datasettet som er tilgjengelig på Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Det er rundt 230 MB ukomprimert. Last det ned til rotmappen `/data` knyttet til disse NLP-leksjonene.

## Utforskende dataanalyse

Denne utfordringen antar at du bygger en hotellanbefalingsbot ved hjelp av sentimentanalyse og gjesteanmeldelsesscorer. Datasettet du vil bruke inkluderer anmeldelser av 1493 forskjellige hoteller i 6 byer.

Ved å bruke Python, et datasett med hotellanmeldelser, og NLTKs sentimentanalyse kan du finne ut:

* Hva er de mest brukte ordene og frasene i anmeldelsene?
* Korresponderer de offisielle *taggene* som beskriver et hotell med anmeldelsesscorer (f.eks. er det flere negative anmeldelser for et bestemt hotell fra *Familie med små barn* enn fra *Alenereisende*, noe som kanskje indikerer at det er bedre for *Alenereisende*)?
* Stemmer NLTKs sentimentanalyseresultater overens med den numeriske scoren fra hotellanmelderen?

#### Datasett

La oss utforske datasettet du har lastet ned og lagret lokalt. Åpne filen i en editor som VS Code eller til og med Excel.

Overskriftene i datasettet er som følger:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Her er de gruppert på en måte som kan være lettere å undersøke: 
##### Hotellkolonner

* `Hotel_Name`, `Hotel_Address`, `lat` (breddegrad), `lng` (lengdegrad)
  * Ved å bruke *lat* og *lng* kan du lage et kart med Python som viser hotellplasseringer (kanskje fargekodet for negative og positive anmeldelser)
  * Hotel_Address er ikke åpenbart nyttig for oss, og vi vil sannsynligvis erstatte det med et land for enklere sortering og søking

**Hotell meta-anmeldelseskolonner**

* `Average_Score`
  * Ifølge datasettets skaper er denne kolonnen *Gjennomsnittsscore for hotellet, beregnet basert på den nyeste kommentaren det siste året*. Dette virker som en uvanlig måte å beregne scoren på, men det er data som er hentet, så vi kan ta det for god fisk for nå. 
  
  ✅ Basert på de andre kolonnene i dette datasettet, kan du tenke på en annen måte å beregne gjennomsnittsscoren på?

* `Total_Number_of_Reviews`
  * Det totale antallet anmeldelser dette hotellet har mottatt - det er ikke klart (uten å skrive litt kode) om dette refererer til anmeldelsene i datasettet.
* `Additional_Number_of_Scoring`
  * Dette betyr at en anmeldelsesscore ble gitt, men ingen positiv eller negativ anmeldelse ble skrevet av anmelderen

**Anmeldelseskolonner**

- `Reviewer_Score`
  - Dette er en numerisk verdi med maksimalt én desimal mellom minimums- og maksimumsverdiene 2.5 og 10
  - Det er ikke forklart hvorfor 2.5 er den laveste mulige scoren
- `Negative_Review`
  - Hvis en anmelder ikke skrev noe, vil dette feltet ha "**No Negative**"
  - Merk at en anmelder kan skrive en positiv anmeldelse i kolonnen for Negative Review (f.eks. "det er ingenting dårlig med dette hotellet")
- `Review_Total_Negative_Word_Counts`
  - Høyere negative ordtellinger indikerer en lavere score (uten å sjekke sentimentet)
- `Positive_Review`
  - Hvis en anmelder ikke skrev noe, vil dette feltet ha "**No Positive**"
  - Merk at en anmelder kan skrive en negativ anmeldelse i kolonnen for Positive Review (f.eks. "det er ingenting bra med dette hotellet i det hele tatt")
- `Review_Total_Positive_Word_Counts`
  - Høyere positive ordtellinger indikerer en høyere score (uten å sjekke sentimentet)
- `Review_Date` og `days_since_review`
  - En ferskhets- eller foreldelsesmåling kan brukes på en anmeldelse (eldre anmeldelser er kanskje ikke like nøyaktige som nyere fordi hotellledelsen har endret seg, renoveringer er gjort, eller et basseng er lagt til osv.)
- `Tags`
  - Dette er korte beskrivelser som en anmelder kan velge for å beskrive typen gjest de var (f.eks. alene eller familie), typen rom de hadde, lengden på oppholdet og hvordan anmeldelsen ble sendt inn. 
  - Dessverre er bruken av disse taggene problematisk, se avsnittet nedenfor som diskuterer deres nytteverdi

**Anmelderkolonner**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dette kan være en faktor i en anbefalingsmodell, for eksempel hvis du kan fastslå at mer produktive anmeldere med hundrevis av anmeldelser er mer tilbøyelige til å være negative enn positive. Imidlertid er anmelderen av en bestemt anmeldelse ikke identifisert med en unik kode, og kan derfor ikke kobles til et sett med anmeldelser. Det er 30 anmeldere med 100 eller flere anmeldelser, men det er vanskelig å se hvordan dette kan hjelpe anbefalingsmodellen.
- `Reviewer_Nationality`
  - Noen kan tro at visse nasjonaliteter er mer tilbøyelige til å gi en positiv eller negativ anmeldelse på grunn av en nasjonal tilbøyelighet. Vær forsiktig med å bygge slike anekdotiske synspunkter inn i modellene dine. Dette er nasjonale (og noen ganger rasemessige) stereotyper, og hver anmelder var en individuell person som skrev en anmeldelse basert på sin opplevelse. Den kan ha blitt filtrert gjennom mange linser som deres tidligere hotellopphold, avstanden de reiste, og deres personlige temperament. Å tro at deres nasjonalitet var årsaken til en anmeldelsesscore er vanskelig å rettferdiggjøre.

##### Eksempler

| Gjennomsnittlig  Score | Totalt Antall   Anmeldelser | Anmelder   Score | Negativ <br />Anmeldelse                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positiv   Anmeldelse                 | Tags                                                                                      |
| ---------------------- | -------------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------------- |
| 7.8                    | 1945                       | 2.5              | Dette er  for øyeblikket ikke et hotell, men en byggeplass Jeg ble terrorisert fra tidlig morgen og hele dagen med uakseptabel byggestøy mens jeg hvilte etter en lang reise og jobbet på rommet Folk jobbet hele dagen med f.eks. trykkluftbor i de tilstøtende rommene Jeg ba om å bytte rom, men ingen stille rom var tilgjengelig For å gjøre ting verre ble jeg overbelastet Jeg sjekket ut på kvelden siden jeg måtte dra veldig tidlig fly og mottok en passende regning En dag senere gjorde hotellet en annen belastning uten mitt samtykke som oversteg den bookede prisen Det er et forferdelig sted Ikke straff deg selv ved å booke her | Ingenting  Forferdelig sted Hold deg unna | Forretningsreise                                Par Standard Dobbeltrom Bodde 2 netter |

Som du kan se, hadde denne gjesten ikke et hyggelig opphold på dette hotellet. Hotellet har en god gjennomsnittsscore på 7.8 og 1945 anmeldelser, men denne anmelderen ga det 2.5 og skrev 115 ord om hvor negativt oppholdet var. Hvis de ikke skrev noe i Positive_Review-kolonnen, kan du anta at det ikke var noe positivt, men de skrev faktisk 7 ord som advarsel. Hvis vi bare teller ord i stedet for betydningen eller sentimentet av ordene, kan vi få et skjevt bilde av anmelderens intensjon. Merkelig nok er deres score på 2.5 forvirrende, fordi hvis hotelloppholdet var så dårlig, hvorfor gi det noen poeng i det hele tatt? Ved å undersøke datasettet nøye, vil du se at den laveste mulige scoren er 2.5, ikke 0. Den høyeste mulige scoren er 10.

##### Tags

Som nevnt ovenfor, ved første øyekast virker ideen om å bruke `Tags` til å kategorisere data fornuftig. Dessverre er disse taggene ikke standardiserte, noe som betyr at i et gitt hotell kan alternativene være *Enkeltrom*, *Tomannsrom* og *Dobbeltrom*, men i neste hotell er de *Deluxe Enkeltrom*, *Klassisk Queen-rom* og *Executive King-rom*. Disse kan være de samme tingene, men det er så mange variasjoner at valget blir:

1. Forsøke å endre alle begrepene til en enkelt standard, noe som er veldig vanskelig, fordi det ikke er klart hva konverteringsbanen vil være i hvert tilfelle (f.eks. *Klassisk enkeltrom* kartlegges til *Enkeltrom*, men *Superior Queen-rom med hage eller byutsikt* er mye vanskeligere å kartlegge)

1. Vi kan ta en NLP-tilnærming og måle frekvensen av visse begreper som *Alene*, *Forretningsreisende* eller *Familie med små barn* slik de gjelder for hvert hotell, og ta dette med i anbefalingen  

Tags er vanligvis (men ikke alltid) et enkelt felt som inneholder en liste med 5 til 6 kommaseparerte verdier som samsvarer med *Type reise*, *Type gjester*, *Type rom*, *Antall netter* og *Type enhet anmeldelsen ble sendt inn fra*. Imidlertid, fordi noen anmeldere ikke fyller ut hvert felt (de kan la ett være tomt), er verdiene ikke alltid i samme rekkefølge.

Som et eksempel, ta *Type gruppe*. Det er 1025 unike muligheter i dette feltet i `Tags`-kolonnen, og dessverre refererer bare noen av dem til en gruppe (noen er typen rom osv.). Hvis du filtrerer bare de som nevner familie, inneholder resultatene mange *Familierom*-type resultater. Hvis du inkluderer begrepet *med*, dvs. teller *Familie med*-verdier, er resultatene bedre, med over 80,000 av de 515,000 resultatene som inneholder frasen "Familie med små barn" eller "Familie med eldre barn".

Dette betyr at tags-kolonnen ikke er helt ubrukelig for oss, men det vil kreve litt arbeid for å gjøre den nyttig.

##### Gjennomsnittlig hotellscore

Det er en rekke rariteter eller avvik med datasettet som jeg ikke kan finne ut av, men som er illustrert her slik at du er klar over dem når du bygger modellene dine. Hvis du finner ut av det, gi oss beskjed i diskusjonsseksjonen!

Datasettet har følgende kolonner relatert til gjennomsnittsscore og antall anmeldelser: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotellet med flest anmeldelser i dette datasettet er *Britannia International Hotel Canary Wharf* med 4789 anmeldelser av 515,000. Men hvis vi ser på verdien `Total_Number_of_Reviews` for dette hotellet, er det 9086. Du kan anta at det er mange flere scorer uten anmeldelser, så kanskje vi bør legge til verdien i kolonnen `Additional_Number_of_Scoring`. Den verdien er 2682, og å legge den til 4789 gir oss 7471, som fortsatt er 1615 mindre enn `Total_Number_of_Reviews`. 

Hvis du tar kolonnen `Average_Score`, kan du anta at det er gjennomsnittet av anmeldelsene i datasettet, men beskrivelsen fra Kaggle er "*Gjennomsnittsscore for hotellet, beregnet basert på den nyeste kommentaren det siste året*". Det virker ikke så nyttig, men vi kan beregne vårt eget gjennomsnitt basert på anmeldelsesscorene i datasettet. Ved å bruke det samme hotellet som et eksempel, er den gjennomsnittlige hotellscoren gitt som 7.1, men den beregnede scoren (gjennomsnittlig anmelder-score *i* datasettet) er 6.8. Dette er nært, men ikke den samme verdien, og vi kan bare gjette at scorene gitt i `Additional_Number_of_Scoring`-anmeldelsene økte gjennomsnittet til 7.1. Dessverre, uten noen måte å teste eller bevise den påstanden, er det vanskelig å bruke eller stole på `Average_Score`, `Additional_Number_of_Scoring` og `Total_Number_of_Reviews` når de er basert på, eller refererer til, data vi ikke har.

For å komplisere ting ytterligere, har hotellet med det nest høyeste antallet anmeldelser en beregnet gjennomsnittsscore på 8.12, og datasettets `Average_Score` er 8.1. Er denne korrekte scoren en tilfeldighet, eller er det første hotellet et avvik? 

På muligheten for at disse hotellene kan være uteliggere, og at kanskje de fleste verdiene stemmer (men noen gjør det ikke av en eller annen grunn), vil vi skrive et kort program neste gang for å utforske verdiene i datasettet og bestemme korrekt bruk (eller ikke-bruk) av verdiene.
> 🚨 En advarsel

> Når du arbeider med dette datasettet, vil du skrive kode som beregner noe ut fra teksten uten å måtte lese eller analysere teksten selv. Dette er essensen av NLP, å tolke mening eller sentiment uten at en menneskelig person trenger å gjøre det. Det er imidlertid mulig at du vil lese noen av de negative anmeldelsene. Jeg vil sterkt oppfordre deg til å la være, fordi du ikke trenger det. Noen av dem er tullete eller irrelevante negative hotellanmeldelser, som for eksempel "Været var ikke bra", noe som er utenfor hotellets, eller noens, kontroll. Men det finnes også en mørk side ved noen anmeldelser. Noen ganger er de negative anmeldelsene rasistiske, sexistiske eller aldersdiskriminerende. Dette er uheldig, men forventet i et datasett hentet fra en offentlig nettside. Noen anmeldere legger igjen anmeldelser som du kan finne smakløse, ubehagelige eller opprørende. Det er bedre å la koden måle sentimentet enn å lese dem selv og bli opprørt. Når det er sagt, er det en minoritet som skriver slike ting, men de finnes likevel.
## Øvelse - Datautforskning
### Last inn dataene

Det er nok å undersøke dataene visuelt, nå skal du skrive litt kode og få noen svar! Denne delen bruker pandas-biblioteket. Din aller første oppgave er å sørge for at du kan laste inn og lese CSV-dataene. Pandas-biblioteket har en rask CSV-laster, og resultatet plasseres i en dataframe, som i tidligere leksjoner. CSV-filen vi laster inn har over en halv million rader, men bare 17 kolonner. Pandas gir deg mange kraftige måter å samhandle med en dataframe på, inkludert muligheten til å utføre operasjoner på hver rad.

Fra nå av i denne leksjonen vil det være kodeeksempler, noen forklaringer av koden og diskusjoner om hva resultatene betyr. Bruk den inkluderte _notebook.ipynb_ for koden din.

La oss starte med å laste inn datafilen du skal bruke:

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

Nå som dataene er lastet inn, kan vi utføre noen operasjoner på dem. Hold denne koden øverst i programmet ditt for neste del.

## Utforsk dataene

I dette tilfellet er dataene allerede *rene*, det betyr at de er klare til å jobbe med og ikke inneholder tegn fra andre språk som kan skape problemer for algoritmer som forventer kun engelske tegn.

✅ Du kan komme til å jobbe med data som krever en viss forhåndsbehandling for å formatere dem før du bruker NLP-teknikker, men ikke denne gangen. Hvis du måtte, hvordan ville du håndtert ikke-engelske tegn?

Ta et øyeblikk for å forsikre deg om at når dataene er lastet inn, kan du utforske dem med kode. Det er veldig lett å ville fokusere på kolonnene `Negative_Review` og `Positive_Review`. De er fylt med naturlig tekst som NLP-algoritmene dine kan prosessere. Men vent! Før du hopper inn i NLP og sentimentanalyse, bør du følge koden nedenfor for å fastslå om verdiene som er gitt i datasettet samsvarer med verdiene du beregner med pandas.

## Dataframe-operasjoner

Den første oppgaven i denne leksjonen er å sjekke om følgende påstander er korrekte ved å skrive litt kode som undersøker dataframen (uten å endre den).

> Som med mange programmeringsoppgaver, finnes det flere måter å løse dette på, men et godt råd er å gjøre det på den enkleste og letteste måten, spesielt hvis det vil være lettere å forstå når du kommer tilbake til denne koden i fremtiden. Med dataframes finnes det et omfattende API som ofte vil ha en effektiv måte å gjøre det du ønsker.

Behandle følgende spørsmål som kodingsoppgaver og prøv å svare på dem uten å se på løsningen.

1. Skriv ut *formen* til dataframen du nettopp har lastet inn (formen er antall rader og kolonner).
2. Beregn frekvensen for anmeldernasjonaliteter:
   1. Hvor mange distinkte verdier finnes det i kolonnen `Reviewer_Nationality`, og hva er de?
   2. Hvilken anmeldernasjonalitet er den vanligste i datasettet (skriv ut land og antall anmeldelser)?
   3. Hva er de neste 10 mest vanlige nasjonalitetene, og deres frekvens?
3. Hvilket hotell ble anmeldt oftest for hver av de 10 mest vanlige anmeldernasjonalitetene?
4. Hvor mange anmeldelser er det per hotell (frekvensen av anmeldelser per hotell) i datasettet?
5. Selv om det finnes en kolonne `Average_Score` for hvert hotell i datasettet, kan du også beregne en gjennomsnittsscore (ved å ta gjennomsnittet av alle anmelderscorer i datasettet for hvert hotell). Legg til en ny kolonne i dataframen med kolonneoverskriften `Calc_Average_Score` som inneholder den beregnede gjennomsnittsscoren.
6. Har noen hoteller samme (avrundet til én desimal) `Average_Score` og `Calc_Average_Score`?
   1. Prøv å skrive en Python-funksjon som tar en Series (rad) som et argument og sammenligner verdiene, og skriver ut en melding når verdiene ikke er like. Bruk deretter `.apply()`-metoden for å prosessere hver rad med funksjonen.
7. Beregn og skriv ut hvor mange rader som har verdien "No Negative" i kolonnen `Negative_Review`.
8. Beregn og skriv ut hvor mange rader som har verdien "No Positive" i kolonnen `Positive_Review`.
9. Beregn og skriv ut hvor mange rader som har verdien "No Positive" i kolonnen `Positive_Review` **og** verdien "No Negative" i kolonnen `Negative_Review`.

### Kodesvar

1. Skriv ut *formen* til dataframen du nettopp har lastet inn (formen er antall rader og kolonner).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Beregn frekvensen for anmeldernasjonaliteter:

   1. Hvor mange distinkte verdier finnes det i kolonnen `Reviewer_Nationality`, og hva er de?
   2. Hvilken anmeldernasjonalitet er den vanligste i datasettet (skriv ut land og antall anmeldelser)?

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

   3. Hva er de neste 10 mest vanlige nasjonalitetene, og deres frekvens?

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

3. Hvilket hotell ble anmeldt oftest for hver av de 10 mest vanlige anmeldernasjonalitetene?

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

4. Hvor mange anmeldelser er det per hotell (frekvensen av anmeldelser per hotell) i datasettet?

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
   
   Du vil kanskje legge merke til at resultatene *telt i datasettet* ikke samsvarer med verdien i `Total_Number_of_Reviews`. Det er uklart om denne verdien i datasettet representerer det totale antallet anmeldelser hotellet hadde, men ikke alle ble skrapet, eller om det er en annen beregning. `Total_Number_of_Reviews` brukes ikke i modellen på grunn av denne uklarheten.

5. Selv om det finnes en kolonne `Average_Score` for hvert hotell i datasettet, kan du også beregne en gjennomsnittsscore (ved å ta gjennomsnittet av alle anmelderscorer i datasettet for hvert hotell). Legg til en ny kolonne i dataframen med kolonneoverskriften `Calc_Average_Score` som inneholder den beregnede gjennomsnittsscoren. Skriv ut kolonnene `Hotel_Name`, `Average_Score` og `Calc_Average_Score`.

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

   Du lurer kanskje også på verdien `Average_Score` og hvorfor den noen ganger er forskjellig fra den beregnede gjennomsnittsscoren. Siden vi ikke kan vite hvorfor noen av verdiene samsvarer, men andre har en forskjell, er det tryggest i dette tilfellet å bruke anmelderscorene vi har for å beregne gjennomsnittet selv. Når det er sagt, er forskjellene vanligvis veldig små. Her er hotellene med størst avvik mellom gjennomsnittet i datasettet og det beregnede gjennomsnittet:

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

   Med bare 1 hotell som har en forskjell i score større enn 1, betyr det at vi sannsynligvis kan ignorere forskjellen og bruke den beregnede gjennomsnittsscoren.

6. Beregn og skriv ut hvor mange rader som har verdien "No Negative" i kolonnen `Negative_Review`.

7. Beregn og skriv ut hvor mange rader som har verdien "No Positive" i kolonnen `Positive_Review`.

8. Beregn og skriv ut hvor mange rader som har verdien "No Positive" i kolonnen `Positive_Review` **og** verdien "No Negative" i kolonnen `Negative_Review`.

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

## En annen måte

En annen måte å telle elementer uten Lambdas, og bruke sum for å telle radene:

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

   Du har kanskje lagt merke til at det er 127 rader som har både "No Negative" og "No Positive" verdier i kolonnene `Negative_Review` og `Positive_Review` henholdsvis. Det betyr at anmelderen ga hotellet en numerisk score, men unnlot å skrive enten en positiv eller negativ anmeldelse. Heldigvis er dette en liten mengde rader (127 av 515738, eller 0,02 %), så det vil sannsynligvis ikke skjevfordele modellen vår eller resultatene i noen bestemt retning, men du hadde kanskje ikke forventet at et datasett med anmeldelser skulle ha rader uten anmeldelser, så det er verdt å utforske dataene for å oppdage slike rader.

Nå som du har utforsket datasettet, vil du i neste leksjon filtrere dataene og legge til litt sentimentanalyse.

---
## 🚀Utfordring

Denne leksjonen viser, som vi så i tidligere leksjoner, hvor kritisk viktig det er å forstå dataene dine og deres særegenheter før du utfører operasjoner på dem. Tekstbaserte data krever spesielt nøye gransking. Grav gjennom ulike teksttunge datasett og se om du kan oppdage områder som kan introdusere skjevhet eller skjev sentimentanalyse i en modell.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Ta [denne læringsstien om NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) for å oppdage verktøy du kan prøve når du bygger tale- og teksttunge modeller.

## Oppgave

[NLTK](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.