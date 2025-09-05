<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T01:26:59+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "da"
}
-->
# Sentimentanalyse med hotelanmeldelser - bearbejdning af data

I denne sektion vil du bruge teknikkerne fra de tidligere lektioner til at lave en udforskende dataanalyse af et stort datasæt. Når du har fået en god forståelse af nytten af de forskellige kolonner, vil du lære:

- hvordan man fjerner unødvendige kolonner
- hvordan man beregner nye data baseret på de eksisterende kolonner
- hvordan man gemmer det resulterende datasæt til brug i den endelige udfordring

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

### Introduktion

Indtil videre har du lært, hvordan tekstdata adskiller sig fra numeriske datatyper. Hvis det er tekst skrevet eller talt af et menneske, kan det analyseres for at finde mønstre, frekvenser, sentiment og mening. Denne lektion introducerer dig til et reelt datasæt med en reel udfordring: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, som inkluderer en [CC0: Public Domain-licens](https://creativecommons.org/publicdomain/zero/1.0/). Det blev indsamlet fra Booking.com fra offentlige kilder. Skaberen af datasættet er Jiashen Liu.

### Forberedelse

Du skal bruge:

* Muligheden for at køre .ipynb-notebooks med Python 3
* pandas
* NLTK, [som du bør installere lokalt](https://www.nltk.org/install.html)
* Datasættet, som er tilgængeligt på Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Det fylder omkring 230 MB, når det er udpakket. Download det til rodmappen `/data`, der er tilknyttet disse NLP-lektioner.

## Udforskende dataanalyse

Denne udfordring antager, at du bygger en hotelanbefalingsbot ved hjælp af sentimentanalyse og gæsteanmeldelsesscores. Datasættet, du vil bruge, indeholder anmeldelser af 1493 forskellige hoteller i 6 byer.

Ved hjælp af Python, et datasæt med hotelanmeldelser og NLTK's sentimentanalyse kan du finde ud af:

* Hvilke ord og sætninger bruges hyppigst i anmeldelserne?
* Korrelerer de officielle *tags*, der beskriver et hotel, med anmeldelsesscores (f.eks. er der flere negative anmeldelser for et bestemt hotel fra *Familie med små børn* end fra *Solorejsende*, hvilket måske indikerer, at det er bedre for *Solorejsende*)?
* Er NLTK's sentiment scores enige med hotelanmelderens numeriske score?

#### Datasæt

Lad os udforske det datasæt, du har downloadet og gemt lokalt. Åbn filen i en editor som VS Code eller endda Excel.

Overskrifterne i datasættet er som følger:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Her er de grupperet på en måde, der måske er lettere at undersøge:  
##### Hotelkolonner

* `Hotel_Name`, `Hotel_Address`, `lat` (breddegrad), `lng` (længdegrad)
  * Ved hjælp af *lat* og *lng* kunne du lave et kort med Python, der viser hotellernes placeringer (måske farvekodet for negative og positive anmeldelser)
  * Hotel_Address er ikke umiddelbart nyttig for os, og vi vil sandsynligvis erstatte den med et land for lettere sortering og søgning

**Hotel Meta-review kolonner**

* `Average_Score`
  * Ifølge datasættets skaber er denne kolonne *Gennemsnitsscore for hotellet, beregnet baseret på den seneste kommentar i det sidste år*. Dette virker som en usædvanlig måde at beregne scoren på, men det er de data, der er indsamlet, så vi må tage det for gode varer for nu.

  ✅ Baseret på de andre kolonner i dette datasæt, kan du komme på en anden måde at beregne gennemsnitsscoren på?

* `Total_Number_of_Reviews`
  * Det samlede antal anmeldelser, dette hotel har modtaget - det er ikke klart (uden at skrive noget kode), om dette refererer til anmeldelserne i datasættet.
* `Additional_Number_of_Scoring`
  * Dette betyder, at en anmeldelsesscore blev givet, men ingen positiv eller negativ anmeldelse blev skrevet af anmelderen.

**Anmeldelseskolonner**

- `Reviewer_Score`
  - Dette er en numerisk værdi med højst 1 decimal mellem minimums- og maksimumsværdierne 2.5 og 10
  - Det er ikke forklaret, hvorfor 2.5 er den lavest mulige score
- `Negative_Review`
  - Hvis en anmelder ikke skrev noget, vil dette felt have "**No Negative**"
  - Bemærk, at en anmelder kan skrive en positiv anmeldelse i kolonnen Negative review (f.eks. "der er intet dårligt ved dette hotel")
- `Review_Total_Negative_Word_Counts`
  - Højere antal negative ord indikerer en lavere score (uden at kontrollere sentimentet)
- `Positive_Review`
  - Hvis en anmelder ikke skrev noget, vil dette felt have "**No Positive**"
  - Bemærk, at en anmelder kan skrive en negativ anmeldelse i kolonnen Positive review (f.eks. "der er absolut intet godt ved dette hotel")
- `Review_Total_Positive_Word_Counts`
  - Højere antal positive ord indikerer en højere score (uden at kontrollere sentimentet)
- `Review_Date` og `days_since_review`
  - En friskheds- eller forældelsesfaktor kan anvendes på en anmeldelse (ældre anmeldelser er måske ikke lige så præcise som nyere, fordi hotelledelsen har ændret sig, renoveringer er blevet udført, eller en pool er blevet tilføjet osv.)
- `Tags`
  - Disse er korte beskrivelser, som en anmelder kan vælge for at beskrive typen af gæst, de var (f.eks. solo eller familie), typen af værelse, de havde, længden af opholdet og hvordan anmeldelsen blev indsendt.
  - Desværre er brugen af disse tags problematisk, se afsnittet nedenfor, der diskuterer deres nytte.

**Anmelderkolonner**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dette kan være en faktor i en anbefalingsmodel, for eksempel hvis du kan afgøre, at mere produktive anmeldere med hundredevis af anmeldelser er mere tilbøjelige til at være negative frem for positive. Dog er anmelderen af en bestemt anmeldelse ikke identificeret med en unik kode og kan derfor ikke knyttes til et sæt anmeldelser. Der er 30 anmeldere med 100 eller flere anmeldelser, men det er svært at se, hvordan dette kan hjælpe anbefalingsmodellen.
- `Reviewer_Nationality`
  - Nogle mennesker kunne tro, at visse nationaliteter er mere tilbøjelige til at give en positiv eller negativ anmeldelse på grund af en national tilbøjelighed. Vær forsigtig med at bygge sådanne anekdotiske synspunkter ind i dine modeller. Disse er nationale (og nogle gange racemæssige) stereotyper, og hver anmelder var en individuel person, der skrev en anmeldelse baseret på deres oplevelse. Det kan være blevet filtreret gennem mange linser såsom deres tidligere hotelophold, den tilbagelagte afstand og deres personlige temperament. At tro, at deres nationalitet var årsagen til en anmeldelsesscore, er svært at retfærdiggøre.

##### Eksempler

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Dette er i øjeblikket ikke et hotel, men en byggeplads. Jeg blev terroriseret fra tidlig morgen og hele dagen med uacceptabel byggestøj, mens jeg hvilede efter en lang rejse og arbejdede på værelset. Folk arbejdede hele dagen, f.eks. med trykluftbor i de tilstødende værelser. Jeg bad om at få skiftet værelse, men der var ikke noget stille værelse tilgængeligt. For at gøre det værre blev jeg overopkrævet. Jeg tjekkede ud om aftenen, da jeg skulle rejse tidligt med fly og modtog en passende regning. En dag senere foretog hotellet en anden opkrævning uden mit samtykke, som oversteg den bookede pris. Det er et forfærdeligt sted. Straf ikke dig selv ved at booke her. | Intet. Forfærdeligt sted. Hold dig væk. | Forretningsrejse Par Standard dobbeltværelse Opholdt sig 2 nætter |

Som du kan se, havde denne gæst ikke et godt ophold på dette hotel. Hotellet har en god gennemsnitsscore på 7.8 og 1945 anmeldelser, men denne anmelder gav det 2.5 og skrev 115 ord om, hvor negativt deres ophold var. Hvis de ikke skrev noget i kolonnen Positive_Review, kunne man antage, at der ikke var noget positivt, men de skrev dog 7 advarselsord. Hvis vi kun tæller ord i stedet for betydningen eller sentimentet af ordene, kunne vi få et skævt billede af anmelderens hensigt. Mærkeligt nok er deres score på 2.5 forvirrende, fordi hvis hotelopholdet var så dårligt, hvorfor give det overhovedet nogen point? Ved at undersøge datasættet nøje vil du se, at den lavest mulige score er 2.5, ikke 0. Den højest mulige score er 10.

##### Tags

Som nævnt ovenfor giver det ved første øjekast mening at bruge `Tags` til at kategorisere dataene. Desværre er disse tags ikke standardiserede, hvilket betyder, at i et givet hotel kan mulighederne være *Single room*, *Twin room* og *Double room*, men i det næste hotel er de *Deluxe Single Room*, *Classic Queen Room* og *Executive King Room*. Disse kan være de samme ting, men der er så mange variationer, at valget bliver:

1. Forsøg på at ændre alle termer til en enkelt standard, hvilket er meget vanskeligt, fordi det ikke er klart, hvad konverteringsvejen ville være i hvert tilfælde (f.eks. *Classic single room* maps til *Single room*, men *Superior Queen Room with Courtyard Garden or City View* er meget sværere at mappe).

1. Vi kan tage en NLP-tilgang og måle hyppigheden af visse termer som *Solo*, *Business Traveller* eller *Family with young kids*, som de gælder for hvert hotel, og inkludere det i anbefalingen.

Tags er normalt (men ikke altid) et enkelt felt, der indeholder en liste over 5 til 6 kommaseparerede værdier, der svarer til *Type of trip*, *Type of guests*, *Type of room*, *Number of nights* og *Type of device review was submitted on*. Men fordi nogle anmeldere ikke udfylder hvert felt (de kan efterlade et tomt), er værdierne ikke altid i samme rækkefølge.

Som et eksempel, tag *Type of group*. Der er 1025 unikke muligheder i dette felt i kolonnen `Tags`, og desværre refererer kun nogle af dem til en gruppe (nogle er typen af værelse osv.). Hvis du filtrerer kun dem, der nævner familie, indeholder resultaterne mange *Family room*-typer. Hvis du inkluderer termen *with*, dvs. tæller *Family with*-værdierne, er resultaterne bedre, med over 80.000 af de 515.000 resultater, der indeholder sætningen "Family with young children" eller "Family with older children".

Dette betyder, at tags-kolonnen ikke er helt ubrugelig for os, men det vil kræve noget arbejde at gøre den nyttig.

##### Gennemsnitlig hotelscore

Der er en række mærkværdigheder eller uoverensstemmelser med datasættet, som jeg ikke kan finde ud af, men de er illustreret her, så du er opmærksom på dem, når du bygger dine modeller. Hvis du finder ud af det, så lad os det vide i diskussionssektionen!

Datasættet har følgende kolonner, der relaterer sig til gennemsnitsscore og antal anmeldelser:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Det hotel med flest anmeldelser i dette datasæt er *Britannia International Hotel Canary Wharf* med 4789 anmeldelser ud af 515.000. Men hvis vi ser på værdien `Total_Number_of_Reviews` for dette hotel, er den 9086. Du kunne antage, at der er mange flere scores uden anmeldelser, så måske skulle vi tilføje værdien i kolonnen `Additional_Number_of_Scoring`. Den værdi er 2682, og hvis vi lægger den til 4789, får vi 7471, hvilket stadig er 1615 mindre end `Total_Number_of_Reviews`.

Hvis du tager kolonnen `Average_Score`, kunne du antage, at det er gennemsnittet af anmeldelserne i datasættet, men beskrivelsen fra Kaggle er "*Gennemsnitsscore for hotellet, beregnet baseret på den seneste kommentar i det sidste år*". Det virker ikke særlig nyttigt, men vi kan beregne vores eget gennemsnit baseret på anmeldelsesscores i datasættet. Ved at bruge det samme hotel som eksempel er den gennemsnitlige hotelscore angivet som 7.1, men den beregnede score (gennemsnitlig anmelder-score *i* datasættet) er 6.8. Dette er tæt på, men ikke den samme værdi, og vi kan kun gætte på, at de scores, der er givet i `Additional_Number_of_Scoring`-anmeldelserne, øgede gennemsnittet til 7.1. Desværre, uden nogen måde at teste eller bevise den påstand, er det svært at bruge eller stole på `Average_Score`, `Additional_Number_of_Scoring` og `Total_Number_of_Reviews`, når de er baseret på eller refererer til data, vi ikke har.

For at komplicere tingene yderligere har hotellet med det næsthøjeste antal anmeldelser en beregnet gennemsnitsscore på 8.12, og datasættets `Average_Score` er 8.1. Er denne korrekte score en tilfældighed, eller er det første hotel en uoverensstemmelse?

På muligheden for, at disse hoteller måske er outliers, og at måske de fleste af værdierne stemmer overens (men nogle gør det ikke af en eller anden grund), vil vi skrive et kort program næste gang for at udforske værdierne i datasættet og bestemme den korrekte brug (eller ikke-brug) af værdierne.
> 🚨 En advarsel  
>  
> Når du arbejder med dette datasæt, vil du skrive kode, der beregner noget ud fra teksten uden at skulle læse eller analysere teksten selv. Dette er essensen af NLP, at fortolke mening eller følelser uden at en menneskelig person behøver at gøre det. Dog er det muligt, at du vil læse nogle af de negative anmeldelser. Jeg vil opfordre dig til ikke at gøre det, fordi det ikke er nødvendigt. Nogle af dem er fjollede eller irrelevante negative hotelanmeldelser, såsom "Vejret var ikke godt", noget der ligger uden for hotellets eller nogen andens kontrol. Men der er også en mørk side ved nogle anmeldelser. Nogle gange er de negative anmeldelser racistiske, sexistiske eller aldersdiskriminerende. Dette er uheldigt, men forventeligt i et datasæt, der er hentet fra en offentlig hjemmeside. Nogle anmeldere efterlader anmeldelser, som du vil finde smagløse, ubehagelige eller oprørende. Det er bedre at lade koden måle følelsen end at læse dem selv og blive ked af det. Når det er sagt, er det en minoritet, der skriver sådanne ting, men de findes stadig.
## Øvelse - Dataudforskning
### Indlæs data

Det er nok med at undersøge data visuelt, nu skal du skrive noget kode og få nogle svar! Denne sektion bruger pandas-biblioteket. Din allerførste opgave er at sikre, at du kan indlæse og læse CSV-data. Pandas-biblioteket har en hurtig CSV-loader, og resultatet placeres i en dataframe, som i tidligere lektioner. Den CSV-fil, vi indlæser, har over en halv million rækker, men kun 17 kolonner. Pandas giver dig mange kraftfulde måder at interagere med en dataframe på, herunder muligheden for at udføre operationer på hver række.

Fra nu af i denne lektion vil der være kodeeksempler og nogle forklaringer af koden samt diskussioner om, hvad resultaterne betyder. Brug den medfølgende _notebook.ipynb_ til din kode.

Lad os starte med at indlæse datafilen, du skal bruge:

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

Nu hvor dataene er indlæst, kan vi udføre nogle operationer på dem. Hold denne kode øverst i dit program til den næste del.

## Udforsk dataene

I dette tilfælde er dataene allerede *rene*, hvilket betyder, at de er klar til at arbejde med og ikke indeholder tegn på andre sprog, der kunne forstyrre algoritmer, der kun forventer engelske tegn.

✅ Du kan komme til at arbejde med data, der kræver en indledende behandling for at formatere dem, før du anvender NLP-teknikker, men ikke denne gang. Hvis du skulle, hvordan ville du håndtere ikke-engelske tegn?

Tag et øjeblik til at sikre, at når dataene er indlæst, kan du udforske dem med kode. Det er meget nemt at fokusere på kolonnerne `Negative_Review` og `Positive_Review`. De er fyldt med naturlig tekst, som dine NLP-algoritmer kan bearbejde. Men vent! Før du springer ind i NLP og sentimentanalyse, bør du følge koden nedenfor for at sikre, at de værdier, der er angivet i datasættet, matcher de værdier, du beregner med pandas.

## Dataframe-operationer

Den første opgave i denne lektion er at kontrollere, om følgende påstande er korrekte ved at skrive noget kode, der undersøger dataframen (uden at ændre den).

> Som med mange programmeringsopgaver er der flere måder at løse dette på, men et godt råd er at gøre det på den enkleste, letteste måde, især hvis det vil være lettere at forstå, når du vender tilbage til denne kode i fremtiden. Med dataframes er der en omfattende API, der ofte har en måde at gøre det, du ønsker, effektivt.

Behandl følgende spørgsmål som kodningsopgaver og forsøg at besvare dem uden at kigge på løsningen.

1. Udskriv *formen* af den dataframe, du lige har indlæst (formen er antallet af rækker og kolonner).
2. Beregn frekvensoptællingen for anmeldernationaliteter:
   1. Hvor mange forskellige værdier er der for kolonnen `Reviewer_Nationality`, og hvad er de?
   2. Hvilken anmeldernationalitet er den mest almindelige i datasættet (udskriv land og antal anmeldelser)?
   3. Hvad er de næste 10 mest hyppigt forekommende nationaliteter og deres frekvensoptælling?
3. Hvilket hotel blev anmeldt mest for hver af de 10 mest hyppige anmeldernationaliteter?
4. Hvor mange anmeldelser er der pr. hotel (frekvensoptælling af hotel) i datasættet?
5. Selvom der er en kolonne `Average_Score` for hvert hotel i datasættet, kan du også beregne en gennemsnitsscore (beregne gennemsnittet af alle anmelderes scorer i datasættet for hvert hotel). Tilføj en ny kolonne til din dataframe med kolonneoverskriften `Calc_Average_Score`, der indeholder det beregnede gennemsnit.
6. Har nogle hoteller samme (afrundet til 1 decimal) `Average_Score` og `Calc_Average_Score`?
   1. Prøv at skrive en Python-funktion, der tager en Series (række) som argument og sammenligner værdierne, og udskriver en besked, når værdierne ikke er ens. Brug derefter `.apply()`-metoden til at behandle hver række med funktionen.
7. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Negative_Review` som "No Negative".
8. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Positive_Review` som "No Positive".
9. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Positive_Review` som "No Positive" **og** `Negative_Review` som "No Negative".

### Kodesvar

1. Udskriv *formen* af den dataframe, du lige har indlæst (formen er antallet af rækker og kolonner).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Beregn frekvensoptællingen for anmeldernationaliteter:

   1. Hvor mange forskellige værdier er der for kolonnen `Reviewer_Nationality`, og hvad er de?
   2. Hvilken anmeldernationalitet er den mest almindelige i datasættet (udskriv land og antal anmeldelser)?

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

   3. Hvad er de næste 10 mest hyppigt forekommende nationaliteter og deres frekvensoptælling?

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

3. Hvilket hotel blev anmeldt mest for hver af de 10 mest hyppige anmeldernationaliteter?

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

4. Hvor mange anmeldelser er der pr. hotel (frekvensoptælling af hotel) i datasættet?

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

   Du bemærker måske, at resultaterne *optalt i datasættet* ikke matcher værdien i `Total_Number_of_Reviews`. Det er uklart, om denne værdi i datasættet repræsenterede det samlede antal anmeldelser, hotellet havde, men ikke alle blev skrabet, eller en anden beregning. `Total_Number_of_Reviews` bruges ikke i modellen på grund af denne uklarhed.

5. Selvom der er en kolonne `Average_Score` for hvert hotel i datasættet, kan du også beregne en gennemsnitsscore (beregne gennemsnittet af alle anmelderes scorer i datasættet for hvert hotel). Tilføj en ny kolonne til din dataframe med kolonneoverskriften `Calc_Average_Score`, der indeholder det beregnede gennemsnit. Udskriv kolonnerne `Hotel_Name`, `Average_Score` og `Calc_Average_Score`.

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

   Du undrer dig måske også over værdien `Average_Score` og hvorfor den nogle gange er forskellig fra den beregnede gennemsnitsscore. Da vi ikke kan vide, hvorfor nogle af værdierne matcher, men andre har en forskel, er det sikrest i dette tilfælde at bruge de anmeldelsesscorer, vi har, til at beregne gennemsnittet selv. Når det er sagt, er forskellene normalt meget små, her er hotellerne med den største afvigelse mellem datasættets gennemsnit og det beregnede gennemsnit:

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

   Med kun 1 hotel, der har en forskel i score større end 1, betyder det, at vi sandsynligvis kan ignorere forskellen og bruge den beregnede gennemsnitsscore.

6. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Negative_Review` som "No Negative".

7. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Positive_Review` som "No Positive".

8. Beregn og udskriv, hvor mange rækker der har kolonneværdien `Positive_Review` som "No Positive" **og** `Negative_Review` som "No Negative".

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

## En anden metode

En anden måde at tælle elementer uden Lambdas og bruge sum til at tælle rækkerne:

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

   Du har måske bemærket, at der er 127 rækker, der har både "No Negative" og "No Positive" værdier for kolonnerne `Negative_Review` og `Positive_Review` henholdsvis. Det betyder, at anmelderen gav hotellet en numerisk score, men undlod at skrive enten en positiv eller negativ anmeldelse. Heldigvis er dette et lille antal rækker (127 ud af 515738, eller 0,02%), så det vil sandsynligvis ikke skævvride vores model eller resultater i nogen bestemt retning, men du havde måske ikke forventet, at et datasæt med anmeldelser ville have rækker uden anmeldelser, så det er værd at udforske dataene for at opdage rækker som denne.

Nu hvor du har udforsket datasættet, vil du i den næste lektion filtrere dataene og tilføje noget sentimentanalyse.

---
## 🚀Udfordring

Denne lektion demonstrerer, som vi så i tidligere lektioner, hvor kritisk vigtigt det er at forstå dine data og deres særheder, før du udfører operationer på dem. Tekstbaserede data kræver især omhyggelig granskning. Grav igennem forskellige teksttunge datasæt og se, om du kan opdage områder, der kunne introducere bias eller skævvredet sentiment i en model.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Tag [denne læringssti om NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) for at opdage værktøjer, du kan prøve, når du bygger tale- og teksttunge modeller.

## Opgave

[NLTK](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.