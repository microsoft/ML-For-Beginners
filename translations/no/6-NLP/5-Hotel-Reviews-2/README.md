<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T22:28:31+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "no"
}
-->
# Sentimentanalyse med hotellanmeldelser

Nå som du har utforsket datasettet i detalj, er det på tide å filtrere kolonnene og deretter bruke NLP-teknikker på datasettet for å få nye innsikter om hotellene.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Filtrering og sentimentanalyse-operasjoner

Som du sikkert har lagt merke til, har datasettet noen problemer. Noen kolonner er fylt med unyttig informasjon, andre virker feil. Selv om de er korrekte, er det uklart hvordan de ble beregnet, og svarene kan ikke uavhengig verifiseres med dine egne beregninger.

## Oppgave: litt mer databehandling

Rens dataene litt mer. Legg til kolonner som vil være nyttige senere, endre verdiene i andre kolonner, og fjern visse kolonner helt.

1. Innledende kolonnebehandling

   1. Fjern `lat` og `lng`

   2. Erstatt verdiene i `Hotel_Address` med følgende verdier (hvis adressen inneholder navnet på byen og landet, endre det til bare byen og landet).

      Dette er de eneste byene og landene i datasettet:

      Amsterdam, Nederland

      Barcelona, Spania

      London, Storbritannia

      Milano, Italia

      Paris, Frankrike

      Wien, Østerrike 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Nå kan du hente data på landsnivå:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nederland   |    105     |
      | Barcelona, Spania      |    211     |
      | London, Storbritannia  |    400     |
      | Milano, Italia         |    162     |
      | Paris, Frankrike       |    458     |
      | Wien, Østerrike        |    158     |

2. Behandle hotell-meta-anmeldelseskolonner

   1. Fjern `Additional_Number_of_Scoring`

   2. Erstatt `Total_Number_of_Reviews` med det totale antallet anmeldelser for det hotellet som faktisk er i datasettet 

   3. Erstatt `Average_Score` med vår egen beregnede score

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Behandle anmeldelseskolonner

   1. Fjern `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` og `days_since_review`

   2. Behold `Reviewer_Score`, `Negative_Review` og `Positive_Review` som de er
     
   3. Behold `Tags` for nå

     - Vi skal gjøre noen ekstra filtreringsoperasjoner på taggene i neste seksjon, og deretter vil taggene bli fjernet

4. Behandle anmelderkolonner

   1. Fjern `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Behold `Reviewer_Nationality`

### Tag-kolonner

`Tag`-kolonnen er problematisk siden den er en liste (i tekstform) lagret i kolonnen. Dessverre er rekkefølgen og antallet underseksjoner i denne kolonnen ikke alltid det samme. Det er vanskelig for et menneske å identifisere de riktige frasene som er interessante, fordi det er 515 000 rader og 1427 hoteller, og hver har litt forskjellige alternativer en anmelder kunne velge. Her kommer NLP til nytte. Du kan skanne teksten og finne de vanligste frasene og telle dem.

Dessverre er vi ikke interessert i enkeltord, men flervordsfraser (f.eks. *Forretningsreise*). Å kjøre en flervordsfrekvensfordelingsalgoritme på så mye data (6762646 ord) kan ta ekstremt lang tid, men uten å se på dataene, virker det som om det er en nødvendig utgift. Her kommer utforskende dataanalyse til nytte, fordi du har sett et utvalg av taggene som `[' Forretningsreise  ', ' Alenereisende ', ' Enkeltrom ', ' Bodde 5 netter ', ' Sendt fra en mobil enhet ']`, kan du begynne å spørre om det er mulig å redusere prosesseringen betydelig. Heldigvis er det det - men først må du følge noen få trinn for å fastslå hvilke tagger som er interessante.

### Filtrering av tagger

Husk at målet med datasettet er å legge til sentiment og kolonner som vil hjelpe deg med å velge det beste hotellet (for deg selv eller kanskje en klient som ber deg lage en hotellanbefalingsbot). Du må spørre deg selv om taggene er nyttige eller ikke i det endelige datasettet. Her er en tolkning (hvis du trengte datasettet av andre grunner, kan forskjellige tagger bli inkludert/utelatt):

1. Typen reise er relevant, og det bør beholdes
2. Typen gjestegruppe er viktig, og det bør beholdes
3. Typen rom, suite eller studio som gjesten bodde i er irrelevant (alle hoteller har stort sett de samme rommene)
4. Enheten anmeldelsen ble sendt fra er irrelevant
5. Antall netter anmelderen bodde *kan* være relevant hvis du tilskriver lengre opphold til at de likte hotellet mer, men det er en svak sammenheng og sannsynligvis irrelevant

Oppsummert, **behold 2 typer tagger og fjern de andre**.

Først vil du ikke telle taggene før de er i et bedre format, så det betyr å fjerne firkantede parenteser og anførselstegn. Du kan gjøre dette på flere måter, men du vil ha den raskeste siden det kan ta lang tid å behandle mye data. Heldigvis har pandas en enkel måte å gjøre hvert av disse trinnene på.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Hver tag blir noe som: `Forretningsreise, Alenereisende, Enkeltrom, Bodde 5 netter, Sendt fra en mobil enhet`. 

Neste finner vi et problem. Noen anmeldelser, eller rader, har 5 kolonner, noen 3, noen 6. Dette er et resultat av hvordan datasettet ble opprettet, og vanskelig å fikse. Du vil få en frekvenstelling av hver frase, men de er i forskjellig rekkefølge i hver anmeldelse, så tellingen kan være feil, og et hotell kan ikke få en tag tilordnet som det fortjente.

I stedet vil du bruke den forskjellige rekkefølgen til vår fordel, fordi hver tag er flervords, men også separert med komma! Den enkleste måten å gjøre dette på er å opprette 6 midlertidige kolonner med hver tag satt inn i kolonnen som tilsvarer dens rekkefølge i taggen. Du kan deretter slå sammen de 6 kolonnene til én stor kolonne og kjøre `value_counts()`-metoden på den resulterende kolonnen. Ved å skrive ut dette, vil du se at det var 2428 unike tagger. Her er et lite utvalg:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Fritidsreise                   | 417778 |
| Sendt fra en mobil enhet       | 307640 |
| Par                            | 252294 |
| Bodde 1 natt                   | 193645 |
| Bodde 2 netter                 | 133937 |
| Alenereisende                  | 108545 |
| Bodde 3 netter                 | 95821  |
| Forretningsreise               | 82939  |
| Gruppe                         | 65392  |
| Familie med små barn           | 61015  |
| Bodde 4 netter                 | 47817  |
| Dobbeltrom                     | 35207  |
| Standard dobbeltrom            | 32248  |
| Superior dobbeltrom            | 31393  |
| Familie med eldre barn         | 26349  |
| Deluxe dobbeltrom              | 24823  |
| Dobbelt- eller tomannsrom      | 22393  |
| Bodde 5 netter                 | 20845  |
| Standard dobbelt- eller tomannsrom | 17483  |
| Klassisk dobbeltrom            | 16989  |
| Superior dobbelt- eller tomannsrom | 13570 |
| 2 rom                          | 12393  |

Noen av de vanlige taggene som `Sendt fra en mobil enhet` er ikke nyttige for oss, så det kan være smart å fjerne dem før du teller fraseforekomst, men det er en så rask operasjon at du kan la dem være og ignorere dem.

### Fjerne tagger for lengden på oppholdet

Å fjerne disse taggene er trinn 1, det reduserer det totale antallet tagger som skal vurderes litt. Merk at du ikke fjerner dem fra datasettet, bare velger å fjerne dem fra vurdering som verdier å telle/beholde i anmeldelsesdatasettet.

| Lengde på opphold | Count  |
| ------------------ | ------ |
| Bodde 1 natt      | 193645 |
| Bodde 2 netter    | 133937 |
| Bodde 3 netter    | 95821  |
| Bodde 4 netter    | 47817  |
| Bodde 5 netter    | 20845  |
| Bodde 6 netter    | 9776   |
| Bodde 7 netter    | 7399   |
| Bodde 8 netter    | 2502   |
| Bodde 9 netter    | 1293   |
| ...               | ...    |

Det finnes et stort utvalg av rom, suiter, studioer, leiligheter og så videre. De betyr stort sett det samme og er ikke relevante for deg, så fjern dem fra vurdering.

| Type rom                     | Count |
| ----------------------------- | ----- |
| Dobbeltrom                   | 35207 |
| Standard dobbeltrom          | 32248 |
| Superior dobbeltrom          | 31393 |
| Deluxe dobbeltrom            | 24823 |
| Dobbelt- eller tomannsrom    | 22393 |
| Standard dobbelt- eller tomannsrom | 17483 |
| Klassisk dobbeltrom          | 16989 |
| Superior dobbelt- eller tomannsrom | 13570 |

Til slutt, og dette er gledelig (fordi det ikke tok mye prosessering i det hele tatt), vil du bli igjen med følgende *nyttige* tagger:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Fritidsreise                                  | 417778 |
| Par                                           | 252294 |
| Alenereisende                                 | 108545 |
| Forretningsreise                              | 82939  |
| Gruppe (kombinert med Reisende med venner)    | 67535  |
| Familie med små barn                          | 61015  |
| Familie med eldre barn                        | 26349  |
| Med et kjæledyr                               | 1405   |

Du kan argumentere for at `Reisende med venner` er det samme som `Gruppe` mer eller mindre, og det ville være rimelig å kombinere de to som ovenfor. Koden for å identifisere de riktige taggene finnes i [Tags-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Det siste trinnet er å opprette nye kolonner for hver av disse taggene. Deretter, for hver anmeldelsesrad, hvis `Tag`-kolonnen samsvarer med en av de nye kolonnene, legg til en 1, hvis ikke, legg til en 0. Sluttresultatet vil være en telling av hvor mange anmeldere som valgte dette hotellet (i aggregat) for, for eksempel, forretning vs fritid, eller for å ta med et kjæledyr, og dette er nyttig informasjon når du skal anbefale et hotell.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Lagre filen

Til slutt, lagre datasettet slik det er nå med et nytt navn.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentanalyse-operasjoner

I denne siste seksjonen skal du bruke sentimentanalyse på anmeldelseskolonnene og lagre resultatene i et datasett.

## Oppgave: last inn og lagre de filtrerte dataene

Merk at du nå laster inn det filtrerte datasettet som ble lagret i forrige seksjon, **ikke** det originale datasettet.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Fjerne stoppord

Hvis du skulle kjøre sentimentanalyse på de negative og positive anmeldelseskolonnene, kan det ta lang tid. Testet på en kraftig test-laptop med rask CPU, tok det 12–14 minutter avhengig av hvilken sentimentbibliotek som ble brukt. Det er en (relativt) lang tid, så det er verdt å undersøke om det kan gjøres raskere. 

Å fjerne stoppord, eller vanlige engelske ord som ikke endrer sentimentet i en setning, er det første trinnet. Ved å fjerne dem, bør sentimentanalysen kjøre raskere, men ikke være mindre nøyaktig (siden stoppordene ikke påvirker sentimentet, men de bremser analysen). 

Den lengste negative anmeldelsen var 395 ord, men etter å ha fjernet stoppordene, er den 195 ord.

Å fjerne stoppordene er også en rask operasjon, å fjerne stoppordene fra 2 anmeldelseskolonner over 515 000 rader tok 3,3 sekunder på test-enheten. Det kan ta litt mer eller mindre tid for deg avhengig av enhetens CPU-hastighet, RAM, om du har SSD eller ikke, og noen andre faktorer. Den relativt korte tiden for operasjonen betyr at hvis det forbedrer sentimentanalysen, er det verdt å gjøre.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Utføre sentimentanalyse

Nå bør du beregne sentimentanalysen for både negative og positive anmeldelseskolonner, og lagre resultatet i 2 nye kolonner. Testen av sentimentet vil være å sammenligne det med anmelderens score for samme anmeldelse. For eksempel, hvis sentimentet mener den negative anmeldelsen hadde et sentiment på 1 (ekstremt positivt sentiment) og et positivt anmeldelsessentiment på 1, men anmelderen ga hotellet den laveste scoren mulig, så stemmer enten ikke anmeldelsesteksten med scoren, eller sentimentanalysatoren kunne ikke gjenkjenne sentimentet korrekt. Du bør forvente at noen sentimentresultater er helt feil, og ofte vil det være forklarbart, f.eks. anmeldelsen kan være ekstremt sarkastisk "Selvfølgelig ELSKET jeg å sove i et rom uten oppvarming" og sentimentanalysatoren tror det er positivt sentiment, selv om et menneske som leser det ville vite at det var sarkasme.
NLTK tilbyr forskjellige sentimentanalysatorer som du kan lære med, og du kan bytte dem ut og se om sentimentet blir mer eller mindre nøyaktig. VADER sentimentanalyse brukes her.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, juni 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Senere i programmet, når du er klar til å beregne sentiment, kan du bruke det på hver anmeldelse som følger:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Dette tar omtrent 120 sekunder på min datamaskin, men det vil variere fra datamaskin til datamaskin. Hvis du vil skrive ut resultatene og se om sentimentet samsvarer med anmeldelsen:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Det aller siste du må gjøre med filen før du bruker den i utfordringen, er å lagre den! Du bør også vurdere å omorganisere alle de nye kolonnene dine slik at de er enkle å jobbe med (for et menneske er dette en kosmetisk endring).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Du bør kjøre hele koden for [analyse-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (etter at du har kjørt [filtrerings-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) for å generere Hotel_Reviews_Filtered.csv-filen).

For å oppsummere, trinnene er:

1. Den originale datasettfilen **Hotel_Reviews.csv** utforskes i forrige leksjon med [utforsker-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv filtreres av [filtrerings-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) og resulterer i **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv behandles av [sentimentanalyse-notatboken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) og resulterer i **Hotel_Reviews_NLP.csv**
4. Bruk Hotel_Reviews_NLP.csv i NLP-utfordringen nedenfor

### Konklusjon

Da du startet, hadde du et datasett med kolonner og data, men ikke alt kunne verifiseres eller brukes. Du har utforsket dataene, filtrert ut det du ikke trenger, konvertert tagger til noe nyttig, beregnet dine egne gjennomsnitt, lagt til noen sentimentkolonner og forhåpentligvis lært noen interessante ting om behandling av naturlig tekst.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Utfordring

Nå som du har analysert datasettet for sentiment, se om du kan bruke strategier du har lært i dette kurset (klustering, kanskje?) for å finne mønstre rundt sentiment.

## Gjennomgang & Selvstudium

Ta [denne Learn-modulen](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) for å lære mer og bruke forskjellige verktøy for å utforske sentiment i tekst.

## Oppgave

[Prøv et annet datasett](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.