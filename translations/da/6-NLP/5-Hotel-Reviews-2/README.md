<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T01:43:43+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "da"
}
-->
# Sentimentanalyse med hotelanmeldelser

Nu hvor du har udforsket datasættet i detaljer, er det tid til at filtrere kolonnerne og derefter bruge NLP-teknikker på datasættet for at få nye indsigter om hotellerne.

## [Quiz før forelæsning](https://ff-quizzes.netlify.app/en/ml/)

### Filtrering & Sentimentanalyse-operationer

Som du sikkert har bemærket, har datasættet nogle problemer. Nogle kolonner er fyldt med unyttige oplysninger, andre virker forkerte. Hvis de er korrekte, er det uklart, hvordan de blev beregnet, og svarene kan ikke uafhængigt verificeres med dine egne beregninger.

## Øvelse: lidt mere databehandling

Rens dataene lidt mere. Tilføj kolonner, der vil være nyttige senere, ændr værdierne i andre kolonner, og fjern visse kolonner helt.

1. Indledende kolonnebehandling

   1. Fjern `lat` og `lng`

   2. Erstat værdierne i `Hotel_Address` med følgende værdier (hvis adressen indeholder navnet på byen og landet, ændres det til kun byen og landet).

      Disse er de eneste byer og lande i datasættet:

      Amsterdam, Nederlandene

      Barcelona, Spanien

      London, Storbritannien

      Milano, Italien

      Paris, Frankrig

      Wien, Østrig 

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

      Nu kan du forespørge data på landeniveau:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nederlandene |    105     |
      | Barcelona, Spanien       |    211     |
      | London, Storbritannien   |    400     |
      | Milano, Italien          |    162     |
      | Paris, Frankrig          |    458     |
      | Wien, Østrig             |    158     |

2. Behandling af hotel-meta-anmeldelseskolonner

  1. Fjern `Additional_Number_of_Scoring`

  1. Erstat `Total_Number_of_Reviews` med det samlede antal anmeldelser for det hotel, der faktisk er i datasættet 

  1. Erstat `Average_Score` med vores egen beregnede score

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Behandling af anmeldelseskolonner

   1. Fjern `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` og `days_since_review`

   2. Behold `Reviewer_Score`, `Negative_Review` og `Positive_Review` som de er,
     
   3. Behold `Tags` for nu

     - Vi vil udføre nogle yderligere filtreringsoperationer på tags i næste afsnit, og derefter vil tags blive fjernet

4. Behandling af anmelderkolonner

  1. Fjern `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Behold `Reviewer_Nationality`

### Tag-kolonner

Kolonnen `Tag` er problematisk, da den er en liste (i tekstform), der er gemt i kolonnen. Desværre er rækkefølgen og antallet af underafsnit i denne kolonne ikke altid det samme. Det er svært for et menneske at identificere de korrekte sætninger, der er interessante, fordi der er 515.000 rækker og 1427 hoteller, og hver har lidt forskellige muligheder, som en anmelder kunne vælge. Her kommer NLP til sin ret. Du kan scanne teksten og finde de mest almindelige sætninger og tælle dem.

Desværre er vi ikke interesserede i enkeltord, men i flertalsord-sætninger (f.eks. *Forretningsrejse*). At køre en flertalsord-frekvensfordelingsalgoritme på så mange data (6762646 ord) kunne tage ekstraordinært lang tid, men uden at se på dataene, ville det virke som en nødvendig udgift. Her er eksplorativ dataanalyse nyttig, fordi du har set et eksempel på tags som `[' Forretningsrejse  ', ' Solorejsende ', ' Enkeltværelse ', ' Opholdt sig 5 nætter ', ' Indsendt fra en mobil enhed ']`, kan du begynde at spørge, om det er muligt at reducere den behandling, du skal udføre, betydeligt. Heldigvis er det muligt - men først skal du følge nogle trin for at fastslå de interessante tags.

### Filtrering af tags

Husk, at målet med datasættet er at tilføje sentiment og kolonner, der vil hjælpe dig med at vælge det bedste hotel (for dig selv eller måske en klient, der beder dig om at lave en hotelanbefalingsbot). Du skal spørge dig selv, om tags er nyttige eller ej i det endelige datasæt. Her er en fortolkning (hvis du havde brug for datasættet af andre grunde, kunne forskellige tags blive inkluderet/udeladt):

1. Rejsetypen er relevant og bør beholdes
2. Gæstegruppens type er vigtig og bør beholdes
3. Typen af værelse, suite eller studio, som gæsten boede i, er irrelevant (alle hoteller har stort set de samme værelser)
4. Enheden, som anmeldelsen blev indsendt fra, er irrelevant
5. Antallet af nætter, som anmelderen opholdt sig, *kunne* være relevant, hvis du tilskriver længere ophold til, at de kunne lide hotellet mere, men det er tvivlsomt og sandsynligvis irrelevant

Kort sagt, **behold 2 slags tags og fjern de andre**.

Først vil du ikke tælle tags, før de er i et bedre format, så det betyder at fjerne firkantede parenteser og citater. Du kan gøre dette på flere måder, men du vil have den hurtigste, da det kan tage lang tid at behandle mange data. Heldigvis har pandas en nem måde at udføre hvert af disse trin.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Hvert tag bliver til noget som: `Forretningsrejse, Solorejsende, Enkeltværelse, Opholdt sig 5 nætter, Indsendt fra en mobil enhed`. 

Næste problem opstår. Nogle anmeldelser, eller rækker, har 5 kolonner, nogle 3, nogle 6. Dette er et resultat af, hvordan datasættet blev oprettet, og det er svært at rette. Du vil gerne have en frekvenstælling af hver sætning, men de er i forskellig rækkefølge i hver anmeldelse, så tællingen kan være forkert, og et hotel kan ikke få et tag tildelt, som det fortjente.

I stedet vil du bruge den forskellige rækkefølge til din fordel, fordi hvert tag er flertalsord, men også adskilt af et komma! Den enkleste måde at gøre dette på er at oprette 6 midlertidige kolonner med hvert tag indsat i kolonnen, der svarer til dets rækkefølge i tagget. Du kan derefter flette de 6 kolonner til én stor kolonne og køre metoden `value_counts()` på den resulterende kolonne. Når du printer det ud, vil du se, at der var 2428 unikke tags. Her er et lille eksempel:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Fritidsrejse                   | 417778 |
| Indsendt fra en mobil enhed    | 307640 |
| Par                            | 252294 |
| Opholdt sig 1 nat              | 193645 |
| Opholdt sig 2 nætter           | 133937 |
| Solorejsende                   | 108545 |
| Opholdt sig 3 nætter           | 95821  |
| Forretningsrejse               | 82939  |
| Gruppe                         | 65392  |
| Familie med små børn           | 61015  |
| Opholdt sig 4 nætter           | 47817  |
| Dobbeltværelse                 | 35207  |
| Standard dobbeltværelse        | 32248  |
| Superior dobbeltværelse        | 31393  |
| Familie med ældre børn         | 26349  |
| Deluxe dobbeltværelse          | 24823  |
| Dobbelt- eller twin-værelse    | 22393  |
| Opholdt sig 5 nætter           | 20845  |
| Standard dobbelt- eller twin-værelse | 17483  |
| Klassisk dobbeltværelse        | 16989  |
| Superior dobbelt- eller twin-værelse | 13570 |
| 2 værelser                     | 12393  |

Nogle af de almindelige tags som `Indsendt fra en mobil enhed` er ikke nyttige for os, så det kunne være smart at fjerne dem, før du tæller sætningens forekomst, men det er en så hurtig operation, at du kan lade dem være og ignorere dem.

### Fjernelse af tags for opholdslængde

Fjernelse af disse tags er trin 1, det reducerer det samlede antal tags, der skal overvejes, en smule. Bemærk, at du ikke fjerner dem fra datasættet, men blot vælger at fjerne dem fra overvejelse som værdier, der skal tælles/beholdes i anmeldelsesdatasættet.

| Opholdslængde | Count  |
| ------------- | ------ |
| Opholdt sig 1 nat | 193645 |
| Opholdt sig 2 nætter | 133937 |
| Opholdt sig 3 nætter | 95821  |
| Opholdt sig 4 nætter | 47817  |
| Opholdt sig 5 nætter | 20845  |
| Opholdt sig 6 nætter | 9776   |
| Opholdt sig 7 nætter | 7399   |
| Opholdt sig 8 nætter | 2502   |
| Opholdt sig 9 nætter | 1293   |
| ...              | ...    |

Der er en enorm variation af værelser, suiter, studios, lejligheder og så videre. De betyder stort set det samme og er ikke relevante for dig, så fjern dem fra overvejelse.

| Værelsestype                  | Count |
| ----------------------------- | ----- |
| Dobbeltværelse                | 35207 |
| Standard dobbeltværelse       | 32248 |
| Superior dobbeltværelse       | 31393 |
| Deluxe dobbeltværelse         | 24823 |
| Dobbelt- eller twin-værelse   | 22393 |
| Standard dobbelt- eller twin-værelse | 17483 |
| Klassisk dobbeltværelse       | 16989 |
| Superior dobbelt- eller twin-værelse | 13570 |

Endelig, og dette er glædeligt (fordi det ikke krævede meget behandling overhovedet), vil du være tilbage med følgende *nyttige* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Fritidsrejse                                  | 417778 |
| Par                                           | 252294 |
| Solorejsende                                  | 108545 |
| Forretningsrejse                              | 82939  |
| Gruppe (kombineret med Rejsende med venner)   | 67535  |
| Familie med små børn                          | 61015  |
| Familie med ældre børn                        | 26349  |
| Med et kæledyr                                | 1405   |

Man kunne argumentere for, at `Rejsende med venner` er det samme som `Gruppe` mere eller mindre, og det ville være rimeligt at kombinere de to som ovenfor. Koden til at identificere de korrekte tags findes i [Tags-notebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Det sidste trin er at oprette nye kolonner for hver af disse tags. Derefter, for hver anmeldelsesrække, hvis kolonnen `Tag` matcher en af de nye kolonner, tilføj en 1, hvis ikke, tilføj en 0. Det endelige resultat vil være en optælling af, hvor mange anmeldere der valgte dette hotel (i aggregat) til f.eks. forretning vs fritid, eller til at tage et kæledyr med, og dette er nyttige oplysninger, når man anbefaler et hotel.

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

### Gem din fil

Til sidst gem datasættet, som det er nu, med et nyt navn.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentanalyse-operationer

I dette sidste afsnit vil du anvende sentimentanalyse på anmeldelseskolonnerne og gemme resultaterne i et datasæt.

## Øvelse: indlæs og gem de filtrerede data

Bemærk, at du nu indlæser det filtrerede datasæt, der blev gemt i det foregående afsnit, **ikke** det originale datasæt.

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

### Fjernelse af stopord

Hvis du skulle køre sentimentanalyse på de negative og positive anmeldelseskolonner, kunne det tage lang tid. Testet på en kraftfuld test-laptop med hurtig CPU tog det 12 - 14 minutter afhængigt af hvilken sentimentbibliotek der blev brugt. Det er en (relativt) lang tid, så det er værd at undersøge, om det kan gøres hurtigere. 

Fjernelse af stopord, eller almindelige engelske ord, der ikke ændrer sentimenten i en sætning, er det første trin. Ved at fjerne dem bør sentimentanalysen køre hurtigere, men ikke være mindre præcis (da stopordene ikke påvirker sentimenten, men de gør analysen langsommere). 

Den længste negative anmeldelse var 395 ord, men efter fjernelse af stopordene er den 195 ord.

Fjernelse af stopordene er også en hurtig operation; fjernelse af stopordene fra 2 anmeldelseskolonner over 515.000 rækker tog 3,3 sekunder på test-enheden. Det kunne tage lidt mere eller mindre tid for dig afhængigt af din enheds CPU-hastighed, RAM, om du har en SSD eller ej, og nogle andre faktorer. Den relativt korte varighed af operationen betyder, at hvis det forbedrer sentimentanalysens tid, så er det værd at gøre.

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

### Udførelse af sentimentanalyse

Nu skal du beregne sentimentanalysen for både negative og positive anmeldelseskolonner og gemme resultatet i 2 nye kolonner. Testen af sentimenten vil være at sammenligne den med anmelderens score for den samme anmeldelse. For eksempel, hvis sentimenten vurderer, at den negative anmeldelse havde en sentiment på 1 (ekstremt positiv sentiment) og en positiv anmeldelse sentiment på 1, men anmelderen gav hotellet den laveste score muligt, så matcher anmeldelsesteksten ikke scoren, eller sentimentanalysatoren kunne ikke genkende sentimenten korrekt. Du bør forvente, at nogle sentimentscorer er helt forkerte, og ofte vil det være forklarligt, f.eks. anmeldelsen kunne være ekstremt sarkastisk "Selvfølgelig ELSKEDE jeg at sove i et rum uden varme", og sentimentanalysatoren tror, det er positiv sentiment, selvom et menneske, der læser det, ville vide, det var sarkasme.
NLTK leverer forskellige sentiment-analyzatorer, som du kan lære med, og du kan udskifte dem og se, om sentimentet er mere eller mindre præcist. VADER sentimentanalyse bruges her.

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

Senere i dit program, når du er klar til at beregne sentiment, kan du anvende det på hver anmeldelse som følger:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Dette tager cirka 120 sekunder på min computer, men det vil variere fra computer til computer. Hvis du vil udskrive resultaterne og se, om sentimentet matcher anmeldelsen:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Det allersidste, du skal gøre med filen, før du bruger den i udfordringen, er at gemme den! Du bør også overveje at omorganisere alle dine nye kolonner, så de er nemme at arbejde med (for et menneske, det er en kosmetisk ændring).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Du bør køre hele koden for [analysenotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (efter du har kørt [din filtreringsnotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) for at generere filen Hotel_Reviews_Filtered.csv).

For at opsummere, trinnene er:

1. Den originale datasetfil **Hotel_Reviews.csv** udforskes i den forrige lektion med [explorer-notebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv filtreres af [filtreringsnotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), hvilket resulterer i **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv behandles af [sentimentanalyse-notebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), hvilket resulterer i **Hotel_Reviews_NLP.csv**
4. Brug Hotel_Reviews_NLP.csv i NLP-udfordringen nedenfor

### Konklusion

Da du startede, havde du et datasæt med kolonner og data, men ikke alt kunne verificeres eller bruges. Du har udforsket dataene, filtreret det, du ikke har brug for, konverteret tags til noget nyttigt, beregnet dine egne gennemsnit, tilføjet nogle sentimentkolonner og forhåbentlig lært nogle interessante ting om behandling af naturlig tekst.

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Udfordring

Nu hvor du har analyseret dit datasæt for sentiment, se om du kan bruge strategier, du har lært i dette pensum (måske clustering?) til at identificere mønstre omkring sentiment.

## Gennemgang & Selvstudie

Tag [dette Learn-modul](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) for at lære mere og bruge forskellige værktøjer til at udforske sentiment i tekst.

## Opgave

[Prøv et andet datasæt](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.