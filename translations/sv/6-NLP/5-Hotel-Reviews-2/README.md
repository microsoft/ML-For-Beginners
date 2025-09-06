<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T22:27:31+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "sv"
}
-->
# Sentimentanalys med hotellrecensioner

Nu när du har utforskat datasetet i detalj är det dags att filtrera kolumnerna och använda NLP-tekniker på datasetet för att få nya insikter om hotellen.

## [Quiz före föreläsning](https://ff-quizzes.netlify.app/en/ml/)

### Filtrering och sentimentanalysoperationer

Som du förmodligen har märkt har datasetet några problem. Vissa kolumner är fyllda med värdelös information, andra verkar felaktiga. Om de är korrekta är det oklart hur de beräknades, och svaren kan inte verifieras självständigt med dina egna beräkningar.

## Övning: lite mer databehandling

Rensa data lite mer. Lägg till kolumner som kommer att vara användbara senare, ändra värden i andra kolumner och ta bort vissa kolumner helt.

1. Inledande kolumnbearbetning

   1. Ta bort `lat` och `lng`

   2. Ersätt värdena i `Hotel_Address` med följande värden (om adressen innehåller stadens och landets namn, ändra det till endast stad och land).

      Dessa är de enda städerna och länderna i datasetet:

      Amsterdam, Nederländerna

      Barcelona, Spanien

      London, Storbritannien

      Milano, Italien

      Paris, Frankrike

      Wien, Österrike 

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

      Nu kan du göra förfrågningar på landsnivå:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nederländerna |    105     |
      | Barcelona, Spanien       |    211     |
      | London, Storbritannien   |    400     |
      | Milano, Italien          |    162     |
      | Paris, Frankrike         |    458     |
      | Wien, Österrike          |    158     |

2. Bearbeta kolumner för hotellens metarecensioner

   1. Ta bort `Additional_Number_of_Scoring`

   2. Ersätt `Total_Number_of_Reviews` med det totala antalet recensioner för det hotellet som faktiskt finns i datasetet 

   3. Ersätt `Average_Score` med vår egen beräknade poäng

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Bearbeta recensionskolumner

   1. Ta bort `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` och `days_since_review`

   2. Behåll `Reviewer_Score`, `Negative_Review` och `Positive_Review` som de är
     
   3. Behåll `Tags` för tillfället

     - Vi kommer att göra ytterligare filtreringsoperationer på taggarna i nästa avsnitt och sedan kommer taggarna att tas bort

4. Bearbeta recensionsskrivarkolumner

   1. Ta bort `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Behåll `Reviewer_Nationality`

### Taggkolumner

Kolumnen `Tag` är problematisk eftersom den är en lista (i textform) som lagras i kolumnen. Tyvärr är ordningen och antalet underavsnitt i denna kolumn inte alltid densamma. Det är svårt för en människa att identifiera de korrekta fraserna att fokusera på, eftersom det finns 515 000 rader och 1427 hotell, och varje hotell har något olika alternativ som en recensent kan välja. Här är där NLP är användbart. Du kan skanna texten och hitta de vanligaste fraserna och räkna dem.

Tyvärr är vi inte intresserade av enskilda ord, utan flerordsfraser (t.ex. *Affärsresa*). Att köra en algoritm för flerordsfrekvensfördelning på så mycket data (6 762 646 ord) kan ta extremt lång tid, men utan att titta på data verkar det vara en nödvändig kostnad. Här är där utforskande dataanalys är användbart, eftersom du har sett ett urval av taggar som `[' Affärsresa  ', ' Ensamresenär ', ' Enkelrum ', ' Stannade 5 nätter ', ' Skickad från en mobil enhet ']`, kan du börja fråga dig om det är möjligt att kraftigt minska den bearbetning du måste göra. Lyckligtvis är det möjligt - men först måste du följa några steg för att fastställa vilka taggar som är av intresse.

### Filtrering av taggar

Kom ihåg att målet med datasetet är att lägga till sentiment och kolumner som hjälper dig att välja det bästa hotellet (för dig själv eller kanske en klient som ber dig skapa en hotellrekommendationsbot). Du måste fråga dig själv om taggarna är användbara eller inte i det slutliga datasetet. Här är en tolkning (om du behövde datasetet för andra ändamål kan olika taggar inkluderas/uteslutas):

1. Typen av resa är relevant och bör behållas
2. Typen av gästgrupp är viktig och bör behållas
3. Typen av rum, svit eller studio som gästen bodde i är irrelevant (alla hotell har i princip samma rum)
4. Enheten som recensionen skickades från är irrelevant
5. Antalet nätter som recensenten stannade *kan* vara relevant om du kopplar längre vistelser till att de gillade hotellet mer, men det är tveksamt och förmodligen irrelevant

Sammanfattningsvis, **behåll 2 typer av taggar och ta bort de andra**.

Först vill du inte räkna taggarna förrän de är i ett bättre format, vilket innebär att ta bort hakparenteser och citattecken. Du kan göra detta på flera sätt, men du vill ha det snabbaste eftersom det kan ta lång tid att bearbeta mycket data. Lyckligtvis har pandas ett enkelt sätt att utföra varje steg.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Varje tagg blir något som: `Affärsresa, Ensamresenär, Enkelrum, Stannade 5 nätter, Skickad från en mobil enhet`. 

Nästa problem är att vissa recensioner, eller rader, har 5 kolumner, vissa 3, vissa 6. Detta är ett resultat av hur datasetet skapades och svårt att åtgärda. Du vill få en frekvensräkning av varje fras, men de är i olika ordning i varje recension, så räkningen kan bli felaktig och ett hotell kanske inte får en tagg tilldelad som det förtjänade.

Istället kommer du att använda den olika ordningen till vår fördel, eftersom varje tagg är flerordig men också separerad med ett komma! Det enklaste sättet att göra detta är att skapa 6 temporära kolumner med varje tagg insatt i kolumnen som motsvarar dess ordning i taggen. Du kan sedan slå samman de 6 kolumnerna till en stor kolumn och köra metoden `value_counts()` på den resulterande kolumnen. När du skriver ut det ser du att det fanns 2428 unika taggar. Här är ett litet urval:

| Tagg                           | Antal  |
| ------------------------------ | ------ |
| Fritidsresa                   | 417778 |
| Skickad från en mobil enhet   | 307640 |
| Par                           | 252294 |
| Stannade 1 natt               | 193645 |
| Stannade 2 nätter             | 133937 |
| Ensamresenär                  | 108545 |
| Stannade 3 nätter             | 95821  |
| Affärsresa                    | 82939  |
| Grupp                         | 65392  |
| Familj med små barn           | 61015  |
| Stannade 4 nätter             | 47817  |
| Dubbelrum                     | 35207  |
| Standard dubbelrum            | 32248  |
| Superior dubbelrum            | 31393  |
| Familj med äldre barn         | 26349  |
| Deluxe dubbelrum              | 24823  |
| Dubbel- eller tvåbäddsrum     | 22393  |
| Stannade 5 nätter             | 20845  |
| Standard dubbel- eller tvåbäddsrum | 17483  |
| Klassiskt dubbelrum           | 16989  |
| Superior dubbel- eller tvåbäddsrum | 13570 |
| 2 rum                         | 12393  |

Vissa av de vanliga taggarna som `Skickad från en mobil enhet` är inte användbara för oss, så det kan vara smart att ta bort dem innan man räknar frasförekomster, men det är en så snabb operation att du kan lämna dem kvar och ignorera dem.

### Ta bort taggar för vistelselängd

Att ta bort dessa taggar är steg 1, det minskar det totala antalet taggar som ska beaktas något. Observera att du inte tar bort dem från datasetet, utan bara väljer att inte beakta dem som värden att räkna/behålla i recensionsdatasetet.

| Vistelselängd | Antal  |
| -------------- | ------ |
| Stannade 1 natt | 193645 |
| Stannade 2 nätter | 133937 |
| Stannade 3 nätter | 95821  |
| Stannade 4 nätter | 47817  |
| Stannade 5 nätter | 20845  |
| Stannade 6 nätter | 9776   |
| Stannade 7 nätter | 7399   |
| Stannade 8 nätter | 2502   |
| Stannade 9 nätter | 1293   |
| ...              | ...    |

Det finns en stor variation av rum, sviter, studios, lägenheter och så vidare. De betyder alla ungefär samma sak och är inte relevanta för dig, så ta bort dem från övervägande.

| Rumstyp                     | Antal |
| --------------------------- | ----- |
| Dubbelrum                   | 35207 |
| Standard dubbelrum          | 32248 |
| Superior dubbelrum          | 31393 |
| Deluxe dubbelrum            | 24823 |
| Dubbel- eller tvåbäddsrum   | 22393 |
| Standard dubbel- eller tvåbäddsrum | 17483 |
| Klassiskt dubbelrum         | 16989 |
| Superior dubbel- eller tvåbäddsrum | 13570 |

Slutligen, och detta är glädjande (eftersom det inte krävde mycket bearbetning alls), kommer du att ha kvar följande *användbara* taggar:

| Tagg                                         | Antal  |
| ------------------------------------------- | ------ |
| Fritidsresa                                 | 417778 |
| Par                                         | 252294 |
| Ensamresenär                                | 108545 |
| Affärsresa                                  | 82939  |
| Grupp (kombinerad med Resenärer med vänner) | 67535  |
| Familj med små barn                         | 61015  |
| Familj med äldre barn                       | 26349  |
| Med ett husdjur                             | 1405   |

Man skulle kunna argumentera att `Resenärer med vänner` är samma sak som `Grupp` mer eller mindre, och det skulle vara rimligt att kombinera de två som ovan. Koden för att identifiera de korrekta taggarna finns i [Tags-notebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Det sista steget är att skapa nya kolumner för var och en av dessa taggar. Sedan, för varje recensionsrad, om kolumnen `Tag` matchar en av de nya kolumnerna, lägg till en 1, om inte, lägg till en 0. Slutresultatet blir en räkning av hur många recensenter som valde detta hotell (i aggregat) för exempelvis affärer kontra fritid, eller för att ta med ett husdjur, och detta är användbar information när man rekommenderar ett hotell.

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

### Spara din fil

Slutligen, spara datasetet som det är nu med ett nytt namn.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentanalysoperationer

I detta sista avsnitt kommer du att tillämpa sentimentanalys på recensionskolumnerna och spara resultaten i ett dataset.

## Övning: ladda och spara den filtrerade datan

Observera att du nu laddar det filtrerade datasetet som sparades i föregående avsnitt, **inte** det ursprungliga datasetet.

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

### Ta bort stoppord

Om du skulle köra sentimentanalys på kolumnerna för negativa och positiva recensioner kan det ta lång tid. Testat på en kraftfull testdator med snabb CPU tog det 12–14 minuter beroende på vilken sentimentbibliotek som användes. Det är en (relativt) lång tid, så det är värt att undersöka om det kan snabbas upp. 

Att ta bort stoppord, eller vanliga engelska ord som inte förändrar sentimentet i en mening, är det första steget. Genom att ta bort dem bör sentimentanalysen gå snabbare, men inte bli mindre exakt (eftersom stopporden inte påverkar sentimentet, men de saktar ner analysen). 

Den längsta negativa recensionen var 395 ord, men efter att ha tagit bort stopporden är den 195 ord.

Att ta bort stopporden är också en snabb operation, att ta bort stopporden från 2 recensionskolumner över 515 000 rader tog 3,3 sekunder på testdatorn. Det kan ta något mer eller mindre tid för dig beroende på din dators CPU-hastighet, RAM, om du har en SSD eller inte, och några andra faktorer. Den relativt korta operationen innebär att om den förbättrar sentimentanalystiden, så är det värt att göra.

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

### Utföra sentimentanalys

Nu ska du beräkna sentimentanalysen för både negativa och positiva recensionskolumner och lagra resultatet i 2 nya kolumner. Testet av sentimentet kommer att vara att jämföra det med recensentens poäng för samma recension. Till exempel, om sentimentet anser att den negativa recensionen hade ett sentiment på 1 (extremt positivt sentiment) och ett positivt recensionssentiment på 1, men recensenten gav hotellet den lägsta möjliga poängen, då matchar antingen recensionsinnehållet inte poängen, eller så kunde sentimentanalysverktyget inte känna igen sentimentet korrekt. Du bör förvänta dig att vissa sentimentpoäng är helt felaktiga, och ofta kommer det att vara förklarligt, t.ex. recensionen kan vara extremt sarkastisk "Självklart ÄLSKADE jag att sova i ett rum utan värme" och sentimentanalysverktyget tror att det är positivt sentiment, även om en människa som läser det skulle förstå att det var sarkasm.
NLTK tillhandahåller olika sentimentanalysverktyg att lära sig med, och du kan byta ut dem och se om sentimentanalysen blir mer eller mindre exakt. Här används VADER för sentimentanalys.

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

Senare i ditt program, när du är redo att beräkna sentiment, kan du tillämpa det på varje recension enligt följande:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Detta tar ungefär 120 sekunder på min dator, men det kan variera beroende på dator. Om du vill skriva ut resultaten och se om sentimentet stämmer överens med recensionen:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Det allra sista du behöver göra med filen innan du använder den i utmaningen är att spara den! Du bör också överväga att omorganisera alla dina nya kolumner så att de blir enkla att arbeta med (för en människa, det är en kosmetisk förändring).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Du bör köra hela koden för [analysnotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (efter att du har kört [din filtreringsnotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) för att generera filen Hotel_Reviews_Filtered.csv).

För att sammanfatta, stegen är:

1. Den ursprungliga datasetfilen **Hotel_Reviews.csv** utforskas i den föregående lektionen med [utforskningsnotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv filtreras med hjälp av [filtreringsnotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) och resulterar i **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv bearbetas med [sentimentanalysnotebooken](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) och resulterar i **Hotel_Reviews_NLP.csv**
4. Använd Hotel_Reviews_NLP.csv i NLP-utmaningen nedan

### Slutsats

När du började hade du ett dataset med kolumner och data, men inte allt kunde verifieras eller användas. Du har utforskat datan, filtrerat bort det du inte behöver, konverterat taggar till något användbart, beräknat dina egna medelvärden, lagt till några sentimentkolumner och förhoppningsvis lärt dig några intressanta saker om bearbetning av naturlig text.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Utmaning

Nu när du har analyserat ditt dataset för sentiment, se om du kan använda strategier du lärt dig i denna kurs (klustring, kanske?) för att identifiera mönster kring sentiment.

## Granskning och självstudier

Ta [denna Learn-modul](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) för att lära dig mer och använda olika verktyg för att utforska sentiment i text.

## Uppgift

[Prova ett annat dataset](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.