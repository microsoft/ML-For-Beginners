<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T20:43:13+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "nl"
}
-->
# Sentimentanalyse met hotelbeoordelingen

Nu je de dataset in detail hebt verkend, is het tijd om de kolommen te filteren en vervolgens NLP-technieken toe te passen op de dataset om nieuwe inzichten over de hotels te verkrijgen.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Filteren & Sentimentanalyse-operaties

Zoals je waarschijnlijk hebt gemerkt, heeft de dataset een paar problemen. Sommige kolommen bevatten nutteloze informatie, andere lijken incorrect. Als ze correct zijn, is het onduidelijk hoe ze zijn berekend, en antwoorden kunnen niet onafhankelijk worden geverifieerd door je eigen berekeningen.

## Oefening: een beetje meer gegevensverwerking

Maak de gegevens nog wat schoner. Voeg kolommen toe die later nuttig zullen zijn, wijzig de waarden in andere kolommen en verwijder bepaalde kolommen volledig.

1. Eerste verwerking van kolommen

   1. Verwijder `lat` en `lng`

   2. Vervang de waarden in `Hotel_Address` door de volgende waarden (als het adres de naam van de stad en het land bevat, verander het dan naar alleen de stad en het land).

      Dit zijn de enige steden en landen in de dataset:

      Amsterdam, Nederland

      Barcelona, Spanje

      Londen, Verenigd Koninkrijk

      Milaan, Italië

      Parijs, Frankrijk

      Wenen, Oostenrijk 

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

      Nu kun je gegevens op landniveau opvragen:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nederland   |    105     |
      | Barcelona, Spanje      |    211     |
      | Londen, Verenigd Koninkrijk | 400   |
      | Milaan, Italië         |    162     |
      | Parijs, Frankrijk      |    458     |
      | Wenen, Oostenrijk      |    158     |

2. Verwerk Hotel Meta-review kolommen

  1. Verwijder `Additional_Number_of_Scoring`

  1. Vervang `Total_Number_of_Reviews` door het totale aantal beoordelingen voor dat hotel dat daadwerkelijk in de dataset staat 

  1. Vervang `Average_Score` door onze eigen berekende score

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Verwerk reviewkolommen

   1. Verwijder `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` en `days_since_review`

   2. Behoud `Reviewer_Score`, `Negative_Review` en `Positive_Review` zoals ze zijn,
     
   3. Behoud `Tags` voorlopig

     - We zullen in de volgende sectie enkele aanvullende filterbewerkingen uitvoeren op de tags en daarna worden de tags verwijderd

4. Verwerk reviewer kolommen

  1. Verwijder `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Behoud `Reviewer_Nationality`

### Tag-kolommen

De `Tag`-kolom is problematisch omdat het een lijst (in tekstvorm) is die in de kolom is opgeslagen. Helaas is de volgorde en het aantal subsecties in deze kolom niet altijd hetzelfde. Het is moeilijk voor een mens om de juiste zinnen te identificeren die interessant zijn, omdat er 515.000 rijen en 1427 hotels zijn, en elk heeft iets andere opties die een beoordelaar zou kunnen kiezen. Hier komt NLP goed van pas. Je kunt de tekst scannen en de meest voorkomende zinnen vinden en tellen.

Helaas zijn we niet geïnteresseerd in losse woorden, maar in meerwoordige zinnen (bijv. *Zakenreis*). Het uitvoeren van een frequentieverdelingsalgoritme voor meerwoordige zinnen op zoveel gegevens (6762646 woorden) kan buitengewoon veel tijd kosten, maar zonder naar de gegevens te kijken, lijkt dat een noodzakelijke uitgave. Hier komt verkennende data-analyse van pas, omdat je een voorbeeld van de tags hebt gezien, zoals `[' Zakenreis  ', ' Soloreiziger ', ' Eenpersoonskamer ', ' Verbleef 5 nachten ', ' Ingediend vanaf een mobiel apparaat ']`, kun je beginnen te vragen of het mogelijk is om de verwerking die je moet doen aanzienlijk te verminderen. Gelukkig is dat zo - maar eerst moet je een paar stappen volgen om de interessante tags vast te stellen.

### Tags filteren

Onthoud dat het doel van de dataset is om sentiment en kolommen toe te voegen die je helpen het beste hotel te kiezen (voor jezelf of misschien een klant die je vraagt een hotelaanbevelingsbot te maken). Je moet jezelf afvragen of de tags nuttig zijn of niet in de uiteindelijke dataset. Hier is een interpretatie (als je de dataset om andere redenen nodig had, zouden verschillende tags in/uit de selectie kunnen blijven):

1. Het type reis is relevant en moet blijven
2. Het type gastgroep is belangrijk en moet blijven
3. Het type kamer, suite of studio waarin de gast verbleef is irrelevant (alle hotels hebben in principe dezelfde kamers)
4. Het apparaat waarop de beoordeling is ingediend is irrelevant
5. Het aantal nachten dat de beoordelaar verbleef *zou* relevant kunnen zijn als je langere verblijven associeert met het meer waarderen van het hotel, maar dat is twijfelachtig en waarschijnlijk irrelevant

Samenvattend, **behoud 2 soorten tags en verwijder de andere**.

Eerst wil je de tags niet tellen totdat ze in een beter formaat zijn, dus dat betekent dat je de vierkante haken en aanhalingstekens verwijdert. Je kunt dit op verschillende manieren doen, maar je wilt de snelste manier omdat het lang kan duren om veel gegevens te verwerken. Gelukkig heeft pandas een eenvoudige manier om elk van deze stappen uit te voeren.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Elke tag wordt iets als: `Zakenreis, Soloreiziger, Eenpersoonskamer, Verbleef 5 nachten, Ingediend vanaf een mobiel apparaat`. 

Vervolgens vinden we een probleem. Sommige beoordelingen, of rijen, hebben 5 kolommen, sommige 3, sommige 6. Dit is een resultaat van hoe de dataset is gemaakt en moeilijk te corrigeren. Je wilt een frequentietelling krijgen van elke zin, maar ze staan in verschillende volgorde in elke beoordeling, dus de telling kan verkeerd zijn en een hotel krijgt mogelijk geen tag toegewezen die het verdiende.

In plaats daarvan gebruik je de verschillende volgorde in ons voordeel, omdat elke tag meerwoordig is maar ook gescheiden door een komma! De eenvoudigste manier om dit te doen is om 6 tijdelijke kolommen te maken met elke tag ingevoegd in de kolom die overeenkomt met de volgorde in de tag. Je kunt vervolgens de 6 kolommen samenvoegen tot één grote kolom en de `value_counts()`-methode uitvoeren op de resulterende kolom. Als je dat afdrukt, zie je dat er 2428 unieke tags waren. Hier is een kleine steekproef:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Vrijetijdsreis                | 417778 |
| Ingediend vanaf een mobiel apparaat | 307640 |
| Koppel                         | 252294 |
| Verbleef 1 nacht               | 193645 |
| Verbleef 2 nachten             | 133937 |
| Soloreiziger                   | 108545 |
| Verbleef 3 nachten             | 95821  |
| Zakenreis                      | 82939  |
| Groep                          | 65392  |
| Gezin met jonge kinderen       | 61015  |
| Verbleef 4 nachten             | 47817  |
| Tweepersoonskamer              | 35207  |
| Standaard tweepersoonskamer    | 32248  |
| Superior tweepersoonskamer     | 31393  |
| Gezin met oudere kinderen      | 26349  |
| Deluxe tweepersoonskamer       | 24823  |
| Tweepersoons- of twin-kamer    | 22393  |
| Verbleef 5 nachten             | 20845  |
| Standaard tweepersoons- of twin-kamer | 17483  |
| Klassieke tweepersoonskamer    | 16989  |
| Superior tweepersoons- of twin-kamer | 13570 |
| 2 kamers                       | 12393  |

Sommige van de veelvoorkomende tags zoals `Ingediend vanaf een mobiel apparaat` zijn voor ons niet nuttig, dus het kan slim zijn om ze te verwijderen voordat je de zinnen gaat tellen, maar het is zo'n snelle operatie dat je ze erin kunt laten en negeren.

### Verwijderen van tags over verblijfsduur

Het verwijderen van deze tags is stap 1, het vermindert het totale aantal tags dat moet worden overwogen enigszins. Merk op dat je ze niet uit de dataset verwijdert, maar ervoor kiest om ze niet mee te nemen als waarden om te tellen/behouden in de beoordelingsdataset.

| Verblijfsduur   | Count  |
| ---------------- | ------ |
| Verbleef 1 nacht | 193645 |
| Verbleef 2 nachten | 133937 |
| Verbleef 3 nachten | 95821  |
| Verbleef 4 nachten | 47817  |
| Verbleef 5 nachten | 20845  |
| Verbleef 6 nachten | 9776   |
| Verbleef 7 nachten | 7399   |
| Verbleef 8 nachten | 2502   |
| Verbleef 9 nachten | 1293   |
| ...              | ...    |

Er is een enorme variëteit aan kamers, suites, studio's, appartementen enzovoort. Ze betekenen allemaal ongeveer hetzelfde en zijn niet relevant voor jou, dus verwijder ze uit de overweging.

| Type kamer                  | Count |
| ----------------------------- | ----- |
| Tweepersoonskamer            | 35207 |
| Standaard tweepersoonskamer  | 32248 |
| Superior tweepersoonskamer   | 31393 |
| Deluxe tweepersoonskamer     | 24823 |
| Tweepersoons- of twin-kamer  | 22393 |
| Standaard tweepersoons- of twin-kamer | 17483 |
| Klassieke tweepersoonskamer  | 16989 |
| Superior tweepersoons- of twin-kamer | 13570 |

Ten slotte, en dit is geweldig (omdat het niet veel verwerking kostte), blijf je over met de volgende *nuttige* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Vrijetijdsreis                                | 417778 |
| Koppel                                        | 252294 |
| Soloreiziger                                  | 108545 |
| Zakenreis                                     | 82939  |
| Groep (gecombineerd met Reizigers met vrienden) | 67535  |
| Gezin met jonge kinderen                      | 61015  |
| Gezin met oudere kinderen                     | 26349  |
| Met een huisdier                              | 1405   |

Je zou kunnen stellen dat `Reizigers met vrienden` min of meer hetzelfde is als `Groep`, en dat zou eerlijk zijn om de twee te combineren zoals hierboven. De code voor het identificeren van de juiste tags is [de Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

De laatste stap is om nieuwe kolommen te maken voor elk van deze tags. Vervolgens, voor elke beoordelingsrij, als de `Tag`-kolom overeenkomt met een van de nieuwe kolommen, voeg een 1 toe, zo niet, voeg een 0 toe. Het eindresultaat zal een telling zijn van hoeveel beoordelaars dit hotel (in totaal) kozen voor bijvoorbeeld zaken versus vrije tijd, of om een huisdier mee te nemen, en dit is nuttige informatie bij het aanbevelen van een hotel.

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

### Sla je bestand op

Sla ten slotte de dataset op zoals deze nu is met een nieuwe naam.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentimentanalyse-operaties

In deze laatste sectie ga je sentimentanalyse toepassen op de beoordelingskolommen en de resultaten opslaan in een dataset.

## Oefening: laad en sla de gefilterde gegevens op

Merk op dat je nu de gefilterde dataset laadt die in de vorige sectie is opgeslagen, **niet** de originele dataset.

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

### Stopwoorden verwijderen

Als je sentimentanalyse zou uitvoeren op de negatieve en positieve beoordelingskolommen, kan dat lang duren. Getest op een krachtige testlaptop met snelle CPU, duurde het 12 - 14 minuten, afhankelijk van welke sentimentbibliotheek werd gebruikt. Dat is een (relatief) lange tijd, dus het is de moeite waard om te onderzoeken of dat kan worden versneld. 

Stopwoorden verwijderen, of veelvoorkomende Engelse woorden die de sentiment van een zin niet veranderen, is de eerste stap. Door ze te verwijderen, zou de sentimentanalyse sneller moeten verlopen, maar niet minder nauwkeurig (aangezien de stopwoorden de sentiment niet beïnvloeden, maar ze vertragen wel de analyse). 

De langste negatieve beoordeling was 395 woorden, maar na het verwijderen van de stopwoorden zijn het er 195.

Het verwijderen van de stopwoorden is ook een snelle operatie, het verwijderen van de stopwoorden uit 2 beoordelingskolommen over 515.000 rijen duurde 3,3 seconden op het testapparaat. Het kan iets meer of minder tijd kosten voor jou, afhankelijk van de snelheid van je apparaat CPU, RAM, of je een SSD hebt of niet, en enkele andere factoren. De relatief korte duur van de operatie betekent dat als het de sentimentanalyse tijd verbetert, het de moeite waard is om te doen.

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

### Sentimentanalyse uitvoeren

Nu moet je de sentimentanalyse berekenen voor zowel negatieve als positieve beoordelingskolommen en het resultaat opslaan in 2 nieuwe kolommen. De test van de sentiment zal zijn om het te vergelijken met de score van de beoordelaar voor dezelfde beoordeling. Bijvoorbeeld, als de sentiment denkt dat de negatieve beoordeling een sentiment van 1 had (extreem positieve sentiment) en een positieve beoordeling sentiment van 1, maar de beoordelaar gaf het hotel de laagste score mogelijk, dan komt de tekst van de beoordeling niet overeen met de score, of kon de sentimentanalyse de sentiment niet correct herkennen. Je kunt verwachten dat sommige sentiment scores volledig verkeerd zijn, en vaak zal dat verklaarbaar zijn, bijvoorbeeld de beoordeling kan extreem sarcastisch zijn: "Natuurlijk VOND ik het GEWELDIG om te slapen in een kamer zonder verwarming" en de sentimentanalyse denkt dat dat positieve sentiment is, terwijl een mens die het leest zou weten dat het sarcasme is.
NLTK biedt verschillende sentimentanalysatoren om mee te experimenteren, en je kunt ze vervangen en zien of de sentimentanalyse meer of minder nauwkeurig is. Hier wordt de VADER-sentimentanalyse gebruikt.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Een zuinig regelgebaseerd model voor sentimentanalyse van sociale media-tekst. Achtste Internationale Conferentie over Weblogs en Sociale Media (ICWSM-14). Ann Arbor, MI, juni 2014.

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

Later in je programma, wanneer je klaar bent om sentiment te berekenen, kun je het toepassen op elke review als volgt:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Dit duurt ongeveer 120 seconden op mijn computer, maar het zal variëren per computer. Als je de resultaten wilt afdrukken en wilt zien of het sentiment overeenkomt met de review:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Het allerlaatste wat je met het bestand moet doen voordat je het in de uitdaging gebruikt, is het opslaan! Je zou ook moeten overwegen om al je nieuwe kolommen opnieuw te ordenen zodat ze gemakkelijk te gebruiken zijn (voor een mens, het is een cosmetische verandering).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Je moet de volledige code uitvoeren voor [het analysenotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (nadat je [je filternotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) hebt uitgevoerd om het bestand Hotel_Reviews_Filtered.csv te genereren).

Samenvattend, de stappen zijn:

1. Het originele datasetbestand **Hotel_Reviews.csv** wordt onderzocht in de vorige les met [het verkenningsnotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv wordt gefilterd door [het filternotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), wat resulteert in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv wordt verwerkt door [het sentimentanalysenotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), wat resulteert in **Hotel_Reviews_NLP.csv**
4. Gebruik Hotel_Reviews_NLP.csv in de NLP-uitdaging hieronder

### Conclusie

Toen je begon, had je een dataset met kolommen en gegevens, maar niet alles kon worden geverifieerd of gebruikt. Je hebt de gegevens verkend, gefilterd wat je niet nodig hebt, tags omgezet in iets bruikbaars, je eigen gemiddelden berekend, enkele sentimentkolommen toegevoegd en hopelijk interessante dingen geleerd over het verwerken van natuurlijke tekst.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Uitdaging

Nu je je dataset hebt geanalyseerd op sentiment, kijk of je strategieën kunt gebruiken die je in deze cursus hebt geleerd (clustering, misschien?) om patronen rond sentiment te bepalen.

## Review & Zelfstudie

Volg [deze Learn-module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) om meer te leren en verschillende tools te gebruiken om sentiment in tekst te verkennen.

## Opdracht

[Probeer een andere dataset](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.