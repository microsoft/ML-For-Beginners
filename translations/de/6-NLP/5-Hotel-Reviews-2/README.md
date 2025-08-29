# Sentiment-Analyse mit Hotelbewertungen

Jetzt, wo Sie den Datensatz im Detail erkundet haben, ist es an der Zeit, die Spalten zu filtern und dann NLP-Techniken auf den Datensatz anzuwenden, um neue Erkenntnisse über die Hotels zu gewinnen.
## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Filter- und Sentiment-Analyse-Operationen

Wie Sie wahrscheinlich bemerkt haben, weist der Datensatz einige Probleme auf. Einige Spalten sind mit nutzlosen Informationen gefüllt, andere scheinen inkorrekt zu sein. Wenn sie korrekt sind, ist unklar, wie sie berechnet wurden, und die Antworten können nicht unabhängig durch eigene Berechnungen verifiziert werden.

## Übung: etwas mehr Datenverarbeitung

Bereinigen Sie die Daten ein wenig mehr. Fügen Sie Spalten hinzu, die später nützlich sein werden, ändern Sie die Werte in anderen Spalten und entfernen Sie bestimmte Spalten vollständig.

1. Erste Spaltenverarbeitung

   1. Entfernen Sie `lat` und `lng`

   2. Ersetzen Sie die Werte von `Hotel_Address` durch die folgenden Werte (wenn die Adresse sowohl die Stadt als auch das Land enthält, ändern Sie sie in nur die Stadt und das Land).

      Dies sind die einzigen Städte und Länder im Datensatz:

      Amsterdam, Niederlande

      Barcelona, Spanien

      London, Vereinigtes Königreich

      Mailand, Italien

      Paris, Frankreich

      Wien, Österreich 

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

      Jetzt können Sie länderspezifische Daten abfragen:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Niederlande |    105     |
      | Barcelona, Spanien       |    211     |
      | London, Vereinigtes Königreich |    400     |
      | Mailand, Italien           |    162     |
      | Paris, Frankreich          |    458     |
      | Wien, Österreich          |    158     |

2. Verarbeiten Sie die Hotel-Meta-Bewertungs-Spalten

  1. Entfernen Sie `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` mit unserem eigenen berechneten Wert

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Verarbeiten Sie die Bewertungs-Spalten

   1. Entfernen Sie `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` , Sie können beginnen zu fragen, ob es möglich ist, die Verarbeitung, die Sie durchführen müssen, erheblich zu reduzieren. Glücklicherweise ist das der Fall - aber zuerst müssen Sie einige Schritte befolgen, um die interessanten Tags zu ermitteln.

### Tags filtern

Denken Sie daran, dass das Ziel des Datensatzes darin besteht, Sentiment und Spalten hinzuzufügen, die Ihnen helfen, das beste Hotel auszuwählen (für sich selbst oder vielleicht für einen Kunden, der Sie beauftragt, einen Hotelempfehlungsbot zu erstellen). Sie müssen sich fragen, ob die Tags im endgültigen Datensatz nützlich sind oder nicht. Hier ist eine Interpretation (wenn Sie den Datensatz aus anderen Gründen benötigten, könnten andere Tags in der Auswahl bleiben oder nicht):

1. Die Art der Reise ist relevant und sollte bleiben
2. Die Art der Gästegruppe ist wichtig und sollte bleiben
3. Die Art des Zimmers, der Suite oder des Studios, in dem der Gast übernachtet hat, ist irrelevant (alle Hotels haben im Grunde die gleichen Zimmer)
4. Das Gerät, auf dem die Bewertung eingereicht wurde, ist irrelevant
5. Die Anzahl der Nächte, die der Rezensent geblieben ist, *könnte* relevant sein, wenn Sie längere Aufenthalte damit in Verbindung bringen, dass sie das Hotel mehr mögen, aber das ist eine Dehnung und wahrscheinlich irrelevant

Zusammenfassend lässt sich sagen, **behalten Sie 2 Arten von Tags und entfernen Sie die anderen**.

Zunächst möchten Sie die Tags nicht zählen, bis sie in einem besseren Format vorliegen, das bedeutet, dass Sie die eckigen Klammern und Anführungszeichen entfernen müssen. Sie können dies auf verschiedene Weise tun, aber Sie möchten die schnellste Methode, da es lange dauern könnte, eine große Menge an Daten zu verarbeiten. Glücklicherweise hat pandas eine einfache Möglichkeit, jeden dieser Schritte durchzuführen.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Jeder Tag wird zu etwas wie: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` Spalte, die mit einer der neuen Spalten übereinstimmt, fügen Sie eine 1 hinzu, wenn nicht, fügen Sie eine 0 hinzu. Das Endergebnis wird eine Zählung sein, wie viele Rezensenten dieses Hotel (insgesamt) für beispielsweise geschäftliche Zwecke oder zur Freizeit gewählt haben, und dies ist nützliche Information bei der Hotelempfehlung.

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

### Speichern Sie Ihre Datei

Speichern Sie schließlich den Datensatz in seinem aktuellen Zustand unter einem neuen Namen.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentiment-Analyse-Operationen

In diesem letzten Abschnitt wenden Sie die Sentiment-Analyse auf die Bewertungs-Spalten an und speichern die Ergebnisse in einem Datensatz.

## Übung: Laden und Speichern der gefilterten Daten

Bitte beachten Sie, dass Sie jetzt den gefilterten Datensatz laden, der im vorherigen Abschnitt gespeichert wurde, **nicht** den ursprünglichen Datensatz.

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

### Entfernen von Stoppwörtern

Wenn Sie Sentiment-Analyse auf den negativen und positiven Bewertungs-Spalten durchführen würden, könnte das lange dauern. Getestet auf einem leistungsstarken Test-Laptop mit schnellem CPU dauerte es 12 bis 14 Minuten, abhängig davon, welche Sentiment-Bibliothek verwendet wurde. Das ist eine (relativ) lange Zeit, also ist es wert, zu untersuchen, ob das beschleunigt werden kann.

Das Entfernen von Stoppwörtern, oder gängigen englischen Wörtern, die die Stimmung eines Satzes nicht verändern, ist der erste Schritt. Durch das Entfernen dieser Wörter sollte die Sentiment-Analyse schneller laufen, ohne weniger genau zu sein (da die Stoppwörter die Stimmung nicht beeinflussen, aber die Analyse verlangsamen).

Die längste negative Bewertung hatte 395 Wörter, aber nach dem Entfernen der Stoppwörter sind es 195 Wörter.

Das Entfernen der Stoppwörter ist auch eine schnelle Operation; das Entfernen der Stoppwörter aus 2 Bewertungs-Spalten über 515.000 Zeilen dauerte auf dem Testgerät 3,3 Sekunden. Es könnte für Sie je nach CPU-Geschwindigkeit Ihres Geräts, RAM, ob Sie eine SSD haben oder nicht, und einigen anderen Faktoren etwas mehr oder weniger Zeit in Anspruch nehmen. Die relative Kürze der Operation bedeutet, dass es sich lohnt, wenn es die Zeit der Sentiment-Analyse verbessert.

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

### Durchführung der Sentiment-Analyse

Jetzt sollten Sie die Sentiment-Analyse für sowohl negative als auch positive Bewertungs-Spalten berechnen und das Ergebnis in 2 neuen Spalten speichern. Der Test des Sentiments wird darin bestehen, es mit dem Score des Rezensenten für dieselbe Bewertung zu vergleichen. Zum Beispiel, wenn das Sentiment denkt, dass die negative Bewertung ein Sentiment von 1 (extrem positives Sentiment) hatte und das positive Bewertungs-Sentiment ebenfalls 1, aber der Rezensent dem Hotel die niedrigste mögliche Punktzahl gegeben hat, dann passt entweder der Bewertungstext nicht zur Punktzahl, oder der Sentiment-Analysator konnte das Sentiment nicht korrekt erkennen. Sie sollten erwarten, dass einige Sentiment-Punkte völlig falsch sind, und oft wird das erklärbar sein, z.B. könnte die Bewertung extrem sarkastisch sein: "Natürlich habe ich es GELIEBT, in einem Zimmer ohne Heizung zu schlafen" und der Sentiment-Analysator denkt, dass das positives Sentiment ist, obwohl ein Mensch, der es liest, wüsste, dass es Sarkasmus war.

NLTK bietet verschiedene Sentiment-Analysatoren zum Lernen an, und Sie können diese austauschen und sehen, ob das Sentiment genauer oder weniger genau ist. Hier wird die VADER-Sentiment-Analyse verwendet.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, Juni 2014.

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

Später in Ihrem Programm, wenn Sie bereit sind, das Sentiment zu berechnen, können Sie es wie folgt auf jede Bewertung anwenden:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Das dauert auf meinem Computer ungefähr 120 Sekunden, kann aber auf jedem Computer variieren. Wenn Sie die Ergebnisse drucken und sehen möchten, ob das Sentiment mit der Bewertung übereinstimmt:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Das letzte, was Sie mit der Datei tun müssen, bevor Sie sie in der Herausforderung verwenden, ist, sie zu speichern! Sie sollten auch in Betracht ziehen, alle Ihre neuen Spalten neu anzuordnen, damit sie einfach zu bearbeiten sind (für einen Menschen ist das eine kosmetische Änderung).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Sie sollten den gesamten Code für [das Analyse-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ausführen (nachdem Sie [Ihr Filter-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ausgeführt haben, um die Datei Hotel_Reviews_Filtered.csv zu generieren).

Zusammenfassend sind die Schritte:

1. Die ursprüngliche Datensatzdatei **Hotel_Reviews.csv** wurde in der vorherigen Lektion mit [dem Explorer-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) erkundet.
2. Hotel_Reviews.csv wird durch [das Filter-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) gefiltert, was zu **Hotel_Reviews_Filtered.csv** führt.
3. Hotel_Reviews_Filtered.csv wird durch [das Sentiment-Analyse-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) verarbeitet, was zu **Hotel_Reviews_NLP.csv** führt.
4. Verwenden Sie Hotel_Reviews_NLP.csv in der untenstehenden NLP-Herausforderung.

### Fazit

Als Sie begonnen haben, hatten Sie einen Datensatz mit Spalten und Daten, aber nicht alles konnte verifiziert oder verwendet werden. Sie haben die Daten erkundet, was Sie nicht benötigen, herausgefiltert, Tags in etwas Nützliches umgewandelt, Ihre eigenen Durchschnitte berechnet, einige Sentiment-Spalten hinzugefügt und hoffentlich interessante Dinge über die Verarbeitung natürlicher Texte gelernt.

## [Nachlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Herausforderung

Jetzt, wo Sie Ihren Datensatz auf Sentiment analysiert haben, sehen Sie, ob Sie Strategien, die Sie in diesem Lehrgang gelernt haben (vielleicht Clustering?), verwenden können, um Muster im Zusammenhang mit Sentiment zu bestimmen.

## Überprüfung & Selbststudium

Nehmen Sie [dieses Lernmodul](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) in Anspruch, um mehr zu lernen und verschiedene Tools zu verwenden, um Sentiment in Texten zu erkunden.
## Aufgabe 

[Versuchen Sie einen anderen Datensatz](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als autoritative Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung resultieren.