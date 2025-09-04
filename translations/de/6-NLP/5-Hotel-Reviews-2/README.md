<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-03T22:04:18+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "de"
}
-->
# Sentimentanalyse mit Hotelbewertungen

Nachdem Sie den Datensatz im Detail untersucht haben, ist es an der Zeit, die Spalten zu filtern und dann NLP-Techniken auf den Datensatz anzuwenden, um neue Einblicke in die Hotels zu gewinnen.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Filter- und Sentimentanalyse-Operationen

Wie Sie wahrscheinlich bemerkt haben, weist der Datensatz einige Probleme auf. Einige Spalten enthalten nutzlose Informationen, andere scheinen fehlerhaft zu sein. Selbst wenn sie korrekt sind, ist unklar, wie sie berechnet wurden, und die Ergebnisse können nicht unabhängig durch eigene Berechnungen überprüft werden.

## Übung: Etwas mehr Datenverarbeitung

Bereinigen Sie die Daten noch ein wenig mehr. Fügen Sie Spalten hinzu, die später nützlich sein werden, ändern Sie die Werte in anderen Spalten und entfernen Sie bestimmte Spalten vollständig.

1. Erste Spaltenverarbeitung

   1. Entfernen Sie `lat` und `lng`.

   2. Ersetzen Sie die Werte in `Hotel_Address` durch die folgenden Werte (wenn die Adresse den Namen der Stadt und des Landes enthält, ändern Sie sie so, dass nur die Stadt und das Land angegeben sind).

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

      Jetzt können Sie Daten auf Länderebene abfragen:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Niederlande |    105     |
      | Barcelona, Spanien     |    211     |
      | London, Vereinigtes Königreich | 400 |
      | Mailand, Italien       |    162     |
      | Paris, Frankreich      |    458     |
      | Wien, Österreich       |    158     |

2. Verarbeitung der Hotel-Meta-Review-Spalten

   1. Entfernen Sie `Additional_Number_of_Scoring`.

   2. Ersetzen Sie `Total_Number_of_Reviews` durch die tatsächliche Anzahl der Bewertungen für dieses Hotel, die im Datensatz enthalten sind.

   3. Ersetzen Sie `Average_Score` durch einen selbst berechneten Durchschnittswert.

      ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Verarbeitung der Bewertungs-Spalten

   1. Entfernen Sie `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` und `days_since_review`.

   2. Behalten Sie `Reviewer_Score`, `Negative_Review` und `Positive_Review` bei.

   3. Behalten Sie `Tags` vorerst bei.

      - In der nächsten Sektion werden wir zusätzliche Filteroperationen auf die Tags anwenden, bevor sie entfernt werden.

4. Verarbeitung der Reviewer-Spalten

   1. Entfernen Sie `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Behalten Sie `Reviewer_Nationality`.

### Tag-Spalten

Die `Tag`-Spalte ist problematisch, da sie eine Liste (im Textformat) enthält, die in der Spalte gespeichert ist. Leider sind die Reihenfolge und die Anzahl der Unterabschnitte in dieser Spalte nicht immer gleich. Es ist für einen Menschen schwierig, die richtigen Phrasen zu identifizieren, die von Interesse sind, da es 515.000 Zeilen und 1427 Hotels gibt, und jede Bewertung leicht unterschiedliche Optionen bietet, die ein Bewerter auswählen könnte. Hier kommt NLP ins Spiel. Sie können den Text scannen, die häufigsten Phrasen finden und sie zählen.

Leider interessieren uns keine einzelnen Wörter, sondern mehrwortige Phrasen (z. B. *Geschäftsreise*). Das Ausführen eines Algorithmus zur Häufigkeitsverteilung von Mehrwortphrasen auf so vielen Daten (6.762.646 Wörter) könnte außergewöhnlich viel Zeit in Anspruch nehmen. Ohne die Daten anzusehen, scheint dies jedoch eine notwendige Maßnahme zu sein. Hier ist explorative Datenanalyse nützlich, denn nachdem Sie eine Stichprobe der Tags wie `[' Geschäftsreise ', ' Alleinreisender ', ' Einzelzimmer ', ' Aufenthalt 5 Nächte ', ' Übermittelt von einem mobilen Gerät ']` gesehen haben, können Sie sich fragen, ob es möglich ist, die Verarbeitung erheblich zu reduzieren. Glücklicherweise ist das möglich – aber zuerst müssen Sie einige Schritte befolgen, um die interessanten Tags zu ermitteln.

### Filtern der Tags

Denken Sie daran, dass das Ziel des Datensatzes darin besteht, Sentiment und Spalten hinzuzufügen, die Ihnen helfen, das beste Hotel auszuwählen (für sich selbst oder vielleicht für einen Kunden, der Sie beauftragt, einen Hotel-Empfehlungsbot zu erstellen). Sie müssen sich fragen, ob die Tags im endgültigen Datensatz nützlich sind oder nicht. Hier ist eine Interpretation (wenn Sie den Datensatz für andere Zwecke benötigen, könnten andere Tags in die Auswahl ein- oder ausgeschlossen werden):

1. Die Art der Reise ist relevant und sollte beibehalten werden.
2. Die Art der Gästegruppe ist wichtig und sollte beibehalten werden.
3. Die Art des Zimmers, der Suite oder des Studios, in dem der Gast übernachtet hat, ist irrelevant (alle Hotels haben im Grunde die gleichen Zimmer).
4. Das Gerät, mit dem die Bewertung übermittelt wurde, ist irrelevant.
5. Die Anzahl der Nächte, die der Bewerter geblieben ist, *könnte* relevant sein, wenn Sie längere Aufenthalte damit in Verbindung bringen, dass der Gast das Hotel mehr mochte, aber das ist unwahrscheinlich und wahrscheinlich irrelevant.

Zusammenfassend: **Behalten Sie 2 Arten von Tags und entfernen Sie die anderen.**

Zuerst möchten Sie die Tags nicht zählen, bis sie in einem besseren Format vorliegen. Das bedeutet, dass Sie die eckigen Klammern und Anführungszeichen entfernen müssen. Es gibt mehrere Möglichkeiten, dies zu tun, aber Sie möchten die schnellste Methode, da die Verarbeitung einer großen Datenmenge lange dauern könnte. Glücklicherweise bietet pandas eine einfache Möglichkeit, jeden dieser Schritte auszuführen.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Jedes Tag wird zu etwas wie: `Geschäftsreise, Alleinreisender, Einzelzimmer, Aufenthalt 5 Nächte, Übermittelt von einem mobilen Gerät`.

Als Nächstes stoßen wir auf ein Problem. Einige Bewertungen oder Zeilen haben 5 Spalten, andere 3, wieder andere 6. Dies ist ein Ergebnis der Art und Weise, wie der Datensatz erstellt wurde, und schwer zu beheben. Sie möchten eine Häufigkeitszählung jeder Phrase erhalten, aber sie sind in jeder Bewertung in unterschiedlicher Reihenfolge, sodass die Zählung möglicherweise ungenau ist und ein Hotel möglicherweise keinen Tag zugewiesen bekommt, den es verdient hätte.

Stattdessen nutzen Sie die unterschiedliche Reihenfolge zu Ihrem Vorteil, da jedes Tag mehrwörtig ist, aber auch durch ein Komma getrennt! Der einfachste Weg, dies zu tun, besteht darin, 6 temporäre Spalten zu erstellen, in die jedes Tag in die Spalte eingefügt wird, die seiner Reihenfolge im Tag entspricht. Sie können dann die 6 Spalten zu einer großen Spalte zusammenführen und die Methode `value_counts()` auf die resultierende Spalte anwenden. Wenn Sie das ausgeben, sehen Sie, dass es 2428 einzigartige Tags gab. Hier ist eine kleine Stichprobe:

| Tag                            | Anzahl  |
| ------------------------------ | ------- |
| Freizeitreise                  | 417778  |
| Übermittelt von einem mobilen Gerät | 307640 |
| Paar                           | 252294  |
| Aufenthalt 1 Nacht             | 193645  |
| Aufenthalt 2 Nächte            | 133937  |
| Alleinreisender                | 108545  |
| Aufenthalt 3 Nächte            | 95821   |
| Geschäftsreise                 | 82939   |
| Gruppe                         | 65392   |
| Familie mit kleinen Kindern    | 61015   |
| Aufenthalt 4 Nächte            | 47817   |
| Doppelzimmer                   | 35207   |
| Standard Doppelzimmer          | 32248   |
| Superior Doppelzimmer          | 31393   |
| Familie mit älteren Kindern    | 26349   |
| Deluxe Doppelzimmer            | 24823   |
| Doppel- oder Zweibettzimmer    | 22393   |
| Aufenthalt 5 Nächte            | 20845   |
| Standard Doppel- oder Zweibettzimmer | 17483 |
| Klassisches Doppelzimmer       | 16989   |
| Superior Doppel- oder Zweibettzimmer | 13570 |
| 2 Zimmer                       | 12393   |

Einige der häufigen Tags wie `Übermittelt von einem mobilen Gerät` sind für uns nutzlos, daher könnte es sinnvoll sein, sie vor der Zählung der Phrasenhäufigkeit zu entfernen. Da dies jedoch eine schnelle Operation ist, können Sie sie belassen und ignorieren.

### Entfernen der Aufenthaltsdauer-Tags

Das Entfernen dieser Tags ist Schritt 1, es reduziert die Gesamtanzahl der zu berücksichtigenden Tags leicht. Beachten Sie, dass Sie sie nicht aus dem Datensatz entfernen, sondern nur entscheiden, sie nicht als Werte in den Bewertungsdaten zu zählen oder zu behalten.

| Aufenthaltsdauer | Anzahl  |
| ---------------- | ------- |
| Aufenthalt 1 Nacht | 193645 |
| Aufenthalt 2 Nächte | 133937 |
| Aufenthalt 3 Nächte | 95821  |
| Aufenthalt 4 Nächte | 47817  |
| Aufenthalt 5 Nächte | 20845  |
| Aufenthalt 6 Nächte | 9776   |
| Aufenthalt 7 Nächte | 7399   |
| Aufenthalt 8 Nächte | 2502   |
| Aufenthalt 9 Nächte | 1293   |
| ...              | ...     |

Es gibt eine große Vielfalt an Zimmern, Suiten, Studios, Apartments und so weiter. Sie bedeuten alle ungefähr dasselbe und sind für Sie nicht relevant, daher entfernen Sie sie aus der Betrachtung.

| Zimmertyp                  | Anzahl |
| -------------------------- | ------ |
| Doppelzimmer               | 35207  |
| Standard Doppelzimmer      | 32248  |
| Superior Doppelzimmer      | 31393  |
| Deluxe Doppelzimmer        | 24823  |
| Doppel- oder Zweibettzimmer | 22393 |
| Standard Doppel- oder Zweibettzimmer | 17483 |
| Klassisches Doppelzimmer   | 16989  |
| Superior Doppel- oder Zweibettzimmer | 13570 |

Schließlich, und das ist erfreulich (weil es kaum Verarbeitung erforderte), bleiben Ihnen die folgenden *nützlichen* Tags:

| Tag                                           | Anzahl  |
| --------------------------------------------- | ------- |
| Freizeitreise                                 | 417778  |
| Paar                                          | 252294  |
| Alleinreisender                               | 108545  |
| Geschäftsreise                                | 82939   |
| Gruppe (kombiniert mit Reisende mit Freunden) | 67535   |
| Familie mit kleinen Kindern                  | 61015   |
| Familie mit älteren Kindern                  | 26349   |
| Mit einem Haustier                           | 1405    |

Man könnte argumentieren, dass `Reisende mit Freunden` mehr oder weniger dasselbe ist wie `Gruppe`, und es wäre sinnvoll, die beiden wie oben zu kombinieren. Der Code zur Identifizierung der richtigen Tags befindet sich im [Tags-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Der letzte Schritt besteht darin, neue Spalten für jedes dieser Tags zu erstellen. Dann fügen Sie für jede Bewertungszeile eine 1 hinzu, wenn die `Tag`-Spalte mit einer der neuen Spalten übereinstimmt, andernfalls eine 0. Das Endergebnis ist eine Zählung, wie viele Bewerter dieses Hotel (in der Gesamtheit) beispielsweise für Geschäftsreisen oder Freizeitaufenthalte ausgewählt haben oder ob sie ein Haustier mitgebracht haben. Diese Informationen sind nützlich, wenn Sie ein Hotel empfehlen möchten.

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

## Sentimentanalyse-Operationen

Im letzten Abschnitt wenden Sie Sentimentanalyse auf die Bewertungsspalten an und speichern die Ergebnisse in einem Datensatz.

## Übung: Laden und Speichern der gefilterten Daten

Beachten Sie, dass Sie jetzt den gefilterten Datensatz laden, der im vorherigen Abschnitt gespeichert wurde, **nicht** den ursprünglichen Datensatz.

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

Wenn Sie die Sentimentanalyse auf die Spalten für negative und positive Bewertungen anwenden würden, könnte dies viel Zeit in Anspruch nehmen. Auf einem leistungsstarken Test-Laptop mit schnellem Prozessor dauerte es je nach verwendeter Sentiment-Bibliothek 12–14 Minuten. Das ist eine (relativ) lange Zeit, daher lohnt es sich zu untersuchen, ob dies beschleunigt werden kann.

Das Entfernen von Stoppwörtern, also häufigen englischen Wörtern, die die Stimmung eines Satzes nicht verändern, ist der erste Schritt. Durch das Entfernen dieser Wörter sollte die Sentimentanalyse schneller ausgeführt werden, ohne an Genauigkeit zu verlieren (da die Stoppwörter die Stimmung nicht beeinflussen, aber die Analyse verlangsamen). 

Die längste negative Bewertung hatte 395 Wörter, aber nach dem Entfernen der Stoppwörter sind es nur noch 195 Wörter.

Das Entfernen der Stoppwörter ist ebenfalls eine schnelle Operation. Das Entfernen der Stoppwörter aus 2 Bewertungsspalten über 515.000 Zeilen dauerte auf dem Testgerät 3,3 Sekunden. Es könnte bei Ihnen etwas mehr oder weniger Zeit in Anspruch nehmen, abhängig von der Geschwindigkeit Ihrer CPU, dem Arbeitsspeicher, ob Sie eine SSD haben oder nicht, und einigen anderen Faktoren. Die relativ kurze Dauer der Operation bedeutet, dass es sich lohnt, sie durchzuführen, wenn sie die Zeit für die Sentimentanalyse verbessert.

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

### Durchführung der Sentimentanalyse
Berechnen Sie nun die Sentiment-Analyse für die Spalten mit negativen und positiven Bewertungen und speichern Sie das Ergebnis in 2 neuen Spalten. Der Test der Sentiment-Analyse besteht darin, sie mit der Bewertung des Rezensenten für dieselbe Bewertung zu vergleichen. Zum Beispiel: Wenn die Sentiment-Analyse ergibt, dass die negative Bewertung ein Sentiment von 1 (extrem positives Sentiment) und die positive Bewertung ebenfalls ein Sentiment von 1 hat, der Rezensent dem Hotel jedoch die niedrigste mögliche Punktzahl gegeben hat, dann passt entweder der Bewertungstext nicht zur Punktzahl, oder der Sentiment-Analysator konnte das Sentiment nicht korrekt erkennen. Sie sollten erwarten, dass einige Sentiment-Werte völlig falsch sind, was oft erklärbar ist, z. B. könnte die Bewertung extrem sarkastisch sein: „Natürlich LIEBTE ich es, in einem Zimmer ohne Heizung zu schlafen“, und der Sentiment-Analysator interpretiert das als positives Sentiment, obwohl ein Mensch beim Lesen erkennen würde, dass es sich um Sarkasmus handelt.

NLTK bietet verschiedene Sentiment-Analysatoren zum Lernen an, und Sie können diese austauschen, um zu sehen, ob das Sentiment genauer oder weniger genau ist. Hier wird die VADER-Sentiment-Analyse verwendet.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, Juni 2014.

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

Später im Programm, wenn Sie bereit sind, das Sentiment zu berechnen, können Sie es auf jede Bewertung wie folgt anwenden:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Dies dauert auf meinem Computer etwa 120 Sekunden, kann jedoch je nach Computer variieren. Wenn Sie die Ergebnisse ausdrucken und überprüfen möchten, ob das Sentiment zur Bewertung passt:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Das Allerletzte, was Sie mit der Datei tun sollten, bevor Sie sie in der Challenge verwenden, ist, sie zu speichern! Sie sollten auch in Betracht ziehen, alle neuen Spalten so anzuordnen, dass sie leicht zu bearbeiten sind (für einen Menschen ist das eine kosmetische Änderung).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Führen Sie den gesamten Code für [das Analyse-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) aus (nachdem Sie [Ihr Filter-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ausgeführt haben, um die Datei Hotel_Reviews_Filtered.csv zu generieren).

Zur Wiederholung, die Schritte sind:

1. Die ursprüngliche Datensatzdatei **Hotel_Reviews.csv** wurde in der vorherigen Lektion mit [dem Explorer-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) untersucht.
2. Hotel_Reviews.csv wird mit [dem Filter-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) gefiltert, was zu **Hotel_Reviews_Filtered.csv** führt.
3. Hotel_Reviews_Filtered.csv wird mit [dem Sentiment-Analyse-Notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) verarbeitet, was zu **Hotel_Reviews_NLP.csv** führt.
4. Verwenden Sie Hotel_Reviews_NLP.csv in der untenstehenden NLP-Challenge.

### Fazit

Zu Beginn hatten Sie einen Datensatz mit Spalten und Daten, aber nicht alle davon konnten überprüft oder verwendet werden. Sie haben die Daten untersucht, Unnötiges herausgefiltert, Tags in etwas Nützliches umgewandelt, eigene Durchschnittswerte berechnet, einige Sentiment-Spalten hinzugefügt und hoffentlich interessante Dinge über die Verarbeitung natürlicher Texte gelernt.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Challenge

Da Sie Ihren Datensatz nun auf Sentiment analysiert haben, versuchen Sie, mit den in diesem Kurs erlernten Strategien (z. B. Clustering) Muster im Zusammenhang mit Sentiment zu erkennen.

## Überprüfung & Selbststudium

Nehmen Sie [dieses Lernmodul](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), um mehr zu erfahren und verschiedene Tools zur Analyse von Sentiment in Texten zu verwenden.

## Aufgabe

[Probieren Sie einen anderen Datensatz aus](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.