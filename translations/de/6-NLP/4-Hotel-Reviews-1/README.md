# Sentiment-Analyse mit Hotelbewertungen - Datenverarbeitung

In diesem Abschnitt wirst du die Techniken aus den vorherigen Lektionen verwenden, um eine explorative Datenanalyse eines großen Datensatzes durchzuführen. Sobald du ein gutes Verständnis für die Nützlichkeit der verschiedenen Spalten hast, wirst du lernen: 

- wie man die überflüssigen Spalten entfernt
- wie man einige neue Daten basierend auf den vorhandenen Spalten berechnet
- wie man den resultierenden Datensatz für die finale Herausforderung speichert

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Einführung

Bisher hast du gelernt, dass Textdaten sich stark von numerischen Datentypen unterscheiden. Wenn es sich um Text handelt, der von einem Menschen geschrieben oder gesprochen wurde, kann er analysiert werden, um Muster und Frequenzen, Emotionen und Bedeutungen zu finden. Diese Lektion führt dich in einen echten Datensatz mit einer echten Herausforderung: **[515K Hotelbewertungen-Daten in Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, der unter einer [CC0: Public Domain-Lizenz](https://creativecommons.org/publicdomain/zero/1.0/) veröffentlicht ist. Die Daten wurden von Booking.com aus öffentlichen Quellen extrahiert. Der Ersteller des Datensatzes ist Jiashen Liu.

### Vorbereitung

Du benötigst:

* Die Fähigkeit, .ipynb-Notebooks mit Python 3 auszuführen
* pandas
* NLTK, [das du lokal installieren solltest](https://www.nltk.org/install.html)
* Den Datensatz, der auf Kaggle verfügbar ist: [515K Hotelbewertungen-Daten in Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Er hat eine Größe von etwa 230 MB im entpackten Zustand. Lade ihn in den Wurzelordner `/data`, der mit diesen NLP-Lektionen verbunden ist.

## Explorative Datenanalyse

Diese Herausforderung geht davon aus, dass du einen Hotelempfehlungsbot mit Hilfe von Sentiment-Analyse und Gästebewertungen aufbaust. Der Datensatz, den du verwenden wirst, enthält Bewertungen von 1493 verschiedenen Hotels in 6 Städten.

Mit Python, einem Datensatz von Hotelbewertungen und der Sentiment-Analyse von NLTK könntest du herausfinden:

* Was sind die am häufigsten verwendeten Wörter und Phrasen in den Bewertungen?
* Korrelieren die offiziellen *Tags*, die ein Hotel beschreiben, mit den Bewertungszahlen (z.B. sind die negativen Bewertungen für ein bestimmtes Hotel von *Familien mit kleinen Kindern* schlechter als von *Alleinreisenden*, was darauf hindeutet, dass es besser für *Alleinreisende* geeignet ist?)
* Stimmen die NLTK-Sentimentwerte mit den numerischen Bewertungen des Hotelbewertenden überein?

#### Datensatz

Lass uns den Datensatz erkunden, den du heruntergeladen und lokal gespeichert hast. Öffne die Datei in einem Editor wie VS Code oder sogar Excel.

Die Überschriften im Datensatz sind wie folgt:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hier sind sie gruppiert, was das Überprüfen erleichtern könnte: 
##### Hotelspalten

* `Hotel_Name`, `Hotel_Address`, `lat` (Breitengrad), `lng` (Längengrad)
  * Mit *lat* und *lng* könntest du eine Karte mit Python erstellen, die die Hotelstandorte anzeigt (vielleicht farblich kodiert für negative und positive Bewertungen)
  * Hotel_Address ist für uns offensichtlich nicht nützlich, und wir werden das wahrscheinlich durch ein Land ersetzen, um das Sortieren und Suchen zu erleichtern

**Hotel-Meta-Bewertungsspalten**

* `Average_Score`
  * Laut dem Ersteller des Datensatzes ist diese Spalte die *Durchschnittsbewertung des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*. Dies scheint eine ungewöhnliche Methode zur Berechnung der Bewertung zu sein, aber es sind die gescrapten Daten, also nehmen wir es vorerst als gegeben hin. 
  
  ✅ Basierend auf den anderen Spalten in diesen Daten, kannst du dir eine andere Methode zur Berechnung der Durchschnittsbewertung vorstellen?

* `Total_Number_of_Reviews`
  * Die Gesamtzahl der Bewertungen, die dieses Hotel erhalten hat - es ist nicht klar (ohne etwas Code zu schreiben), ob sich dies auf die Bewertungen im Datensatz bezieht.
* `Additional_Number_of_Scoring`
  * Dies bedeutet, dass eine Bewertungszahl vergeben wurde, aber keine positive oder negative Bewertung vom Bewerter geschrieben wurde

**Bewertungsspalten**

- `Reviewer_Score`
  - Dies ist ein numerischer Wert mit maximal 1 Dezimalstelle zwischen den minimalen und maximalen Werten 2.5 und 10
  - Es wird nicht erklärt, warum 2.5 die niedrigste mögliche Bewertung ist
- `Negative_Review`
  - Wenn ein Bewerter nichts geschrieben hat, wird dieses Feld mit "**No Negative**" gefüllt
  - Beachte, dass ein Bewerter in der negativen Bewertungsspalte eine positive Bewertung schreiben kann (z.B. "es gibt nichts Schlechtes an diesem Hotel")
- `Review_Total_Negative_Word_Counts`
  - Höhere negative Wortanzahlen deuten auf eine niedrigere Bewertung hin (ohne die Sentimentalität zu überprüfen)
- `Positive_Review`
  - Wenn ein Bewerter nichts geschrieben hat, wird dieses Feld mit "**No Positive**" gefüllt
  - Beachte, dass ein Bewerter in der positiven Bewertungsspalte eine negative Bewertung schreiben kann (z.B. "es gibt nichts Gutes an diesem Hotel")
- `Review_Total_Positive_Word_Counts`
  - Höhere positive Wortanzahlen deuten auf eine höhere Bewertung hin (ohne die Sentimentalität zu überprüfen)
- `Review_Date` und `days_since_review`
  - Ein Frische- oder Alterungsmaß könnte auf eine Bewertung angewendet werden (ältere Bewertungen könnten weniger genau sein als neuere, weil sich das Hotelmanagement geändert hat, Renovierungen vorgenommen wurden oder ein Pool hinzugefügt wurde etc.)
- `Tags`
  - Dies sind kurze Beschreibungen, die ein Bewerter auswählen kann, um die Art des Gastes zu beschreiben, der sie waren (z.B. allein oder Familie), die Art des Zimmers, das sie hatten, die Dauer des Aufenthalts und wie die Bewertung eingereicht wurde. 
  - Leider ist die Verwendung dieser Tags problematisch, siehe den Abschnitt unten, der ihre Nützlichkeit diskutiert.

**Bewertersäulen**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dies könnte ein Faktor in einem Empfehlungsmodell sein, beispielsweise wenn du feststellen könntest, dass produktivere Bewerter mit Hunderten von Bewertungen eher negativ als positiv sind. Allerdings ist der Bewerter einer bestimmten Bewertung nicht mit einem eindeutigen Code identifiziert und kann daher nicht mit einem Set von Bewertungen verknüpft werden. Es gibt 30 Bewerter mit 100 oder mehr Bewertungen, aber es ist schwer zu sehen, wie dies dem Empfehlungsmodell helfen kann.
- `Reviewer_Nationality`
  - Einige Leute könnten denken, dass bestimmte Nationalitäten eher eine positive oder negative Bewertung abgeben, aufgrund einer nationalen Neigung. Sei vorsichtig, solche anekdotischen Ansichten in deine Modelle einzubauen. Dies sind nationale (und manchmal rassistische) Stereotypen, und jeder Bewerter war ein Individuum, das eine Bewertung basierend auf seinen Erfahrungen geschrieben hat. Es könnte durch viele Linsen gefiltert worden sein, wie z.B. ihre vorherigen Hotelaufenthalte, die zurückgelegte Distanz und ihr persönliches Temperament. Zu denken, dass ihre Nationalität der Grund für eine Bewertungszahl war, ist schwer zu rechtfertigen.

##### Beispiele

| Durchschnittliche Bewertung | Gesamtzahl der Bewertungen | Bewerter Bewertung | Negative <br />Bewertung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Bewertung                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Dies ist derzeit kein Hotel, sondern eine Baustelle. Ich wurde von frühmorgens bis den ganzen Tag mit inakzeptablem Baulärm terrorisiert, während ich nach einer langen Reise und der Arbeit im Zimmer ausruhen wollte. Leute haben den ganzen Tag mit Presslufthämmern in den angrenzenden Zimmern gearbeitet. Ich bat um einen Zimmerwechsel, aber kein ruhiges Zimmer war verfügbar. Um die Sache noch schlimmer zu machen, wurde ich über den Tisch gezogen. Ich checkte am Abend aus, da ich sehr früh einen Flug hatte und erhielt eine angemessene Rechnung. Einen Tag später machte das Hotel ohne meine Zustimmung eine weitere Belastung in Höhe des gebuchten Preises. Es ist ein schrecklicher Ort. Bestrafe dich nicht, indem du hier buchst. | Nichts  Schrecklicher Ort. Halte dich fern. | Geschäftsreise                                Paar Standard Doppelzimmer. 2 Nächte geblieben. |

Wie du sehen kannst, hatte dieser Gast keinen glücklichen Aufenthalt in diesem Hotel. Das Hotel hat eine gute Durchschnittsbewertung von 7.8 und 1945 Bewertungen, aber dieser Bewerter gab ihm 2.5 und schrieb 115 Wörter darüber, wie negativ ihr Aufenthalt war. Wenn sie in der Spalte Positive_Review überhaupt nichts geschrieben hätten, könnte man annehmen, dass es nichts Positives gab, aber leider schrieben sie 7 Worte der Warnung. Wenn wir nur die Wörter zählen würden, anstatt die Bedeutung oder Sentiment der Wörter zu betrachten, könnten wir eine verzerrte Sicht auf die Absicht des Bewerters haben. Seltsamerweise ist ihre Bewertung von 2.5 verwirrend, denn wenn dieser Hotelaufenthalt so schlecht war, warum sollten sie dann überhaupt Punkte vergeben? Bei genauerer Untersuchung des Datensatzes wirst du feststellen, dass die niedrigste mögliche Bewertung 2.5 beträgt, nicht 0. Die höchste mögliche Bewertung ist 10.

##### Tags

Wie oben erwähnt, macht die Idee, `Tags` zu verwenden, um die Daten zu kategorisieren, auf den ersten Blick Sinn. Leider sind diese Tags nicht standardisiert, was bedeutet, dass in einem bestimmten Hotel die Optionen *Einzelzimmer*, *Zweibettzimmer* und *Doppelzimmer* sein könnten, aber im nächsten Hotel sind sie *Deluxe Einzelzimmer*, *Klassisches Queensize-Zimmer* und *Executive Kingsize-Zimmer*. Diese könnten die gleichen Dinge sein, aber es gibt so viele Variationen, dass die Wahl wird:

1. Versuchen, alle Begriffe auf einen einheitlichen Standard zu ändern, was sehr schwierig ist, da nicht klar ist, wie der Umwandlungsweg in jedem Fall aussehen würde (z.B. *Klassisches Einzelzimmer* entspricht *Einzelzimmer*, aber *Superior Queensize-Zimmer mit Innenhofgarten oder Stadtblick* ist viel schwerer zuzuordnen)

2. Wir können einen NLP-Ansatz wählen und die Häufigkeit bestimmter Begriffe wie *Alleinreisender*, *Geschäftsreisender* oder *Familie mit kleinen Kindern* messen, wie sie auf jedes Hotel zutreffen, und dies in die Empfehlung einfließen lassen.  

Tags sind normalerweise (aber nicht immer) ein einzelnes Feld, das eine Liste von 5 bis 6 durch Kommas getrennten Werten enthält, die sich auf *Art der Reise*, *Art der Gäste*, *Art des Zimmers*, *Anzahl der Nächte* und *Art des Geräts, auf dem die Bewertung eingereicht wurde* beziehen. Da einige Bewerter jedoch nicht jedes Feld ausfüllen (sie könnten eines leer lassen), sind die Werte nicht immer in derselben Reihenfolge.

Nehmen wir als Beispiel *Art der Gruppe*. Es gibt 1025 einzigartige Möglichkeiten in diesem Feld in der `Tags`-Spalte, und leider beziehen sich nur einige von ihnen auf eine Gruppe (einige sind die Art des Zimmers usw.). Wenn du nur die filterst, die Familie erwähnen, enthalten die Ergebnisse viele *Familienzimmer*-Typen. Wenn du den Begriff *mit* einbeziehst, d.h. die *Familie mit* Werte zählst, sind die Ergebnisse besser, mit über 80.000 der 515.000 Ergebnisse, die die Phrase "Familie mit kleinen Kindern" oder "Familie mit älteren Kindern" enthalten.

Das bedeutet, dass die Tags-Spalte für uns nicht völlig nutzlos ist, aber es wird einige Arbeit erfordern, um sie nützlich zu machen.

##### Durchschnittliche Hotelbewertung

Es gibt eine Reihe von Eigenheiten oder Diskrepanzen mit dem Datensatz, die ich nicht herausfinden kann, aber hier illustriert werden, damit du dir dessen bewusst bist, wenn du deine Modelle erstellst. Wenn du es herausfindest, lass es uns bitte im Diskussionsbereich wissen!

Der Datensatz hat die folgenden Spalten, die sich auf die durchschnittliche Bewertung und die Anzahl der Bewertungen beziehen: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Das einzelne Hotel mit den meisten Bewertungen in diesem Datensatz ist *Britannia International Hotel Canary Wharf* mit 4789 Bewertungen von 515.000. Aber wenn wir den `Total_Number_of_Reviews`-Wert für dieses Hotel betrachten, beträgt er 9086. Du könntest annehmen, dass es viele weitere Bewertungen ohne Rezensionen gibt, also sollten wir vielleicht den Wert aus der `Additional_Number_of_Scoring`-Spalte hinzufügen. Dieser Wert beträgt 2682, und wenn wir ihn zu 4789 hinzufügen, erhalten wir 7471, was immer noch 1615 weniger als der `Total_Number_of_Reviews` ist. 

Wenn du die `Average_Score`-Spalten betrachtest, könntest du annehmen, dass es der Durchschnitt der Bewertungen im Datensatz ist, aber die Beschreibung von Kaggle lautet: "*Durchschnittsbewertung des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*". Das scheint nicht sehr nützlich zu sein, aber wir können unseren eigenen Durchschnitt basierend auf den Bewertungen im Datensatz berechnen. Wenn wir dasselbe Hotel als Beispiel nehmen, wird die durchschnittliche Hotelbewertung mit 7.1 angegeben, aber die berechnete Bewertung (Durchschnitt der Bewerterbewertungen *im* Datensatz) beträgt 6.8. Dies ist nah, aber nicht derselbe Wert, und wir können nur raten, dass die in den `Additional_Number_of_Scoring`-Bewertungen vergebenen Bewertungen den Durchschnitt auf 7.1 erhöht haben. Leider ist es mit keinen Möglichkeiten, diese Annahme zu testen oder zu beweisen, schwierig, `Average_Score`, `Additional_Number_of_Scoring` und `Total_Number_of_Reviews` zu verwenden oder ihnen zu vertrauen, wenn sie auf Daten basieren oder sich auf Daten beziehen, die wir nicht haben.

Um die Sache weiter zu komplizieren, hat das Hotel mit der zweithöchsten Anzahl von Bewertungen eine berechnete Durchschnittsbewertung von 8.12 und der Datensatz `Average_Score` beträgt 8.1. Ist dieser korrekte Wert ein Zufall oder ist das erste Hotel eine Diskrepanz? 

Angesichts der Möglichkeit, dass diese Hotels Ausreißer sein könnten und dass vielleicht die meisten Werte übereinstimmen (aber einige aus irgendeinem Grund nicht), werden wir ein kurzes Programm schreiben, um die Werte im Datensatz zu erkunden und die korrekte Verwendung (oder Nichtverwendung) der Werte zu bestimmen.

> 🚨 Eine Warnung
>
> Wenn du mit diesem Datensatz arbeitest, wirst du Code schreiben, der etwas aus dem Text berechnet, ohne den Text selbst lesen oder analysieren zu müssen. Das ist das Wesen von NLP, Bedeutung oder Sentiment zu interpretieren, ohne dass ein Mensch dies tun muss. Es ist jedoch möglich, dass du einige der negativen Bewertungen liest. Ich würde dir raten, das nicht zu tun, denn du musst es nicht. Einige von ihnen sind albern oder irrelevante negative Hotelbewertungen, wie "Das Wetter war nicht toll", etwas, das außerhalb der Kontrolle des Hotels oder tatsächlich von irgendjemandem liegt. Aber es gibt auch eine dunkle Seite zu einigen Bewertungen. Manchmal sind die negativen Bewertungen rassistisch, sexistisch oder altersdiskriminierend. Das ist bedauerlich, aber in einem Datensatz, der von einer öffentlichen Website gescrapet wurde, zu erwarten. Einige Bewerter hinterlassen Bewertungen, die du als unangenehm, unangenehm oder verstörend empfinden würdest. Es ist besser, den Code das Sentiment messen zu lassen, als sie selbst zu lesen und verärgert zu sein. Das gesagt, es ist eine Minderheit, die solche Dinge schreibt, aber sie existieren trotzdem. 

## Übung - Datenexploration
### Daten laden

Das reicht für die visuelle Untersuchung der Daten, jetzt wirst du etwas Code schreiben und Antworten erhalten! Dieser Abschnitt verwendet die pandas-Bibliothek. Deine erste Aufgabe ist es sicherzustellen, dass du die CSV-Daten laden und lesen kannst. Die pandas-Bibliothek hat einen schnellen CSV-Loader, und das Ergebnis wird in einem DataFrame platziert, wie in den vorherigen Lektionen. Die CSV, die wir laden, hat über eine halbe Million Zeilen, aber nur 17 Spalten. Pandas bietet dir viele leistungsstarke Möglichkeiten, mit einem DataFrame zu interagieren, einschließlich der Fähigkeit, Operationen auf jeder Zeile durchzuführen. 

Von hier an wird es in dieser Lektion Code-Schnipsel und einige Erklärungen zum Code sowie einige Diskussionen darüber geben, was die Ergebnisse bedeuten. Verwende das beigefügte _notebook.ipynb_ für deinen Code.

Lass uns mit dem Laden der Datendatei beginnen, die du verwenden wirst:

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

Jetzt, da die Daten geladen sind, können wir einige Operationen darauf durchführen. Halte diesen Code am Anfang deines Programms für den nächsten Teil.

## Daten erkunden

In diesem Fall sind die Daten bereits *sauber*, das bedeutet, dass sie bereit sind, damit zu arbeiten, und keine Zeichen in anderen Sprachen enthalten, die Algorithmen, die nur englische Zeichen erwarten, in Schwierigkeiten bringen könnten. 

✅ Möglicherweise musst du mit Daten arbeiten, die eine anfängliche Verarbeitung benötigten, um sie zu formatieren, bevor du NLP-Techniken anwendest, aber nicht dieses Mal. Wenn du es tun müsstest, wie würdest du mit nicht-englischen Zeichen umgehen?

Nimm dir einen Moment Zeit, um sicherzustellen, dass du, sobald die Daten geladen
Zeilen haben Spaltenwerte von `Positive_Review` "Keine Positiven" 9. Berechnen und drucken Sie aus, wie viele Zeilen Spaltenwerte von `Positive_Review` "Keine Positiven" **und** `Negative_Review` "Keine Negativen" haben ### Code-Antworten 1. Drucken Sie die *Form* des Datenrahmens aus, den Sie gerade geladen haben (die Form ist die Anzahl der Zeilen und Spalten) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Berechnen Sie die Häufigkeitszählung für die Nationalitäten der Rezensenten: 1. Wie viele unterschiedliche Werte gibt es für die Spalte `Reviewer_Nationality` und was sind sie? 2. Welche Rezensenten-Nationalität ist die häufigste im Datensatz (geben Sie das Land und die Anzahl der Bewertungen an)? ```python
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
   ``` 3. Was sind die nächsten 10 häufigsten Nationalitäten und ihre Häufigkeitszählung? ```python
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
      ``` 3. Was war das am häufigsten bewertete Hotel für jede der 10 häufigsten Rezensenten-Nationalitäten? ```python
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
   ``` 4. Wie viele Bewertungen gibt es pro Hotel (Häufigkeitszählung des Hotels) im Datensatz? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | Sie werden möglicherweise feststellen, dass die *im Datensatz gezählten* Ergebnisse nicht mit dem Wert in `Total_Number_of_Reviews` übereinstimmen. Es ist unklar, ob dieser Wert im Datensatz die Gesamtzahl der Bewertungen darstellt, die das Hotel hatte, aber nicht alle wurden erfasst oder eine andere Berechnung. `Total_Number_of_Reviews` wird aufgrund dieser Unklarheit nicht im Modell verwendet. 5. Während es eine `Average_Score`-Spalte für jedes Hotel im Datensatz gibt, können Sie auch einen Durchschnittswert berechnen (den Durchschnitt aller Rezensentennoten im Datensatz für jedes Hotel). Fügen Sie Ihrem Datenrahmen eine neue Spalte mit der Spaltenüberschrift `Calc_Average_Score` hinzu, die diesen berechneten Durchschnitt enthält. Drucken Sie die Spalten `Hotel_Name`, `Average_Score` und `Calc_Average_Score` aus. ```python
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
   ``` Sie fragen sich vielleicht auch über den `Average_Score`-Wert und warum er manchmal von der berechneten Durchschnittsbewertung abweicht. Da wir nicht wissen können, warum einige der Werte übereinstimmen, andere jedoch einen Unterschied aufweisen, ist es in diesem Fall am sichersten, die Bewertungsnoten, die wir haben, zu verwenden, um den Durchschnitt selbst zu berechnen. Das gesagt, die Unterschiede sind normalerweise sehr klein, hier sind die Hotels mit der größten Abweichung vom durchschnittlichen Datensatz und dem berechneten Durchschnitt: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | Mit nur 1 Hotel, das einen Punktunterschied von mehr als 1 aufweist, bedeutet dies, dass wir den Unterschied wahrscheinlich ignorieren und den berechneten Durchschnittswert verwenden können. 6. Berechnen und drucken Sie aus, wie viele Zeilen Spaltenwerte von `Negative_Review` "Keine Negativen" haben 7. Berechnen und drucken Sie aus, wie viele Zeilen Spaltenwerte von `Positive_Review` "Keine Positiven" haben 8. Berechnen und drucken Sie aus, wie viele Zeilen Spaltenwerte von `Positive_Review` "Keine Positiven" **und** `Negative_Review` "Keine Negativen" haben ```python
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
   ``` ## Eine andere Möglichkeit Eine andere Möglichkeit, Elemente ohne Lambdas zu zählen und die Summe zu verwenden, um die Zeilen zu zählen: ```python
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
   ``` Sie haben möglicherweise bemerkt, dass es 127 Zeilen gibt, die sowohl "Keine Negativen" als auch "Keine Positiven" Werte für die Spalten `Negative_Review` und `Positive_Review` haben. Das bedeutet, dass der Rezensent dem Hotel eine numerische Bewertung gegeben hat, sich jedoch geweigert hat, eine positive oder negative Bewertung zu schreiben. Glücklicherweise handelt es sich um eine kleine Anzahl von Zeilen (127 von 515738 oder 0,02 %), sodass es wahrscheinlich unser Modell oder unsere Ergebnisse in keine bestimmte Richtung verzerren wird, aber Sie hätten möglicherweise nicht erwartet, dass ein Datensatz von Bewertungen Zeilen ohne Bewertungen enthält, daher ist es wert, die Daten zu erkunden, um solche Zeilen zu entdecken. Jetzt, da Sie den Datensatz erkundet haben, werden Sie in der nächsten Lektion die Daten filtern und eine Sentimentanalyse hinzufügen. --- ## 🚀Herausforderung Diese Lektion zeigt, wie kritisch wichtig es ist, Ihre Daten und deren Eigenheiten zu verstehen, bevor Sie Operationen darauf durchführen. Textbasierte Daten erfordern besonders sorgfältige Prüfung. Durchsuchen Sie verschiedene textlastige Datensätze und sehen Sie, ob Sie Bereiche entdecken können, die Vorurteile oder verzerrte Sentimente in ein Modell einführen könnten. ## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Überprüfung & Selbststudium Nehmen Sie [diesen Lernpfad zu NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) in Anspruch, um Werkzeuge zu entdecken, die Sie beim Aufbau von Sprach- und textlastigen Modellen ausprobieren können. ## Aufgabe [NLTK](assignment.md) Bitte schreiben Sie die Ausgabe von links nach rechts.

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.