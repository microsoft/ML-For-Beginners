<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-09-03T22:01:41+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "de"
}
-->
# Sentimentanalyse mit Hotelbewertungen – Datenverarbeitung

In diesem Abschnitt wirst du die in den vorherigen Lektionen erlernten Techniken nutzen, um eine explorative Datenanalyse eines großen Datensatzes durchzuführen. Sobald du ein gutes Verständnis für die Nützlichkeit der verschiedenen Spalten hast, wirst du lernen:

- wie man unnötige Spalten entfernt
- wie man neue Daten basierend auf den vorhandenen Spalten berechnet
- wie man den resultierenden Datensatz speichert, um ihn in der abschließenden Herausforderung zu verwenden

## [Quiz vor der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Einführung

Bisher hast du gelernt, dass Textdaten sich stark von numerischen Datentypen unterscheiden. Wenn es sich um Text handelt, der von einem Menschen geschrieben oder gesprochen wurde, kann er analysiert werden, um Muster, Häufigkeiten, Stimmungen und Bedeutungen zu erkennen. Diese Lektion führt dich in einen realen Datensatz mit einer echten Herausforderung ein: **[515K Hotelbewertungen in Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, der unter einer [CC0: Public Domain Lizenz](https://creativecommons.org/publicdomain/zero/1.0/) steht. Die Daten wurden von Booking.com aus öffentlichen Quellen extrahiert. Der Ersteller des Datensatzes ist Jiashen Liu.

### Vorbereitung

Du benötigst:

* Die Möglichkeit, .ipynb-Notebooks mit Python 3 auszuführen
* pandas
* NLTK, [das du lokal installieren solltest](https://www.nltk.org/install.html)
* Den Datensatz, der auf Kaggle verfügbar ist: [515K Hotelbewertungen in Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Er ist etwa 230 MB groß, wenn er entpackt ist. Lade ihn in den Root-Ordner `/data`, der mit diesen NLP-Lektionen verbunden ist.

## Explorative Datenanalyse

Diese Herausforderung geht davon aus, dass du einen Hotel-Empfehlungsbot basierend auf Sentimentanalyse und Gästebewertungen erstellst. Der Datensatz, den du verwenden wirst, enthält Bewertungen von 1493 verschiedenen Hotels in 6 Städten.

Mit Python, einem Datensatz von Hotelbewertungen und der Sentimentanalyse von NLTK könntest du herausfinden:

* Welche Wörter und Phrasen werden in Bewertungen am häufigsten verwendet?
* Korrelieren die offiziellen *Tags*, die ein Hotel beschreiben, mit den Bewertungsergebnissen (z. B. gibt es mehr negative Bewertungen für ein bestimmtes Hotel von *Familien mit kleinen Kindern* als von *Alleinreisenden*, was darauf hindeuten könnte, dass es besser für *Alleinreisende* geeignet ist)?
* Stimmen die Sentiment-Scores von NLTK mit den numerischen Bewertungen der Hotelgäste überein?

#### Datensatz

Lass uns den Datensatz, den du heruntergeladen und lokal gespeichert hast, erkunden. Öffne die Datei in einem Editor wie VS Code oder sogar Excel.

Die Spaltenüberschriften im Datensatz sind wie folgt:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hier sind sie gruppiert, um sie leichter zu untersuchen: 
##### Hotel-Spalten

* `Hotel_Name`, `Hotel_Address`, `lat` (Breitengrad), `lng` (Längengrad)
  * Mit *lat* und *lng* könntest du eine Karte mit Python erstellen, die die Hotelstandorte anzeigt (vielleicht farblich codiert für negative und positive Bewertungen).
  * Hotel_Address ist für uns nicht offensichtlich nützlich, und wir werden es wahrscheinlich durch ein Land ersetzen, um die Sortierung und Suche zu erleichtern.

**Hotel-Meta-Bewertungsspalten**

* `Average_Score`
  * Laut dem Ersteller des Datensatzes ist diese Spalte der *Durchschnittliche Score des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*. Dies scheint eine ungewöhnliche Methode zur Berechnung des Scores zu sein, aber es sind die extrahierten Daten, daher nehmen wir sie vorerst so hin.
  
  ✅ Basierend auf den anderen Spalten in diesen Daten: Kannst du dir eine andere Methode vorstellen, um den Durchschnittsscore zu berechnen?

* `Total_Number_of_Reviews`
  * Die Gesamtanzahl der Bewertungen, die dieses Hotel erhalten hat – es ist unklar (ohne Code zu schreiben), ob sich dies auf die Bewertungen im Datensatz bezieht.
* `Additional_Number_of_Scoring`
  * Dies bedeutet, dass eine Bewertung abgegeben wurde, aber keine positive oder negative Bewertung vom Bewerter geschrieben wurde.

**Bewertungsspalten**

- `Reviewer_Score`
  - Dies ist ein numerischer Wert mit maximal einer Dezimalstelle zwischen den Minimal- und Maximalwerten 2,5 und 10.
  - Es wird nicht erklärt, warum 2,5 der niedrigste mögliche Score ist.
- `Negative_Review`
  - Wenn ein Bewerter nichts geschrieben hat, enthält dieses Feld "**No Negative**".
  - Beachte, dass ein Bewerter möglicherweise eine positive Bewertung in der Spalte Negative Review schreibt (z. B. "Es gibt nichts Schlechtes an diesem Hotel").
- `Review_Total_Negative_Word_Counts`
  - Höhere negative Wortzahlen deuten auf einen niedrigeren Score hin (ohne die Sentimentalität zu überprüfen).
- `Positive_Review`
  - Wenn ein Bewerter nichts geschrieben hat, enthält dieses Feld "**No Positive**".
  - Beachte, dass ein Bewerter möglicherweise eine negative Bewertung in der Spalte Positive Review schreibt (z. B. "Es gibt absolut nichts Gutes an diesem Hotel").
- `Review_Total_Positive_Word_Counts`
  - Höhere positive Wortzahlen deuten auf einen höheren Score hin (ohne die Sentimentalität zu überprüfen).
- `Review_Date` und `days_since_review`
  - Ein Frische- oder Veraltungsmaß könnte auf eine Bewertung angewendet werden (ältere Bewertungen könnten weniger genau sein als neuere, da sich das Hotelmanagement geändert hat, Renovierungen durchgeführt wurden oder ein Pool hinzugefügt wurde usw.).
- `Tags`
  - Dies sind kurze Beschreibungen, die ein Bewerter möglicherweise auswählt, um die Art des Gastes zu beschreiben, der er war (z. B. allein oder Familie), die Art des Zimmers, das er hatte, die Aufenthaltsdauer und wie die Bewertung eingereicht wurde.
  - Leider ist die Verwendung dieser Tags problematisch. Siehe den Abschnitt unten, der ihre Nützlichkeit diskutiert.

**Bewerterspalten**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dies könnte ein Faktor in einem Empfehlungsmodell sein, z. B. wenn du feststellen könntest, dass produktivere Bewerter mit Hunderten von Bewertungen eher negativ als positiv sind. Der Bewerter einer bestimmten Bewertung wird jedoch nicht mit einem eindeutigen Code identifiziert und kann daher nicht mit einer Reihe von Bewertungen verknüpft werden. Es gibt 30 Bewerter mit 100 oder mehr Bewertungen, aber es ist schwer zu erkennen, wie dies das Empfehlungsmodell unterstützen kann.
- `Reviewer_Nationality`
  - Manche Leute könnten denken, dass bestimmte Nationalitäten eher positive oder negative Bewertungen abgeben, basierend auf einer nationalen Neigung. Sei vorsichtig, solche anekdotischen Ansichten in deine Modelle einzubauen. Dies sind nationale (und manchmal rassistische) Stereotypen, und jeder Bewerter war eine Einzelperson, die eine Bewertung basierend auf ihrer Erfahrung geschrieben hat. Diese könnte durch viele Filter wie frühere Hotelaufenthalte, die zurückgelegte Entfernung und ihr persönliches Temperament beeinflusst worden sein. Zu denken, dass ihre Nationalität der Grund für eine Bewertung war, ist schwer zu rechtfertigen.

##### Beispiele

| Durchschnittlicher Score | Gesamtanzahl Bewertungen | Bewerter-Score | Negative <br />Bewertung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive Bewertung               | Tags                                                                                      |
| ------------------------ | ------------------------ | -------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8                      | 1945                     | 2.5            | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Wie du sehen kannst, hatte dieser Gast keinen angenehmen Aufenthalt in diesem Hotel. Das Hotel hat einen guten Durchschnittsscore von 7,8 und 1945 Bewertungen, aber dieser Bewerter gab ihm 2,5 und schrieb 115 Wörter darüber, wie negativ sein Aufenthalt war. Wenn er überhaupt nichts in der Spalte Positive_Review geschrieben hätte, könnte man vermuten, dass es nichts Positives gab, aber er schrieb 7 Worte der Warnung. Wenn wir nur Wörter zählen würden, anstatt die Bedeutung oder Stimmung der Wörter zu berücksichtigen, könnten wir ein verzerrtes Bild der Absicht des Bewerters erhalten. Seltsamerweise ist ihre Bewertung von 2,5 verwirrend, denn wenn der Hotelaufenthalt so schlecht war, warum überhaupt Punkte vergeben? Wenn du den Datensatz genau untersuchst, wirst du feststellen, dass der niedrigste mögliche Score 2,5 und nicht 0 ist. Der höchste mögliche Score ist 10.

##### Tags

Wie oben erwähnt, scheint die Idee, `Tags` zur Kategorisierung der Daten zu verwenden, auf den ersten Blick sinnvoll. Leider sind diese Tags nicht standardisiert, was bedeutet, dass in einem bestimmten Hotel die Optionen *Einzelzimmer*, *Zweibettzimmer* und *Doppelzimmer* sein könnten, während sie im nächsten Hotel *Deluxe Einzelzimmer*, *Klassisches Queen-Zimmer* und *Executive King-Zimmer* heißen. Dies könnten dieselben Dinge sein, aber es gibt so viele Variationen, dass die Wahl besteht zwischen:

1. Der Versuch, alle Begriffe auf einen einzigen Standard zu ändern, was sehr schwierig ist, da nicht klar ist, wie der Konvertierungspfad in jedem Fall aussehen würde (z. B. *Klassisches Einzelzimmer* wird zu *Einzelzimmer*, aber *Superior Queen Room with Courtyard Garden or City View* ist viel schwieriger zuzuordnen).

1. Wir können einen NLP-Ansatz verfolgen und die Häufigkeit bestimmter Begriffe wie *Alleinreisender*, *Geschäftsreisender* oder *Familie mit kleinen Kindern* messen, wie sie auf jedes Hotel zutreffen, und dies in das Empfehlungsmodell einfließen lassen.

Tags sind normalerweise (aber nicht immer) ein einzelnes Feld, das eine Liste von 5 bis 6 durch Kommas getrennten Werten enthält, die sich auf *Art der Reise*, *Art der Gäste*, *Art des Zimmers*, *Anzahl der Nächte* und *Art des Geräts, auf dem die Bewertung eingereicht wurde* beziehen. Da jedoch einige Bewerter nicht jedes Feld ausfüllen (sie könnten eines leer lassen), sind die Werte nicht immer in derselben Reihenfolge.

Als Beispiel nehmen wir *Art der Gruppe*. Es gibt 1025 einzigartige Möglichkeiten in diesem Feld in der Spalte `Tags`, und leider beziehen sich nur einige davon auf eine Gruppe (einige beziehen sich auf die Art des Zimmers usw.). Wenn du nur die herausfilterst, die Familie erwähnen, enthalten die Ergebnisse viele *Familienzimmer*-Typen. Wenn du den Begriff *mit* einbeziehst, d. h. die Werte *Familie mit* zählst, sind die Ergebnisse besser, mit über 80.000 der 515.000 Ergebnisse, die den Ausdruck "Familie mit kleinen Kindern" oder "Familie mit älteren Kindern" enthalten.

Das bedeutet, dass die Tags-Spalte für uns nicht völlig nutzlos ist, aber es wird einige Arbeit erfordern, sie nützlich zu machen.

##### Durchschnittlicher Hotelscore

Es gibt eine Reihe von Unstimmigkeiten oder Diskrepanzen im Datensatz, die ich nicht erklären kann, die aber hier illustriert werden, damit du dir ihrer bewusst bist, wenn du deine Modelle erstellst. Wenn du es herausfindest, lass es uns bitte im Diskussionsbereich wissen!

Der Datensatz enthält die folgenden Spalten, die sich auf den Durchschnittsscore und die Anzahl der Bewertungen beziehen:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Das einzelne Hotel mit den meisten Bewertungen in diesem Datensatz ist *Britannia International Hotel Canary Wharf* mit 4789 Bewertungen von insgesamt 515.000. Wenn wir uns jedoch den Wert `Total_Number_of_Reviews` für dieses Hotel ansehen, beträgt er 9086. Du könntest vermuten, dass es viele weitere Scores ohne Bewertungen gibt, also sollten wir vielleicht den Wert der Spalte `Additional_Number_of_Scoring` hinzufügen. Dieser Wert beträgt 2682, und wenn wir ihn zu 4789 addieren, erhalten wir 7471, was immer noch 1615 weniger als der Wert von `Total_Number_of_Reviews` ist.

Wenn du die Spalte `Average_Score` betrachtest, könntest du vermuten, dass es sich um den Durchschnitt der Bewertungen im Datensatz handelt, aber die Beschreibung von Kaggle lautet: "*Durchschnittlicher Score des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*". Das scheint nicht sehr nützlich zu sein, aber wir können unseren eigenen Durchschnitt basierend auf den Bewertungsscores im Datensatz berechnen. Wenn wir dasselbe Hotel als Beispiel nehmen, wird der durchschnittliche Hotelscore mit 7,1 angegeben, aber der berechnete Score (durchschnittlicher Bewerterscore *im* Datensatz) beträgt 6,8. Das ist nah dran, aber nicht derselbe Wert, und wir können nur vermuten, dass die Scores in den `Additional_Number_of_Scoring`-Bewertungen den Durchschnitt auf 7,1 erhöht haben. Leider ist es ohne Möglichkeit, diese Annahme zu testen oder zu beweisen, schwierig, `Average_Score`, `Additional_Number_of_Scoring` und `Total_Number_of_Reviews` zu verwenden oder ihnen zu vertrauen, wenn sie auf Daten basieren oder sich auf Daten beziehen, die wir nicht haben.

Um die Sache weiter zu verkomplizieren: Das Hotel mit der zweithöchsten Anzahl an Bewertungen hat einen berechneten Durchschnittsscore von 8,12, und der `Average_Score` im Datensatz beträgt 8,1. Ist dieser korrekte Score ein Zufall, oder ist das erste Hotel eine Diskrepanz?
Auf die Möglichkeit hin, dass dieses Hotel ein Ausreißer sein könnte und dass die meisten Werte übereinstimmen (aber einige aus irgendeinem Grund nicht), werden wir als Nächstes ein kurzes Programm schreiben, um die Werte im Datensatz zu untersuchen und die korrekte Verwendung (oder Nichtverwendung) der Werte zu bestimmen.

> 🚨 Ein Hinweis zur Vorsicht
>
> Beim Arbeiten mit diesem Datensatz werden Sie Code schreiben, der etwas aus dem Text berechnet, ohne den Text selbst lesen oder analysieren zu müssen. Das ist das Wesen von NLP: Bedeutung oder Stimmung interpretieren, ohne dass ein Mensch dies tun muss. Es ist jedoch möglich, dass Sie einige der negativen Bewertungen lesen. Ich würde Ihnen davon abraten, da es nicht notwendig ist. Einige davon sind albern oder irrelevante negative Hotelbewertungen, wie "Das Wetter war nicht gut", etwas, das außerhalb der Kontrolle des Hotels oder überhaupt von jemandem liegt. Aber es gibt auch eine dunkle Seite bei einigen Bewertungen. Manchmal sind die negativen Bewertungen rassistisch, sexistisch oder altersdiskriminierend. Das ist bedauerlich, aber zu erwarten bei einem Datensatz, der von einer öffentlichen Website abgerufen wurde. Einige Bewerter hinterlassen Bewertungen, die Sie geschmacklos, unangenehm oder verstörend finden könnten. Es ist besser, den Code die Stimmung messen zu lassen, als sie selbst zu lesen und sich zu ärgern. Das gesagt, es ist nur eine Minderheit, die solche Dinge schreibt, aber sie existieren dennoch.

## Übung - Datenexploration
### Daten laden

Das visuelle Untersuchen der Daten reicht jetzt aus, Sie werden Code schreiben und Antworten erhalten! Dieser Abschnitt verwendet die Pandas-Bibliothek. Ihre allererste Aufgabe ist sicherzustellen, dass Sie die CSV-Daten laden und lesen können. Die Pandas-Bibliothek hat einen schnellen CSV-Loader, und das Ergebnis wird in einem Dataframe platziert, wie in den vorherigen Lektionen. Die CSV, die wir laden, hat über eine halbe Million Zeilen, aber nur 17 Spalten. Pandas bietet viele leistungsstarke Möglichkeiten, mit einem Dataframe zu interagieren, einschließlich der Fähigkeit, Operationen auf jeder Zeile auszuführen.

Von hier an in dieser Lektion wird es Code-Snippets geben sowie einige Erklärungen zum Code und Diskussionen darüber, was die Ergebnisse bedeuten. Verwenden Sie das enthaltene _notebook.ipynb_ für Ihren Code.

Beginnen wir mit dem Laden der Datendatei, die Sie verwenden werden:

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

Jetzt, da die Daten geladen sind, können wir einige Operationen darauf ausführen. Halten Sie diesen Code oben in Ihrem Programm für den nächsten Teil.

## Daten erkunden

In diesem Fall sind die Daten bereits *sauber*, das bedeutet, dass sie bereit sind, verarbeitet zu werden, und keine Zeichen in anderen Sprachen enthalten, die Algorithmen, die nur englische Zeichen erwarten, durcheinanderbringen könnten.

✅ Möglicherweise müssen Sie mit Daten arbeiten, die eine anfängliche Verarbeitung erfordern, um sie zu formatieren, bevor Sie NLP-Techniken anwenden können, aber diesmal nicht. Wenn Sie es müssten, wie würden Sie mit nicht-englischen Zeichen umgehen?

Nehmen Sie sich einen Moment Zeit, um sicherzustellen, dass Sie die Daten, sobald sie geladen sind, mit Code erkunden können. Es ist sehr verlockend, sich auf die Spalten `Negative_Review` und `Positive_Review` zu konzentrieren. Sie sind mit natürlichem Text gefüllt, den Ihre NLP-Algorithmen verarbeiten können. Aber warten Sie! Bevor Sie in NLP und Stimmungsanalyse eintauchen, sollten Sie den folgenden Code verwenden, um festzustellen, ob die im Datensatz angegebenen Werte mit den Werten übereinstimmen, die Sie mit Pandas berechnen.

## Dataframe-Operationen

Die erste Aufgabe in dieser Lektion besteht darin, zu überprüfen, ob die folgenden Annahmen korrekt sind, indem Sie Code schreiben, der den Dataframe untersucht (ohne ihn zu ändern).

> Wie bei vielen Programmieraufgaben gibt es mehrere Möglichkeiten, dies zu erledigen, aber ein guter Rat ist, es auf die einfachste und leichteste Weise zu tun, insbesondere wenn es einfacher zu verstehen ist, wenn Sie später zu diesem Code zurückkehren. Mit Dataframes gibt es eine umfassende API, die oft eine Möglichkeit bietet, das, was Sie wollen, effizient zu erledigen.

Behandeln Sie die folgenden Fragen als Programmieraufgaben und versuchen Sie, sie zu beantworten, ohne die Lösung anzusehen.

1. Geben Sie die *Form* des Dataframes aus, den Sie gerade geladen haben (die Form ist die Anzahl der Zeilen und Spalten).
2. Berechnen Sie die Häufigkeit der Nationalitäten der Bewerter:
   1. Wie viele unterschiedliche Werte gibt es in der Spalte `Reviewer_Nationality` und welche sind das?
   2. Welche Nationalität der Bewerter ist die häufigste im Datensatz (Land und Anzahl der Bewertungen ausgeben)?
   3. Was sind die nächsten 10 am häufigsten vorkommenden Nationalitäten und ihre Häufigkeitszählung?
3. Welches Hotel wurde am häufigsten von jeder der 10 häufigsten Nationalitäten bewertet?
4. Wie viele Bewertungen gibt es pro Hotel (Häufigkeitszählung der Hotels) im Datensatz?
5. Obwohl es eine Spalte `Average_Score` für jedes Hotel im Datensatz gibt, können Sie auch eine durchschnittliche Bewertung berechnen (indem Sie den Durchschnitt aller Bewerterbewertungen im Datensatz für jedes Hotel berechnen). Fügen Sie Ihrem Dataframe eine neue Spalte mit der Spaltenüberschrift `Calc_Average_Score` hinzu, die diesen berechneten Durchschnitt enthält.
6. Haben einige Hotels denselben (auf eine Dezimalstelle gerundeten) `Average_Score` und `Calc_Average_Score`?
   1. Versuchen Sie, eine Python-Funktion zu schreiben, die eine Serie (Zeile) als Argument nimmt und die Werte vergleicht, wobei eine Nachricht ausgegeben wird, wenn die Werte nicht gleich sind. Verwenden Sie dann die `.apply()`-Methode, um jede Zeile mit der Funktion zu verarbeiten.
7. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Negative_Review` den Wert "No Negative" haben.
8. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" haben.
9. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" **und** in der Spalte `Negative_Review` den Wert "No Negative" haben.

### Code-Antworten

1. Geben Sie die *Form* des Dataframes aus, den Sie gerade geladen haben (die Form ist die Anzahl der Zeilen und Spalten).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Berechnen Sie die Häufigkeit der Nationalitäten der Bewerter:

   1. Wie viele unterschiedliche Werte gibt es in der Spalte `Reviewer_Nationality` und welche sind das?
   2. Welche Nationalität der Bewerter ist die häufigste im Datensatz (Land und Anzahl der Bewertungen ausgeben)?

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

   3. Was sind die nächsten 10 am häufigsten vorkommenden Nationalitäten und ihre Häufigkeitszählung?

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

3. Welches Hotel wurde am häufigsten von jeder der 10 häufigsten Nationalitäten bewertet?

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

4. Wie viele Bewertungen gibt es pro Hotel (Häufigkeitszählung der Hotels) im Datensatz?

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
   
   Sie werden vielleicht bemerken, dass die *im Datensatz gezählten* Ergebnisse nicht mit dem Wert in `Total_Number_of_Reviews` übereinstimmen. Es ist unklar, ob dieser Wert im Datensatz die Gesamtzahl der Bewertungen darstellt, die das Hotel hatte, aber nicht alle wurden abgerufen, oder ob es sich um eine andere Berechnung handelt. `Total_Number_of_Reviews` wird im Modell aufgrund dieser Unklarheit nicht verwendet.

5. Obwohl es eine Spalte `Average_Score` für jedes Hotel im Datensatz gibt, können Sie auch eine durchschnittliche Bewertung berechnen (indem Sie den Durchschnitt aller Bewerterbewertungen im Datensatz für jedes Hotel berechnen). Fügen Sie Ihrem Dataframe eine neue Spalte mit der Spaltenüberschrift `Calc_Average_Score` hinzu, die diesen berechneten Durchschnitt enthält. Geben Sie die Spalten `Hotel_Name`, `Average_Score` und `Calc_Average_Score` aus.

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

   Sie könnten sich auch über den Wert `Average_Score` wundern und warum er manchmal von der berechneten durchschnittlichen Bewertung abweicht. Da wir nicht wissen können, warum einige Werte übereinstimmen, andere jedoch eine Abweichung aufweisen, ist es in diesem Fall am sichersten, die Bewertungswerte zu verwenden, die wir haben, um den Durchschnitt selbst zu berechnen. Das gesagt, die Unterschiede sind normalerweise sehr klein, hier sind die Hotels mit der größten Abweichung zwischen dem Durchschnitt des Datensatzes und dem berechneten Durchschnitt:

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

   Da nur 1 Hotel eine Bewertungsabweichung von mehr als 1 hat, können wir die Abweichung wahrscheinlich ignorieren und den berechneten Durchschnitt verwenden.

6. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Negative_Review` den Wert "No Negative" haben.

7. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" haben.

8. Berechnen und geben Sie aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" **und** in der Spalte `Negative_Review` den Wert "No Negative" haben.

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

## Eine andere Methode

Eine andere Möglichkeit, Elemente ohne Lambdas zu zählen, und die Summe zu verwenden, um die Zeilen zu zählen:

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

   Sie haben vielleicht bemerkt, dass es 127 Zeilen gibt, die sowohl "No Negative" als auch "No Positive" Werte für die Spalten `Negative_Review` und `Positive_Review` haben. Das bedeutet, dass der Bewerter dem Hotel eine numerische Bewertung gegeben hat, aber darauf verzichtet hat, eine positive oder negative Bewertung zu schreiben. Glücklicherweise handelt es sich um eine kleine Anzahl von Zeilen (127 von 515738, oder 0,02%), sodass es wahrscheinlich unser Modell oder die Ergebnisse in keine bestimmte Richtung verzerren wird. Aber Sie hätten vielleicht nicht erwartet, dass ein Datensatz mit Bewertungen Zeilen ohne Bewertungen enthält, daher lohnt es sich, die Daten zu erkunden, um solche Zeilen zu entdecken.

Jetzt, da Sie den Datensatz erkundet haben, werden Sie in der nächsten Lektion die Daten filtern und eine Stimmungsanalyse hinzufügen.

---
## 🚀 Herausforderung

Diese Lektion zeigt, wie wir in den vorherigen Lektionen gesehen haben, wie wichtig es ist, Ihre Daten und ihre Eigenheiten zu verstehen, bevor Sie Operationen darauf ausführen. Textbasierte Daten erfordern besonders sorgfältige Prüfung. Durchsuchen Sie verschiedene textlastige Datensätze und sehen Sie, ob Sie Bereiche entdecken können, die Vorurteile oder verzerrte Stimmungen in ein Modell einführen könnten.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## Überprüfung & Selbststudium

Nehmen Sie [diesen Lernpfad zu NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), um Werkzeuge zu entdecken, die Sie ausprobieren können, wenn Sie sprach- und textlastige Modelle erstellen.

## Aufgabe 

[NLTK](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.