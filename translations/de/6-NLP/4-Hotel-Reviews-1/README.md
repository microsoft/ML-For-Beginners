<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-04T22:06:46+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "de"
}
-->
# Sentimentanalyse mit Hotelbewertungen - Datenverarbeitung

In diesem Abschnitt verwenden Sie die Techniken aus den vorherigen Lektionen, um eine explorative Datenanalyse eines großen Datensatzes durchzuführen. Sobald Sie ein gutes Verständnis für die Nützlichkeit der verschiedenen Spalten haben, lernen Sie:

- wie man unnötige Spalten entfernt
- wie man neue Daten basierend auf den vorhandenen Spalten berechnet
- wie man den resultierenden Datensatz speichert, um ihn in der finalen Herausforderung zu verwenden

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

### Einführung

Bisher haben Sie gelernt, dass Textdaten sich stark von numerischen Datentypen unterscheiden. Wenn es sich um Text handelt, der von einem Menschen geschrieben oder gesprochen wurde, kann er analysiert werden, um Muster, Häufigkeiten, Stimmungen und Bedeutungen zu erkennen. Diese Lektion führt Sie in einen realen Datensatz mit einer echten Herausforderung ein: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, der unter einer [CC0: Public Domain Lizenz](https://creativecommons.org/publicdomain/zero/1.0/) steht. Er wurde von Booking.com aus öffentlichen Quellen extrahiert. Der Ersteller des Datensatzes ist Jiashen Liu.

### Vorbereitung

Sie benötigen:

* Die Fähigkeit, .ipynb-Notebooks mit Python 3 auszuführen
* pandas
* NLTK, [das Sie lokal installieren sollten](https://www.nltk.org/install.html)
* Den Datensatz, der auf Kaggle verfügbar ist: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Er ist etwa 230 MB groß, wenn er entpackt ist. Laden Sie ihn in den Root-Ordner `/data` herunter, der mit diesen NLP-Lektionen verbunden ist.

## Explorative Datenanalyse

Diese Herausforderung geht davon aus, dass Sie einen Hotel-Empfehlungsbot mit Sentimentanalyse und Gästebewertungen erstellen. Der Datensatz, den Sie verwenden, enthält Bewertungen von 1493 verschiedenen Hotels in 6 Städten.

Mit Python, einem Datensatz von Hotelbewertungen und der Sentimentanalyse von NLTK könnten Sie herausfinden:

* Welche Wörter und Phrasen werden in Bewertungen am häufigsten verwendet?
* Korrelieren die offiziellen *Tags*, die ein Hotel beschreiben, mit den Bewertungsergebnissen (z. B. gibt es mehr negative Bewertungen für ein bestimmtes Hotel von *Familien mit kleinen Kindern* als von *Alleinreisenden*, was darauf hindeuten könnte, dass es besser für *Alleinreisende* geeignet ist)?
* Stimmen die Sentiment-Scores von NLTK mit den numerischen Bewertungen der Hotelgäste überein?

#### Datensatz

Lassen Sie uns den Datensatz, den Sie heruntergeladen und lokal gespeichert haben, erkunden. Öffnen Sie die Datei in einem Editor wie VS Code oder sogar Excel.

Die Kopfzeilen des Datensatzes sind wie folgt:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hier sind sie gruppiert, um sie leichter zu untersuchen: 
##### Hotel-Spalten

* `Hotel_Name`, `Hotel_Address`, `lat` (Breitengrad), `lng` (Längengrad)
  * Mit *lat* und *lng* könnten Sie eine Karte mit Python erstellen, die die Hotelstandorte anzeigt (vielleicht farblich codiert für negative und positive Bewertungen).
  * Hotel_Address ist für uns nicht offensichtlich nützlich und wir werden es wahrscheinlich durch ein Land ersetzen, um die Sortierung und Suche zu erleichtern.

**Hotel-Meta-Bewertungsspalten**

* `Average_Score`
  * Laut dem Ersteller des Datensatzes ist diese Spalte der *Durchschnittliche Score des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*. Dies scheint eine ungewöhnliche Methode zur Berechnung des Scores zu sein, aber es sind die extrahierten Daten, daher nehmen wir sie zunächst so hin. 
  
  ✅ Basierend auf den anderen Spalten in diesen Daten: Können Sie sich eine andere Methode vorstellen, um den Durchschnittsscore zu berechnen?

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
  - Beachten Sie, dass ein Bewerter möglicherweise eine positive Bewertung in der Spalte Negative Review schreibt (z. B. "there is nothing bad about this hotel").
- `Review_Total_Negative_Word_Counts`
  - Höhere negative Wortanzahlen deuten auf einen niedrigeren Score hin (ohne die Sentimentalität zu überprüfen).
- `Positive_Review`
  - Wenn ein Bewerter nichts geschrieben hat, enthält dieses Feld "**No Positive**".
  - Beachten Sie, dass ein Bewerter möglicherweise eine negative Bewertung in der Spalte Positive Review schreibt (z. B. "there is nothing good about this hotel at all").
- `Review_Total_Positive_Word_Counts`
  - Höhere positive Wortanzahlen deuten auf einen höheren Score hin (ohne die Sentimentalität zu überprüfen).
- `Review_Date` und `days_since_review`
  - Eine Frische- oder Veraltungsmaßnahme könnte auf eine Bewertung angewendet werden (ältere Bewertungen könnten weniger genau sein als neuere, da sich das Hotelmanagement geändert hat, Renovierungen durchgeführt wurden oder ein Pool hinzugefügt wurde usw.).
- `Tags`
  - Dies sind kurze Beschreibungen, die ein Bewerter möglicherweise auswählt, um die Art des Gastes zu beschreiben, der er war (z. B. allein oder Familie), die Art des Zimmers, die Aufenthaltsdauer und wie die Bewertung eingereicht wurde.
  - Leider ist die Verwendung dieser Tags problematisch. Siehe den Abschnitt unten, der ihre Nützlichkeit diskutiert.

**Bewerterspalten**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dies könnte ein Faktor in einem Empfehlungsmodell sein, z. B. wenn Sie feststellen könnten, dass produktivere Bewerter mit Hunderten von Bewertungen eher negativ als positiv sind. Der Bewerter einer bestimmten Bewertung wird jedoch nicht mit einem eindeutigen Code identifiziert und kann daher nicht mit einer Reihe von Bewertungen verknüpft werden. Es gibt 30 Bewerter mit 100 oder mehr Bewertungen, aber es ist schwer zu erkennen, wie dies das Empfehlungsmodell unterstützen könnte.
- `Reviewer_Nationality`
  - Manche Leute könnten denken, dass bestimmte Nationalitäten eher positive oder negative Bewertungen abgeben, basierend auf einer nationalen Neigung. Seien Sie vorsichtig, solche anekdotischen Ansichten in Ihre Modelle einzubauen. Dies sind nationale (und manchmal rassistische) Stereotypen, und jeder Bewerter war eine Einzelperson, die eine Bewertung basierend auf ihrer Erfahrung geschrieben hat. Diese könnte durch viele Filter wie frühere Hotelaufenthalte, die zurückgelegte Entfernung und ihr persönliches Temperament beeinflusst worden sein. Zu denken, dass ihre Nationalität der Grund für eine Bewertung war, ist schwer zu rechtfertigen.

##### Beispiele

| Durchschnittlicher Score | Gesamtanzahl Bewertungen | Bewerter-Score | Negative <br />Bewertung                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive Bewertung                 | Tags                                                                                      |
| ------------------------- | ------------------------ | -------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8                       | 1945                    | 2.5            | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Wie Sie sehen können, hatte dieser Gast keinen angenehmen Aufenthalt in diesem Hotel. Das Hotel hat einen guten Durchschnittsscore von 7,8 und 1945 Bewertungen, aber dieser Bewerter gab ihm 2,5 und schrieb 115 Wörter darüber, wie negativ sein Aufenthalt war. Wenn sie überhaupt nichts in der Spalte Positive_Review geschrieben hätten, könnten Sie vermuten, dass es nichts Positives gab, aber sie schrieben 7 Worte der Warnung. Wenn wir nur Wörter zählen würden, anstatt die Bedeutung oder Stimmung der Wörter zu berücksichtigen, könnten wir eine verzerrte Sicht auf die Absicht des Bewerters haben. Seltsamerweise ist ihre Bewertung von 2,5 verwirrend, denn wenn der Hotelaufenthalt so schlecht war, warum überhaupt Punkte vergeben? Wenn Sie den Datensatz genau untersuchen, werden Sie feststellen, dass der niedrigste mögliche Score 2,5 und nicht 0 ist. Der höchste mögliche Score ist 10.

##### Tags

Wie oben erwähnt, scheint die Idee, `Tags` zur Kategorisierung der Daten zu verwenden, auf den ersten Blick sinnvoll. Leider sind diese Tags nicht standardisiert, was bedeutet, dass in einem bestimmten Hotel die Optionen *Single room*, *Twin room* und *Double room* sein könnten, während sie im nächsten Hotel *Deluxe Single Room*, *Classic Queen Room* und *Executive King Room* sind. Dies könnten dieselben Dinge sein, aber es gibt so viele Variationen, dass die Wahl besteht zwischen:

1. Der Versuch, alle Begriffe auf einen einzigen Standard zu ändern, was sehr schwierig ist, da nicht klar ist, wie der Konvertierungspfad in jedem Fall aussehen würde (z. B. *Classic single room* wird zu *Single room*, aber *Superior Queen Room with Courtyard Garden or City View* ist viel schwieriger zuzuordnen).

1. Wir können einen NLP-Ansatz verfolgen und die Häufigkeit bestimmter Begriffe wie *Solo*, *Business Traveller* oder *Family with young kids* messen, wie sie auf jedes Hotel angewendet werden, und dies in das Empfehlungsmodell einfließen lassen.

Tags sind normalerweise (aber nicht immer) ein einzelnes Feld, das eine Liste von 5 bis 6 durch Kommas getrennten Werten enthält, die sich auf *Art der Reise*, *Art der Gäste*, *Art des Zimmers*, *Anzahl der Nächte* und *Art des Geräts, auf dem die Bewertung eingereicht wurde* beziehen. Da jedoch einige Bewerter nicht jedes Feld ausfüllen (sie könnten eines leer lassen), sind die Werte nicht immer in derselben Reihenfolge.

Ein Beispiel: Nehmen Sie *Art der Gruppe*. Es gibt 1025 einzigartige Möglichkeiten in diesem Feld in der Spalte `Tags`, und leider beziehen sich nur einige davon auf eine Gruppe (einige sind die Art des Zimmers usw.). Wenn Sie nur die Ergebnisse filtern, die Familie erwähnen, enthalten die Ergebnisse viele *Family room*-Typen. Wenn Sie den Begriff *with* einbeziehen, d. h. die *Family with*-Werte zählen, sind die Ergebnisse besser, mit über 80.000 der 515.000 Ergebnisse, die die Phrase "Family with young children" oder "Family with older children" enthalten.

Das bedeutet, dass die Tags-Spalte für uns nicht völlig nutzlos ist, aber es wird einige Arbeit erfordern, sie nützlich zu machen.

##### Durchschnittlicher Hotelscore

Es gibt eine Reihe von Unstimmigkeiten oder Diskrepanzen im Datensatz, die ich nicht erklären kann, die aber hier illustriert werden, damit Sie sich ihrer bewusst sind, wenn Sie Ihre Modelle erstellen. Wenn Sie es herausfinden, lassen Sie es uns bitte im Diskussionsbereich wissen!

Der Datensatz enthält die folgenden Spalten, die sich auf den Durchschnittsscore und die Anzahl der Bewertungen beziehen:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Das einzelne Hotel mit den meisten Bewertungen in diesem Datensatz ist *Britannia International Hotel Canary Wharf* mit 4789 Bewertungen von insgesamt 515.000. Wenn wir jedoch den Wert `Total_Number_of_Reviews` für dieses Hotel betrachten, beträgt er 9086. Sie könnten vermuten, dass es viele weitere Scores ohne Bewertungen gibt, daher sollten wir vielleicht den Wert der Spalte `Additional_Number_of_Scoring` hinzufügen. Dieser Wert beträgt 2682, und wenn wir ihn zu 4789 addieren, erhalten wir 7471, was immer noch 1615 weniger ist als der Wert von `Total_Number_of_Reviews`.

Wenn Sie die Spalte `Average_Score` betrachten, könnten Sie vermuten, dass es sich um den Durchschnitt der Bewertungen im Datensatz handelt, aber die Beschreibung von Kaggle lautet: "*Durchschnittlicher Score des Hotels, berechnet basierend auf dem neuesten Kommentar im letzten Jahr*". Das scheint nicht sehr nützlich zu sein, aber wir können unseren eigenen Durchschnitt basierend auf den Bewertungsscores im Datensatz berechnen. Am Beispiel desselben Hotels wird der durchschnittliche Hotelscore mit 7,1 angegeben, aber der berechnete Score (durchschnittlicher Bewerterscore *im* Datensatz) beträgt 6,8. Das ist nah dran, aber nicht derselbe Wert, und wir können nur vermuten, dass die in den `Additional_Number_of_Scoring`-Bewertungen angegebenen Scores den Durchschnitt auf 7,1 erhöht haben. Leider ist es ohne Möglichkeit, diese Annahme zu testen oder zu beweisen, schwierig, `Average_Score`, `Additional_Number_of_Scoring` und `Total_Number_of_Reviews` zu verwenden oder ihnen zu vertrauen, wenn sie auf Daten basieren oder sich auf Daten beziehen, die wir nicht haben.

Um die Sache weiter zu verkomplizieren, hat das Hotel mit der zweithöchsten Anzahl an Bewertungen einen berechneten Durchschnittsscore von 8,12, und der `Average_Score` im Datensatz beträgt 8,1. Ist dieser korrekte Score ein Zufall, oder ist das erste Hotel eine Diskrepanz?

In der Annahme, dass dieses Hotel ein Ausreißer sein könnte und dass vielleicht die meisten Werte übereinstimmen (aber einige aus irgendeinem Grund nicht), werden wir als Nächstes ein kurzes Programm schreiben, um die Werte im Datensatz zu untersuchen und die korrekte Verwendung (oder Nichtverwendung) der Werte zu bestimmen.
> 🚨 Ein Hinweis zur Vorsicht
>
> Bei der Arbeit mit diesem Datensatz werden Sie Code schreiben, der etwas aus dem Text berechnet, ohne dass Sie den Text selbst lesen oder analysieren müssen. Das ist das Wesen von NLP: Bedeutung oder Stimmung zu interpretieren, ohne dass ein Mensch dies tun muss. Es ist jedoch möglich, dass Sie einige der negativen Bewertungen lesen. Ich möchte Sie dringend bitten, dies nicht zu tun, da es nicht notwendig ist. Einige davon sind albern oder irrelevante negative Hotelbewertungen, wie zum Beispiel: "Das Wetter war nicht gut" – etwas, das außerhalb der Kontrolle des Hotels oder überhaupt irgendjemandes liegt. Aber es gibt auch eine dunkle Seite bei einigen Bewertungen. Manchmal sind die negativen Bewertungen rassistisch, sexistisch oder altersdiskriminierend. Das ist bedauerlich, aber zu erwarten, wenn ein Datensatz von einer öffentlichen Website extrahiert wird. Einige Rezensenten hinterlassen Bewertungen, die Sie geschmacklos, unangenehm oder verstörend finden könnten. Es ist besser, den Code die Stimmung messen zu lassen, als sie selbst zu lesen und sich darüber aufzuregen. Das gesagt, es ist nur eine Minderheit, die solche Dinge schreibt, aber sie existiert dennoch.
## Übung – Datenexploration
### Daten laden

Genug mit der visuellen Untersuchung der Daten – jetzt wirst du etwas Code schreiben und Antworten finden! In diesem Abschnitt verwenden wir die Bibliothek pandas. Deine erste Aufgabe ist sicherzustellen, dass du die CSV-Daten laden und lesen kannst. Die pandas-Bibliothek verfügt über einen schnellen CSV-Loader, und das Ergebnis wird, wie in den vorherigen Lektionen, in einem DataFrame gespeichert. Die CSV-Datei, die wir laden, enthält über eine halbe Million Zeilen, aber nur 17 Spalten. Pandas bietet dir viele leistungsstarke Möglichkeiten, mit einem DataFrame zu interagieren, einschließlich der Möglichkeit, Operationen auf jeder Zeile auszuführen.

Ab hier enthält diese Lektion Code-Snippets, Erklärungen zum Code und Diskussionen darüber, was die Ergebnisse bedeuten. Verwende das beigefügte _notebook.ipynb_ für deinen Code.

Beginnen wir mit dem Laden der Datendatei, die du verwenden wirst:

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

Sobald die Daten geladen sind, können wir einige Operationen darauf ausführen. Halte diesen Code für den nächsten Abschnitt oben in deinem Programm.

## Daten erkunden

In diesem Fall sind die Daten bereits *sauber*, das bedeutet, dass sie bereit zur Verarbeitung sind und keine Zeichen in anderen Sprachen enthalten, die Algorithmen, die nur englische Zeichen erwarten, verwirren könnten.

✅ Es könnte sein, dass du mit Daten arbeiten musst, die eine anfängliche Verarbeitung erfordern, um sie zu formatieren, bevor du NLP-Techniken anwendest – aber diesmal nicht. Wenn du es müsstest, wie würdest du mit nicht-englischen Zeichen umgehen?

Nimm dir einen Moment Zeit, um sicherzustellen, dass du die Daten nach dem Laden mit Code erkunden kannst. Es ist sehr verlockend, sich auf die Spalten `Negative_Review` und `Positive_Review` zu konzentrieren. Diese enthalten natürlichen Text, den deine NLP-Algorithmen verarbeiten können. Aber warte! Bevor du mit NLP und Sentiment-Analyse beginnst, solltest du den folgenden Code verwenden, um sicherzustellen, dass die im Datensatz angegebenen Werte mit den Werten übereinstimmen, die du mit pandas berechnest.

## DataFrame-Operationen

Die erste Aufgabe in dieser Lektion besteht darin, zu überprüfen, ob die folgenden Annahmen korrekt sind, indem du Code schreibst, der den DataFrame untersucht (ohne ihn zu ändern).

> Wie bei vielen Programmieraufgaben gibt es mehrere Möglichkeiten, dies zu lösen. Ein guter Rat ist jedoch, es auf die einfachste und verständlichste Weise zu tun, insbesondere wenn es einfacher zu verstehen ist, wenn du später zu diesem Code zurückkehrst. Mit DataFrames gibt es eine umfassende API, die oft eine effiziente Möglichkeit bietet, das zu tun, was du möchtest.

Behandle die folgenden Fragen als Programmieraufgaben und versuche, sie zu beantworten, ohne die Lösung anzusehen.

1. Gib die *Form* des gerade geladenen DataFrames aus (die Form ist die Anzahl der Zeilen und Spalten).
2. Berechne die Häufigkeit der Rezensenten-Nationalitäten:
   1. Wie viele unterschiedliche Werte gibt es in der Spalte `Reviewer_Nationality` und welche sind das?
   2. Welche Nationalität der Rezensenten ist im Datensatz am häufigsten (Land und Anzahl der Bewertungen ausgeben)?
   3. Was sind die nächsten 10 am häufigsten vorkommenden Nationalitäten und deren Häufigkeit?
3. Welches Hotel wurde für jede der 10 häufigsten Rezensenten-Nationalitäten am häufigsten bewertet?
4. Wie viele Bewertungen gibt es pro Hotel (Häufigkeit der Bewertungen pro Hotel im Datensatz)?
5. Obwohl es im Datensatz eine Spalte `Average_Score` für jedes Hotel gibt, kannst du auch einen Durchschnittswert berechnen (indem du den Durchschnitt aller Rezensentenbewertungen im Datensatz für jedes Hotel berechnest). Füge deinem DataFrame eine neue Spalte mit der Überschrift `Calc_Average_Score` hinzu, die diesen berechneten Durchschnitt enthält.
6. Haben einige Hotels denselben (auf eine Dezimalstelle gerundeten) `Average_Score` und `Calc_Average_Score`?
   1. Versuche, eine Python-Funktion zu schreiben, die eine Series (Zeile) als Argument nimmt und die Werte vergleicht. Gib eine Nachricht aus, wenn die Werte nicht übereinstimmen. Verwende dann die `.apply()`-Methode, um jede Zeile mit der Funktion zu verarbeiten.
7. Berechne und gib aus, wie viele Zeilen in der Spalte `Negative_Review` den Wert "No Negative" haben.
8. Berechne und gib aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" haben.
9. Berechne und gib aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" **und** in der Spalte `Negative_Review` den Wert "No Negative" haben.

### Code-Antworten

1. Gib die *Form* des gerade geladenen DataFrames aus (die Form ist die Anzahl der Zeilen und Spalten).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Berechne die Häufigkeit der Rezensenten-Nationalitäten:

   1. Wie viele unterschiedliche Werte gibt es in der Spalte `Reviewer_Nationality` und welche sind das?
   2. Welche Nationalität der Rezensenten ist im Datensatz am häufigsten (Land und Anzahl der Bewertungen ausgeben)?

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

   3. Was sind die nächsten 10 am häufigsten vorkommenden Nationalitäten und deren Häufigkeit?

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

3. Welches Hotel wurde für jede der 10 häufigsten Rezensenten-Nationalitäten am häufigsten bewertet?

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

4. Wie viele Bewertungen gibt es pro Hotel (Häufigkeit der Bewertungen pro Hotel im Datensatz)?

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

   Du wirst feststellen, dass die *im Datensatz gezählten* Ergebnisse nicht mit dem Wert in `Total_Number_of_Reviews` übereinstimmen. Es ist unklar, ob dieser Wert im Datensatz die Gesamtzahl der Bewertungen des Hotels darstellt, von denen nicht alle erfasst wurden, oder ob eine andere Berechnung vorliegt. `Total_Number_of_Reviews` wird aufgrund dieser Unklarheit nicht im Modell verwendet.

5. Obwohl es im Datensatz eine Spalte `Average_Score` für jedes Hotel gibt, kannst du auch einen Durchschnittswert berechnen (indem du den Durchschnitt aller Rezensentenbewertungen im Datensatz für jedes Hotel berechnest). Füge deinem DataFrame eine neue Spalte mit der Überschrift `Calc_Average_Score` hinzu, die diesen berechneten Durchschnitt enthält. Gib die Spalten `Hotel_Name`, `Average_Score` und `Calc_Average_Score` aus.

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

   Du fragst dich vielleicht, warum der Wert in `Average_Score` manchmal vom berechneten Durchschnittswert abweicht. Da wir nicht wissen, warum einige Werte übereinstimmen, andere jedoch abweichen, ist es in diesem Fall sicherer, die Bewertungswerte zu verwenden, die wir haben, um den Durchschnitt selbst zu berechnen. Die Abweichungen sind jedoch in der Regel sehr gering. Hier sind die Hotels mit der größten Abweichung zwischen dem Datensatzdurchschnitt und dem berechneten Durchschnitt:

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

   Da nur ein Hotel eine Abweichung von mehr als 1 Punkt hat, können wir die Abweichung wahrscheinlich ignorieren und den berechneten Durchschnittswert verwenden.

6. Berechne und gib aus, wie viele Zeilen in der Spalte `Negative_Review` den Wert "No Negative" haben.

7. Berechne und gib aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" haben.

8. Berechne und gib aus, wie viele Zeilen in der Spalte `Positive_Review` den Wert "No Positive" **und** in der Spalte `Negative_Review` den Wert "No Negative" haben.

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

Eine andere Möglichkeit, Elemente ohne Lambdas zu zählen, ist die Verwendung von `sum`, um die Zeilen zu zählen:

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

   Du hast vielleicht bemerkt, dass es 127 Zeilen gibt, die sowohl "No Negative" als auch "No Positive" in den Spalten `Negative_Review` und `Positive_Review` enthalten. Das bedeutet, dass der Rezensent dem Hotel eine numerische Bewertung gegeben hat, aber darauf verzichtet hat, eine positive oder negative Bewertung zu schreiben. Glücklicherweise handelt es sich hierbei um eine kleine Anzahl von Zeilen (127 von 515738, also 0,02 %), sodass dies unser Modell oder die Ergebnisse wahrscheinlich nicht in eine bestimmte Richtung verzerren wird. Dennoch hättest du vielleicht nicht erwartet, dass ein Datensatz mit Bewertungen Zeilen ohne Bewertungen enthält. Es lohnt sich also, die Daten zu erkunden, um solche Zeilen zu entdecken.

Nachdem du den Datensatz erkundet hast, wirst du in der nächsten Lektion die Daten filtern und eine Sentiment-Analyse hinzufügen.

---
## 🚀 Herausforderung

Diese Lektion zeigt, wie wir bereits in früheren Lektionen gesehen haben, wie wichtig es ist, die Daten und ihre Eigenheiten genau zu verstehen, bevor man Operationen darauf ausführt. Insbesondere textbasierte Daten erfordern eine sorgfältige Prüfung. Durchsuche verschiedene textlastige Datensätze und finde heraus, ob du Bereiche entdecken kannst, die Vorurteile oder verzerrte Stimmungen in ein Modell einbringen könnten.

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Überprüfung & Selbststudium

Nimm an [diesem Lernpfad zu NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) teil, um Werkzeuge zu entdecken, die du beim Erstellen von sprach- und textlastigen Modellen ausprobieren kannst.

## Aufgabe

[NLTK](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.