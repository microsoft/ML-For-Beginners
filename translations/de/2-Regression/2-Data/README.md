<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-04T21:52:58+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "de"
}
-->
# Erstellen eines Regressionsmodells mit Scikit-learn: Daten vorbereiten und visualisieren

![Infografik zur Datenvisualisierung](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz vor der Lektion](https://ff-quizzes.netlify.app/en/ml/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Einführung

Nachdem Sie nun die notwendigen Werkzeuge eingerichtet haben, um mit dem Aufbau von Machine-Learning-Modellen in Scikit-learn zu beginnen, können Sie anfangen, Fragen an Ihre Daten zu stellen. Wenn Sie mit Daten arbeiten und ML-Lösungen anwenden, ist es entscheidend, die richtigen Fragen zu stellen, um das volle Potenzial Ihres Datensatzes auszuschöpfen.

In dieser Lektion lernen Sie:

- Wie Sie Ihre Daten für den Modellaufbau vorbereiten.
- Wie Sie Matplotlib für die Datenvisualisierung nutzen.

## Die richtigen Fragen an Ihre Daten stellen

Die Frage, die Sie beantwortet haben möchten, bestimmt, welche Art von ML-Algorithmen Sie verwenden. Und die Qualität der Antwort hängt stark von der Beschaffenheit Ihrer Daten ab.

Werfen Sie einen Blick auf die [Daten](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), die für diese Lektion bereitgestellt wurden. Sie können diese .csv-Datei in VS Code öffnen. Ein kurzer Blick zeigt sofort, dass es Lücken gibt und eine Mischung aus Zeichenketten und numerischen Daten vorliegt. Es gibt auch eine seltsame Spalte namens 'Package', in der die Daten eine Mischung aus 'sacks', 'bins' und anderen Werten sind. Die Daten sind tatsächlich ein bisschen chaotisch.

[![ML für Anfänger - Wie man einen Datensatz analysiert und bereinigt](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML für Anfänger - Wie man einen Datensatz analysiert und bereinigt")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zur Vorbereitung der Daten für diese Lektion anzusehen.

Es ist tatsächlich nicht sehr häufig, dass man einen Datensatz erhält, der vollständig bereit ist, um direkt ein ML-Modell zu erstellen. In dieser Lektion lernen Sie, wie Sie einen Rohdatensatz mit Standard-Python-Bibliotheken vorbereiten. Außerdem lernen Sie verschiedene Techniken zur Visualisierung der Daten kennen.

## Fallstudie: 'Der Kürbismarkt'

In diesem Ordner finden Sie eine .csv-Datei im Stammverzeichnis des `data`-Ordners namens [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), die 1757 Zeilen mit Daten über den Kürbismarkt enthält, sortiert nach Städten. Dies sind Rohdaten, die aus den [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) des US-Landwirtschaftsministeriums extrahiert wurden.

### Daten vorbereiten

Diese Daten sind gemeinfrei. Sie können in vielen separaten Dateien, nach Städten sortiert, von der USDA-Website heruntergeladen werden. Um zu vermeiden, dass zu viele separate Dateien vorliegen, haben wir alle Städtedaten in eine Tabelle zusammengeführt, sodass die Daten bereits etwas _vorbereitet_ wurden. Schauen wir uns die Daten nun genauer an.

### Die Kürbis-Daten - erste Eindrücke

Was fällt Ihnen an diesen Daten auf? Sie haben bereits gesehen, dass es eine Mischung aus Zeichenketten, Zahlen, Lücken und seltsamen Werten gibt, die Sie interpretieren müssen.

Welche Frage könnten Sie mit diesen Daten unter Verwendung einer Regressionsmethode stellen? Wie wäre es mit: "Den Preis eines Kürbisses für einen bestimmten Monat vorhersagen". Wenn Sie die Daten erneut betrachten, gibt es einige Änderungen, die Sie vornehmen müssen, um die für diese Aufgabe erforderliche Datenstruktur zu erstellen.

## Übung - Analysieren der Kürbis-Daten

Verwenden wir [Pandas](https://pandas.pydata.org/), ein sehr nützliches Tool zur Datenanalyse, um diese Kürbis-Daten zu analysieren und vorzubereiten.

### Zuerst fehlende Daten überprüfen

Zunächst müssen Sie Schritte unternehmen, um fehlende Daten zu überprüfen:

1. Konvertieren Sie die Daten in ein Monatsformat (es handelt sich um US-Daten, daher ist das Format `MM/DD/YYYY`).
2. Extrahieren Sie den Monat in eine neue Spalte.

Öffnen Sie die Datei _notebook.ipynb_ in Visual Studio Code und importieren Sie die Tabelle in ein neues Pandas-Dataframe.

1. Verwenden Sie die Funktion `head()`, um die ersten fünf Zeilen anzuzeigen.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Welche Funktion würden Sie verwenden, um die letzten fünf Zeilen anzuzeigen?

1. Überprüfen Sie, ob im aktuellen Dataframe fehlende Daten vorhanden sind:

    ```python
    pumpkins.isnull().sum()
    ```

    Es gibt fehlende Daten, aber möglicherweise sind diese für die aktuelle Aufgabe nicht relevant.

1. Um Ihr Dataframe einfacher zu gestalten, wählen Sie nur die benötigten Spalten aus, indem Sie die Funktion `loc` verwenden, die aus dem ursprünglichen Dataframe eine Gruppe von Zeilen (als erster Parameter übergeben) und Spalten (als zweiter Parameter übergeben) extrahiert. Der Ausdruck `:` im folgenden Fall bedeutet "alle Zeilen".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Zweitens, den Durchschnittspreis eines Kürbisses bestimmen

Überlegen Sie, wie Sie den Durchschnittspreis eines Kürbisses in einem bestimmten Monat bestimmen können. Welche Spalten würden Sie für diese Aufgabe auswählen? Hinweis: Sie benötigen 3 Spalten.

Lösung: Nehmen Sie den Durchschnitt der Spalten `Low Price` und `High Price`, um die neue Spalte `Price` zu füllen, und konvertieren Sie die Spalte `Date`, sodass nur der Monat angezeigt wird. Glücklicherweise gibt es laut der oben durchgeführten Überprüfung keine fehlenden Daten für Daten oder Preise.

1. Um den Durchschnitt zu berechnen, fügen Sie den folgenden Code hinzu:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Sie können beliebige Daten mit `print(month)` überprüfen.

2. Kopieren Sie nun Ihre konvertierten Daten in ein neues Pandas-Dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Wenn Sie Ihr Dataframe ausgeben, sehen Sie einen sauberen, aufgeräumten Datensatz, auf dessen Grundlage Sie Ihr neues Regressionsmodell erstellen können.

### Aber Moment! Etwas ist hier seltsam

Wenn Sie sich die Spalte `Package` ansehen, werden Kürbisse in vielen verschiedenen Konfigurationen verkauft. Einige werden in '1 1/9 bushel'-Maßen verkauft, andere in '1/2 bushel'-Maßen, einige pro Kürbis, einige pro Pfund und einige in großen Kisten mit unterschiedlichen Breiten.

> Kürbisse scheinen schwer konsistent zu wiegen

Wenn man sich die Originaldaten ansieht, ist es interessant, dass alles mit `Unit of Sale` gleich 'EACH' oder 'PER BIN' auch den `Package`-Typ pro Zoll, pro Kiste oder 'each' hat. Kürbisse scheinen schwer konsistent zu wiegen, daher filtern wir sie, indem wir nur Kürbisse mit dem String 'bushel' in ihrer `Package`-Spalte auswählen.

1. Fügen Sie einen Filter am Anfang der Datei unter dem ersten .csv-Import hinzu:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Wenn Sie die Daten jetzt ausgeben, sehen Sie, dass Sie nur die etwa 415 Zeilen mit Daten erhalten, die Kürbisse nach dem Bushel enthalten.

### Aber Moment! Es gibt noch etwas zu tun

Haben Sie bemerkt, dass die Bushel-Menge pro Zeile variiert? Sie müssen die Preise normalisieren, sodass Sie die Preise pro Bushel anzeigen. Führen Sie also einige Berechnungen durch, um dies zu standardisieren.

1. Fügen Sie diese Zeilen nach dem Block hinzu, der das `new_pumpkins`-Dataframe erstellt:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Laut [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) hängt das Gewicht eines Bushels von der Art des Produkts ab, da es sich um eine Volumenmessung handelt. "Ein Bushel Tomaten wiegt beispielsweise 56 Pfund... Blätter und Grünzeug nehmen mehr Platz mit weniger Gewicht ein, daher wiegt ein Bushel Spinat nur 20 Pfund." Das ist alles ziemlich kompliziert! Lassen Sie uns die Umrechnung von Bushel zu Pfund ignorieren und stattdessen den Preis pro Bushel berechnen. All diese Studien zu Bushels von Kürbissen zeigen jedoch, wie wichtig es ist, die Natur Ihrer Daten zu verstehen!

Nun können Sie die Preise pro Einheit basierend auf ihrer Bushel-Messung analysieren. Wenn Sie die Daten noch einmal ausgeben, sehen Sie, wie sie standardisiert wurden.

✅ Haben Sie bemerkt, dass Kürbisse, die nach dem halben Bushel verkauft werden, sehr teuer sind? Können Sie herausfinden, warum? Hinweis: Kleine Kürbisse sind viel teurer als große, wahrscheinlich weil es viel mehr davon pro Bushel gibt, da ein großer hohler Kürbis viel Platz einnimmt.

## Visualisierungsstrategien

Ein Teil der Aufgabe eines Datenwissenschaftlers besteht darin, die Qualität und Beschaffenheit der Daten, mit denen er arbeitet, zu demonstrieren. Dazu erstellen sie oft interessante Visualisierungen, wie Diagramme, Grafiken und Charts, die verschiedene Aspekte der Daten zeigen. Auf diese Weise können sie Beziehungen und Lücken visuell darstellen, die sonst schwer zu erkennen wären.

[![ML für Anfänger - Wie man Daten mit Matplotlib visualisiert](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML für Anfänger - Wie man Daten mit Matplotlib visualisiert")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zur Visualisierung der Daten für diese Lektion anzusehen.

Visualisierungen können auch dabei helfen, die am besten geeignete Machine-Learning-Technik für die Daten zu bestimmen. Ein Streudiagramm, das einer Linie zu folgen scheint, deutet beispielsweise darauf hin, dass die Daten gut für eine lineare Regression geeignet sind.

Eine Datenvisualisierungsbibliothek, die gut in Jupyter-Notebooks funktioniert, ist [Matplotlib](https://matplotlib.org/) (die Sie auch in der vorherigen Lektion gesehen haben).

> Erhalten Sie mehr Erfahrung mit der Datenvisualisierung in [diesen Tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Übung - Experimentieren mit Matplotlib

Versuchen Sie, einige grundlegende Diagramme zu erstellen, um das neue Dataframe anzuzeigen, das Sie gerade erstellt haben. Was würde ein einfaches Liniendiagramm zeigen?

1. Importieren Sie Matplotlib am Anfang der Datei, unter dem Pandas-Import:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Führen Sie das gesamte Notebook erneut aus, um es zu aktualisieren.
1. Fügen Sie am Ende des Notebooks eine Zelle hinzu, um die Daten als Box-Diagramm darzustellen:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Ein Streudiagramm, das die Beziehung zwischen Preis und Monat zeigt](../../../../2-Regression/2-Data/images/scatterplot.png)

    Ist dies ein nützliches Diagramm? Überrascht Sie etwas daran?

    Es ist nicht besonders nützlich, da es nur die Verteilung Ihrer Daten in einem bestimmten Monat anzeigt.

### Machen Sie es nützlich

Um Diagramme nützlich zu machen, müssen Sie die Daten in der Regel irgendwie gruppieren. Versuchen wir, ein Diagramm zu erstellen, bei dem die y-Achse die Monate zeigt und die Daten die Verteilung darstellen.

1. Fügen Sie eine Zelle hinzu, um ein gruppiertes Balkendiagramm zu erstellen:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Ein Balkendiagramm, das die Beziehung zwischen Preis und Monat zeigt](../../../../2-Regression/2-Data/images/barchart.png)

    Dies ist eine nützlichere Datenvisualisierung! Es scheint darauf hinzudeuten, dass die höchsten Preise für Kürbisse im September und Oktober auftreten. Entspricht das Ihrer Erwartung? Warum oder warum nicht?

---

## 🚀 Herausforderung

Erforschen Sie die verschiedenen Arten von Visualisierungen, die Matplotlib bietet. Welche Arten sind am besten für Regressionsprobleme geeignet?

## [Quiz nach der Lektion](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Schauen Sie sich die vielen Möglichkeiten zur Datenvisualisierung an. Erstellen Sie eine Liste der verschiedenen verfügbaren Bibliotheken und notieren Sie, welche für bestimmte Aufgaben am besten geeignet sind, z. B. 2D-Visualisierungen vs. 3D-Visualisierungen. Was entdecken Sie?

## Aufgabe

[Visualisierung erkunden](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.