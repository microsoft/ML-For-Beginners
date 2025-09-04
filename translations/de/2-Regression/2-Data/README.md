<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a683e1fe430bb0d4a10b68f6ca15e0a6",
  "translation_date": "2025-09-03T21:42:05+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "de"
}
-->
# Erstellen eines Regressionsmodells mit Scikit-learn: Daten vorbereiten und visualisieren

![Infografik zur Datenvisualisierung](../../../../translated_images/data-visualization.54e56dded7c1a804d00d027543f2881cb32da73aeadda2d4a4f10f3497526114.de.png)

Infografik von [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [Diese Lektion ist auch in R verfügbar!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Einführung

Jetzt, da Sie mit den notwendigen Tools ausgestattet sind, um mit dem Aufbau von Machine-Learning-Modellen mit Scikit-learn zu beginnen, können Sie anfangen, Fragen zu Ihren Daten zu stellen. Wenn Sie mit Daten arbeiten und ML-Lösungen anwenden, ist es äußerst wichtig, die richtigen Fragen zu stellen, um das Potenzial Ihres Datensatzes voll auszuschöpfen.

In dieser Lektion lernen Sie:

- Wie Sie Ihre Daten für den Modellaufbau vorbereiten.
- Wie Sie Matplotlib für die Datenvisualisierung nutzen.

## Die richtigen Fragen an Ihre Daten stellen

Die Frage, die Sie beantwortet haben möchten, bestimmt, welche Art von ML-Algorithmen Sie verwenden werden. Und die Qualität der Antwort hängt stark von der Beschaffenheit Ihrer Daten ab.

Werfen Sie einen Blick auf die [Daten](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), die für diese Lektion bereitgestellt wurden. Sie können diese .csv-Datei in VS Code öffnen. Ein kurzer Blick zeigt sofort, dass es Leerstellen und eine Mischung aus Zeichenketten und numerischen Daten gibt. Es gibt auch eine seltsame Spalte namens 'Package', in der die Daten eine Mischung aus 'sacks', 'bins' und anderen Werten sind. Die Daten sind tatsächlich ein wenig chaotisch.

[![ML für Anfänger - Wie man einen Datensatz analysiert und bereinigt](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML für Anfänger - Wie man einen Datensatz analysiert und bereinigt")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zur Vorbereitung der Daten für diese Lektion anzusehen.

Es ist tatsächlich nicht sehr üblich, einen Datensatz zu erhalten, der vollständig bereit ist, um direkt ein ML-Modell zu erstellen. In dieser Lektion lernen Sie, wie Sie einen Rohdatensatz mit Standard-Python-Bibliotheken vorbereiten. Sie lernen auch verschiedene Techniken zur Visualisierung der Daten.

## Fallstudie: 'Der Kürbismarkt'

In diesem Ordner finden Sie eine .csv-Datei im Stammordner `data` namens [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), die 1757 Zeilen Daten über den Markt für Kürbisse enthält, sortiert nach Städten. Dies sind Rohdaten, die aus den [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) des US-Landwirtschaftsministeriums extrahiert wurden.

### Daten vorbereiten

Diese Daten sind gemeinfrei. Sie können auf der USDA-Website in vielen separaten Dateien, nach Städten sortiert, heruntergeladen werden. Um zu vermeiden, dass zu viele separate Dateien entstehen, haben wir alle Städtedaten in eine Tabelle zusammengeführt, sodass die Daten bereits _etwas_ vorbereitet sind. Schauen wir uns die Daten nun genauer an.

### Die Kürbisdaten - erste Schlussfolgerungen

Was fällt Ihnen an diesen Daten auf? Sie haben bereits gesehen, dass es eine Mischung aus Zeichenketten, Zahlen, Leerstellen und seltsamen Werten gibt, die Sie verstehen müssen.

Welche Frage können Sie mit diesen Daten unter Verwendung einer Regressionstechnik stellen? Wie wäre es mit "Den Preis eines Kürbisses für den Verkauf in einem bestimmten Monat vorhersagen"? Wenn Sie die Daten erneut betrachten, gibt es einige Änderungen, die Sie vornehmen müssen, um die für die Aufgabe erforderliche Datenstruktur zu erstellen.

## Übung - Die Kürbisdaten analysieren

Verwenden wir [Pandas](https://pandas.pydata.org/) (der Name steht für `Python Data Analysis`), ein sehr nützliches Tool zur Datenaufbereitung, um diese Kürbisdaten zu analysieren und vorzubereiten.

### Zuerst fehlende Daten überprüfen

Zunächst müssen Sie Schritte unternehmen, um fehlende Daten zu überprüfen:

1. Konvertieren Sie die Daten in ein Monatsformat (dies sind US-Daten, das Format ist `MM/DD/YYYY`).
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

    Es gibt fehlende Daten, aber vielleicht spielt das für die Aufgabe keine Rolle.

1. Um Ihr Dataframe einfacher zu bearbeiten, wählen Sie nur die benötigten Spalten aus, indem Sie die Funktion `loc` verwenden, die aus dem ursprünglichen Dataframe eine Gruppe von Zeilen (als erster Parameter übergeben) und Spalten (als zweiter Parameter übergeben) extrahiert. Der Ausdruck `:` bedeutet in diesem Fall "alle Zeilen".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Zweitens den Durchschnittspreis eines Kürbisses bestimmen

Überlegen Sie, wie Sie den Durchschnittspreis eines Kürbisses in einem bestimmten Monat bestimmen können. Welche Spalten würden Sie für diese Aufgabe auswählen? Hinweis: Sie benötigen 3 Spalten.

Lösung: Nehmen Sie den Durchschnitt der Spalten `Low Price` und `High Price`, um die neue Spalte Price zu füllen, und konvertieren Sie die Spalte Date so, dass nur der Monat angezeigt wird. Glücklicherweise gibt es laut der obigen Überprüfung keine fehlenden Daten für Daten oder Preise.

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

    Wenn Sie Ihr Dataframe ausdrucken, sehen Sie einen sauberen, aufgeräumten Datensatz, auf dem Sie Ihr neues Regressionsmodell aufbauen können.

### Aber Moment! Hier ist etwas Seltsames

Wenn Sie sich die Spalte `Package` ansehen, werden Kürbisse in vielen verschiedenen Konfigurationen verkauft. Einige werden in '1 1/9 bushel'-Maßen verkauft, andere in '1/2 bushel'-Maßen, einige pro Kürbis, einige pro Pfund und einige in großen Kisten mit unterschiedlichen Breiten.

> Kürbisse scheinen sehr schwer konsistent zu wiegen

Wenn man sich die Originaldaten ansieht, ist es interessant, dass alles mit `Unit of Sale` gleich 'EACH' oder 'PER BIN' auch den `Package`-Typ pro Zoll, pro Bin oder 'each' hat. Kürbisse scheinen sehr schwer konsistent zu wiegen, daher filtern wir sie, indem wir nur Kürbisse mit dem String 'bushel' in ihrer `Package`-Spalte auswählen.

1. Fügen Sie einen Filter oben in der Datei unter dem ursprünglichen .csv-Import hinzu:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Wenn Sie die Daten jetzt ausdrucken, sehen Sie, dass Sie nur die etwa 415 Zeilen mit Daten erhalten, die Kürbisse nach dem Bushel enthalten.

### Aber Moment! Es gibt noch etwas zu tun

Haben Sie bemerkt, dass die Bushel-Menge pro Zeile variiert? Sie müssen die Preise normalisieren, sodass Sie die Preise pro Bushel anzeigen. Machen Sie also einige Berechnungen, um dies zu standardisieren.

1. Fügen Sie diese Zeilen nach dem Block hinzu, der das neue_pumpkins-Dataframe erstellt:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Laut [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) hängt das Gewicht eines Bushels von der Art des Produkts ab, da es sich um eine Volumenmessung handelt. "Ein Bushel Tomaten soll beispielsweise 56 Pfund wiegen... Blätter und Grünzeug nehmen mehr Platz mit weniger Gewicht ein, sodass ein Bushel Spinat nur 20 Pfund wiegt." Das ist alles ziemlich kompliziert! Lassen Sie uns die Umrechnung von Bushel zu Pfund ignorieren und stattdessen nach Bushel berechnen. All diese Studien zu Bushels von Kürbissen zeigen jedoch, wie wichtig es ist, die Natur Ihrer Daten zu verstehen!

Jetzt können Sie die Preise pro Einheit basierend auf ihrer Bushel-Messung analysieren. Wenn Sie die Daten noch einmal ausdrucken, können Sie sehen, wie sie standardisiert sind.

✅ Haben Sie bemerkt, dass Kürbisse, die nach dem halben Bushel verkauft werden, sehr teuer sind? Können Sie herausfinden, warum? Hinweis: Kleine Kürbisse sind viel teurer als große, wahrscheinlich weil es so viel mehr von ihnen pro Bushel gibt, angesichts des ungenutzten Raums, den ein großer hohler Kürbis für Kuchen einnimmt.

## Visualisierungsstrategien

Ein Teil der Rolle eines Data Scientists besteht darin, die Qualität und Natur der Daten, mit denen er arbeitet, zu demonstrieren. Dazu erstellen sie oft interessante Visualisierungen, wie Diagramme, Grafiken und Charts, die verschiedene Aspekte der Daten zeigen. Auf diese Weise können sie visuell Beziehungen und Lücken aufzeigen, die sonst schwer zu erkennen wären.

[![ML für Anfänger - Wie man Daten mit Matplotlib visualisiert](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML für Anfänger - Wie man Daten mit Matplotlib visualisiert")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zur Visualisierung der Daten für diese Lektion anzusehen.

Visualisierungen können auch helfen, die am besten geeignete Machine-Learning-Technik für die Daten zu bestimmen. Ein Streudiagramm, das einer Linie zu folgen scheint, zeigt beispielsweise, dass die Daten ein guter Kandidat für eine lineare Regression sind.

Eine Datenvisualisierungsbibliothek, die gut in Jupyter-Notebooks funktioniert, ist [Matplotlib](https://matplotlib.org/) (die Sie auch in der vorherigen Lektion gesehen haben).

> Sammeln Sie mehr Erfahrung mit Datenvisualisierung in [diesen Tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Übung - Mit Matplotlib experimentieren

Versuchen Sie, einige grundlegende Diagramme zu erstellen, um das neue Dataframe anzuzeigen, das Sie gerade erstellt haben. Was würde ein einfaches Liniendiagramm zeigen?

1. Importieren Sie Matplotlib oben in der Datei, unter dem Pandas-Import:

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

    ![Ein Streudiagramm, das die Beziehung zwischen Preis und Monat zeigt](../../../../translated_images/scatterplot.b6868f44cbd2051c6680ccdbb1510697d06a3ff6cd4abda656f5009c0ed4e3fc.de.png)

    Ist dies ein nützliches Diagramm? Überrascht Sie etwas daran?

    Es ist nicht besonders nützlich, da es Ihre Daten nur als Punktverteilung in einem bestimmten Monat anzeigt.

### Machen Sie es nützlich

Um Diagramme nützliche Daten anzeigen zu lassen, müssen Sie die Daten normalerweise irgendwie gruppieren. Versuchen wir, ein Diagramm zu erstellen, bei dem die y-Achse die Monate zeigt und die Daten die Verteilung der Daten darstellen.

1. Fügen Sie eine Zelle hinzu, um ein gruppiertes Balkendiagramm zu erstellen:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Ein Balkendiagramm, das die Beziehung zwischen Preis und Monat zeigt](../../../../translated_images/barchart.a833ea9194346d769c77a3a870f7d8aee51574cd1138ca902e5500830a41cbce.de.png)

    Dies ist eine nützlichere Datenvisualisierung! Es scheint darauf hinzudeuten, dass die höchsten Preise für Kürbisse im September und Oktober auftreten. Entspricht das Ihrer Erwartung? Warum oder warum nicht?

---

## 🚀 Herausforderung

Erforschen Sie die verschiedenen Arten von Visualisierungen, die Matplotlib bietet. Welche Typen sind am besten für Regressionsprobleme geeignet?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## Rückblick & Selbststudium

Schauen Sie sich die vielen Möglichkeiten zur Visualisierung von Daten an. Erstellen Sie eine Liste der verschiedenen verfügbaren Bibliotheken und notieren Sie, welche für bestimmte Arten von Aufgaben am besten geeignet sind, z. B. 2D-Visualisierungen vs. 3D-Visualisierungen. Was entdecken Sie?

## Aufgabe

[Visualisierung erkunden](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.