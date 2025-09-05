<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-04T21:54:27+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "de"
}
-->
# Einführung in die Zeitreihenprognose

![Zusammenfassung von Zeitreihen in einer Sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

In dieser und der nächsten Lektion wirst du etwas über Zeitreihenprognosen lernen, ein interessantes und wertvolles Werkzeug im Repertoire eines ML-Wissenschaftlers, das weniger bekannt ist als andere Themen. Zeitreihenprognosen sind eine Art „Kristallkugel“: Basierend auf der bisherigen Entwicklung einer Variablen, wie z. B. einem Preis, kannst du ihren potenziellen zukünftigen Wert vorhersagen.

[![Einführung in die Zeitreihenprognose](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Einführung in die Zeitreihenprognose")

> 🎥 Klicke auf das Bild oben, um ein Video über Zeitreihenprognosen anzusehen

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

Es ist ein nützliches und interessantes Feld mit echtem Mehrwert für Unternehmen, da es direkt auf Probleme wie Preisgestaltung, Bestandsmanagement und Lieferkettenfragen angewendet werden kann. Während Deep-Learning-Techniken zunehmend eingesetzt werden, um bessere Einblicke zu gewinnen und zukünftige Entwicklungen besser vorherzusagen, bleibt die Zeitreihenprognose ein Bereich, der stark von klassischen ML-Techniken geprägt ist.

> Das nützliche Zeitreihen-Curriculum der Penn State University findest du [hier](https://online.stat.psu.edu/stat510/lesson/1)

## Einführung

Angenommen, du verwaltest eine Reihe von intelligenten Parkuhren, die Daten darüber liefern, wie oft und wie lange sie im Laufe der Zeit genutzt werden.

> Was wäre, wenn du basierend auf der bisherigen Nutzung der Parkuhr ihren zukünftigen Wert gemäß den Gesetzen von Angebot und Nachfrage vorhersagen könntest?

Die genaue Vorhersage, wann gehandelt werden muss, um ein Ziel zu erreichen, ist eine Herausforderung, die mit Zeitreihenprognosen angegangen werden kann. Es würde die Leute zwar nicht glücklich machen, in Stoßzeiten mehr zahlen zu müssen, wenn sie einen Parkplatz suchen, aber es wäre eine sichere Möglichkeit, Einnahmen zu generieren, um die Straßen zu reinigen!

Lass uns einige Arten von Zeitreihenalgorithmen erkunden und ein Notebook starten, um einige Daten zu bereinigen und vorzubereiten. Die Daten, die du analysieren wirst, stammen aus dem GEFCom2014-Vorhersagewettbewerb. Sie umfassen 3 Jahre stündliche Daten zu Stromverbrauch und Temperatur zwischen 2012 und 2014. Anhand der historischen Muster des Stromverbrauchs und der Temperatur kannst du zukünftige Werte des Stromverbrauchs vorhersagen.

In diesem Beispiel lernst du, wie man einen Zeitschritt in die Zukunft vorhersagt, indem nur historische Verbrauchsdaten verwendet werden. Bevor du jedoch beginnst, ist es hilfreich, zu verstehen, was im Hintergrund passiert.

## Einige Definitionen

Wenn du auf den Begriff „Zeitreihe“ stößt, musst du verstehen, wie er in verschiedenen Kontexten verwendet wird.

🎓 **Zeitreihe**

In der Mathematik ist eine „Zeitreihe eine Reihe von Datenpunkten, die in zeitlicher Reihenfolge indiziert (oder aufgelistet oder grafisch dargestellt) sind. Am häufigsten ist eine Zeitreihe eine Sequenz, die in gleichmäßigen zeitlichen Abständen aufgenommen wurde.“ Ein Beispiel für eine Zeitreihe ist der tägliche Schlusswert des [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Zeitreihendiagramme und statistische Modellierung werden häufig in der Signalverarbeitung, Wettervorhersage, Erdbebenvorhersage und anderen Bereichen verwendet, in denen Ereignisse auftreten und Datenpunkte über die Zeit hinweg dargestellt werden können.

🎓 **Zeitreihenanalyse**

Die Zeitreihenanalyse ist die Analyse der oben genannten Zeitreihendaten. Zeitreihendaten können unterschiedliche Formen annehmen, einschließlich „unterbrochener Zeitreihen“, die Muster in der Entwicklung einer Zeitreihe vor und nach einem unterbrechenden Ereignis erkennen. Die Art der Analyse, die für die Zeitreihe erforderlich ist, hängt von der Natur der Daten ab. Zeitreihendaten selbst können in Form von Zahlen- oder Zeichenfolgen vorliegen.

Die durchzuführende Analyse verwendet eine Vielzahl von Methoden, einschließlich Frequenz- und Zeitbereich, linear und nichtlinear und mehr. [Erfahre mehr](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) über die vielen Möglichkeiten, diese Art von Daten zu analysieren.

🎓 **Zeitreihenprognose**

Die Zeitreihenprognose ist die Verwendung eines Modells, um zukünftige Werte basierend auf Mustern vorherzusagen, die aus zuvor gesammelten Daten abgeleitet wurden. Während es möglich ist, Regressionsmodelle zu verwenden, um Zeitreihendaten zu untersuchen, bei denen Zeitindizes als x-Variablen in einem Diagramm dargestellt werden, werden solche Daten am besten mit speziellen Modelltypen analysiert.

Zeitreihendaten sind eine Liste geordneter Beobachtungen, im Gegensatz zu Daten, die durch lineare Regression analysiert werden können. Das am häufigsten verwendete Modell ist ARIMA, ein Akronym für „Autoregressive Integrated Moving Average“.

[ARIMA-Modelle](https://online.stat.psu.edu/stat510/lesson/1/1.1) „beziehen den aktuellen Wert einer Serie auf vergangene Werte und frühere Vorhersagefehler.“ Sie eignen sich am besten zur Analyse von Zeitbereichsdaten, bei denen Daten in zeitlicher Reihenfolge angeordnet sind.

> Es gibt verschiedene Arten von ARIMA-Modellen, über die du [hier](https://people.duke.edu/~rnau/411arim.htm) mehr erfahren kannst und die in der nächsten Lektion behandelt werden.

In der nächsten Lektion wirst du ein ARIMA-Modell mit [Univariaten Zeitreihen](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) erstellen, das sich auf eine Variable konzentriert, deren Wert sich im Laufe der Zeit ändert. Ein Beispiel für diese Art von Daten ist [dieser Datensatz](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), der die monatliche CO2-Konzentration am Mauna Loa Observatory aufzeichnet:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifiziere die Variable, die sich in diesem Datensatz im Laufe der Zeit ändert.

## Zu berücksichtigende Eigenschaften von Zeitreihendaten

Wenn du Zeitreihendaten betrachtest, wirst du möglicherweise feststellen, dass sie [bestimmte Eigenschaften](https://online.stat.psu.edu/stat510/lesson/1/1.1) aufweisen, die du berücksichtigen und mildern musst, um ihre Muster besser zu verstehen. Wenn du Zeitreihendaten als potenzielles „Signal“ betrachtest, das du analysieren möchtest, können diese Eigenschaften als „Rauschen“ angesehen werden. Oft musst du dieses „Rauschen“ reduzieren, indem du einige statistische Techniken anwendest.

Hier sind einige Konzepte, die du kennen solltest, um mit Zeitreihen zu arbeiten:

🎓 **Trends**

Trends sind messbare Zunahmen und Abnahmen im Laufe der Zeit. [Lies mehr](https://machinelearningmastery.com/time-series-trends-in-python). Im Kontext von Zeitreihen geht es darum, wie man Trends nutzt und, falls erforderlich, aus den Zeitreihen entfernt.

🎓 **[Saisonalität](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Saisonalität ist definiert als periodische Schwankungen, wie z. B. Feiertagsanstürme, die sich auf den Umsatz auswirken könnten. [Schau dir an](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), wie verschiedene Diagrammtypen Saisonalität in Daten darstellen.

🎓 **Ausreißer**

Ausreißer sind Datenpunkte, die weit außerhalb der Standardvarianz liegen.

🎓 **Langfristige Zyklen**

Unabhängig von der Saisonalität können Daten langfristige Zyklen aufweisen, wie z. B. eine wirtschaftliche Rezession, die länger als ein Jahr dauert.

🎓 **Konstante Varianz**

Im Laufe der Zeit zeigen einige Daten konstante Schwankungen, wie z. B. der tägliche und nächtliche Energieverbrauch.

🎓 **Abrupte Änderungen**

Die Daten können abrupte Änderungen aufweisen, die einer weiteren Analyse bedürfen. Die plötzliche Schließung von Unternehmen aufgrund von COVID führte beispielsweise zu Veränderungen in den Daten.

✅ Hier ist ein [Beispiel für ein Zeitreihendiagramm](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), das die täglichen Ausgaben für In-Game-Währung über einige Jahre zeigt. Kannst du eine der oben genannten Eigenschaften in diesen Daten erkennen?

![In-Game-Währungsausgaben](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Übung – Einstieg in Stromverbrauchsdaten

Lass uns beginnen, ein Zeitreihenmodell zu erstellen, um den zukünftigen Stromverbrauch basierend auf vergangenen Verbrauchsdaten vorherzusagen.

> Die Daten in diesem Beispiel stammen aus dem GEFCom2014-Vorhersagewettbewerb. Sie umfassen 3 Jahre stündliche Daten zu Stromverbrauch und Temperaturwerten zwischen 2012 und 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli und Rob J. Hyndman, „Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond“, International Journal of Forecasting, vol.32, no.3, pp 896-913, Juli-September, 2016.

1. Öffne im Ordner `working` dieser Lektion die Datei _notebook.ipynb_. Beginne damit, Bibliotheken hinzuzufügen, die dir beim Laden und Visualisieren von Daten helfen:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Beachte, dass du die Dateien aus dem enthaltenen Ordner `common` verwendest, die deine Umgebung einrichten und das Herunterladen der Daten übernehmen.

2. Untersuche als Nächstes die Daten als DataFrame, indem du `load_data()` und `head()` aufrufst:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Du kannst sehen, dass es zwei Spalten gibt, die Datum und Verbrauch darstellen:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Zeichne nun die Daten, indem du `plot()` aufrufst:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Energie-Diagramm](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Zeichne nun die erste Juliwoche 2014, indem du sie als Eingabe im Muster `[von Datum]:[bis Datum]` an `energy` übergibst:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Juli](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Ein wunderschönes Diagramm! Sieh dir diese Diagramme an und überlege, ob du eine der oben genannten Eigenschaften erkennen kannst. Was können wir durch die Visualisierung der Daten ableiten?

In der nächsten Lektion wirst du ein ARIMA-Modell erstellen, um einige Vorhersagen zu treffen.

---

## 🚀 Herausforderung

Erstelle eine Liste aller Branchen und Forschungsbereiche, die deiner Meinung nach von Zeitreihenprognosen profitieren könnten. Kannst du dir eine Anwendung dieser Techniken in den Künsten vorstellen? In der Ökonometrie? In der Ökologie? Im Einzelhandel? In der Industrie? In der Finanzwelt? Wo noch?

## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Rückblick & Selbststudium

Obwohl wir sie hier nicht behandeln, werden neuronale Netze manchmal verwendet, um klassische Methoden der Zeitreihenprognose zu verbessern. Lies mehr darüber [in diesem Artikel](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412).

## Aufgabe

[Visualisiere weitere Zeitreihen](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mithilfe des KI-Übersetzungsdienstes [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen.