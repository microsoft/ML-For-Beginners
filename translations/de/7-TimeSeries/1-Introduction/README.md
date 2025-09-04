<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3150d40f36a77857316ecaed5f31e856",
  "translation_date": "2025-09-03T21:43:40+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "de"
}
-->
# Einführung in die Zeitreihenprognose

![Zusammenfassung von Zeitreihen in einer Sketchnote](../../../../translated_images/ml-timeseries.fb98d25f1013fc0c59090030080b5d1911ff336427bec31dbaf1ad08193812e9.de.png)

> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

In dieser und der nächsten Lektion wirst du etwas über Zeitreihenprognosen lernen, ein interessantes und wertvolles Werkzeug im Repertoire eines ML-Wissenschaftlers, das weniger bekannt ist als andere Themen. Zeitreihenprognosen sind eine Art „Kristallkugel“: Basierend auf der bisherigen Entwicklung einer Variablen wie dem Preis kannst du ihren zukünftigen potenziellen Wert vorhersagen.

[![Einführung in die Zeitreihenprognose](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Einführung in die Zeitreihenprognose")

> 🎥 Klicke auf das Bild oben, um ein Video über Zeitreihenprognosen anzusehen

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41/)

Es ist ein nützliches und interessantes Feld mit echtem Mehrwert für Unternehmen, da es direkt auf Probleme wie Preisgestaltung, Lagerbestände und Lieferkettenfragen angewendet werden kann. Während Techniken des Deep Learning zunehmend eingesetzt werden, um bessere Einblicke zu gewinnen und zukünftige Entwicklungen besser vorherzusagen, bleibt die Zeitreihenprognose ein Bereich, der stark von klassischen ML-Techniken geprägt ist.

> Das hilfreiche Zeitreihen-Curriculum der Penn State findest du [hier](https://online.stat.psu.edu/stat510/lesson/1)

## Einführung

Angenommen, du verwaltest eine Reihe von intelligenten Parkuhren, die Daten darüber liefern, wie oft und wie lange sie im Laufe der Zeit genutzt werden.

> Was wäre, wenn du basierend auf der bisherigen Leistung der Parkuhr ihren zukünftigen Wert gemäß den Gesetzen von Angebot und Nachfrage vorhersagen könntest?

Die genaue Vorhersage, wann gehandelt werden sollte, um dein Ziel zu erreichen, ist eine Herausforderung, die durch Zeitreihenprognosen angegangen werden könnte. Es würde die Leute zwar nicht glücklich machen, in Stoßzeiten mehr zahlen zu müssen, wenn sie einen Parkplatz suchen, aber es wäre eine sichere Möglichkeit, Einnahmen zu generieren, um die Straßen zu reinigen!

Lass uns einige Arten von Zeitreihenalgorithmen erkunden und ein Notebook starten, um einige Daten zu bereinigen und vorzubereiten. Die Daten, die du analysieren wirst, stammen aus dem GEFCom2014-Vorhersagewettbewerb. Sie bestehen aus 3 Jahren stündlicher Stromlast- und Temperaturwerte zwischen 2012 und 2014. Basierend auf den historischen Mustern von Stromlast und Temperatur kannst du zukünftige Werte der Stromlast vorhersagen.

In diesem Beispiel wirst du lernen, wie man einen Zeitschritt voraus prognostiziert, indem nur historische Lastdaten verwendet werden. Bevor du jedoch beginnst, ist es hilfreich zu verstehen, was hinter den Kulissen passiert.

## Einige Definitionen

Wenn du auf den Begriff „Zeitreihe“ stößt, musst du seine Verwendung in verschiedenen Kontexten verstehen.

🎓 **Zeitreihe**

In der Mathematik ist „eine Zeitreihe eine Reihe von Datenpunkten, die in zeitlicher Reihenfolge indiziert (oder aufgelistet oder grafisch dargestellt) sind. Am häufigsten ist eine Zeitreihe eine Sequenz, die zu aufeinanderfolgenden, gleichmäßig verteilten Zeitpunkten aufgenommen wird.“ Ein Beispiel für eine Zeitreihe ist der tägliche Schlusswert des [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Die Verwendung von Zeitreihendiagrammen und statistischen Modellen wird häufig in der Signalverarbeitung, Wettervorhersage, Erdbebenprognose und anderen Bereichen angewendet, in denen Ereignisse auftreten und Datenpunkte über die Zeit hinweg dargestellt werden können.

🎓 **Zeitreihenanalyse**

Die Zeitreihenanalyse ist die Analyse der oben genannten Zeitreihendaten. Zeitreihendaten können verschiedene Formen annehmen, einschließlich „unterbrochener Zeitreihen“, die Muster in der Entwicklung einer Zeitreihe vor und nach einem unterbrechenden Ereignis erkennen. Die Art der Analyse, die für die Zeitreihe erforderlich ist, hängt von der Natur der Daten ab. Zeitreihendaten selbst können die Form von Zahlen- oder Zeichenfolgen annehmen.

Die durchzuführende Analyse verwendet eine Vielzahl von Methoden, einschließlich Frequenzbereich und Zeitbereich, linear und nichtlinear und mehr. [Erfahre mehr](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) über die vielen Möglichkeiten, diese Art von Daten zu analysieren.

🎓 **Zeitreihenprognose**

Die Zeitreihenprognose ist die Verwendung eines Modells, um zukünftige Werte basierend auf Mustern vorherzusagen, die durch zuvor gesammelte Daten angezeigt werden, wie sie in der Vergangenheit aufgetreten sind. Während es möglich ist, Regressionsmodelle zu verwenden, um Zeitreihendaten zu untersuchen, mit Zeitindizes als x-Variablen in einem Diagramm, werden solche Daten am besten mit speziellen Arten von Modellen analysiert.

Zeitreihendaten sind eine Liste geordneter Beobachtungen, im Gegensatz zu Daten, die durch lineare Regression analysiert werden können. Das am häufigsten verwendete Modell ist ARIMA, ein Akronym für „Autoregressive Integrated Moving Average“.

[ARIMA-Modelle](https://online.stat.psu.edu/stat510/lesson/1/1.1) „beziehen den aktuellen Wert einer Serie auf vergangene Werte und vergangene Vorhersagefehler.“ Sie eignen sich am besten zur Analyse von Zeitbereichsdaten, bei denen Daten über die Zeit hinweg geordnet sind.

> Es gibt verschiedene Arten von ARIMA-Modellen, über die du [hier](https://people.duke.edu/~rnau/411arim.htm) mehr erfahren kannst und die du in der nächsten Lektion behandeln wirst.

In der nächsten Lektion wirst du ein ARIMA-Modell mit [Univariaten Zeitreihen](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) erstellen, das sich auf eine Variable konzentriert, die ihren Wert im Laufe der Zeit ändert. Ein Beispiel für diese Art von Daten ist [dieser Datensatz](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), der die monatliche CO2-Konzentration am Mauna Loa Observatory aufzeichnet:

|   CO2   | YearMonth | Year  | Month |
| :-----: | :-------: | :---: | :---: |
| 330.62  |  1975.04  | 1975  |   1   |
| 331.40  |  1975.13  | 1975  |   2   |
| 331.87  |  1975.21  | 1975  |   3   |
| 333.18  |  1975.29  | 1975  |   4   |
| 333.92  |  1975.38  | 1975  |   5   |
| 333.43  |  1975.46  | 1975  |   6   |
| 331.85  |  1975.54  | 1975  |   7   |
| 330.01  |  1975.63  | 1975  |   8   |
| 328.51  |  1975.71  | 1975  |   9   |
| 328.41  |  1975.79  | 1975  |  10   |
| 329.25  |  1975.88  | 1975  |  11   |
| 330.97  |  1975.96  | 1975  |  12   |

✅ Identifiziere die Variable, die sich in diesem Datensatz im Laufe der Zeit ändert.

## Zu berücksichtigende Eigenschaften von Zeitreihendaten

Wenn du Zeitreihendaten betrachtest, wirst du möglicherweise feststellen, dass sie [bestimmte Eigenschaften](https://online.stat.psu.edu/stat510/lesson/1/1.1) aufweisen, die du berücksichtigen und abschwächen musst, um ihre Muster besser zu verstehen. Wenn du Zeitreihendaten als potenzielles „Signal“ betrachtest, das du analysieren möchtest, können diese Eigenschaften als „Rauschen“ angesehen werden. Oft musst du dieses „Rauschen“ reduzieren, indem du einige dieser Eigenschaften mit statistischen Techniken ausgleichst.

Hier sind einige Konzepte, die du kennen solltest, um mit Zeitreihen arbeiten zu können:

🎓 **Trends**

Trends sind messbare Zunahmen und Abnahmen im Laufe der Zeit. [Lies mehr](https://machinelearningmastery.com/time-series-trends-in-python). Im Kontext von Zeitreihen geht es darum, wie man Trends nutzt und, falls erforderlich, aus Zeitreihen entfernt.

🎓 **[Saisonalität](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Saisonalität ist definiert als periodische Schwankungen, wie beispielsweise der Ansturm während der Feiertage, der sich auf den Umsatz auswirken könnte. [Schau dir an](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), wie verschiedene Arten von Diagrammen Saisonalität in Daten darstellen.

🎓 **Ausreißer**

Ausreißer liegen weit außerhalb der normalen Datenvarianz.

🎓 **Langfristige Zyklen**

Unabhängig von der Saisonalität können Daten langfristige Zyklen aufweisen, wie beispielsweise einen wirtschaftlichen Abschwung, der länger als ein Jahr anhält.

🎓 **Konstante Varianz**

Im Laufe der Zeit zeigen einige Daten konstante Schwankungen, wie beispielsweise der tägliche Energieverbrauch zwischen Tag und Nacht.

🎓 **Abrupte Veränderungen**

Die Daten könnten eine abrupte Veränderung zeigen, die einer weiteren Analyse bedarf. Die plötzliche Schließung von Unternehmen aufgrund von COVID hat beispielsweise Veränderungen in den Daten verursacht.

✅ Hier ist ein [Beispiel für ein Zeitreihendiagramm](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), das die täglichen Ausgaben für In-Game-Währung über einige Jahre zeigt. Kannst du eine der oben genannten Eigenschaften in diesen Daten erkennen?

![In-Game-Währungsausgaben](../../../../translated_images/currency.e7429812bfc8c6087b2d4c410faaa4aaa11b2fcaabf6f09549b8249c9fbdb641.de.png)

## Übung – Einstieg in Stromverbrauchsdaten

Lass uns beginnen, ein Zeitreihenmodell zu erstellen, um den zukünftigen Stromverbrauch basierend auf vergangenen Verbrauchsdaten vorherzusagen.

> Die Daten in diesem Beispiel stammen aus dem GEFCom2014-Vorhersagewettbewerb. Sie bestehen aus 3 Jahren stündlicher Stromlast- und Temperaturwerte zwischen 2012 und 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli und Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. Öffne im `working`-Ordner dieser Lektion die Datei _notebook.ipynb_. Beginne damit, Bibliotheken hinzuzufügen, die dir beim Laden und Visualisieren von Daten helfen:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Beachte, dass du die Dateien aus dem enthaltenen `common`-Ordner verwendest, die deine Umgebung einrichten und das Herunterladen der Daten übernehmen.

2. Untersuche als Nächstes die Daten als DataFrame, indem du `load_data()` und `head()` aufrufst:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Du kannst sehen, dass es zwei Spalten gibt, die Datum und Last darstellen:

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

    ![Energie-Diagramm](../../../../translated_images/energy-plot.5fdac3f397a910bc6070602e9e45bea8860d4c239354813fa8fc3c9d556f5bad.de.png)

4. Zeichne nun die erste Woche im Juli 2014, indem du sie als Eingabe für `energy` im Muster `[von Datum]:[bis Datum]` angibst:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Juli](../../../../translated_images/july-2014.9e1f7c318ec6d5b30b0d7e1e20be3643501f64a53f3d426d7c7d7b62addb335e.de.png)

    Ein wunderschönes Diagramm! Schau dir diese Diagramme an und sieh, ob du eine der oben genannten Eigenschaften erkennen kannst. Was können wir durch die Visualisierung der Daten ableiten?

In der nächsten Lektion wirst du ein ARIMA-Modell erstellen, um einige Prognosen zu erstellen.

---

## 🚀 Herausforderung

Erstelle eine Liste aller Branchen und Forschungsbereiche, die deiner Meinung nach von Zeitreihenprognosen profitieren könnten. Kannst du eine Anwendung dieser Techniken in den Künsten, der Ökonometrie, der Ökologie, dem Einzelhandel, der Industrie oder der Finanzwelt finden? Wo sonst?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/)

## Rückblick & Selbststudium

Obwohl wir sie hier nicht behandeln, werden neuronale Netzwerke manchmal verwendet, um klassische Methoden der Zeitreihenprognose zu verbessern. Lies mehr darüber [in diesem Artikel](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412).

## Aufgabe

[Visualisiere weitere Zeitreihen](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.