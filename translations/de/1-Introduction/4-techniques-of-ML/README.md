<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "dc4575225da159f2b06706e103ddba2a",
  "translation_date": "2025-09-03T21:50:54+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "de"
}
-->
# Techniken des maschinellen Lernens

Der Prozess des Erstellens, Nutzens und Wartens von Modellen des maschinellen Lernens (ML) und der Daten, die sie verwenden, unterscheidet sich stark von vielen anderen Entwicklungs-Workflows. In dieser Lektion werden wir den Prozess entmystifizieren und die wichtigsten Techniken skizzieren, die Sie kennen müssen. Sie werden:

- Die grundlegenden Prozesse des maschinellen Lernens auf hoher Ebene verstehen.
- Grundlegende Konzepte wie "Modelle", "Vorhersagen" und "Trainingsdaten" erkunden.

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/)

[![ML für Anfänger - Techniken des maschinellen Lernens](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML für Anfänger - Techniken des maschinellen Lernens")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zu dieser Lektion anzusehen.

## Einführung

Auf hoher Ebene besteht die Kunst, Prozesse des maschinellen Lernens zu erstellen, aus mehreren Schritten:

1. **Die Frage festlegen**. Die meisten ML-Prozesse beginnen mit einer Frage, die nicht durch ein einfaches bedingtes Programm oder eine regelbasierte Engine beantwortet werden kann. Diese Fragen drehen sich oft um Vorhersagen, die auf einer Sammlung von Daten basieren.
2. **Daten sammeln und vorbereiten**. Um Ihre Frage beantworten zu können, benötigen Sie Daten. Die Qualität und manchmal auch die Menge Ihrer Daten bestimmen, wie gut Sie Ihre ursprüngliche Frage beantworten können. Die Visualisierung von Daten ist ein wichtiger Aspekt dieser Phase. Diese Phase umfasst auch das Aufteilen der Daten in eine Trainings- und Testgruppe, um ein Modell zu erstellen.
3. **Eine Trainingsmethode wählen**. Abhängig von Ihrer Frage und der Art Ihrer Daten müssen Sie entscheiden, wie Sie ein Modell trainieren möchten, um Ihre Daten bestmöglich zu reflektieren und genaue Vorhersagen zu treffen. Dieser Teil des ML-Prozesses erfordert spezifisches Fachwissen und oft eine beträchtliche Menge an Experimenten.
4. **Das Modell trainieren**. Mithilfe Ihrer Trainingsdaten verwenden Sie verschiedene Algorithmen, um ein Modell zu trainieren, das Muster in den Daten erkennt. Das Modell kann interne Gewichte nutzen, die angepasst werden können, um bestimmte Teile der Daten gegenüber anderen zu bevorzugen, um ein besseres Modell zu erstellen.
5. **Das Modell bewerten**. Sie verwenden bisher unbekannte Daten (Ihre Testdaten) aus Ihrem gesammelten Satz, um zu sehen, wie das Modell abschneidet.
6. **Parameteranpassung**. Basierend auf der Leistung Ihres Modells können Sie den Prozess mit unterschiedlichen Parametern oder Variablen, die das Verhalten der Algorithmen steuern, wiederholen.
7. **Vorhersagen**. Verwenden Sie neue Eingaben, um die Genauigkeit Ihres Modells zu testen.

## Welche Frage soll gestellt werden?

Computer sind besonders gut darin, versteckte Muster in Daten zu entdecken. Diese Fähigkeit ist sehr hilfreich für Forscher, die Fragen zu einem bestimmten Bereich haben, die nicht leicht durch die Erstellung einer regelbasierten Engine beantwortet werden können. Bei einer versicherungsmathematischen Aufgabe könnte ein Datenwissenschaftler beispielsweise handgefertigte Regeln zur Sterblichkeit von Rauchern im Vergleich zu Nichtrauchern erstellen.

Wenn jedoch viele andere Variablen in die Gleichung einbezogen werden, könnte sich ein ML-Modell als effizienter erweisen, um zukünftige Sterblichkeitsraten basierend auf vergangenen Gesundheitsdaten vorherzusagen. Ein fröhlicheres Beispiel könnte die Wettervorhersage für den Monat April an einem bestimmten Ort sein, basierend auf Daten wie Breitengrad, Längengrad, Klimawandel, Nähe zum Ozean, Jetstream-Mustern und mehr.

✅ Diese [Präsentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) zu Wettermodellen bietet eine historische Perspektive zur Nutzung von ML in der Wetteranalyse.

## Aufgaben vor dem Modellaufbau

Bevor Sie mit dem Aufbau Ihres Modells beginnen, gibt es mehrere Aufgaben, die Sie erledigen müssen. Um Ihre Frage zu testen und eine Hypothese basierend auf den Vorhersagen eines Modells zu bilden, müssen Sie mehrere Elemente identifizieren und konfigurieren.

### Daten

Um Ihre Frage mit einer gewissen Sicherheit beantworten zu können, benötigen Sie eine ausreichende Menge an Daten des richtigen Typs. Es gibt zwei Dinge, die Sie an diesem Punkt tun müssen:

- **Daten sammeln**. Denken Sie an die vorherige Lektion zur Fairness in der Datenanalyse und sammeln Sie Ihre Daten sorgfältig. Achten Sie auf die Quellen dieser Daten, mögliche inhärente Verzerrungen und dokumentieren Sie deren Herkunft.
- **Daten vorbereiten**. Es gibt mehrere Schritte im Datenvorbereitungsprozess. Sie müssen möglicherweise Daten zusammenführen und normalisieren, wenn sie aus verschiedenen Quellen stammen. Sie können die Qualität und Quantität der Daten durch verschiedene Methoden verbessern, wie z. B. das Konvertieren von Zeichenfolgen in Zahlen (wie wir es in [Clustering](../../5-Clustering/1-Visualize/README.md) tun). Sie könnten auch neue Daten basierend auf den ursprünglichen Daten generieren (wie wir es in [Klassifikation](../../4-Classification/1-Introduction/README.md) tun). Sie können die Daten bereinigen und bearbeiten (wie wir es vor der [Web-App](../../3-Web-App/README.md)-Lektion tun). Schließlich müssen Sie die Daten möglicherweise auch zufällig anordnen und mischen, je nach Ihren Trainingstechniken.

✅ Nachdem Sie Ihre Daten gesammelt und verarbeitet haben, nehmen Sie sich einen Moment Zeit, um zu prüfen, ob deren Struktur es Ihnen ermöglicht, Ihre beabsichtigte Frage zu beantworten. Es könnte sein, dass die Daten für Ihre Aufgabe nicht gut geeignet sind, wie wir in unseren [Clustering](../../5-Clustering/1-Visualize/README.md)-Lektionen herausfinden!

### Merkmale und Ziel

Ein [Merkmal](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) ist eine messbare Eigenschaft Ihrer Daten. In vielen Datensätzen wird es als Spaltenüberschrift wie 'Datum', 'Größe' oder 'Farbe' ausgedrückt. Ihre Merkmalsvariable, die in der Regel als `X` im Code dargestellt wird, repräsentiert die Eingabevariable, die verwendet wird, um das Modell zu trainieren.

Ein Ziel ist das, was Sie vorhersagen möchten. Das Ziel, in der Regel als `y` im Code dargestellt, repräsentiert die Antwort auf die Frage, die Sie an Ihre Daten stellen möchten: Im Dezember, welche **Farbe** haben die günstigsten Kürbisse? In San Francisco, welche Stadtteile haben die besten Immobilien-**preise**? Manchmal wird das Ziel auch als Label-Attribut bezeichnet.

### Auswahl Ihrer Merkmalsvariablen

🎓 **Merkmalsauswahl und Merkmalsextraktion** Wie wissen Sie, welche Variable Sie beim Aufbau eines Modells auswählen sollen? Sie werden wahrscheinlich einen Prozess der Merkmalsauswahl oder Merkmalsextraktion durchlaufen, um die richtigen Variablen für das leistungsfähigste Modell auszuwählen. Sie sind jedoch nicht dasselbe: "Die Merkmalsextraktion erstellt neue Merkmale aus Funktionen der ursprünglichen Merkmale, während die Merkmalsauswahl eine Teilmenge der Merkmale zurückgibt." ([Quelle](https://wikipedia.org/wiki/Feature_selection))

### Visualisieren Sie Ihre Daten

Ein wichtiger Aspekt im Werkzeugkasten eines Datenwissenschaftlers ist die Fähigkeit, Daten mit mehreren hervorragenden Bibliotheken wie Seaborn oder MatPlotLib zu visualisieren. Die visuelle Darstellung Ihrer Daten könnte es Ihnen ermöglichen, versteckte Korrelationen zu entdecken, die Sie nutzen können. Ihre Visualisierungen könnten Ihnen auch helfen, Verzerrungen oder unausgewogene Daten aufzudecken (wie wir in [Klassifikation](../../4-Classification/2-Classifiers-1/README.md) herausfinden).

### Teilen Sie Ihren Datensatz

Vor dem Training müssen Sie Ihren Datensatz in zwei oder mehr ungleiche Teile aufteilen, die die Daten dennoch gut repräsentieren.

- **Training**. Dieser Teil des Datensatzes wird an Ihr Modell angepasst, um es zu trainieren. Dieser Satz macht den Großteil des ursprünglichen Datensatzes aus.
- **Testen**. Ein Testdatensatz ist eine unabhängige Gruppe von Daten, die oft aus den ursprünglichen Daten entnommen wird und die Sie verwenden, um die Leistung des erstellten Modells zu bestätigen.
- **Validieren**. Ein Validierungssatz ist eine kleinere unabhängige Gruppe von Beispielen, die Sie verwenden, um die Hyperparameter oder die Architektur des Modells zu optimieren, um das Modell zu verbessern. Abhängig von der Größe Ihrer Daten und der Frage, die Sie stellen, müssen Sie diesen dritten Satz möglicherweise nicht erstellen (wie wir in [Zeitreihenprognosen](../../7-TimeSeries/1-Introduction/README.md) feststellen).

## Ein Modell erstellen

Mithilfe Ihrer Trainingsdaten besteht Ihr Ziel darin, ein Modell oder eine statistische Darstellung Ihrer Daten zu erstellen, indem Sie verschiedene Algorithmen verwenden, um es zu **trainieren**. Das Training eines Modells setzt es Daten aus und ermöglicht es ihm, Annahmen über wahrgenommene Muster zu treffen, diese zu validieren und anzunehmen oder abzulehnen.

### Eine Trainingsmethode wählen

Abhängig von Ihrer Frage und der Art Ihrer Daten wählen Sie eine Methode, um sie zu trainieren. Wenn Sie [Scikit-learn's Dokumentation](https://scikit-learn.org/stable/user_guide.html) durchgehen - die wir in diesem Kurs verwenden - können Sie viele Möglichkeiten erkunden, ein Modell zu trainieren. Abhängig von Ihrer Erfahrung müssen Sie möglicherweise mehrere Methoden ausprobieren, um das beste Modell zu erstellen. Sie werden wahrscheinlich einen Prozess durchlaufen, bei dem Datenwissenschaftler die Leistung eines Modells bewerten, indem sie ihm unbekannte Daten zuführen, die Genauigkeit, Verzerrungen und andere qualitätsmindernde Probleme überprüfen und die am besten geeignete Trainingsmethode für die jeweilige Aufgabe auswählen.

### Ein Modell trainieren

Mit Ihren Trainingsdaten sind Sie bereit, sie zu "fitten", um ein Modell zu erstellen. Sie werden feststellen, dass Sie in vielen ML-Bibliotheken den Code 'model.fit' finden - zu diesem Zeitpunkt senden Sie Ihre Merkmalsvariable als Array von Werten (in der Regel 'X') und eine Zielvariable (in der Regel 'y').

### Das Modell bewerten

Sobald der Trainingsprozess abgeschlossen ist (es kann viele Iterationen oder 'Epochen' dauern, um ein großes Modell zu trainieren), können Sie die Qualität des Modells bewerten, indem Sie Testdaten verwenden, um seine Leistung zu messen. Diese Daten sind ein Teil der ursprünglichen Daten, die das Modell zuvor nicht analysiert hat. Sie können eine Tabelle mit Metriken zur Qualität Ihres Modells ausgeben.

🎓 **Modellanpassung**

Im Kontext des maschinellen Lernens bezieht sich die Modellanpassung auf die Genauigkeit der zugrunde liegenden Funktion des Modells, während es versucht, Daten zu analysieren, mit denen es nicht vertraut ist.

🎓 **Underfitting** und **Overfitting** sind häufige Probleme, die die Qualität des Modells beeinträchtigen, da das Modell entweder nicht gut genug oder zu gut passt. Dies führt dazu, dass das Modell Vorhersagen entweder zu eng oder zu locker an seine Trainingsdaten anpasst. Ein überangepasstes Modell sagt Trainingsdaten zu gut voraus, da es die Details und das Rauschen der Daten zu gut gelernt hat. Ein unterangepasstes Modell ist nicht genau, da es weder seine Trainingsdaten noch Daten, die es noch nicht "gesehen" hat, genau analysieren kann.

![überangepasstes Modell](../../../../translated_images/overfitting.1c132d92bfd93cb63240baf63ebdf82c30e30a0a44e1ad49861b82ff600c2b5c.de.png)
> Infografik von [Jen Looper](https://twitter.com/jenlooper)

## Parameteranpassung

Sobald Ihr erstes Training abgeschlossen ist, beobachten Sie die Qualität des Modells und überlegen, wie Sie es durch Anpassung seiner 'Hyperparameter' verbessern können. Lesen Sie mehr über den Prozess [in der Dokumentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Vorhersage

Dies ist der Moment, in dem Sie völlig neue Daten verwenden können, um die Genauigkeit Ihres Modells zu testen. In einem 'angewandten' ML-Setting, in dem Sie Webanwendungen erstellen, um das Modell in der Produktion zu verwenden, könnte dieser Prozess das Sammeln von Benutzereingaben (z. B. einen Knopfdruck) umfassen, um eine Variable festzulegen und sie an das Modell zur Inferenz oder Bewertung zu senden.

In diesen Lektionen werden Sie entdecken, wie Sie diese Schritte nutzen, um vorzubereiten, zu erstellen, zu testen, zu bewerten und vorherzusagen - all die Aufgaben eines Datenwissenschaftlers und mehr, während Sie auf Ihrem Weg zum 'Full-Stack'-ML-Ingenieur voranschreiten.

---

## 🚀 Herausforderung

Erstellen Sie ein Flussdiagramm, das die Schritte eines ML-Praktikers darstellt. Wo sehen Sie sich derzeit im Prozess? Wo erwarten Sie Schwierigkeiten? Was erscheint Ihnen einfach?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/)

## Rückblick & Selbststudium

Suchen Sie online nach Interviews mit Datenwissenschaftlern, die über ihre tägliche Arbeit sprechen. Hier ist [eins](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Aufgabe

[Führen Sie ein Interview mit einem Datenwissenschaftler](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.