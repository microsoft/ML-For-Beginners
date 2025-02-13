# Aufbau von Machine Learning-Lösungen mit verantwortungsbewusster KI

![Zusammenfassung von verantwortungsbewusster KI im Machine Learning in einer Sketchnote](../../../../translated_images/ml-fairness.ef296ebec6afc98a44566d7b6c1ed18dc2bf1115c13ec679bb626028e852fa1d.de.png)
> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Vorlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Einführung

In diesem Lehrplan werden Sie entdecken, wie Machine Learning unser tägliches Leben beeinflussen kann und bereits beeinflusst. Schon jetzt sind Systeme und Modelle in täglichen Entscheidungsprozessen involviert, wie z.B. bei medizinischen Diagnosen, Kreditgenehmigungen oder der Betrugserkennung. Daher ist es wichtig, dass diese Modelle gut funktionieren, um vertrauenswürdige Ergebnisse zu liefern. Wie jede Softwareanwendung werden auch KI-Systeme Erwartungen nicht erfüllen oder unerwünschte Ergebnisse liefern. Deshalb ist es entscheidend, das Verhalten eines KI-Modells zu verstehen und erklären zu können.

Stellen Sie sich vor, was passieren kann, wenn die Daten, die Sie verwenden, um diese Modelle zu erstellen, bestimmte demografische Merkmale wie Rasse, Geschlecht, politische Ansichten oder Religion nicht berücksichtigen oder diese demografischen Merkmale unverhältnismäßig repräsentieren. Was passiert, wenn die Ausgabe des Modells so interpretiert wird, dass sie eine bestimmte demografische Gruppe begünstigt? Was sind die Konsequenzen für die Anwendung? Und was geschieht, wenn das Modell ein nachteilhaftes Ergebnis hat und Menschen schadet? Wer ist verantwortlich für das Verhalten der KI-Systeme? Dies sind einige Fragen, die wir in diesem Lehrplan untersuchen werden.

In dieser Lektion werden Sie:

- Ihr Bewusstsein für die Bedeutung von Fairness im Machine Learning und damit verbundenen Schäden schärfen.
- Sich mit der Praxis vertrautmachen, Ausreißer und ungewöhnliche Szenarien zu erkunden, um Zuverlässigkeit und Sicherheit zu gewährleisten.
- Verständnis dafür gewinnen, wie wichtig es ist, alle zu ermächtigen, indem inklusive Systeme entworfen werden.
- Erkunden, wie entscheidend es ist, die Privatsphäre und Sicherheit von Daten und Personen zu schützen.
- Die Bedeutung eines „Glasbox“-Ansatzes erkennen, um das Verhalten von KI-Modellen zu erklären.
- Achtsam sein, wie wichtig Verantwortung ist, um Vertrauen in KI-Systeme aufzubauen.

## Voraussetzungen

Als Voraussetzung sollten Sie den Lernpfad "Verantwortungsbewusste KI-Prinzipien" absolvieren und das folgende Video zu diesem Thema ansehen:

Erfahren Sie mehr über verantwortungsbewusste KI, indem Sie diesem [Lernpfad](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) folgen.

[![Microsofts Ansatz zur verantwortungsbewussten KI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsofts Ansatz zur verantwortungsbewussten KI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Microsofts Ansatz zur verantwortungsbewussten KI

## Fairness

KI-Systeme sollten alle fair behandeln und vermeiden, ähnliche Gruppen von Menschen unterschiedlich zu beeinflussen. Zum Beispiel sollten KI-Systeme, die Empfehlungen zu medizinischen Behandlungen, Kreditanträgen oder Beschäftigung abgeben, allen mit ähnlichen Symptomen, finanziellen Umständen oder beruflichen Qualifikationen dieselben Empfehlungen geben. Jeder von uns trägt ererbte Vorurteile in sich, die unsere Entscheidungen und Handlungen beeinflussen. Diese Vorurteile können in den Daten, die wir zur Schulung von KI-Systemen verwenden, offensichtlich werden. Solche Manipulation kann manchmal unbeabsichtigt geschehen. Es ist oft schwierig, sich bewusst zu sein, wenn man Vorurteile in Daten einführt.

**„Unfairness“** umfasst negative Auswirkungen oder „Schäden“ für eine Gruppe von Menschen, wie z.B. solche, die in Bezug auf Rasse, Geschlecht, Alter oder Behinderungsstatus definiert sind. Die Hauptschäden, die mit Fairness verbunden sind, können klassifiziert werden als:

- **Zuteilung**, wenn beispielsweise ein Geschlecht oder eine Ethnie bevorzugt wird.
- **Qualität des Services**. Wenn Sie die Daten für ein bestimmtes Szenario trainieren, die Realität jedoch viel komplexer ist, führt dies zu einem schlecht funktionierenden Service. Zum Beispiel ein Handseifenspender, der anscheinend nicht in der Lage ist, Personen mit dunkler Haut zu erkennen. [Referenz](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Herabwürdigung**. Etwas oder jemanden unfair zu kritisieren und zu kennzeichnen. Zum Beispiel wurde eine Bildkennzeichnungstechnologie berüchtigt dafür, Bilder von dunkelhäutigen Menschen als Gorillas zu kennzeichnen.
- **Über- oder Unterrepräsentation**. Die Idee ist, dass eine bestimmte Gruppe in einem bestimmten Beruf nicht gesehen wird, und jeder Service oder jede Funktion, die dies weiterhin fördert, trägt zu Schäden bei.
- **Stereotypisierung**. Eine bestimmte Gruppe mit vorab zugewiesenen Eigenschaften zu assoziieren. Zum Beispiel kann ein Sprachübersetzungssystem zwischen Englisch und Türkisch Ungenauigkeiten aufweisen, aufgrund von Wörtern mit stereotypischen Assoziationen zum Geschlecht.

![Übersetzung ins Türkische](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d4372912f2b690f6aaddd306ffbb49d795ad8d12a4bf141e7af0.de.png)
> Übersetzung ins Türkische

![Übersetzung zurück ins Englische](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e13a8abbc379209bc8032714169e585bdeac75af09b1752aa.de.png)
> Übersetzung zurück ins Englische

Beim Entwerfen und Testen von KI-Systemen müssen wir sicherstellen, dass KI fair ist und nicht darauf programmiert ist, voreingenommene oder diskriminierende Entscheidungen zu treffen, die auch Menschen verboten sind. Die Gewährleistung von Fairness in KI und Machine Learning bleibt eine komplexe soziotechnische Herausforderung.

### Zuverlässigkeit und Sicherheit

Um Vertrauen aufzubauen, müssen KI-Systeme zuverlässig, sicher und konsistent unter normalen und unerwarteten Bedingungen sein. Es ist wichtig zu wissen, wie KI-Systeme in verschiedenen Situationen reagieren, insbesondere wenn sie Ausreißer sind. Beim Aufbau von KI-Lösungen muss ein erheblicher Fokus darauf gelegt werden, wie eine Vielzahl von Umständen, mit denen die KI-Lösungen konfrontiert werden könnten, zu bewältigen ist. Zum Beispiel muss ein selbstfahrendes Auto die Sicherheit der Menschen an oberste Stelle setzen. Daher muss die KI, die das Auto antreibt, alle möglichen Szenarien berücksichtigen, mit denen das Auto konfrontiert werden könnte, wie Nacht, Gewitter oder Schneestürme, Kinder, die über die Straße laufen, Haustiere, Straßenbau usw. Wie gut ein KI-System eine breite Palette von Bedingungen zuverlässig und sicher bewältigen kann, spiegelt das Maß an Voraussicht wider, das der Datenwissenschaftler oder KI-Entwickler während des Designs oder der Tests des Systems berücksichtigt hat.

> [🎥 Klicken Sie hier für ein Video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusivität

KI-Systeme sollten so gestaltet sein, dass sie alle einbeziehen und ermächtigen. Bei der Gestaltung und Implementierung von KI-Systemen identifizieren und beheben Datenwissenschaftler und KI-Entwickler potenzielle Barrieren im System, die unbeabsichtigt Menschen ausschließen könnten. Zum Beispiel gibt es weltweit 1 Milliarde Menschen mit Behinderungen. Mit dem Fortschritt der KI können sie in ihrem täglichen Leben leichter auf eine Vielzahl von Informationen und Möglichkeiten zugreifen. Indem Barrieren angesprochen werden, entstehen Chancen für Innovation und Entwicklung von KI-Produkten mit besseren Erfahrungen, die allen zugutekommen.

> [🎥 Klicken Sie hier für ein Video: Inklusivität in KI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Sicherheit und Datenschutz

KI-Systeme sollten sicher sein und die Privatsphäre der Menschen respektieren. Menschen haben weniger Vertrauen in Systeme, die ihre Privatsphäre, Informationen oder Leben gefährden. Bei der Schulung von Machine Learning-Modellen verlassen wir uns auf Daten, um die besten Ergebnisse zu erzielen. Dabei müssen die Herkunft der Daten und die Integrität berücksichtigt werden. Zum Beispiel, wurden die Daten vom Benutzer eingereicht oder sind sie öffentlich verfügbar? Darüber hinaus ist es beim Arbeiten mit Daten entscheidend, KI-Systeme zu entwickeln, die vertrauliche Informationen schützen und Angriffen widerstehen können. Da KI immer verbreiteter wird, wird der Schutz der Privatsphäre und die Sicherung wichtiger persönlicher und geschäftlicher Informationen zunehmend kritischer und komplexer. Datenschutz- und Datensicherheitsprobleme erfordern besonders viel Aufmerksamkeit für KI, da der Zugang zu Daten für KI-Systeme entscheidend ist, um genaue und informierte Vorhersagen und Entscheidungen über Menschen zu treffen.

> [🎥 Klicken Sie hier für ein Video: Sicherheit in KI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Als Branche haben wir bedeutende Fortschritte im Bereich Datenschutz und Sicherheit gemacht, die maßgeblich durch Vorschriften wie die DSGVO (Datenschutz-Grundverordnung) gefördert wurden.
- Dennoch müssen wir bei KI-Systemen die Spannung zwischen dem Bedarf an mehr persönlichen Daten, um Systeme persönlicher und effektiver zu machen, und dem Datenschutz anerkennen.
- Ähnlich wie bei der Geburt vernetzter Computer mit dem Internet sehen wir auch einen enormen Anstieg der Anzahl von Sicherheitsproblemen im Zusammenhang mit KI.
- Gleichzeitig haben wir gesehen, dass KI zur Verbesserung der Sicherheit eingesetzt wird. Ein Beispiel sind die meisten modernen Antiviren-Scanner, die heute von KI-Heuristiken gesteuert werden.
- Wir müssen sicherstellen, dass unsere Data-Science-Prozesse harmonisch mit den neuesten Datenschutz- und Sicherheitspraktiken kombiniert werden.

### Transparenz

KI-Systeme sollten verständlich sein. Ein entscheidender Teil der Transparenz besteht darin, das Verhalten von KI-Systemen und ihren Komponenten zu erklären. Das Verständnis von KI-Systemen zu verbessern, erfordert, dass die Stakeholder nachvollziehen, wie und warum sie funktionieren, damit sie potenzielle Leistungsprobleme, Sicherheits- und Datenschutzbedenken, Vorurteile, ausschließende Praktiken oder unbeabsichtigte Ergebnisse identifizieren können. Wir glauben auch, dass diejenigen, die KI-Systeme nutzen, ehrlich und offen darüber sein sollten, wann, warum und wie sie diese einsetzen, sowie über die Einschränkungen der Systeme, die sie verwenden. Zum Beispiel, wenn eine Bank ein KI-System zur Unterstützung ihrer Verbraucherentscheidungen verwendet, ist es wichtig, die Ergebnisse zu überprüfen und zu verstehen, welche Daten die Empfehlungen des Systems beeinflussen. Regierungen beginnen, KI in verschiedenen Branchen zu regulieren, sodass Datenwissenschaftler und Organisationen erklären müssen, ob ein KI-System die regulatorischen Anforderungen erfüllt, insbesondere wenn es zu einem unerwünschten Ergebnis kommt.

> [🎥 Klicken Sie hier für ein Video: Transparenz in KI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Da KI-Systeme so komplex sind, ist es schwer zu verstehen, wie sie funktionieren und die Ergebnisse zu interpretieren.
- Dieser Mangel an Verständnis beeinflusst, wie diese Systeme verwaltet, operationalisiert und dokumentiert werden.
- Dieser Mangel an Verständnis beeinflusst insbesondere die Entscheidungen, die auf der Grundlage der Ergebnisse getroffen werden, die diese Systeme produzieren.

### Verantwortung

Die Personen, die KI-Systeme entwerfen und implementieren, müssen für das Verhalten ihrer Systeme verantwortlich sein. Die Notwendigkeit von Verantwortung ist besonders wichtig bei sensiblen Technologien wie Gesichtserkennung. Kürzlich gab es eine wachsende Nachfrage nach Gesichtserkennungstechnologie, insbesondere von Strafverfolgungsbehörden, die das Potenzial dieser Technologie zur Auffindung vermisster Kinder sehen. Diese Technologien könnten jedoch von einer Regierung genutzt werden, um die grundlegenden Freiheiten ihrer Bürger zu gefährden, indem sie beispielsweise die kontinuierliche Überwachung bestimmter Personen ermöglichen. Daher müssen Datenwissenschaftler und Organisationen verantwortlich dafür sein, wie ihr KI-System Individuen oder die Gesellschaft beeinflusst.

[![Führender KI-Forscher warnt vor Massenüberwachung durch Gesichtserkennung](../../../../translated_images/accountability.41d8c0f4b85b6231301d97f17a450a805b7a07aaeb56b34015d71c757cad142e.de.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsofts Ansatz zur verantwortungsbewussten KI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Warnungen vor Massenüberwachung durch Gesichtserkennung

Letztendlich ist eine der größten Fragen für unsere Generation, die erste Generation, die KI in die Gesellschaft bringt, wie sichergestellt werden kann, dass Computer den Menschen gegenüber verantwortlich bleiben und wie sichergestellt werden kann, dass die Menschen, die Computer entwerfen, allen anderen gegenüber verantwortlich bleiben.

## Auswirkungen bewerten

Vor der Schulung eines Machine Learning-Modells ist es wichtig, eine Auswirkungenbewertung durchzuführen, um den Zweck des KI-Systems zu verstehen; was die beabsichtigte Nutzung ist; wo es eingesetzt wird; und wer mit dem System interagiert. Diese Informationen sind hilfreich für Gutachter oder Tester, die das System bewerten, um zu wissen, welche Faktoren bei der Identifizierung potenzieller Risiken und erwarteter Konsequenzen zu berücksichtigen sind.

Die folgenden Bereiche sind bei der Durchführung einer Auswirkungenbewertung zu beachten:

* **Negative Auswirkungen auf Einzelpersonen**. Es ist wichtig, sich über Einschränkungen oder Anforderungen, nicht unterstützte Nutzungen oder bekannte Einschränkungen, die die Leistung des Systems behindern, bewusst zu sein, um sicherzustellen, dass das System nicht in einer Weise verwendet wird, die Einzelpersonen schaden könnte.
* **Datenanforderungen**. Ein Verständnis darüber, wie und wo das System Daten verwenden wird, ermöglicht es Gutachtern, etwaige Datenanforderungen zu erkunden, die Sie beachten sollten (z.B. DSGVO oder HIPAA-Datenvorschriften). Darüber hinaus sollte geprüft werden, ob die Quelle oder Menge der Daten ausreichend für das Training ist.
* **Zusammenfassung der Auswirkungen**. Erstellen Sie eine Liste potenzieller Schäden, die durch die Nutzung des Systems entstehen könnten. Überprüfen Sie im Verlauf des ML-Lebenszyklus, ob die identifizierten Probleme gemildert oder angesprochen werden.
* **Anwendbare Ziele** für jedes der sechs Kernprinzipien. Bewerten Sie, ob die Ziele jedes der Prinzipien erfüllt werden und ob es Lücken gibt.

## Debugging mit verantwortungsbewusster KI

Ähnlich wie beim Debugging einer Softwareanwendung ist das Debugging eines KI-Systems ein notwendiger Prozess, um Probleme im System zu identifizieren und zu beheben. Es gibt viele Faktoren, die dazu führen können, dass ein Modell nicht wie erwartet oder verantwortungsvoll funktioniert. Die meisten traditionellen Leistungsmetriken für Modelle sind quantitative Aggregationen der Leistung eines Modells, die nicht ausreichen, um zu analysieren, wie ein Modell gegen die Prinzipien verantwortungsbewusster KI verstößt. Darüber hinaus ist ein Machine Learning-Modell eine Black Box, die es schwierig macht zu verstehen, was seine Ergebnisse beeinflusst oder eine Erklärung zu liefern, wenn es einen Fehler macht. Später in diesem Kurs werden wir lernen, wie wir das Responsible AI Dashboard verwenden können, um KI-Systeme zu debuggen. Das Dashboard bietet ein ganzheitliches Werkzeug für Datenwissenschaftler und KI-Entwickler, um Folgendes durchzuführen:

* **Fehleranalyse**. Um die Fehlerverteilung des Modells zu identifizieren, die die Fairness oder Zuverlässigkeit des Systems beeinträchtigen kann.
* **Modellübersicht**. Um herauszufinden, wo es Ungleichheiten in der Leistung des Modells über Datenkohorten hinweg gibt.
* **Datenanalyse**. Um die Datenverteilung zu verstehen und potenzielle Vorurteile in den Daten zu identifizieren, die zu Fairness-, Inklusivitäts- und Zuverlässigkeitsproblemen führen könnten.
* **Modellinterpretierbarkeit**. Um zu verstehen, was die Vorhersagen des Modells beeinflusst oder beeinflusst. Dies hilft, das Verhalten des Modells zu erklären, was wichtig für Transparenz und Verantwortung ist.

## 🚀 Herausforderung

Um zu verhindern, dass Schäden von vornherein entstehen, sollten wir:

- eine Vielfalt von Hintergründen und Perspektiven unter den Menschen haben, die an den Systemen arbeiten
- in Datensätze investieren, die die Vielfalt unserer Gesellschaft widerspiegeln
- bessere Methoden im gesamten Lebenszyklus des Machine Learning entwickeln, um verantwortungsbewusste KI zu erkennen und zu korrigieren, wenn sie auftritt

Denken Sie an reale Szenarien, in denen das Misstrauen gegenüber einem Modell offensichtlich ist, sowohl beim Modellaufbau als auch bei der Nutzung. Was sollten wir noch berücksichtigen?

## [Nachlesungsquiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)

## Überprüfung & Selbststudium

In dieser Lektion haben Sie einige Grundlagen der Konzepte von Fairness und Unfairness im Machine Learning gelernt.

Sehen Sie sich diesen Workshop an, um tiefer in die Themen einzutauchen:

- Auf der Suche nach verantwortungsbewusster KI: Prinzipien in die Praxis umsetzen von Besmira Nushi, Mehrnoosh Sameki und Amit Sharma

[![Responsible AI Toolbox: Ein Open-Source-Rahmenwerk für den Aufbau verantwortungsbewusster KI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Ein Open-Source-Rahmenwerk für den Aufbau verantwortungsbewusster KI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: RAI Toolbox: Ein Open-Source-Rahmenwerk für den Aufbau verantwortungsbewusster KI von Besmira Nushi, Mehrnoosh Sameki und Amit Sharma

Lesen Sie auch:

- Microsofts RAI-Ressourcenzentrum: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsofts FATE-Forschungsgruppe: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub-Repository](https://github.com/microsoft/responsible-ai-toolbox)

Lesen Sie über die Tools von Azure Machine Learning, um Fairness sicherzustellen:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Aufgabe

[RAI Toolbox erkunden](assignment.md)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, sollten Sie sich bewusst sein, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als autoritative Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.