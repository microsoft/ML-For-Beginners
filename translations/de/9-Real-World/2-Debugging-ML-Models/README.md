<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ba0f6e1019351351c8ee4c92867b6a0b",
  "translation_date": "2025-09-03T21:49:04+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "de"
}
-->
# Postskriptum: Modell-Debugging im maschinellen Lernen mit Komponenten des Responsible AI Dashboards

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Einführung

Maschinelles Lernen beeinflusst unser tägliches Leben. KI findet ihren Weg in einige der wichtigsten Systeme, die uns als Individuen und unsere Gesellschaft betreffen, wie Gesundheitswesen, Finanzen, Bildung und Beschäftigung. Beispielsweise sind Systeme und Modelle an alltäglichen Entscheidungsprozessen beteiligt, wie etwa bei medizinischen Diagnosen oder der Betrugserkennung. Folglich werden die Fortschritte in der KI und deren beschleunigte Einführung von sich entwickelnden gesellschaftlichen Erwartungen und wachsender Regulierung begleitet. Wir sehen ständig Bereiche, in denen KI-Systeme weiterhin Erwartungen verfehlen; sie bringen neue Herausforderungen mit sich, und Regierungen beginnen, KI-Lösungen zu regulieren. Daher ist es wichtig, dass diese Modelle analysiert werden, um faire, zuverlässige, inklusive, transparente und verantwortungsvolle Ergebnisse für alle zu gewährleisten.

In diesem Lehrplan werden wir praktische Werkzeuge betrachten, die verwendet werden können, um zu beurteilen, ob ein Modell Probleme mit verantwortungsvoller KI hat. Traditionelle Debugging-Techniken im maschinellen Lernen basieren oft auf quantitativen Berechnungen wie aggregierter Genauigkeit oder durchschnittlichem Fehlerverlust. Stellen Sie sich vor, was passieren kann, wenn die Daten, die Sie zur Erstellung dieser Modelle verwenden, bestimmte demografische Merkmale wie Rasse, Geschlecht, politische Ansichten oder Religion nicht enthalten oder diese überproportional repräsentieren. Was passiert, wenn die Ausgabe des Modells so interpretiert wird, dass sie eine bestimmte demografische Gruppe bevorzugt? Dies kann zu einer Über- oder Unterrepräsentation dieser sensiblen Merkmale führen, was zu Problemen in Bezug auf Fairness, Inklusivität oder Zuverlässigkeit des Modells führt. Ein weiterer Faktor ist, dass maschinelle Lernmodelle oft als Black Boxes betrachtet werden, was es schwierig macht, zu verstehen und zu erklären, was die Vorhersagen eines Modells antreibt. All dies sind Herausforderungen, denen sich Datenwissenschaftler und KI-Entwickler gegenübersehen, wenn sie nicht über geeignete Werkzeuge verfügen, um die Fairness oder Vertrauenswürdigkeit eines Modells zu debuggen und zu bewerten.

In dieser Lektion lernen Sie, wie Sie Ihre Modelle debuggen können, indem Sie:

- **Fehleranalyse**: Identifizieren, wo in Ihrer Datenverteilung das Modell hohe Fehlerraten aufweist.
- **Modellübersicht**: Vergleichende Analysen über verschiedene Datenkohorten durchführen, um Diskrepanzen in den Leistungsmetriken Ihres Modells zu entdecken.
- **Datenanalyse**: Untersuchen, wo es eine Über- oder Unterrepräsentation Ihrer Daten geben könnte, die Ihr Modell dazu bringen kann, eine Daten-Demografie gegenüber einer anderen zu bevorzugen.
- **Feature-Wichtigkeit**: Verstehen, welche Merkmale die Vorhersagen Ihres Modells auf globaler oder lokaler Ebene beeinflussen.

## Voraussetzung

Als Voraussetzung lesen Sie bitte die Übersicht [Responsible AI tools for developers](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif zu Responsible AI Tools](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Fehleranalyse

Traditionelle Leistungsmetriken von Modellen zur Messung der Genauigkeit basieren meist auf Berechnungen von korrekten vs. falschen Vorhersagen. Beispielsweise kann die Feststellung, dass ein Modell zu 89 % genau ist und einen Fehlerverlust von 0,001 aufweist, als gute Leistung angesehen werden. Fehler sind jedoch oft nicht gleichmäßig in Ihrem zugrunde liegenden Datensatz verteilt. Sie könnten eine Modellgenauigkeit von 89 % erzielen, aber feststellen, dass es in verschiedenen Bereichen Ihrer Daten Gruppen gibt, bei denen das Modell zu 42 % fehlerhaft ist. Die Konsequenzen dieser Fehlermuster bei bestimmten Datengruppen können zu Problemen in Bezug auf Fairness oder Zuverlässigkeit führen. Es ist entscheidend, Bereiche zu verstehen, in denen das Modell gut oder schlecht abschneidet. Die Datenbereiche, in denen Ihr Modell viele Ungenauigkeiten aufweist, könnten sich als wichtige demografische Daten herausstellen.

![Analyse und Debugging von Modellfehlern](../../../../translated_images/ea-error-distribution.117452e1177c1dd84fab2369967a68bcde787c76c6ea7fdb92fcf15d1fce8206.de.png)

Die Fehleranalyse-Komponente des RAI Dashboards zeigt, wie Modellfehler über verschiedene Kohorten mit einer Baumvisualisierung verteilt sind. Dies ist nützlich, um Merkmale oder Bereiche zu identifizieren, in denen Ihre Daten eine hohe Fehlerrate aufweisen. Indem Sie sehen, wo die meisten Ungenauigkeiten des Modells auftreten, können Sie beginnen, die Ursache zu untersuchen. Sie können auch Datenkohorten erstellen, um Analysen durchzuführen. Diese Datenkohorten helfen im Debugging-Prozess, um festzustellen, warum die Modellleistung in einer Kohorte gut, aber in einer anderen fehlerhaft ist.

![Fehleranalyse](../../../../translated_images/ea-error-cohort.6886209ea5d438c4daa8bfbf5ce3a7042586364dd3eccda4a4e3d05623ac702a.de.png)

Die visuellen Indikatoren auf der Baumkarte helfen, Problemstellen schneller zu lokalisieren. Beispielsweise zeigt ein dunklerer Rotton eines Baumknotens eine höhere Fehlerrate an.

Eine Heatmap ist eine weitere Visualisierungsfunktion, die Benutzer verwenden können, um die Fehlerrate anhand eines oder zweier Merkmale zu untersuchen und so einen Beitrag zu den Modellfehlern über den gesamten Datensatz oder Kohorten hinweg zu finden.

![Fehleranalyse Heatmap](../../../../translated_images/ea-heatmap.8d27185e28cee3830c85e1b2e9df9d2d5e5c8c940f41678efdb68753f2f7e56c.de.png)

Verwenden Sie die Fehleranalyse, wenn Sie:

* Ein tiefes Verständnis dafür gewinnen möchten, wie Modellfehler über einen Datensatz und mehrere Eingabe- und Merkmalsdimensionen verteilt sind.
* Die aggregierten Leistungsmetriken aufschlüsseln möchten, um fehlerhafte Kohorten automatisch zu entdecken und gezielte Maßnahmen zur Fehlerbehebung zu ergreifen.

## Modellübersicht

Die Bewertung der Leistung eines maschinellen Lernmodells erfordert ein ganzheitliches Verständnis seines Verhaltens. Dies kann erreicht werden, indem mehr als eine Metrik wie Fehlerrate, Genauigkeit, Recall, Präzision oder MAE (Mean Absolute Error) überprüft wird, um Diskrepanzen zwischen den Leistungsmetriken zu finden. Eine Leistungsmetrik mag großartig aussehen, aber Ungenauigkeiten können in einer anderen Metrik aufgedeckt werden. Darüber hinaus hilft der Vergleich der Metriken über den gesamten Datensatz oder Kohorten hinweg, Licht darauf zu werfen, wo das Modell gut oder schlecht abschneidet. Dies ist besonders wichtig, um die Leistung des Modells zwischen sensiblen und unsensiblen Merkmalen (z. B. Rasse, Geschlecht oder Alter von Patienten) zu sehen, um potenzielle Unfairness des Modells aufzudecken. Beispielsweise kann die Entdeckung, dass das Modell in einer Kohorte mit sensiblen Merkmalen fehlerhafter ist, potenzielle Unfairness des Modells offenbaren.

Die Modellübersicht-Komponente des RAI Dashboards hilft nicht nur bei der Analyse der Leistungsmetriken der Datenrepräsentation in einer Kohorte, sondern gibt den Benutzern auch die Möglichkeit, das Verhalten des Modells über verschiedene Kohorten hinweg zu vergleichen.

![Datensatzkohorten - Modellübersicht im RAI Dashboard](../../../../translated_images/model-overview-dataset-cohorts.dfa463fb527a35a0afc01b7b012fc87bf2cad756763f3652bbd810cac5d6cf33.de.png)

Die Funktionalität der merkmalsbasierten Analyse der Komponente ermöglicht es Benutzern, Datenuntergruppen innerhalb eines bestimmten Merkmals einzugrenzen, um Anomalien auf granularer Ebene zu identifizieren. Beispielsweise verfügt das Dashboard über eine eingebaute Intelligenz, um automatisch Kohorten für ein vom Benutzer ausgewähltes Merkmal zu generieren (z. B. *"time_in_hospital < 3"* oder *"time_in_hospital >= 7"*). Dies ermöglicht es einem Benutzer, ein bestimmtes Merkmal aus einer größeren Datengruppe zu isolieren, um zu sehen, ob es ein Schlüsselbeeinflusser der fehlerhaften Ergebnisse des Modells ist.

![Merkmalskohorten - Modellübersicht im RAI Dashboard](../../../../translated_images/model-overview-feature-cohorts.c5104d575ffd0c80b7ad8ede7703fab6166bfc6f9125dd395dcc4ace2f522f70.de.png)

Die Modellübersicht-Komponente unterstützt zwei Klassen von Diskrepanzmetriken:

**Diskrepanz in der Modellleistung**: Diese Metriken berechnen die Diskrepanz (Unterschied) in den Werten der ausgewählten Leistungsmetrik über Untergruppen von Daten. Hier einige Beispiele:

* Diskrepanz in der Genauigkeitsrate
* Diskrepanz in der Fehlerrate
* Diskrepanz in der Präzision
* Diskrepanz im Recall
* Diskrepanz im mittleren absoluten Fehler (MAE)

**Diskrepanz in der Auswahlrate**: Diese Metrik enthält den Unterschied in der Auswahlrate (günstige Vorhersage) zwischen Untergruppen. Ein Beispiel hierfür ist die Diskrepanz in den Kreditgenehmigungsraten. Auswahlrate bedeutet den Anteil der Datenpunkte in jeder Klasse, die als 1 klassifiziert werden (bei binärer Klassifikation) oder die Verteilung der Vorhersagewerte (bei Regression).

## Datenanalyse

> "Wenn Sie die Daten lange genug foltern, werden sie alles gestehen" - Ronald Coase

Diese Aussage klingt extrem, aber es stimmt, dass Daten manipuliert werden können, um jede Schlussfolgerung zu unterstützen. Solche Manipulationen können manchmal unbeabsichtigt geschehen. Als Menschen haben wir alle Vorurteile, und es ist oft schwierig, bewusst zu erkennen, wann man Vorurteile in Daten einführt. Fairness in KI und maschinellem Lernen zu garantieren bleibt eine komplexe Herausforderung.

Daten sind ein großes blinder Fleck für traditionelle Leistungsmetriken von Modellen. Sie könnten hohe Genauigkeitswerte haben, aber dies spiegelt nicht immer die zugrunde liegende Datenverzerrung wider, die in Ihrem Datensatz vorhanden sein könnte. Beispielsweise könnte ein Datensatz von Mitarbeitern, der 27 % Frauen in Führungspositionen und 73 % Männer auf derselben Ebene enthält, dazu führen, dass ein KI-Modell für Stellenanzeigen, das auf diesen Daten trainiert wurde, hauptsächlich eine männliche Zielgruppe für Führungspositionen anspricht. Diese Ungleichheit in den Daten hat die Vorhersage des Modells verzerrt, sodass eine Geschlechterbevorzugung vorliegt. Dies zeigt ein Fairness-Problem, bei dem ein Geschlechterbias im KI-Modell vorhanden ist.

Die Datenanalyse-Komponente des RAI Dashboards hilft, Bereiche zu identifizieren, in denen es eine Über- und Unterrepräsentation im Datensatz gibt. Sie hilft Benutzern, die Ursache von Fehlern und Fairness-Problemen zu diagnostizieren, die durch Datenungleichgewichte oder mangelnde Repräsentation einer bestimmten Datengruppe eingeführt wurden. Dies gibt Benutzern die Möglichkeit, Datensätze basierend auf vorhergesagten und tatsächlichen Ergebnissen, Fehlergruppen und spezifischen Merkmalen zu visualisieren. Manchmal kann die Entdeckung einer unterrepräsentierten Datengruppe auch aufdecken, dass das Modell nicht gut lernt, was zu hohen Ungenauigkeiten führt. Ein Modell mit Datenverzerrung ist nicht nur ein Fairness-Problem, sondern zeigt auch, dass das Modell nicht inklusiv oder zuverlässig ist.

![Datenanalyse-Komponente im RAI Dashboard](../../../../translated_images/dataanalysis-cover.8d6d0683a70a5c1e274e5a94b27a71137e3d0a3b707761d7170eb340dd07f11d.de.png)

Verwenden Sie die Datenanalyse, wenn Sie:

* Die Statistiken Ihres Datensatzes erkunden möchten, indem Sie verschiedene Filter auswählen, um Ihre Daten in verschiedene Dimensionen (auch Kohorten genannt) aufzuteilen.
* Die Verteilung Ihres Datensatzes über verschiedene Kohorten und Merkmalsgruppen verstehen möchten.
* Feststellen möchten, ob Ihre Erkenntnisse zu Fairness, Fehleranalyse und Kausalität (abgeleitet aus anderen Dashboard-Komponenten) auf die Verteilung Ihres Datensatzes zurückzuführen sind.
* Entscheiden möchten, in welchen Bereichen Sie mehr Daten sammeln sollten, um Fehler zu beheben, die durch Repräsentationsprobleme, Label-Rauschen, Merkmalsrauschen, Label-Bias und ähnliche Faktoren entstehen.

## Modellinterpretierbarkeit

Maschinelle Lernmodelle neigen dazu, Black Boxes zu sein. Es kann schwierig sein zu verstehen, welche Schlüsselmerkmale die Vorhersagen eines Modells antreiben. Es ist wichtig, Transparenz darüber zu schaffen, warum ein Modell eine bestimmte Vorhersage trifft. Beispielsweise sollte ein KI-System, das vorhersagt, dass ein Diabetespatient Gefahr läuft, innerhalb von weniger als 30 Tagen wieder ins Krankenhaus eingeliefert zu werden, unterstützende Daten liefern können, die zu seiner Vorhersage geführt haben. Das Vorhandensein unterstützender Datenindikatoren bringt Transparenz, die Kliniken oder Krankenhäusern hilft, fundierte Entscheidungen zu treffen. Darüber hinaus ermöglicht die Fähigkeit, zu erklären, warum ein Modell eine Vorhersage für einen einzelnen Patienten getroffen hat, Verantwortlichkeit im Hinblick auf Gesundheitsvorschriften. Wenn Sie maschinelle Lernmodelle auf eine Weise verwenden, die das Leben von Menschen beeinflusst, ist es entscheidend zu verstehen und zu erklären, was das Verhalten eines Modells beeinflusst. Modell-Erklärbarkeit und Interpretierbarkeit helfen, Fragen in Szenarien wie diesen zu beantworten:

* Modell-Debugging: Warum hat mein Modell diesen Fehler gemacht? Wie kann ich mein Modell verbessern?
* Mensch-KI-Zusammenarbeit: Wie kann ich die Entscheidungen des Modells verstehen und ihm vertrauen?
* Gesetzliche Einhaltung: Erfüllt mein Modell die gesetzlichen Anforderungen?

Die Feature-Wichtigkeit-Komponente des RAI Dashboards hilft Ihnen, Ihr Modell zu debuggen und ein umfassendes Verständnis dafür zu gewinnen, wie ein Modell Vorhersagen trifft. Es ist auch ein nützliches Werkzeug für Fachleute im maschinellen Lernen und Entscheidungsträger, um zu erklären und Beweise für Merkmale zu liefern, die das Verhalten eines Modells beeinflussen, um gesetzliche Anforderungen zu erfüllen. Benutzer können sowohl globale als auch lokale Erklärungen untersuchen, um zu validieren, welche Merkmale die Vorhersagen eines Modells beeinflussen. Globale Erklärungen listen die wichtigsten Merkmale auf, die die Gesamtvorhersage eines Modells beeinflusst haben. Lokale Erklärungen zeigen, welche Merkmale zu einer Vorhersage des Modells für einen einzelnen Fall geführt haben. Die Fähigkeit, lokale Erklärungen zu bewerten, ist auch hilfreich beim Debugging oder bei der Prüfung eines bestimmten Falls, um besser zu verstehen und zu interpretieren, warum ein Modell eine korrekte oder fehlerhafte Vorhersage getroffen hat.

![Feature-Wichtigkeit-Komponente des RAI Dashboards](../../../../translated_images/9-feature-importance.cd3193b4bba3fd4bccd415f566c2437fb3298c4824a3dabbcab15270d783606e.de.png)

* Globale Erklärungen: Zum Beispiel, welche Merkmale beeinflussen das Gesamtverhalten eines Modells zur Krankenhauswiederaufnahme von Diabetespatienten?
* Lokale Erklärungen: Zum Beispiel, warum wurde ein Diabetespatient über 60 Jahre mit vorherigen Krankenhausaufenthalten vorhergesagt, innerhalb von 30 Tagen wieder eingeliefert oder nicht eingeliefert zu werden?

Im Debugging-Prozess der Untersuchung der Modellleistung über verschiedene Kohorten zeigt die Feature-Wichtigkeit, welchen Einfluss ein Merkmal auf die Kohorten hat. Sie hilft, Anomalien aufzudecken, wenn man den Einflussgrad des Merkmals vergleicht, das die fehlerhaften Vorhersagen des Modells antreibt. Die Feature-Wichtigkeit-Komponente kann zeigen, welche Werte in einem Merkmal die Ergebnisse des Modells positiv oder negativ beeinflusst haben. Wenn ein Modell beispielsweise eine fehlerhafte Vorhersage gemacht hat, gibt die Komponente Ihnen die Möglichkeit, ins Detail zu gehen und zu bestimmen, welche Merkmale oder Merkmalswerte die Vorhersage beeinflusst haben. Diese Detailgenauigkeit hilft nicht nur beim Debugging, sondern bietet auch Transparenz und Verantwortlichkeit in Prüfungssituationen. Schließlich kann die Komponente helfen, Fairness-Probleme zu identifizieren. Wenn ein sensibles Merkmal wie Ethnizität oder Geschlecht einen hohen Einfluss auf die Vorhersage eines Modells hat, könnte dies ein Hinweis auf Rassen- oder Geschlechterbias im Modell sein.

![Feature-Wichtigkeit](../../../../translated_images/9-features-influence.3ead3d3f68a84029f1e40d3eba82107445d3d3b6975d4682b23d8acc905da6d0.de.png)

Verwenden Sie Interpretierbarkeit, wenn Sie:

* Bestimmen möchten, wie vertrauenswürdig die Vorhersagen Ihres KI-Systems sind, indem Sie verstehen, welche Merkmale für die Vorhersagen am wichtigsten sind.
* Das Debugging Ihres Modells angehen möchten, indem Sie es zuerst verstehen und feststellen, ob das Modell gesunde Merkmale verwendet oder lediglich falsche Korrelationen.
* Potenzielle Quellen von Unfairness aufdecken möchten, indem Sie verstehen, ob das Modell Vorhersagen auf sensiblen Merkmalen oder auf Merkmalen, die stark mit ihnen korreliert sind, basiert.
* Das Vertrauen der Benutzer in die Entscheidungen Ihres Modells aufbauen möchten, indem Sie lokale Erklärungen generieren, um deren Ergebnisse zu veranschaulichen.
* Eine gesetzliche Prüfung eines KI-Systems abschließen möchten, um Modelle zu validieren und die Auswirkungen von Modellentscheidungen auf Menschen zu überwachen.

## Fazit

Alle Komponenten des RAI Dashboards sind praktische Werkzeuge, die Ihnen helfen, maschinelle Lernmodelle zu erstellen, die weniger schädlich und vertrauenswürdiger für die Gesellschaft sind. Sie verbessern die Prävention von Bedrohungen für Menschenrechte; Diskriminierung oder Ausschluss bestimmter Gruppen von Lebensmöglichkeiten; und das Risiko physischer oder psychologischer Schäden. Sie helfen auch, Vertrauen in die Entscheidungen Ihres Modells aufzubauen, indem sie lokale Erklärungen generieren, um deren Ergebnisse zu veranschaulichen. Einige der potenziellen Schäden können wie folgt klassifiziert werden:

- **Zuweisung**, wenn beispielsweise ein Geschlecht oder eine Ethnizität gegenüber einer anderen bevorzugt wird.
- **Qualität des Dienstes**. Wenn Sie die Daten für ein spezifisches Szenario trainieren, aber die Realität viel komplexer ist, führt dies zu einem schlecht funktionierenden Dienst.
- **Stereotypisierung**. Die Zuordnung einer bestimmten Gruppe zu vorgegebenen Eigenschaften.
- **Herabsetzung**. Unfaire Kritik und Etikettierung von etwas oder jemandem.
- **Über- oder Unterrepräsentation**. Die Idee dahinter ist, dass eine bestimmte Gruppe in einem bestimmten Beruf nicht vertreten ist, und jede Dienstleistung oder Funktion, die dies weiterhin fördert, trägt zu Schaden bei.

### Azure RAI-Dashboard

Das [Azure RAI-Dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu), das auf Open-Source-Tools führender akademischer Institutionen und Organisationen, einschließlich Microsoft, basiert, ist entscheidend für Datenwissenschaftler und KI-Entwickler, um das Verhalten von Modellen besser zu verstehen, unerwünschte Probleme in KI-Modellen zu erkennen und zu beheben.

- Erfahren Sie, wie Sie die verschiedenen Komponenten nutzen können, indem Sie die [Dokumentation des RAI-Dashboards](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) lesen.

- Schauen Sie sich einige [Beispiel-Notebooks des RAI-Dashboards](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) an, um verantwortungsvollere KI-Szenarien in Azure Machine Learning zu debuggen.

---
## 🚀 Herausforderung

Um statistische oder datenbezogene Verzerrungen von Anfang an zu vermeiden, sollten wir:

- eine Vielfalt an Hintergründen und Perspektiven unter den Menschen haben, die an den Systemen arbeiten
- in Datensätze investieren, die die Vielfalt unserer Gesellschaft widerspiegeln
- bessere Methoden entwickeln, um Verzerrungen zu erkennen und zu korrigieren, wenn sie auftreten

Denken Sie über reale Szenarien nach, in denen Ungerechtigkeit beim Modellaufbau und der Nutzung offensichtlich ist. Was sollten wir noch berücksichtigen?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Überprüfung & Selbststudium

In dieser Lektion haben Sie einige praktische Werkzeuge kennengelernt, um verantwortungsvolle KI in maschinelles Lernen zu integrieren.

Sehen Sie sich diesen Workshop an, um tiefer in die Themen einzutauchen:

- Responsible AI Dashboard: Eine zentrale Anlaufstelle für die praktische Umsetzung von RAI von Besmira Nushi und Mehrnoosh Sameki

[![Responsible AI Dashboard: Eine zentrale Anlaufstelle für die praktische Umsetzung von RAI](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Eine zentrale Anlaufstelle für die praktische Umsetzung von RAI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Responsible AI Dashboard: Eine zentrale Anlaufstelle für die praktische Umsetzung von RAI von Besmira Nushi und Mehrnoosh Sameki

Nutzen Sie die folgenden Materialien, um mehr über verantwortungsvolle KI zu erfahren und vertrauenswürdigere Modelle zu entwickeln:

- Microsofts RAI-Dashboard-Tools zur Fehlerbehebung bei ML-Modellen: [Ressourcen für Responsible AI-Tools](https://aka.ms/rai-dashboard)

- Erkunden Sie das Responsible AI Toolkit: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsofts RAI-Ressourcenzentrum: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsofts FATE-Forschungsgruppe: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Aufgabe

[Erkunden Sie das RAI-Dashboard](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.