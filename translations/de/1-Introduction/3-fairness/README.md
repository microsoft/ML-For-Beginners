<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8f819813b2ca08ec7b9f60a2c9336045",
  "translation_date": "2025-09-03T21:50:06+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "de"
}
-->
# Entwicklung von Machine-Learning-Lösungen mit verantwortungsbewusster KI

![Zusammenfassung der verantwortungsbewussten KI im Machine Learning in einer Sketchnote](../../../../translated_images/ml-fairness.ef296ebec6afc98a44566d7b6c1ed18dc2bf1115c13ec679bb626028e852fa1d.de.png)
> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Einführung

In diesem Lehrplan werden Sie beginnen zu entdecken, wie Machine Learning unser tägliches Leben beeinflussen kann und bereits beeinflusst. Schon jetzt sind Systeme und Modelle in alltägliche Entscheidungsprozesse eingebunden, wie etwa bei medizinischen Diagnosen, Kreditgenehmigungen oder der Betrugserkennung. Daher ist es wichtig, dass diese Modelle zuverlässig arbeiten, um vertrauenswürdige Ergebnisse zu liefern. Wie jede Softwareanwendung können auch KI-Systeme Erwartungen nicht erfüllen oder unerwünschte Ergebnisse liefern. Deshalb ist es entscheidend, das Verhalten eines KI-Modells verstehen und erklären zu können.

Stellen Sie sich vor, was passieren kann, wenn die Daten, die Sie zur Erstellung dieser Modelle verwenden, bestimmte demografische Gruppen wie Rasse, Geschlecht, politische Ansichten oder Religion nicht berücksichtigen oder diese unverhältnismäßig repräsentieren. Was passiert, wenn die Ergebnisse des Modells so interpretiert werden, dass sie eine bestimmte demografische Gruppe bevorzugen? Welche Konsequenzen hat das für die Anwendung? Und was passiert, wenn das Modell ein schädliches Ergebnis liefert? Wer ist für das Verhalten des KI-Systems verantwortlich? Dies sind einige der Fragen, die wir in diesem Lehrplan untersuchen werden.

In dieser Lektion werden Sie:

- Ihr Bewusstsein für die Bedeutung von Fairness im Machine Learning und die damit verbundenen Schäden schärfen.
- Sich mit der Praxis vertraut machen, Ausreißer und ungewöhnliche Szenarien zu untersuchen, um Zuverlässigkeit und Sicherheit zu gewährleisten.
- Ein Verständnis dafür gewinnen, wie wichtig es ist, alle Menschen durch die Gestaltung inklusiver Systeme zu stärken.
- Erkunden, wie entscheidend es ist, die Privatsphäre und Sicherheit von Daten und Menschen zu schützen.
- Die Bedeutung eines transparenten Ansatzes erkennen, um das Verhalten von KI-Modellen zu erklären.
- Sich bewusst machen, wie essenziell Verantwortlichkeit ist, um Vertrauen in KI-Systeme aufzubauen.

## Voraussetzungen

Als Voraussetzung sollten Sie den "Responsible AI Principles"-Lernpfad absolvieren und das folgende Video zum Thema ansehen:

Erfahren Sie mehr über verantwortungsbewusste KI, indem Sie diesem [Lernpfad](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) folgen.

[![Microsofts Ansatz für verantwortungsbewusste KI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsofts Ansatz für verantwortungsbewusste KI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Microsofts Ansatz für verantwortungsbewusste KI

## Fairness

KI-Systeme sollten alle Menschen fair behandeln und vermeiden, ähnliche Gruppen unterschiedlich zu beeinflussen. Beispielsweise sollten KI-Systeme bei medizinischen Behandlungen, Kreditanträgen oder Beschäftigungsentscheidungen die gleichen Empfehlungen für alle mit ähnlichen Symptomen, finanziellen Umständen oder beruflichen Qualifikationen geben. Jeder von uns trägt als Mensch ererbte Vorurteile mit sich, die unsere Entscheidungen und Handlungen beeinflussen. Diese Vorurteile können sich in den Daten widerspiegeln, die wir zur Schulung von KI-Systemen verwenden. Solche Manipulationen können manchmal unbeabsichtigt geschehen. Es ist oft schwierig, bewusst zu erkennen, wann man Vorurteile in Daten einführt.

**„Unfairness“** umfasst negative Auswirkungen oder „Schäden“ für eine Gruppe von Menschen, wie etwa solche, die durch Rasse, Geschlecht, Alter oder Behinderungsstatus definiert sind. Die Hauptschäden im Zusammenhang mit Fairness können wie folgt klassifiziert werden:

- **Zuweisung**, wenn beispielsweise ein Geschlecht oder eine Ethnie gegenüber einer anderen bevorzugt wird.
- **Qualität des Dienstes**. Wenn die Daten für ein spezifisches Szenario trainiert werden, die Realität jedoch viel komplexer ist, führt dies zu einem schlecht funktionierenden Dienst. Zum Beispiel ein Seifenspender, der scheinbar keine Menschen mit dunkler Haut erkennen konnte. [Referenz](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Herabsetzung**. Unfaire Kritik oder Etikettierung von etwas oder jemandem. Ein Beispiel ist eine Bildkennzeichnungstechnologie, die Bilder von dunkelhäutigen Menschen fälschlicherweise als Gorillas bezeichnete.
- **Über- oder Unterrepräsentation**. Die Idee, dass eine bestimmte Gruppe in einem bestimmten Beruf nicht gesehen wird, und jede Funktion oder Dienstleistung, die dies weiter fördert, trägt zu Schaden bei.
- **Stereotypisierung**. Die Zuordnung vorgefertigter Attribute zu einer bestimmten Gruppe. Zum Beispiel kann ein Sprachübersetzungssystem zwischen Englisch und Türkisch Ungenauigkeiten aufweisen, die auf stereotypische Geschlechtsassoziationen zurückzuführen sind.

![Übersetzung ins Türkische](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d4372912f2b690f6aaddd306ffbb49d795ad8d12a4bf141e7af0.de.png)
> Übersetzung ins Türkische

![Übersetzung zurück ins Englische](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e13a8abbc379209bc8032714169e585bdeac75af09b1752aa.de.png)
> Übersetzung zurück ins Englische

Beim Entwerfen und Testen von KI-Systemen müssen wir sicherstellen, dass KI fair ist und nicht so programmiert wird, dass sie voreingenommene oder diskriminierende Entscheidungen trifft, die auch Menschen untersagt sind. Fairness in KI und Machine Learning zu garantieren bleibt eine komplexe soziotechnische Herausforderung.

### Zuverlässigkeit und Sicherheit

Um Vertrauen aufzubauen, müssen KI-Systeme zuverlässig, sicher und konsistent unter normalen und unerwarteten Bedingungen sein. Es ist wichtig zu wissen, wie sich KI-Systeme in einer Vielzahl von Situationen verhalten, insbesondere bei Ausreißern. Beim Aufbau von KI-Lösungen muss ein erheblicher Fokus darauf gelegt werden, wie eine breite Palette von Umständen gehandhabt werden kann, denen die KI-Lösungen begegnen könnten. Zum Beispiel muss ein selbstfahrendes Auto die Sicherheit der Menschen als oberste Priorität betrachten. Folglich muss die KI, die das Auto antreibt, alle möglichen Szenarien berücksichtigen, denen das Auto begegnen könnte, wie Nacht, Gewitter oder Schneestürme, Kinder, die über die Straße laufen, Haustiere, Straßenbauarbeiten usw. Wie gut ein KI-System eine Vielzahl von Bedingungen zuverlässig und sicher handhaben kann, spiegelt das Maß an Antizipation wider, das der Datenwissenschaftler oder KI-Entwickler während des Designs oder Tests des Systems berücksichtigt hat.

> [🎥 Klicken Sie hier für ein Video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusivität

KI-Systeme sollten so gestaltet sein, dass sie alle einbeziehen und stärken. Beim Entwerfen und Implementieren von KI-Systemen identifizieren und adressieren Datenwissenschaftler und KI-Entwickler potenzielle Barrieren im System, die Menschen unbeabsichtigt ausschließen könnten. Zum Beispiel gibt es weltweit 1 Milliarde Menschen mit Behinderungen. Mit den Fortschritten in der KI können sie in ihrem täglichen Leben leichter auf eine Vielzahl von Informationen und Möglichkeiten zugreifen. Durch die Beseitigung von Barrieren entstehen Chancen, KI-Produkte mit besseren Erfahrungen zu entwickeln, die allen zugutekommen.

> [🎥 Klicken Sie hier für ein Video: Inklusivität in KI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Sicherheit und Datenschutz

KI-Systeme sollten sicher sein und die Privatsphäre der Menschen respektieren. Menschen haben weniger Vertrauen in Systeme, die ihre Privatsphäre, Informationen oder ihr Leben gefährden. Beim Training von Machine-Learning-Modellen verlassen wir uns auf Daten, um die besten Ergebnisse zu erzielen. Dabei muss die Herkunft und Integrität der Daten berücksichtigt werden. Zum Beispiel: Wurden die Daten von Nutzern eingereicht oder waren sie öffentlich verfügbar? Während der Arbeit mit den Daten ist es entscheidend, KI-Systeme zu entwickeln, die vertrauliche Informationen schützen und Angriffen widerstehen können. Da KI immer häufiger eingesetzt wird, wird der Schutz der Privatsphäre und die Sicherung wichtiger persönlicher und geschäftlicher Informationen immer wichtiger und komplexer. Datenschutz- und Datensicherheitsfragen erfordern besonders große Aufmerksamkeit bei KI, da der Zugang zu Daten entscheidend ist, damit KI-Systeme genaue und fundierte Vorhersagen und Entscheidungen über Menschen treffen können.

> [🎥 Klicken Sie hier für ein Video: Sicherheit in KI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Als Branche haben wir bedeutende Fortschritte im Bereich Datenschutz und Sicherheit gemacht, die maßgeblich durch Vorschriften wie die DSGVO (Datenschutz-Grundverordnung) vorangetrieben wurden.
- Dennoch müssen wir bei KI-Systemen die Spannung zwischen dem Bedarf an mehr persönlichen Daten, um Systeme persönlicher und effektiver zu machen, und dem Datenschutz anerkennen.
- Ähnlich wie bei der Geburt vernetzter Computer mit dem Internet sehen wir auch einen enormen Anstieg der Sicherheitsprobleme im Zusammenhang mit KI.
- Gleichzeitig wird KI genutzt, um die Sicherheit zu verbessern. Ein Beispiel: Die meisten modernen Antiviren-Scanner werden heute von KI-Heuristiken betrieben.
- Wir müssen sicherstellen, dass unsere Datenwissenschaftsprozesse harmonisch mit den neuesten Datenschutz- und Sicherheitspraktiken zusammenarbeiten.

### Transparenz

KI-Systeme sollten verständlich sein. Ein wesentlicher Bestandteil der Transparenz ist die Erklärung des Verhaltens von KI-Systemen und ihrer Komponenten. Die Verbesserung des Verständnisses von KI-Systemen erfordert, dass Interessengruppen verstehen, wie und warum sie funktionieren, damit sie potenzielle Leistungsprobleme, Sicherheits- und Datenschutzbedenken, Vorurteile, ausschließende Praktiken oder unbeabsichtigte Ergebnisse identifizieren können. Wir glauben auch, dass diejenigen, die KI-Systeme nutzen, ehrlich und offen darüber sein sollten, wann, warum und wie sie sich entscheiden, diese einzusetzen. Ebenso über die Grenzen der Systeme, die sie verwenden. Zum Beispiel: Wenn eine Bank ein KI-System zur Unterstützung ihrer Kreditentscheidungen einsetzt, ist es wichtig, die Ergebnisse zu prüfen und zu verstehen, welche Daten die Empfehlungen des Systems beeinflussen. Regierungen beginnen, KI branchenübergreifend zu regulieren, daher müssen Datenwissenschaftler und Organisationen erklären, ob ein KI-System die regulatorischen Anforderungen erfüllt, insbesondere wenn es zu einem unerwünschten Ergebnis kommt.

> [🎥 Klicken Sie hier für ein Video: Transparenz in KI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Da KI-Systeme so komplex sind, ist es schwierig zu verstehen, wie sie funktionieren und die Ergebnisse zu interpretieren.
- Dieses mangelnde Verständnis beeinflusst die Art und Weise, wie diese Systeme verwaltet, operationalisiert und dokumentiert werden.
- Noch wichtiger ist, dass dieses mangelnde Verständnis die Entscheidungen beeinflusst, die auf Grundlage der von diesen Systemen erzeugten Ergebnisse getroffen werden.

### Verantwortlichkeit

Die Menschen, die KI-Systeme entwerfen und einsetzen, müssen für die Funktionsweise ihrer Systeme verantwortlich sein. Die Notwendigkeit von Verantwortlichkeit ist besonders wichtig bei sensiblen Technologien wie Gesichtserkennung. In letzter Zeit gibt es eine wachsende Nachfrage nach Gesichtserkennungstechnologie, insbesondere von Strafverfolgungsbehörden, die das Potenzial der Technologie in Anwendungen wie der Suche nach vermissten Kindern sehen. Diese Technologien könnten jedoch von einer Regierung genutzt werden, um die Grundfreiheiten ihrer Bürger zu gefährden, indem sie beispielsweise eine kontinuierliche Überwachung bestimmter Personen ermöglichen. Daher müssen Datenwissenschaftler und Organisationen verantwortlich dafür sein, wie ihr KI-System Einzelpersonen oder die Gesellschaft beeinflusst.

[![Führender KI-Forscher warnt vor Massenüberwachung durch Gesichtserkennung](../../../../translated_images/accountability.41d8c0f4b85b6231301d97f17a450a805b7a07aaeb56b34015d71c757cad142e.de.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsofts Ansatz für verantwortungsbewusste KI")

> 🎥 Klicken Sie auf das Bild oben für ein Video: Warnungen vor Massenüberwachung durch Gesichtserkennung

Letztendlich ist eine der größten Fragen für unsere Generation, als die erste Generation, die KI in die Gesellschaft bringt, wie wir sicherstellen können, dass Computer weiterhin den Menschen gegenüber verantwortlich bleiben und wie wir sicherstellen können, dass die Menschen, die Computer entwerfen, allen anderen gegenüber verantwortlich bleiben.

## Auswirkungen bewerten

Bevor ein Machine-Learning-Modell trainiert wird, ist es wichtig, eine Auswirkungsbewertung durchzuführen, um den Zweck des KI-Systems zu verstehen; wie es verwendet werden soll; wo es eingesetzt wird; und wer mit dem System interagieren wird. Diese Bewertungen sind hilfreich für Prüfer oder Tester, die das System evaluieren, um zu wissen, welche Faktoren bei der Identifizierung potenzieller Risiken und erwarteter Konsequenzen berücksichtigt werden müssen.

Die folgenden Bereiche sollten bei der Durchführung einer Auswirkungsbewertung berücksichtigt werden:

* **Negative Auswirkungen auf Einzelpersonen**. Es ist wichtig, sich über Einschränkungen oder Anforderungen, nicht unterstützte Verwendungen oder bekannte Einschränkungen, die die Leistung des Systems beeinträchtigen könnten, bewusst zu sein, um sicherzustellen, dass das System nicht auf eine Weise verwendet wird, die Einzelpersonen schaden könnte.
* **Datenanforderungen**. Ein Verständnis dafür, wie und wo das System Daten verwendet, ermöglicht es Prüfern, mögliche Datenanforderungen zu untersuchen, die berücksichtigt werden müssen (z. B. DSGVO- oder HIPPA-Datenvorschriften). Darüber hinaus sollte geprüft werden, ob die Quelle oder Menge der Daten für das Training ausreichend ist.
* **Zusammenfassung der Auswirkungen**. Eine Liste potenzieller Schäden erstellen, die durch die Nutzung des Systems entstehen könnten. Während des gesamten ML-Lebenszyklus überprüfen, ob die identifizierten Probleme gemindert oder adressiert wurden.
* **Anwendbare Ziele** für jedes der sechs Kernprinzipien. Bewerten, ob die Ziele jedes Prinzips erreicht wurden und ob es Lücken gibt.

## Debugging mit verantwortungsbewusster KI

Ähnlich wie beim Debugging einer Softwareanwendung ist das Debugging eines KI-Systems ein notwendiger Prozess zur Identifizierung und Behebung von Problemen im System. Es gibt viele Faktoren, die dazu führen können, dass ein Modell nicht wie erwartet oder verantwortungsvoll funktioniert. Die meisten traditionellen Leistungsmetriken für Modelle sind quantitative Zusammenfassungen der Leistung eines Modells, die nicht ausreichen, um zu analysieren, wie ein Modell gegen die Prinzipien der verantwortungsbewussten KI verstößt. Darüber hinaus ist ein Machine-Learning-Modell eine Blackbox, die es schwierig macht, zu verstehen, was seine Ergebnisse antreibt oder eine Erklärung zu liefern, wenn es einen Fehler macht. Später in diesem Kurs werden wir lernen, wie man das Responsible AI-Dashboard verwendet, um KI-Systeme zu debuggen. Das Dashboard bietet ein ganzheitliches Werkzeug für Datenwissenschaftler und KI-Entwickler, um:

* **Fehleranalyse**. Um die Fehlerverteilung des Modells zu identifizieren, die die Fairness oder Zuverlässigkeit des Systems beeinflussen kann.
* **Modellübersicht**. Um herauszufinden, wo es Leistungsunterschiede des Modells über verschiedene Datenkohorten gibt.
* **Datenanalyse**. Um die Datenverteilung zu verstehen und mögliche Vorurteile in den Daten zu identifizieren, die zu Fairness-, Inklusivitäts- und Zuverlässigkeitsproblemen führen könnten.
* **Modellinterpretierbarkeit**. Um zu verstehen, was die Vorhersagen des Modells beeinflusst. Dies hilft, das Verhalten des Modells zu erklären, was für Transparenz und Verantwortlichkeit wichtig ist.

## 🚀 Herausforderung

Um Schäden von Anfang an zu verhindern, sollten wir:

- eine Vielfalt an Hintergründen und Perspektiven unter den Menschen haben, die an den Systemen arbeiten
- in Datensätze investieren, die die Vielfalt unserer Gesellschaft widerspiegeln
- bessere Methoden im gesamten Machine-Learning-Lebenszyklus entwickeln, um verantwortungsbewusste KI zu erkennen und zu korrigieren, wenn sie auftritt

Denken Sie über reale Szenarien nach, in denen die Unzuverlässigkeit eines Modells beim Modellaufbau und -einsatz offensichtlich ist. Was sollten wir noch berücksichtigen?

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Überprüfung & Selbststudium
In dieser Lektion haben Sie einige Grundlagen der Konzepte von Fairness und Unfairness im maschinellen Lernen kennengelernt.  

Sehen Sie sich diesen Workshop an, um tiefer in die Themen einzutauchen: 

- Auf der Suche nach verantwortungsvoller KI: Prinzipien in die Praxis umsetzen von Besmira Nushi, Mehrnoosh Sameki und Amit Sharma

[![Responsible AI Toolbox: Ein Open-Source-Framework für verantwortungsvolle KI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Ein Open-Source-Framework für verantwortungsvolle KI")


> 🎥 Klicken Sie auf das Bild oben für ein Video: RAI Toolbox: Ein Open-Source-Framework für verantwortungsvolle KI von Besmira Nushi, Mehrnoosh Sameki und Amit Sharma

Lesen Sie außerdem: 

- Microsofts RAI-Ressourcenzentrum: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsofts FATE-Forschungsgruppe: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Responsible AI Toolbox GitHub Repository](https://github.com/microsoft/responsible-ai-toolbox)

Lesen Sie über die Tools von Azure Machine Learning, um Fairness sicherzustellen:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Aufgabe

[Erkunden Sie die RAI Toolbox](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.