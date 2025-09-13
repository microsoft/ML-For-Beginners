<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-04T21:56:58+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "de"
}
-->
# Nachwort: Maschinelles Lernen in der realen Welt

![Zusammenfassung des maschinellen Lernens in der realen Welt in einer Sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

In diesem Lehrplan haben Sie viele Möglichkeiten kennengelernt, Daten für das Training vorzubereiten und maschinelle Lernmodelle zu erstellen. Sie haben eine Reihe klassischer Modelle für Regression, Clustering, Klassifikation, Verarbeitung natürlicher Sprache und Zeitreihen erstellt. Herzlichen Glückwunsch! Nun fragen Sie sich vielleicht, wofür das alles gut ist... Welche Anwendungen gibt es für diese Modelle in der realen Welt?

Obwohl in der Industrie viel Interesse an KI besteht, die oft auf Deep Learning basiert, gibt es immer noch wertvolle Anwendungen für klassische maschinelle Lernmodelle. Vielleicht nutzen Sie einige dieser Anwendungen bereits heute! In dieser Lektion werden Sie erkunden, wie acht verschiedene Branchen und Fachgebiete diese Modelle nutzen, um ihre Anwendungen leistungsfähiger, zuverlässiger, intelligenter und wertvoller für die Nutzer zu machen.

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanzen

Der Finanzsektor bietet viele Möglichkeiten für maschinelles Lernen. Viele Probleme in diesem Bereich lassen sich modellieren und mit ML lösen.

### Erkennung von Kreditkartenbetrug

Wir haben früher im Kurs [k-means Clustering](../../5-Clustering/2-K-Means/README.md) kennengelernt, aber wie kann es verwendet werden, um Probleme im Zusammenhang mit Kreditkartenbetrug zu lösen?

K-means Clustering ist nützlich bei einer Technik zur Erkennung von Kreditkartenbetrug, die als **Ausreißererkennung** bezeichnet wird. Ausreißer oder Abweichungen in Beobachtungen über einen Datensatz können uns zeigen, ob eine Kreditkarte normal verwendet wird oder ob etwas Ungewöhnliches vor sich geht. Wie im unten verlinkten Artikel gezeigt, können Sie Kreditkartendaten mit einem k-means Clustering-Algorithmus sortieren und jede Transaktion einem Cluster zuordnen, basierend darauf, wie sehr sie als Ausreißer erscheint. Anschließend können Sie die riskantesten Cluster auf betrügerische oder legitime Transaktionen bewerten.
[Referenz](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Vermögensverwaltung

In der Vermögensverwaltung verwaltet eine Einzelperson oder Firma Investitionen im Namen ihrer Kunden. Ihre Aufgabe ist es, langfristig Vermögen zu erhalten und zu vermehren, daher ist es entscheidend, Investitionen auszuwählen, die gut abschneiden.

Eine Möglichkeit, die Leistung einer bestimmten Investition zu bewerten, ist die statistische Regression. [Lineare Regression](../../2-Regression/1-Tools/README.md) ist ein wertvolles Werkzeug, um zu verstehen, wie ein Fonds im Vergleich zu einer Benchmark abschneidet. Wir können auch ableiten, ob die Ergebnisse der Regression statistisch signifikant sind oder wie stark sie die Investitionen eines Kunden beeinflussen würden. Sie könnten Ihre Analyse sogar mit multipler Regression erweitern, bei der zusätzliche Risikofaktoren berücksichtigt werden können. Ein Beispiel dafür, wie dies für einen bestimmten Fonds funktionieren würde, finden Sie im unten verlinkten Artikel zur Bewertung der Fondsleistung mithilfe von Regression.
[Referenz](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Bildung

Der Bildungssektor ist ebenfalls ein sehr interessanter Bereich, in dem ML angewendet werden kann. Es gibt interessante Probleme zu lösen, wie das Erkennen von Betrug bei Tests oder Aufsätzen oder das Verwalten von Vorurteilen, ob absichtlich oder nicht, im Korrekturprozess.

### Vorhersage des Schülerverhaltens

[Coursera](https://coursera.com), ein Anbieter von Online-Kursen, hat einen großartigen Tech-Blog, in dem viele technische Entscheidungen diskutiert werden. In dieser Fallstudie haben sie eine Regressionslinie geplottet, um eine mögliche Korrelation zwischen einer niedrigen NPS-Bewertung (Net Promoter Score) und Kursbindung oder -abbruch zu untersuchen.
[Referenz](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Vorurteile mindern

[Grammarly](https://grammarly.com), ein Schreibassistent, der Rechtschreib- und Grammatikfehler überprüft, verwendet ausgeklügelte [Systeme zur Verarbeitung natürlicher Sprache](../../6-NLP/README.md) in seinen Produkten. Sie haben in ihrem Tech-Blog eine interessante Fallstudie veröffentlicht, wie sie mit Geschlechtervorurteilen im maschinellen Lernen umgegangen sind, was Sie in unserer [Einführungslektion zur Fairness](../../1-Introduction/3-fairness/README.md) gelernt haben.
[Referenz](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Einzelhandel

Der Einzelhandelssektor kann definitiv von der Nutzung von ML profitieren, sei es durch die Schaffung einer besseren Kundenreise oder die optimale Lagerhaltung.

### Personalisierung der Kundenreise

Bei Wayfair, einem Unternehmen, das Haushaltswaren wie Möbel verkauft, ist es entscheidend, den Kunden zu helfen, die richtigen Produkte für ihren Geschmack und ihre Bedürfnisse zu finden. In diesem Artikel beschreiben Ingenieure des Unternehmens, wie sie ML und NLP nutzen, um "die richtigen Ergebnisse für Kunden zu präsentieren". Insbesondere wurde ihre Query Intent Engine entwickelt, um Entitätsextraktion, Klassifikatortraining, Asset- und Meinungsextraktion sowie Sentiment-Tagging bei Kundenbewertungen zu nutzen. Dies ist ein klassischer Anwendungsfall dafür, wie NLP im Online-Einzelhandel funktioniert.
[Referenz](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Lagerverwaltung

Innovative, agile Unternehmen wie [StitchFix](https://stitchfix.com), ein Box-Service, der Kleidung an Verbraucher versendet, verlassen sich stark auf ML für Empfehlungen und Lagerverwaltung. Ihre Styling-Teams arbeiten tatsächlich mit ihren Merchandising-Teams zusammen: "Einer unserer Datenwissenschaftler hat mit einem genetischen Algorithmus experimentiert und ihn auf Kleidung angewendet, um vorherzusagen, welches Kleidungsstück erfolgreich sein könnte, das heute noch nicht existiert. Wir haben das dem Merchandising-Team vorgestellt, und jetzt können sie das als Werkzeug nutzen."
[Referenz](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Gesundheitswesen

Der Gesundheitssektor kann ML nutzen, um Forschungsaufgaben und logistische Probleme wie die Wiederaufnahme von Patienten oder die Eindämmung von Krankheiten zu optimieren.

### Verwaltung klinischer Studien

Toxizität in klinischen Studien ist ein großes Anliegen für Arzneimittelhersteller. Wie viel Toxizität ist tolerierbar? In dieser Studie führte die Analyse verschiedener klinischer Studienmethoden zur Entwicklung eines neuen Ansatzes zur Vorhersage der Wahrscheinlichkeit von Ergebnissen klinischer Studien. Insbesondere konnten sie Random Forest verwenden, um einen [Klassifikator](../../4-Classification/README.md) zu erstellen, der zwischen Gruppen von Medikamenten unterscheidet.
[Referenz](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Krankenhaus-Wiederaufnahme-Management

Krankenhauspflege ist teuer, insbesondere wenn Patienten wieder aufgenommen werden müssen. In diesem Artikel wird ein Unternehmen diskutiert, das ML verwendet, um das Potenzial für Wiederaufnahmen mithilfe von [Clustering](../../5-Clustering/README.md)-Algorithmen vorherzusagen. Diese Cluster helfen Analysten, "Gruppen von Wiederaufnahmen zu entdecken, die möglicherweise eine gemeinsame Ursache teilen".
[Referenz](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Krankheitsmanagement

Die jüngste Pandemie hat deutlich gemacht, wie maschinelles Lernen dazu beitragen kann, die Ausbreitung von Krankheiten zu stoppen. In diesem Artikel erkennen Sie die Verwendung von ARIMA, logistischen Kurven, linearer Regression und SARIMA. "Diese Arbeit ist ein Versuch, die Ausbreitungsrate dieses Virus zu berechnen und somit die Todesfälle, Genesungen und bestätigten Fälle vorherzusagen, damit wir uns besser vorbereiten und überleben können."
[Referenz](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ökologie und grüne Technologie

Die Natur und Ökologie bestehen aus vielen empfindlichen Systemen, bei denen das Zusammenspiel zwischen Tieren und Natur im Fokus steht. Es ist wichtig, diese Systeme genau zu messen und angemessen zu handeln, wenn etwas passiert, wie ein Waldbrand oder ein Rückgang der Tierpopulation.

### Waldmanagement

Sie haben in früheren Lektionen [Reinforcement Learning](../../8-Reinforcement/README.md) kennengelernt. Es kann sehr nützlich sein, wenn versucht wird, Muster in der Natur vorherzusagen. Insbesondere kann es verwendet werden, um ökologische Probleme wie Waldbrände und die Ausbreitung invasiver Arten zu verfolgen. In Kanada hat eine Gruppe von Forschern Reinforcement Learning verwendet, um Modelle für die Dynamik von Waldbränden aus Satellitenbildern zu erstellen. Mithilfe eines innovativen "räumlich ausbreitenden Prozesses (SSP)" stellten sie sich einen Waldbrand als "den Agenten an jeder Zelle in der Landschaft" vor. "Die Menge an Aktionen, die das Feuer von einem Standort zu einem beliebigen Zeitpunkt ausführen kann, umfasst die Ausbreitung nach Norden, Süden, Osten oder Westen oder keine Ausbreitung."

Dieser Ansatz kehrt das übliche RL-Setup um, da die Dynamik des entsprechenden Markov Decision Process (MDP) eine bekannte Funktion für die unmittelbare Ausbreitung von Waldbränden ist. Lesen Sie mehr über die klassischen Algorithmen, die von dieser Gruppe verwendet wurden, unter dem unten stehenden Link.
[Referenz](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Bewegungserkennung von Tieren

Während Deep Learning eine Revolution in der visuellen Verfolgung von Tierbewegungen ausgelöst hat (Sie können Ihren eigenen [Eisbären-Tracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) hier erstellen), hat klassisches ML immer noch einen Platz in dieser Aufgabe.

Sensoren zur Verfolgung von Bewegungen von Nutztieren und IoT nutzen diese Art der visuellen Verarbeitung, aber grundlegende ML-Techniken sind nützlich, um Daten vorzuverarbeiten. Zum Beispiel wurden in diesem Artikel die Haltungen von Schafen überwacht und analysiert, indem verschiedene Klassifikator-Algorithmen verwendet wurden. Sie könnten die ROC-Kurve auf Seite 335 erkennen.
[Referenz](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energiemanagement

In unseren Lektionen über [Zeitreihenprognosen](../../7-TimeSeries/README.md) haben wir das Konzept intelligenter Parkuhren eingeführt, um Einnahmen für eine Stadt basierend auf dem Verständnis von Angebot und Nachfrage zu generieren. Dieser Artikel diskutiert ausführlich, wie Clustering, Regression und Zeitreihenprognosen kombiniert wurden, um den zukünftigen Energieverbrauch in Irland vorherzusagen, basierend auf intelligenten Zählern.
[Referenz](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Versicherungen

Der Versicherungssektor ist ein weiterer Bereich, der ML nutzt, um tragfähige finanzielle und versicherungsmathematische Modelle zu erstellen und zu optimieren.

### Volatilitätsmanagement

MetLife, ein Anbieter von Lebensversicherungen, ist offen darüber, wie sie Volatilität in ihren Finanzmodellen analysieren und mindern. In diesem Artikel werden Sie binäre und ordinale Klassifikationsvisualisierungen bemerken. Sie werden auch Prognosevisualisierungen entdecken.
[Referenz](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, Kultur und Literatur

In den Künsten, beispielsweise im Journalismus, gibt es viele interessante Probleme. Die Erkennung von Fake News ist ein großes Problem, da nachgewiesen wurde, dass sie die Meinung der Menschen beeinflussen und sogar Demokratien stürzen können. Museen können ebenfalls von der Nutzung von ML profitieren, sei es bei der Suche nach Verbindungen zwischen Artefakten oder der Ressourcenplanung.

### Erkennung von Fake News

Die Erkennung von Fake News ist heute ein Katz-und-Maus-Spiel in den Medien. In diesem Artikel schlagen Forscher vor, dass ein System, das mehrere der ML-Techniken kombiniert, die wir studiert haben, getestet und das beste Modell eingesetzt werden kann: "Dieses System basiert auf der Verarbeitung natürlicher Sprache, um Merkmale aus den Daten zu extrahieren, und diese Merkmale werden dann für das Training von maschinellen Lernklassifikatoren wie Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) und Logistic Regression (LR) verwendet."
[Referenz](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Dieser Artikel zeigt, wie die Kombination verschiedener ML-Bereiche interessante Ergebnisse liefern kann, die helfen können, die Verbreitung von Fake News zu stoppen und echten Schaden zu verhindern; in diesem Fall war der Anstoß die Verbreitung von Gerüchten über COVID-Behandlungen, die zu Gewalt durch Menschenmengen führten.

### Museum ML

Museen stehen am Beginn einer KI-Revolution, bei der das Katalogisieren und Digitalisieren von Sammlungen sowie das Finden von Verbindungen zwischen Artefakten einfacher wird, da die Technologie voranschreitet. Projekte wie [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) helfen, die Geheimnisse unzugänglicher Sammlungen wie der Vatikanischen Archive zu entschlüsseln. Aber auch der geschäftliche Aspekt von Museen profitiert von ML-Modellen.

Zum Beispiel hat das Art Institute of Chicago Modelle entwickelt, um vorherzusagen, woran Besucher interessiert sind und wann sie Ausstellungen besuchen werden. Das Ziel ist es, jedes Mal, wenn der Nutzer das Museum besucht, individuelle und optimierte Besuchererlebnisse zu schaffen. "Im Geschäftsjahr 2017 sagte das Modell die Besucherzahlen und Einnahmen mit einer Genauigkeit von 1 Prozent voraus, sagt Andrew Simnick, Senior Vice President am Art Institute."
[Referenz](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Kundensegmentierung

Die effektivsten Marketingstrategien zielen auf Kunden auf unterschiedliche Weise ab, basierend auf verschiedenen Gruppierungen. In diesem Artikel werden die Einsatzmöglichkeiten von Clustering-Algorithmen diskutiert, um differenziertes Marketing zu unterstützen. Differenziertes Marketing hilft Unternehmen, die Markenbekanntheit zu verbessern, mehr Kunden zu erreichen und mehr Geld zu verdienen.
[Referenz](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Herausforderung

Identifizieren Sie einen weiteren Sektor, der von einigen der Techniken profitiert, die Sie in diesem Lehrplan gelernt haben, und entdecken Sie, wie er ML nutzt.
## [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

## Wiederholung & Selbststudium

Das Data-Science-Team von Wayfair hat mehrere interessante Videos darüber, wie sie ML in ihrem Unternehmen einsetzen. Es lohnt sich, [einen Blick darauf zu werfen](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Aufgabe

[Eine ML-Schnitzeljagd](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.