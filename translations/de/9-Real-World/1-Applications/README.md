<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20f18ff565638be615df4174858e4a7f",
  "translation_date": "2025-09-03T21:48:11+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "de"
}
-->
# Nachwort: Maschinelles Lernen in der realen Welt

![Zusammenfassung des maschinellen Lernens in der realen Welt in einer Sketchnote](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.de.png)  
> Sketchnote von [Tomomi Imura](https://www.twitter.com/girlie_mac)

In diesem Lehrplan hast du viele Möglichkeiten kennengelernt, Daten für das Training vorzubereiten und Modelle des maschinellen Lernens zu erstellen. Du hast eine Reihe klassischer Modelle für Regression, Clustering, Klassifikation, Verarbeitung natürlicher Sprache und Zeitreihen erstellt. Herzlichen Glückwunsch! Jetzt fragst du dich vielleicht, wofür das alles gut ist... Was sind die realen Anwendungen dieser Modelle?

Obwohl in der Industrie viel Interesse an KI besteht, die in der Regel auf Deep Learning basiert, gibt es immer noch wertvolle Anwendungen für klassische Modelle des maschinellen Lernens. Vielleicht nutzt du einige dieser Anwendungen sogar schon heute! In dieser Lektion wirst du erkunden, wie acht verschiedene Branchen und Fachgebiete diese Modelle nutzen, um ihre Anwendungen leistungsfähiger, zuverlässiger, intelligenter und wertvoller für die Nutzer zu machen.

## [Quiz vor der Lektion](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 Finanzen

Der Finanzsektor bietet viele Möglichkeiten für maschinelles Lernen. Viele Probleme in diesem Bereich lassen sich modellieren und mit ML lösen.

### Erkennung von Kreditkartenbetrug

Wir haben im Kurs bereits über [k-means Clustering](../../5-Clustering/2-K-Means/README.md) gelernt, aber wie kann es genutzt werden, um Probleme im Zusammenhang mit Kreditkartenbetrug zu lösen?

K-means Clustering ist nützlich bei einer Technik zur Erkennung von Kreditkartenbetrug, die als **Ausreißererkennung** bezeichnet wird. Ausreißer oder Abweichungen in Beobachtungen eines Datensatzes können uns zeigen, ob eine Kreditkarte normal genutzt wird oder ob etwas Ungewöhnliches vor sich geht. Wie in dem unten verlinkten Artikel gezeigt, kannst du Kreditkartendaten mithilfe eines k-means Clustering-Algorithmus sortieren und jede Transaktion einem Cluster zuordnen, basierend darauf, wie sehr sie als Ausreißer erscheint. Anschließend kannst du die risikoreichsten Cluster auf betrügerische oder legitime Transaktionen untersuchen.  
[Referenz](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Vermögensverwaltung

In der Vermögensverwaltung verwaltet eine Einzelperson oder ein Unternehmen die Investitionen ihrer Kunden. Ihre Aufgabe ist es, das Vermögen langfristig zu erhalten und zu vermehren, daher ist es entscheidend, Investitionen auszuwählen, die gut abschneiden.

Eine Möglichkeit, die Leistung einer bestimmten Investition zu bewerten, ist die statistische Regression. [Lineare Regression](../../2-Regression/1-Tools/README.md) ist ein wertvolles Werkzeug, um zu verstehen, wie ein Fonds im Vergleich zu einem Benchmark abschneidet. Wir können auch ableiten, ob die Ergebnisse der Regression statistisch signifikant sind oder wie stark sie die Investitionen eines Kunden beeinflussen würden. Du könntest deine Analyse sogar erweitern, indem du eine multiple Regression verwendest, bei der zusätzliche Risikofaktoren berücksichtigt werden. Ein Beispiel dafür, wie dies für einen bestimmten Fonds funktionieren würde, findest du im unten verlinkten Artikel zur Bewertung der Fondsleistung mithilfe von Regression.  
[Referenz](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Bildung

Auch der Bildungssektor ist ein sehr interessantes Gebiet, in dem ML angewendet werden kann. Es gibt spannende Probleme zu lösen, wie das Erkennen von Betrug bei Tests oder Aufsätzen oder das Verwalten von (unbeabsichtigten oder absichtlichen) Verzerrungen im Korrekturprozess.

### Vorhersage des Schülerverhaltens

[Coursera](https://coursera.com), ein Anbieter von Online-Kursen, hat einen großartigen Technik-Blog, in dem sie viele ihrer technischen Entscheidungen diskutieren. In dieser Fallstudie haben sie eine Regressionslinie gezeichnet, um mögliche Korrelationen zwischen einer niedrigen NPS-Bewertung (Net Promoter Score) und der Kursbindung oder dem Abbruch zu untersuchen.  
[Referenz](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Verzerrungen minimieren

[Grammarly](https://grammarly.com), ein Schreibassistent, der Rechtschreib- und Grammatikfehler überprüft, verwendet in seinen Produkten ausgeklügelte [Systeme zur Verarbeitung natürlicher Sprache](../../6-NLP/README.md). In ihrem Technik-Blog haben sie eine interessante Fallstudie veröffentlicht, in der sie beschreiben, wie sie mit Geschlechterverzerrungen im maschinellen Lernen umgegangen sind, was du in unserer [Einführungslektion zur Fairness](../../1-Introduction/3-fairness/README.md) gelernt hast.  
[Referenz](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Einzelhandel

Der Einzelhandelssektor kann definitiv vom Einsatz von ML profitieren, sei es durch die Schaffung einer besseren Customer Journey oder die optimale Lagerhaltung.

### Personalisierung der Customer Journey

Bei Wayfair, einem Unternehmen, das Haushaltswaren wie Möbel verkauft, ist es von größter Bedeutung, den Kunden zu helfen, die richtigen Produkte für ihren Geschmack und ihre Bedürfnisse zu finden. In diesem Artikel beschreiben Ingenieure des Unternehmens, wie sie ML und NLP einsetzen, um "die richtigen Ergebnisse für Kunden zu präsentieren". Insbesondere wurde ihre Query Intent Engine entwickelt, um Entitätsextraktion, Klassifikator-Training, Extraktion von Assets und Meinungen sowie Sentiment-Tagging in Kundenbewertungen zu nutzen. Dies ist ein klassisches Anwendungsbeispiel dafür, wie NLP im Online-Einzelhandel funktioniert.  
[Referenz](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Bestandsmanagement

Innovative, agile Unternehmen wie [StitchFix](https://stitchfix.com), ein Box-Service, der Kleidung an Verbraucher verschickt, verlassen sich stark auf ML für Empfehlungen und Bestandsmanagement. Ihre Styling-Teams arbeiten sogar mit ihren Merchandising-Teams zusammen: "Einer unserer Datenwissenschaftler hat mit einem genetischen Algorithmus experimentiert und ihn auf Bekleidung angewendet, um vorherzusagen, welches Kleidungsstück erfolgreich sein könnte, das heute noch nicht existiert. Wir haben das dem Merchandising-Team vorgestellt, und jetzt können sie es als Werkzeug nutzen."  
[Referenz](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Gesundheitswesen

Der Gesundheitssektor kann ML nutzen, um Forschungsaufgaben zu optimieren und auch logistische Probleme wie die Wiederaufnahme von Patienten oder die Eindämmung von Krankheiten zu lösen.

### Management klinischer Studien

Toxizität in klinischen Studien ist ein großes Anliegen für Arzneimittelhersteller. Wie viel Toxizität ist tolerierbar? In dieser Studie führte die Analyse verschiedener Methoden klinischer Studien zur Entwicklung eines neuen Ansatzes zur Vorhersage der Wahrscheinlichkeit von Studienergebnissen. Insbesondere konnten sie Random Forest verwenden, um einen [Klassifikator](../../4-Classification/README.md) zu erstellen, der zwischen Gruppen von Medikamenten unterscheidet.  
[Referenz](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Management von Krankenhauswiederaufnahmen

Krankenhausaufenthalte sind teuer, insbesondere wenn Patienten wieder aufgenommen werden müssen. Dieses Papier beschreibt ein Unternehmen, das ML einsetzt, um das Potenzial für Wiederaufnahmen mithilfe von [Clustering](../../5-Clustering/README.md)-Algorithmen vorherzusagen. Diese Cluster helfen Analysten, "Gruppen von Wiederaufnahmen zu entdecken, die möglicherweise eine gemeinsame Ursache teilen".  
[Referenz](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Krankheitsmanagement

Die jüngste Pandemie hat deutlich gemacht, wie maschinelles Lernen helfen kann, die Ausbreitung von Krankheiten zu stoppen. In diesem Artikel wirst du den Einsatz von ARIMA, logistischen Kurven, linearer Regression und SARIMA erkennen. "Diese Arbeit ist ein Versuch, die Ausbreitungsrate dieses Virus zu berechnen und so die Todesfälle, Genesungen und bestätigten Fälle vorherzusagen, damit wir besser vorbereitet sind und überleben können."  
[Referenz](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ökologie und grüne Technologien

Die Natur und Ökologie bestehen aus vielen empfindlichen Systemen, bei denen das Zusammenspiel zwischen Tieren und der Natur im Fokus steht. Es ist wichtig, diese Systeme genau zu messen und angemessen zu handeln, wenn etwas passiert, wie ein Waldbrand oder ein Rückgang der Tierpopulation.

### Waldmanagement

Du hast in früheren Lektionen über [Reinforcement Learning](../../8-Reinforcement/README.md) gelernt. Es kann sehr nützlich sein, wenn es darum geht, Muster in der Natur vorherzusagen. Insbesondere kann es verwendet werden, um ökologische Probleme wie Waldbrände und die Ausbreitung invasiver Arten zu verfolgen. In Kanada nutzte eine Gruppe von Forschern Reinforcement Learning, um Modelle für die Dynamik von Waldbränden aus Satellitenbildern zu erstellen. Mithilfe eines innovativen "räumlich ausbreitenden Prozesses (SSP)" stellten sie sich einen Waldbrand als "Agenten an jeder Zelle in der Landschaft" vor. "Die Menge an Aktionen, die das Feuer von einem Standort aus zu einem bestimmten Zeitpunkt ausführen kann, umfasst die Ausbreitung nach Norden, Süden, Osten oder Westen oder keine Ausbreitung."  

Dieser Ansatz kehrt das übliche RL-Setup um, da die Dynamik des entsprechenden Markov Decision Process (MDP) eine bekannte Funktion für die unmittelbare Ausbreitung von Waldbränden ist. Lies mehr über die klassischen Algorithmen, die diese Gruppe verwendet hat, unter dem unten stehenden Link.  
[Referenz](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Bewegungserkennung bei Tieren

Während Deep Learning eine Revolution bei der visuellen Verfolgung von Tierbewegungen ausgelöst hat (du kannst deinen eigenen [Eisbären-Tracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) hier erstellen), hat klassisches ML immer noch seinen Platz in dieser Aufgabe.

Sensoren zur Verfolgung von Bewegungen von Nutztieren und IoT nutzen diese Art der visuellen Verarbeitung, aber grundlegendere ML-Techniken sind nützlich, um Daten vorzuverarbeiten. Zum Beispiel wurden in diesem Artikel Schafhaltungen überwacht und analysiert, indem verschiedene Klassifikator-Algorithmen verwendet wurden. Du wirst die ROC-Kurve auf Seite 335 erkennen.  
[Referenz](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energiemanagement

In unseren Lektionen über [Zeitreihenprognosen](../../7-TimeSeries/README.md) haben wir das Konzept intelligenter Parkuhren eingeführt, um Einnahmen für eine Stadt zu generieren, basierend auf dem Verständnis von Angebot und Nachfrage. Dieser Artikel beschreibt im Detail, wie Clustering, Regression und Zeitreihenprognosen kombiniert wurden, um den zukünftigen Energieverbrauch in Irland vorherzusagen, basierend auf intelligenten Messsystemen.  
[Referenz](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Versicherungen

Der Versicherungssektor ist ein weiterer Bereich, der ML nutzt, um tragfähige finanzielle und versicherungsmathematische Modelle zu erstellen und zu optimieren.

### Volatilitätsmanagement

MetLife, ein Anbieter von Lebensversicherungen, ist offen darüber, wie sie Volatilität in ihren Finanzmodellen analysieren und mindern. In diesem Artikel wirst du Visualisierungen zur binären und ordinalen Klassifikation bemerken. Du wirst auch Visualisierungen zur Prognose entdecken.  
[Referenz](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, Kultur und Literatur

In den Künsten, beispielsweise im Journalismus, gibt es viele interessante Probleme. Die Erkennung von Fake News ist ein großes Problem, da nachgewiesen wurde, dass sie die Meinung der Menschen beeinflussen und sogar Demokratien destabilisieren können. Auch Museen können von der Nutzung von ML profitieren, sei es bei der Verknüpfung von Artefakten oder der Ressourcenplanung.

### Erkennung von Fake News

Die Erkennung von Fake News ist in der heutigen Medienlandschaft zu einem Katz-und-Maus-Spiel geworden. In diesem Artikel schlagen Forscher vor, ein System zu testen, das mehrere der ML-Techniken kombiniert, die wir gelernt haben, und das beste Modell einzusetzen: "Dieses System basiert auf der Verarbeitung natürlicher Sprache, um Merkmale aus den Daten zu extrahieren, und diese Merkmale werden dann für das Training von maschinellen Lernklassifikatoren wie Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) und Logistic Regression (LR) verwendet."  
[Referenz](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Dieser Artikel zeigt, wie die Kombination verschiedener ML-Bereiche interessante Ergebnisse liefern kann, die helfen können, die Verbreitung von Fake News zu stoppen und echten Schaden zu verhindern; in diesem Fall war der Anlass die Verbreitung von Gerüchten über COVID-Behandlungen, die zu Gewaltakten führten.

### Museum-ML

Museen stehen an der Schwelle einer KI-Revolution, bei der das Katalogisieren und Digitalisieren von Sammlungen sowie das Finden von Verbindungen zwischen Artefakten durch technologische Fortschritte erleichtert wird. Projekte wie [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) helfen dabei, die Geheimnisse unzugänglicher Sammlungen wie der Vatikanischen Archive zu entschlüsseln. Aber auch der geschäftliche Aspekt von Museen profitiert von ML-Modellen.

Zum Beispiel hat das Art Institute of Chicago Modelle entwickelt, um vorherzusagen, wofür sich das Publikum interessiert und wann es Ausstellungen besuchen wird. Ziel ist es, bei jedem Besuch des Nutzers ein individuelles und optimiertes Besuchserlebnis zu schaffen. "Im Geschäftsjahr 2017 sagte das Modell die Besucherzahlen und Einnahmen mit einer Genauigkeit von 1 Prozent voraus, sagt Andrew Simnick, Senior Vice President am Art Institute."  
[Referenz](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Kundensegmentierung

Die effektivsten Marketingstrategien sprechen Kunden auf unterschiedliche Weise an, basierend auf verschiedenen Gruppierungen. In diesem Artikel werden die Einsatzmöglichkeiten von Clustering-Algorithmen zur Unterstützung differenzierten Marketings diskutiert. Differenziertes Marketing hilft Unternehmen, die Markenbekanntheit zu steigern, mehr Kunden zu erreichen und mehr Umsatz zu erzielen.  
[Referenz](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Herausforderung
Identifizieren Sie einen weiteren Bereich, der von einigen der in diesem Lehrplan erlernten Techniken profitiert, und entdecken Sie, wie er ML einsetzt.

## [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## Überprüfung & Selbststudium

Das Data-Science-Team von Wayfair hat mehrere interessante Videos darüber, wie sie ML in ihrem Unternehmen einsetzen. Es lohnt sich, [einen Blick darauf zu werfen](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Aufgabe

[Eine ML-Schnitzeljagd](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.