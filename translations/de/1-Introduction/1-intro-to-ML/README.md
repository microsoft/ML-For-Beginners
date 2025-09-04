<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "73e9a7245aa57f00cd413ffd22c0ccb6",
  "translation_date": "2025-09-03T21:51:32+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "de"
}
-->
# Einführung in maschinelles Lernen

## [Quiz vor der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![ML für Anfänger - Einführung in maschinelles Lernen für Anfänger](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML für Anfänger - Einführung in maschinelles Lernen für Anfänger")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zu dieser Lektion anzusehen.

Willkommen zu diesem Kurs über klassisches maschinelles Lernen für Anfänger! Egal, ob Sie völlig neu in diesem Thema sind oder ein erfahrener ML-Praktiker, der sein Wissen auffrischen möchte – wir freuen uns, dass Sie dabei sind! Wir möchten einen freundlichen Einstiegspunkt für Ihr ML-Studium schaffen und freuen uns über Ihr [Feedback](https://github.com/microsoft/ML-For-Beginners/discussions), das wir gerne bewerten, beantworten und einarbeiten.

[![Einführung in ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Einführung in ML")

> 🎥 Klicken Sie auf das Bild oben für ein Video: John Guttag vom MIT führt in maschinelles Lernen ein.

---
## Einstieg in maschinelles Lernen

Bevor Sie mit diesem Lehrplan beginnen, müssen Sie Ihren Computer so einrichten, dass Sie Notebooks lokal ausführen können.

- **Konfigurieren Sie Ihren Computer mit diesen Videos**. Verwenden Sie die folgenden Links, um zu erfahren, [wie Sie Python installieren](https://youtu.be/CXZYvNRIAKM) und [einen Texteditor einrichten](https://youtu.be/EU8eayHWoZg) können.
- **Lernen Sie Python**. Es wird empfohlen, ein grundlegendes Verständnis von [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) zu haben, einer Programmiersprache, die für Datenwissenschaftler nützlich ist und die wir in diesem Kurs verwenden.
- **Lernen Sie Node.js und JavaScript**. Wir verwenden JavaScript auch einige Male in diesem Kurs, wenn wir Webanwendungen erstellen. Daher müssen Sie [Node](https://nodejs.org) und [npm](https://www.npmjs.com/) installieren sowie [Visual Studio Code](https://code.visualstudio.com/) für die Entwicklung mit Python und JavaScript verfügbar haben.
- **Erstellen Sie ein GitHub-Konto**. Da Sie uns hier auf [GitHub](https://github.com) gefunden haben, haben Sie möglicherweise bereits ein Konto. Falls nicht, erstellen Sie eines und forken Sie dann diesen Lehrplan, um ihn selbst zu nutzen. (Geben Sie uns gerne auch einen Stern 😊)
- **Entdecken Sie Scikit-learn**. Machen Sie sich mit [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) vertraut, einer Sammlung von ML-Bibliotheken, die wir in diesen Lektionen verwenden.

---
## Was ist maschinelles Lernen?

Der Begriff „maschinelles Lernen“ gehört zu den beliebtesten und am häufigsten verwendeten Begriffen der heutigen Zeit. Es ist nicht unwahrscheinlich, dass Sie diesen Begriff mindestens einmal gehört haben, wenn Sie mit Technologie vertraut sind, unabhängig davon, in welchem Bereich Sie arbeiten. Die Mechanik des maschinellen Lernens ist jedoch für die meisten Menschen ein Rätsel. Für einen Anfänger im Bereich maschinelles Lernen kann das Thema manchmal überwältigend wirken. Daher ist es wichtig, zu verstehen, was maschinelles Lernen tatsächlich ist, und es Schritt für Schritt anhand praktischer Beispiele zu lernen.

---
## Der Hype-Zyklus

![ml hype curve](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.de.png)

> Google Trends zeigt den aktuellen „Hype-Zyklus“ des Begriffs „maschinelles Lernen“

---
## Ein geheimnisvolles Universum

Wir leben in einem Universum voller faszinierender Geheimnisse. Große Wissenschaftler wie Stephen Hawking, Albert Einstein und viele andere haben ihr Leben der Suche nach bedeutungsvollen Informationen gewidmet, die die Geheimnisse der Welt um uns herum entschlüsseln. Dies ist die menschliche Bedingung des Lernens: Ein Kind lernt neue Dinge und entdeckt die Struktur seiner Welt Jahr für Jahr, während es erwachsen wird.

---
## Das Gehirn eines Kindes

Das Gehirn eines Kindes und seine Sinne nehmen die Fakten seiner Umgebung wahr und lernen nach und nach die verborgenen Muster des Lebens, die dem Kind helfen, logische Regeln zu entwickeln, um die erlernten Muster zu erkennen. Der Lernprozess des menschlichen Gehirns macht den Menschen zum raffiniertesten Lebewesen dieser Welt. Das kontinuierliche Lernen durch das Entdecken verborgener Muster und das anschließende Innovieren auf diesen Mustern ermöglicht es uns, uns im Laufe unseres Lebens immer weiter zu verbessern. Diese Lernfähigkeit und Weiterentwicklungskapazität hängt mit einem Konzept namens [Gehirnplastizität](https://www.simplypsychology.org/brain-plasticity.html) zusammen. Oberflächlich betrachtet können wir einige motivierende Ähnlichkeiten zwischen dem Lernprozess des menschlichen Gehirns und den Konzepten des maschinellen Lernens ziehen.

---
## Das menschliche Gehirn

Das [menschliche Gehirn](https://www.livescience.com/29365-human-brain.html) nimmt Dinge aus der realen Welt wahr, verarbeitet die wahrgenommene Information, trifft rationale Entscheidungen und führt bestimmte Handlungen basierend auf den Umständen aus. Dies nennen wir intelligentes Verhalten. Wenn wir einen Nachbau des intelligenten Verhaltensprozesses für eine Maschine programmieren, nennen wir das künstliche Intelligenz (KI).

---
## Einige Begriffe

Obwohl die Begriffe oft verwechselt werden, ist maschinelles Lernen (ML) ein wichtiger Teilbereich der künstlichen Intelligenz. **ML beschäftigt sich mit der Verwendung spezialisierter Algorithmen, um bedeutungsvolle Informationen zu entdecken und verborgene Muster aus wahrgenommenen Daten zu finden, um den rationalen Entscheidungsprozess zu unterstützen**.

---
## KI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.de.png)

> Ein Diagramm, das die Beziehungen zwischen KI, ML, Deep Learning und Datenwissenschaft zeigt. Infografik von [Jen Looper](https://twitter.com/jenlooper), inspiriert von [dieser Grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Zu behandelnde Konzepte

In diesem Lehrplan behandeln wir nur die Kernkonzepte des maschinellen Lernens, die ein Anfänger kennen muss. Wir konzentrieren uns auf das sogenannte „klassische maschinelle Lernen“, hauptsächlich mit Scikit-learn, einer hervorragenden Bibliothek, die viele Studenten nutzen, um die Grundlagen zu lernen. Um breitere Konzepte der künstlichen Intelligenz oder des Deep Learning zu verstehen, ist ein starkes Grundwissen im maschinellen Lernen unerlässlich, und wir möchten es hier anbieten.

---
## In diesem Kurs lernen Sie:

- Kernkonzepte des maschinellen Lernens
- die Geschichte des ML
- ML und Fairness
- Regressionstechniken im ML
- Klassifikationstechniken im ML
- Clustering-Techniken im ML
- Techniken der Verarbeitung natürlicher Sprache im ML
- Zeitreihenprognose-Techniken im ML
- Verstärkendes Lernen
- Anwendungen des ML in der realen Welt

---
## Was wir nicht behandeln

- Deep Learning
- Neuronale Netze
- KI

Um das Lernen zu erleichtern, vermeiden wir die Komplexität neuronaler Netze, „Deep Learning“ – den Aufbau von Modellen mit vielen Schichten unter Verwendung neuronaler Netze – und KI, die wir in einem anderen Lehrplan behandeln werden. Wir werden auch einen bevorstehenden Lehrplan zur Datenwissenschaft anbieten, um diesen Aspekt dieses größeren Feldes zu fokussieren.

---
## Warum maschinelles Lernen studieren?

Maschinelles Lernen wird aus einer Systemperspektive definiert als die Erstellung automatisierter Systeme, die verborgene Muster aus Daten lernen können, um intelligente Entscheidungen zu unterstützen.

Diese Motivation ist lose inspiriert von der Art und Weise, wie das menschliche Gehirn bestimmte Dinge basierend auf den Daten lernt, die es aus der Außenwelt wahrnimmt.

✅ Denken Sie einen Moment darüber nach, warum ein Unternehmen maschinelle Lernstrategien anstelle eines fest codierten regelbasierten Systems verwenden möchte.

---
## Anwendungen des maschinellen Lernens

Anwendungen des maschinellen Lernens sind mittlerweile fast überall und so allgegenwärtig wie die Daten, die in unseren Gesellschaften fließen, generiert durch unsere Smartphones, vernetzte Geräte und andere Systeme. Angesichts des enormen Potenzials moderner maschineller Lernalgorithmen erforschen Forscher ihre Fähigkeit, multidimensionale und interdisziplinäre Probleme des realen Lebens mit großartigen positiven Ergebnissen zu lösen.

---
## Beispiele für angewandtes ML

**Maschinelles Lernen kann auf viele Arten genutzt werden**:

- Um die Wahrscheinlichkeit einer Krankheit anhand der medizinischen Vorgeschichte oder Berichte eines Patienten vorherzusagen.
- Um Wetterdaten zu nutzen, um Wetterereignisse vorherzusagen.
- Um die Stimmung eines Textes zu verstehen.
- Um Fake News zu erkennen und die Verbreitung von Propaganda zu stoppen.

Finanzen, Wirtschaft, Geowissenschaften, Weltraumforschung, biomedizinische Technik, Kognitionswissenschaften und sogar Bereiche der Geisteswissenschaften haben maschinelles Lernen adaptiert, um die mühsamen, datenintensiven Probleme ihrer Domäne zu lösen.

---
## Fazit

Maschinelles Lernen automatisiert den Prozess der Mustererkennung, indem es bedeutungsvolle Erkenntnisse aus realen oder generierten Daten gewinnt. Es hat sich als äußerst wertvoll in Bereichen wie Wirtschaft, Gesundheit und Finanzen erwiesen.

In naher Zukunft wird das Verständnis der Grundlagen des maschinellen Lernens für Menschen aus jedem Bereich aufgrund seiner weit verbreiteten Anwendung unverzichtbar sein.

---
# 🚀 Herausforderung

Skizzieren Sie auf Papier oder mit einer Online-App wie [Excalidraw](https://excalidraw.com/) Ihr Verständnis der Unterschiede zwischen KI, ML, Deep Learning und Datenwissenschaft. Fügen Sie einige Ideen hinzu, welche Probleme mit diesen Techniken gut gelöst werden können.

# [Quiz nach der Vorlesung](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# Überprüfung & Selbststudium

Um mehr darüber zu erfahren, wie Sie mit ML-Algorithmen in der Cloud arbeiten können, folgen Sie diesem [Lernpfad](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Machen Sie einen [Lernpfad](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) über die Grundlagen des ML.

---
# Aufgabe

[Starten Sie durch](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.