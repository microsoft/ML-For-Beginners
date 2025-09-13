<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-04T22:00:01+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "de"
}
-->
# Einführung in Machine Learning

## [Quiz vor der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML für Anfänger - Einführung in Machine Learning für Anfänger](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML für Anfänger - Einführung in Machine Learning für Anfänger")

> 🎥 Klicken Sie auf das Bild oben, um ein kurzes Video zu dieser Lektion anzusehen.

Willkommen zu diesem Kurs über klassisches Machine Learning für Anfänger! Egal, ob Sie völlig neu in diesem Thema sind oder ein erfahrener ML-Praktiker, der sein Wissen auffrischen möchte – wir freuen uns, dass Sie dabei sind! Wir möchten einen freundlichen Ausgangspunkt für Ihr ML-Studium schaffen und freuen uns über Ihr [Feedback](https://github.com/microsoft/ML-For-Beginners/discussions), das wir gerne bewerten, beantworten und einarbeiten.

[![Einführung in ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Einführung in ML")

> 🎥 Klicken Sie auf das Bild oben, um ein Video anzusehen: John Guttag vom MIT stellt Machine Learning vor.

---
## Einstieg in Machine Learning

Bevor Sie mit diesem Lehrplan beginnen, sollten Sie Ihren Computer so einrichten, dass Sie Notebooks lokal ausführen können.

- **Richten Sie Ihren Computer mit diesen Videos ein**. Nutzen Sie die folgenden Links, um zu erfahren, [wie Sie Python installieren](https://youtu.be/CXZYvNRIAKM) und [einen Texteditor einrichten](https://youtu.be/EU8eayHWoZg) können.
- **Lernen Sie Python**. Es wird empfohlen, ein grundlegendes Verständnis von [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) zu haben, einer Programmiersprache, die für Datenwissenschaftler nützlich ist und die wir in diesem Kurs verwenden.
- **Lernen Sie Node.js und JavaScript**. Wir verwenden JavaScript in diesem Kurs gelegentlich beim Erstellen von Webanwendungen. Daher sollten Sie [Node](https://nodejs.org) und [npm](https://www.npmjs.com/) installiert haben sowie [Visual Studio Code](https://code.visualstudio.com/) für die Entwicklung mit Python und JavaScript.
- **Erstellen Sie ein GitHub-Konto**. Da Sie uns hier auf [GitHub](https://github.com) gefunden haben, haben Sie möglicherweise bereits ein Konto. Falls nicht, erstellen Sie eines und forken Sie diesen Lehrplan, um ihn selbst zu nutzen. (Geben Sie uns gerne auch einen Stern 😊)
- **Entdecken Sie Scikit-learn**. Machen Sie sich mit [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) vertraut, einer Sammlung von ML-Bibliotheken, die wir in diesen Lektionen verwenden.

---
## Was ist Machine Learning?

Der Begriff 'Machine Learning' gehört zu den beliebtesten und am häufigsten verwendeten Begriffen unserer Zeit. Es ist sehr wahrscheinlich, dass Sie diesen Begriff mindestens einmal gehört haben, wenn Sie irgendeine Art von Berührungspunkten mit Technologie haben, unabhängig von Ihrem Arbeitsbereich. Die Mechanismen des Machine Learning sind jedoch für die meisten Menschen ein Rätsel. Für einen Anfänger kann das Thema manchmal überwältigend wirken. Daher ist es wichtig, zu verstehen, was Machine Learning tatsächlich ist, und es Schritt für Schritt anhand praktischer Beispiele zu erlernen.

---
## Der Hype-Zyklus

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends zeigt den aktuellen 'Hype-Zyklus' des Begriffs 'Machine Learning'

---
## Ein geheimnisvolles Universum

Wir leben in einem Universum voller faszinierender Geheimnisse. Große Wissenschaftler wie Stephen Hawking, Albert Einstein und viele andere haben ihr Leben der Suche nach bedeutungsvollen Informationen gewidmet, die die Geheimnisse der Welt um uns herum entschlüsseln. Dies ist die menschliche Bedingung des Lernens: Ein Kind lernt Jahr für Jahr neue Dinge und entdeckt die Struktur seiner Welt, während es erwachsen wird.

---
## Das Gehirn eines Kindes

Das Gehirn eines Kindes und seine Sinne nehmen die Fakten seiner Umgebung wahr und lernen nach und nach die verborgenen Muster des Lebens, die dem Kind helfen, logische Regeln zu entwickeln, um diese Muster zu erkennen. Der Lernprozess des menschlichen Gehirns macht den Menschen zum komplexesten Lebewesen dieser Welt. Indem wir kontinuierlich lernen, verborgene Muster entdecken und auf diesen Mustern aufbauen, können wir uns im Laufe unseres Lebens immer weiter verbessern. Diese Lernfähigkeit und Weiterentwicklungsmöglichkeit steht im Zusammenhang mit einem Konzept namens [Gehirnplastizität](https://www.simplypsychology.org/brain-plasticity.html). Oberflächlich betrachtet können wir einige motivierende Ähnlichkeiten zwischen dem Lernprozess des menschlichen Gehirns und den Konzepten des Machine Learning ziehen.

---
## Das menschliche Gehirn

Das [menschliche Gehirn](https://www.livescience.com/29365-human-brain.html) nimmt Dinge aus der realen Welt wahr, verarbeitet die wahrgenommenen Informationen, trifft rationale Entscheidungen und führt bestimmte Handlungen basierend auf den Umständen aus. Dies nennen wir intelligentes Verhalten. Wenn wir einen Nachbau dieses intelligenten Verhaltensprozesses in eine Maschine programmieren, nennen wir das künstliche Intelligenz (KI).

---
## Einige Begriffe

Obwohl die Begriffe oft verwechselt werden, ist Machine Learning (ML) ein wichtiger Teilbereich der künstlichen Intelligenz. **ML beschäftigt sich mit der Verwendung spezialisierter Algorithmen, um bedeutungsvolle Informationen zu entdecken und verborgene Muster aus wahrgenommenen Daten zu finden, um den rationalen Entscheidungsprozess zu unterstützen**.

---
## KI, ML, Deep Learning

![KI, ML, Deep Learning, Data Science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Ein Diagramm, das die Beziehungen zwischen KI, ML, Deep Learning und Data Science zeigt. Infografik von [Jen Looper](https://twitter.com/jenlooper), inspiriert von [dieser Grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Zu behandelnde Konzepte

In diesem Lehrplan behandeln wir nur die grundlegenden Konzepte des Machine Learning, die ein Anfänger kennen muss. Wir konzentrieren uns auf das sogenannte 'klassische Machine Learning', hauptsächlich unter Verwendung von Scikit-learn, einer hervorragenden Bibliothek, die viele Studenten nutzen, um die Grundlagen zu erlernen. Um breitere Konzepte der künstlichen Intelligenz oder des Deep Learning zu verstehen, ist ein solides Grundwissen im Machine Learning unverzichtbar, und genau das möchten wir hier vermitteln.

---
## In diesem Kurs lernen Sie:

- grundlegende Konzepte des Machine Learning
- die Geschichte des ML
- ML und Fairness
- Regressionstechniken im ML
- Klassifikationstechniken im ML
- Clustering-Techniken im ML
- Techniken zur Verarbeitung natürlicher Sprache im ML
- Zeitreihenprognosen im ML
- Reinforcement Learning
- reale Anwendungen für ML

---
## Was wir nicht behandeln

- Deep Learning
- Neuronale Netze
- KI

Um das Lernen zu erleichtern, vermeiden wir die Komplexität neuronaler Netze, des 'Deep Learning' – des Modellbaus mit vielen Schichten unter Verwendung neuronaler Netze – und der KI, die wir in einem anderen Lehrplan behandeln werden. Wir werden auch einen bevorstehenden Lehrplan zur Datenwissenschaft anbieten, um diesen Aspekt dieses größeren Feldes zu vertiefen.

---
## Warum Machine Learning studieren?

Machine Learning wird aus einer Systemperspektive als die Erstellung automatisierter Systeme definiert, die verborgene Muster aus Daten lernen können, um intelligente Entscheidungen zu unterstützen.

Diese Motivation ist lose inspiriert von der Art und Weise, wie das menschliche Gehirn bestimmte Dinge basierend auf den Daten lernt, die es aus der Außenwelt wahrnimmt.

✅ Überlegen Sie einen Moment, warum ein Unternehmen Machine Learning-Strategien einsetzen möchte, anstatt eine fest codierte regelbasierte Engine zu erstellen.

---
## Anwendungen von Machine Learning

Anwendungen von Machine Learning sind mittlerweile fast überall und so allgegenwärtig wie die Daten, die in unseren Gesellschaften durch Smartphones, vernetzte Geräte und andere Systeme generiert werden. Angesichts des enormen Potenzials moderner Machine Learning-Algorithmen erforschen Forscher ihre Fähigkeit, multidimensionale und multidisziplinäre reale Probleme mit großartigen positiven Ergebnissen zu lösen.

---
## Beispiele für angewandtes ML

**Machine Learning kann auf viele Arten genutzt werden**:

- Um die Wahrscheinlichkeit einer Krankheit anhand der Krankengeschichte oder Berichte eines Patienten vorherzusagen.
- Um Wetterdaten zu nutzen, um Wetterereignisse vorherzusagen.
- Um die Stimmung eines Textes zu verstehen.
- Um Fake News zu erkennen und die Verbreitung von Propaganda zu stoppen.

Finanzen, Wirtschaft, Erdwissenschaften, Weltraumforschung, biomedizinische Technik, Kognitionswissenschaften und sogar Geisteswissenschaften haben Machine Learning adaptiert, um die mühsamen, datenintensiven Probleme ihrer Domänen zu lösen.

---
## Fazit

Machine Learning automatisiert den Prozess der Mustererkennung, indem es bedeutungsvolle Einblicke aus realen oder generierten Daten gewinnt. Es hat sich in Bereichen wie Wirtschaft, Gesundheit und Finanzen als äußerst wertvoll erwiesen.

In naher Zukunft wird das Verständnis der Grundlagen des Machine Learning für Menschen aus allen Bereichen aufgrund seiner weit verbreiteten Anwendung unverzichtbar sein.

---
# 🚀 Herausforderung

Skizzieren Sie auf Papier oder mit einer Online-App wie [Excalidraw](https://excalidraw.com/) Ihr Verständnis der Unterschiede zwischen KI, ML, Deep Learning und Data Science. Fügen Sie einige Ideen hinzu, welche Probleme mit diesen Techniken gut gelöst werden können.

# [Quiz nach der Vorlesung](https://ff-quizzes.netlify.app/en/ml/)

---
# Überprüfung & Selbststudium

Um mehr darüber zu erfahren, wie Sie mit ML-Algorithmen in der Cloud arbeiten können, folgen Sie diesem [Lernpfad](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Machen Sie einen [Lernpfad](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) über die Grundlagen des ML.

---
# Aufgabe

[Starten Sie durch](assignment.md)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, weisen wir darauf hin, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.