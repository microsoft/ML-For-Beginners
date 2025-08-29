# Einführung in das Reinforcement Learning

Reinforcement Learning (RL) wird als eines der grundlegenden Paradigmen des maschinellen Lernens angesehen, neben dem überwachten und unüberwachten Lernen. RL dreht sich um Entscheidungen: die richtigen Entscheidungen zu treffen oder zumindest aus ihnen zu lernen.

Stellen Sie sich vor, Sie haben eine simulierte Umgebung wie den Aktienmarkt. Was passiert, wenn Sie eine bestimmte Regelung auferlegen? Hat sie einen positiven oder negativen Effekt? Wenn etwas Negatives passiert, müssen Sie diese _negative Verstärkung_ annehmen, daraus lernen und den Kurs ändern. Wenn das Ergebnis positiv ist, sollten Sie auf dieser _positiven Verstärkung_ aufbauen.

![peter und der wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.de.png)

> Peter und seine Freunde müssen dem hungrigen Wolf entkommen! Bild von [Jen Looper](https://twitter.com/jenlooper)

## Regionales Thema: Peter und der Wolf (Russland)

[Peter und der Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) ist ein musikalisches Märchen, das von dem russischen Komponisten [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) geschrieben wurde. Es ist die Geschichte des jungen Pioniers Peter, der mutig aus seinem Haus auf die Lichtung im Wald geht, um den Wolf zu jagen. In diesem Abschnitt werden wir Algorithmen des maschinellen Lernens trainieren, die Peter helfen werden:

- **Die Umgebung** zu erkunden und eine optimale Navigationskarte zu erstellen.
- **Zu lernen**, wie man ein Skateboard benutzt und darauf balanciert, um schneller voranzukommen.

[![Peter und der Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klicken Sie auf das Bild oben, um Peter und den Wolf von Prokofiev zu hören.

## Reinforcement Learning

In den vorherigen Abschnitten haben Sie zwei Beispiele für Probleme des maschinellen Lernens gesehen:

- **Überwachtes Lernen**, bei dem wir Datensätze haben, die Beispiel-Lösungen für das Problem vorschlagen, das wir lösen möchten. [Klassifikation](../4-Classification/README.md) und [Regression](../2-Regression/README.md) sind Aufgaben des überwachten Lernens.
- **Unüberwachtes Lernen**, bei dem wir keine beschrifteten Trainingsdaten haben. Das Hauptbeispiel für unüberwachtes Lernen ist [Clustering](../5-Clustering/README.md).

In diesem Abschnitt werden wir Ihnen eine neue Art von Lernproblem vorstellen, das keine beschrifteten Trainingsdaten benötigt. Es gibt mehrere Arten solcher Probleme:

- **[Semi-überwachtes Lernen](https://wikipedia.org/wiki/Semi-supervised_learning)**, bei dem wir eine große Menge an unbeschrifteten Daten haben, die verwendet werden können, um das Modell vorzutrainieren.
- **[Reinforcement Learning](https://wikipedia.org/wiki/Reinforcement_learning)**, bei dem ein Agent lernt, wie er sich verhalten soll, indem er Experimente in einer simulierten Umgebung durchführt.

### Beispiel - Computerspiel

Angenommen, Sie möchten einem Computer beibringen, ein Spiel zu spielen, wie Schach oder [Super Mario](https://wikipedia.org/wiki/Super_Mario). Damit der Computer ein Spiel spielen kann, muss er vorhersagen, welchen Zug er in jedem der Spielzustände machen soll. Auch wenn dies wie ein Klassifikationsproblem erscheinen mag, ist es das nicht - weil wir keinen Datensatz mit Zuständen und entsprechenden Aktionen haben. Auch wenn wir einige Daten wie bestehende Schachpartien oder Aufzeichnungen von Spielern, die Super Mario spielen, haben, ist es wahrscheinlich, dass diese Daten nicht ausreichend eine große Anzahl möglicher Zustände abdecken.

Anstatt nach vorhandenen Spieldaten zu suchen, basiert **Reinforcement Learning** (RL) auf der Idee, *den Computer viele Male spielen zu lassen* und das Ergebnis zu beobachten. Um Reinforcement Learning anzuwenden, benötigen wir daher zwei Dinge:

- **Eine Umgebung** und **einen Simulator**, die es uns ermöglichen, ein Spiel viele Male zu spielen. Dieser Simulator würde alle Spielregeln sowie mögliche Zustände und Aktionen definieren.

- **Eine Belohnungsfunktion**, die uns sagt, wie gut wir während jedes Zuges oder Spiels abgeschnitten haben.

Der Hauptunterschied zwischen anderen Arten des maschinellen Lernens und RL besteht darin, dass wir im RL typischerweise nicht wissen, ob wir gewinnen oder verlieren, bis wir das Spiel beendet haben. Daher können wir nicht sagen, ob ein bestimmter Zug allein gut oder schlecht ist - wir erhalten erst am Ende des Spiels eine Belohnung. Unser Ziel ist es, Algorithmen zu entwerfen, die es uns ermöglichen, ein Modell unter unsicheren Bedingungen zu trainieren. Wir werden über einen RL-Algorithmus namens **Q-Learning** lernen.

## Lektionen

1. [Einführung in Reinforcement Learning und Q-Learning](1-QLearning/README.md)
2. [Verwendung einer Gym-Simulationsumgebung](2-Gym/README.md)

## Danksagungen

"Einführung in Reinforcement Learning" wurde mit ♥️ von [Dmitry Soshnikov](http://soshnikov.com) geschrieben.

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, sollten Sie sich bewusst sein, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle betrachtet werden. Für wichtige Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die aus der Nutzung dieser Übersetzung entstehen.