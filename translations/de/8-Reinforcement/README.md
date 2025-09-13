<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-03T21:58:11+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "de"
}
-->
# Einführung in das Reinforcement Learning

Reinforcement Learning, RL, wird als eines der grundlegenden Paradigmen des maschinellen Lernens angesehen, neben dem überwachten und unüberwachten Lernen. RL dreht sich um Entscheidungen: die richtigen Entscheidungen treffen oder zumindest aus ihnen lernen.

Stellen Sie sich vor, Sie haben eine simulierte Umgebung wie den Aktienmarkt. Was passiert, wenn Sie eine bestimmte Regulierung einführen? Hat dies eine positive oder negative Wirkung? Wenn etwas Negatives passiert, müssen Sie diese _negative Verstärkung_ nutzen, daraus lernen und den Kurs ändern. Wenn es ein positives Ergebnis ist, müssen Sie darauf aufbauen und die _positive Verstärkung_ nutzen.

![Peter und der Wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.de.png)

> Peter und seine Freunde müssen dem hungrigen Wolf entkommen! Bild von [Jen Looper](https://twitter.com/jenlooper)

## Regionales Thema: Peter und der Wolf (Russland)

[Peter und der Wolf](https://de.wikipedia.org/wiki/Peter_und_der_Wolf) ist ein musikalisches Märchen, geschrieben von dem russischen Komponisten [Sergei Prokofjew](https://de.wikipedia.org/wiki/Sergei_Prokofjew). Es ist die Geschichte des jungen Pioniers Peter, der mutig sein Haus verlässt, um auf der Waldlichtung den Wolf zu jagen. In diesem Abschnitt werden wir maschinelle Lernalgorithmen trainieren, die Peter helfen:

- **Die Umgebung erkunden** und eine optimale Navigationskarte erstellen.
- **Lernen**, wie man ein Skateboard benutzt und darauf balanciert, um sich schneller fortzubewegen.

[![Peter und der Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klicken Sie auf das Bild oben, um Peter und der Wolf von Prokofjew zu hören.

## Reinforcement Learning

In den vorherigen Abschnitten haben Sie zwei Beispiele für maschinelle Lernprobleme gesehen:

- **Überwacht**, bei dem wir Datensätze haben, die mögliche Lösungen für das Problem vorschlagen, das wir lösen möchten. [Klassifikation](../4-Classification/README.md) und [Regression](../2-Regression/README.md) sind Aufgaben des überwachten Lernens.
- **Unüberwacht**, bei dem wir keine gelabelten Trainingsdaten haben. Das Hauptbeispiel für unüberwachtes Lernen ist [Clustering](../5-Clustering/README.md).

In diesem Abschnitt führen wir Sie in eine neue Art von Lernproblem ein, das keine gelabelten Trainingsdaten erfordert. Es gibt mehrere Arten solcher Probleme:

- **[Semi-überwachtes Lernen](https://de.wikipedia.org/wiki/Semi-überwachtes_Lernen)**, bei dem wir viele ungelabelte Daten haben, die verwendet werden können, um das Modell vorzutrainieren.
- **[Reinforcement Learning](https://de.wikipedia.org/wiki/Reinforcement_Learning)**, bei dem ein Agent lernt, sich zu verhalten, indem er Experimente in einer simulierten Umgebung durchführt.

### Beispiel - Computerspiel

Angenommen, Sie möchten einem Computer beibringen, ein Spiel zu spielen, wie Schach oder [Super Mario](https://de.wikipedia.org/wiki/Super_Mario). Damit der Computer ein Spiel spielen kann, muss er vorhersagen, welchen Zug er in jedem Spielzustand machen soll. Obwohl dies wie ein Klassifikationsproblem erscheinen mag, ist es keines – denn wir haben keinen Datensatz mit Zuständen und entsprechenden Aktionen. Während wir einige Daten wie bestehende Schachpartien oder Aufzeichnungen von Spielern, die Super Mario spielen, haben könnten, ist es wahrscheinlich, dass diese Daten nicht ausreichend viele mögliche Zustände abdecken.

Anstatt nach bestehenden Spieldaten zu suchen, basiert **Reinforcement Learning** (RL) auf der Idee, dass der Computer *das Spiel viele Male spielt* und das Ergebnis beobachtet. Um Reinforcement Learning anzuwenden, benötigen wir daher zwei Dinge:

- **Eine Umgebung** und **einen Simulator**, die es uns ermöglichen, ein Spiel viele Male zu spielen. Dieser Simulator würde alle Spielregeln sowie mögliche Zustände und Aktionen definieren.

- **Eine Belohnungsfunktion**, die uns sagt, wie gut wir bei jedem Zug oder Spiel abgeschnitten haben.

Der Hauptunterschied zwischen anderen Arten des maschinellen Lernens und RL besteht darin, dass wir bei RL normalerweise nicht wissen, ob wir gewinnen oder verlieren, bis wir das Spiel beendet haben. Daher können wir nicht sagen, ob ein bestimmter Zug allein gut oder schlecht ist – wir erhalten die Belohnung erst am Ende des Spiels. Unser Ziel ist es, Algorithmen zu entwickeln, die es uns ermöglichen, ein Modell unter unsicheren Bedingungen zu trainieren. Wir werden einen RL-Algorithmus namens **Q-Learning** kennenlernen.

## Lektionen

1. [Einführung in Reinforcement Learning und Q-Learning](1-QLearning/README.md)
2. [Verwendung einer Gym-Simulationsumgebung](2-Gym/README.md)

## Credits

"Einführung in Reinforcement Learning" wurde mit ♥️ geschrieben von [Dmitry Soshnikov](http://soshnikov.com)

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.