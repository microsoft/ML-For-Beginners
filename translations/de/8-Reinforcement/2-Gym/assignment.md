<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-03T22:00:43+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "de"
}
-->
# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) wurde so konzipiert, dass alle Umgebungen dieselbe API bereitstellen – d.h. dieselben Methoden `reset`, `step` und `render` sowie dieselben Abstraktionen von **Action Space** und **Observation Space**. Daher sollte es möglich sein, dieselben Reinforcement-Learning-Algorithmen mit minimalen Codeänderungen an verschiedene Umgebungen anzupassen.

## Eine Mountain-Car-Umgebung

Die [Mountain-Car-Umgebung](https://gym.openai.com/envs/MountainCar-v0/) enthält ein Auto, das in einem Tal feststeckt:

Das Ziel ist es, aus dem Tal herauszukommen und die Fahne zu erreichen, indem man bei jedem Schritt eine der folgenden Aktionen ausführt:

| Wert | Bedeutung |
|---|---|
| 0 | Nach links beschleunigen |
| 1 | Nicht beschleunigen |
| 2 | Nach rechts beschleunigen |

Der Hauptkniff bei diesem Problem ist jedoch, dass der Motor des Autos nicht stark genug ist, um den Berg in einem einzigen Anlauf zu erklimmen. Daher besteht die einzige Möglichkeit, erfolgreich zu sein, darin, hin- und herzufahren, um Schwung aufzubauen.

Der Observation Space besteht aus nur zwei Werten:

| Nr. | Beobachtung  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Position des Autos | -1.2 | 0.6 |
|  1  | Geschwindigkeit des Autos | -0.07 | 0.07 |

Das Belohnungssystem für das Mountain Car ist ziemlich knifflig:

 * Eine Belohnung von 0 wird vergeben, wenn der Agent die Fahne (Position = 0.5) auf dem Berggipfel erreicht.
 * Eine Belohnung von -1 wird vergeben, wenn die Position des Agenten kleiner als 0.5 ist.

Die Episode endet, wenn die Position des Autos größer als 0.5 ist oder die Episodenlänge 200 überschreitet.

## Anweisungen

Passen Sie unseren Reinforcement-Learning-Algorithmus an, um das Mountain-Car-Problem zu lösen. Beginnen Sie mit dem bestehenden Code in [notebook.ipynb](notebook.ipynb), ersetzen Sie die Umgebung, ändern Sie die Funktionen zur Diskretisierung des Zustands und versuchen Sie, den bestehenden Algorithmus mit minimalen Codeänderungen zu trainieren. Optimieren Sie das Ergebnis, indem Sie die Hyperparameter anpassen.

> **Hinweis**: Es wird wahrscheinlich notwendig sein, die Hyperparameter anzupassen, damit der Algorithmus konvergiert.

## Bewertungskriterien

| Kriterien | Vorbildlich | Angemessen | Verbesserungswürdig |
| --------- | ----------- | ---------- | -------------------- |
|           | Der Q-Learning-Algorithmus wurde erfolgreich aus dem CartPole-Beispiel übernommen, mit minimalen Codeänderungen, und ist in der Lage, das Problem des Erreichens der Fahne in weniger als 200 Schritten zu lösen. | Ein neuer Q-Learning-Algorithmus wurde aus dem Internet übernommen, aber gut dokumentiert; oder ein bestehender Algorithmus wurde übernommen, erreicht jedoch nicht die gewünschten Ergebnisse. | Der Student war nicht in der Lage, erfolgreich einen Algorithmus zu übernehmen, hat jedoch wesentliche Schritte zur Lösung unternommen (z. B. Implementierung der Zustandsdiskretisierung, Q-Table-Datenstruktur usw.). |

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.