<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-03T21:59:42+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "de"
}
-->
# Eine realistischere Welt

In unserer Situation konnte Peter sich fast ohne Müdigkeit oder Hunger bewegen. In einer realistischeren Welt muss er sich von Zeit zu Zeit hinsetzen und ausruhen sowie sich ernähren. Lassen Sie uns unsere Welt realistischer gestalten, indem wir die folgenden Regeln umsetzen:

1. Durch das Bewegen von einem Ort zum anderen verliert Peter **Energie** und sammelt **Müdigkeit**.
2. Peter kann mehr Energie gewinnen, indem er Äpfel isst.
3. Peter kann Müdigkeit loswerden, indem er sich unter einem Baum oder auf dem Gras ausruht (d. h. indem er auf ein Feld mit einem Baum oder Gras - grünes Feld - geht).
4. Peter muss den Wolf finden und töten.
5. Um den Wolf zu töten, muss Peter bestimmte Energie- und Müdigkeitslevel haben, andernfalls verliert er den Kampf.

## Anweisungen

Verwenden Sie das ursprüngliche [notebook.ipynb](notebook.ipynb) Notebook als Ausgangspunkt für Ihre Lösung.

Modifizieren Sie die Belohnungsfunktion gemäß den Spielregeln, führen Sie den Reinforcement-Learning-Algorithmus aus, um die beste Strategie zum Gewinnen des Spiels zu erlernen, und vergleichen Sie die Ergebnisse des Zufallswegs mit Ihrem Algorithmus in Bezug auf die Anzahl der gewonnenen und verlorenen Spiele.

> **Note**: In Ihrer neuen Welt ist der Zustand komplexer und umfasst neben der menschlichen Position auch Müdigkeits- und Energielevel. Sie können den Zustand als ein Tupel (Board, Energie, Müdigkeit) darstellen oder eine Klasse für den Zustand definieren (Sie können diese auch von `Board` ableiten) oder sogar die ursprüngliche `Board`-Klasse in [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py) modifizieren.

In Ihrer Lösung behalten Sie bitte den Code für die Zufallsweg-Strategie bei und vergleichen die Ergebnisse Ihres Algorithmus am Ende mit dem Zufallsweg.

> **Note**: Sie müssen möglicherweise Hyperparameter anpassen, damit es funktioniert, insbesondere die Anzahl der Epochen. Da der Erfolg des Spiels (Kampf gegen den Wolf) ein seltenes Ereignis ist, können Sie mit einer deutlich längeren Trainingszeit rechnen.

## Bewertungskriterien

| Kriterien | Vorbildlich                                                                                                                                                                                             | Angemessen                                                                                                                                                                                | Verbesserungswürdig                                                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
|           | Ein Notebook wird präsentiert mit der Definition der neuen Weltregeln, Q-Learning-Algorithmus und einigen textlichen Erklärungen. Q-Learning kann die Ergebnisse im Vergleich zum Zufallsweg deutlich verbessern. | Ein Notebook wird präsentiert, Q-Learning ist implementiert und verbessert die Ergebnisse im Vergleich zum Zufallsweg, jedoch nicht signifikant; oder das Notebook ist schlecht dokumentiert und der Code ist nicht gut strukturiert. | Es wurden einige Versuche unternommen, die Regeln der Welt neu zu definieren, aber der Q-Learning-Algorithmus funktioniert nicht oder die Belohnungsfunktion ist nicht vollständig definiert. |

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.